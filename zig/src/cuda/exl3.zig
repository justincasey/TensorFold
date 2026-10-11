//! The EXL3 decode-row linear (exl3/linear.cu's device code): the plan, a layer's tensors, and the two kernels.
const std = @import("std");
const driver = @import("driver.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const kernels = @import("kernels.zig");

const Driver = driver.Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

/// x's and y's element types (linear.cu DType).
pub const DType = enum(i32) { f16 = 0, bf16 = 1, f32 = 2 };

/// The three codebooks (exl3_format.zig's, numbered as the kernel's).
pub const Codebook = enum(u3) { inst3 = 0, mcg = 1, mul1 = 2 };

/// One compiled kernel: 2 * bits halves, a codebook, and the warps one block runs.
pub const Variant = struct { halves: u32, codebook: Codebook, warps: u32 };

/// Every instantiation of exl3_linear.cu: mul1 for 4 to 8 bits and the 4.5/5.5 half steps, mcg and inst3 at 4 bits.
pub const variants = [_]Variant{
    .{ .halves = 8, .codebook = .mul1, .warps = 4 },
    .{ .halves = 8, .codebook = .mul1, .warps = 8 },
    .{ .halves = 10, .codebook = .mul1, .warps = 4 },
    .{ .halves = 10, .codebook = .mul1, .warps = 8 },
    .{ .halves = 12, .codebook = .mul1, .warps = 4 },
    .{ .halves = 12, .codebook = .mul1, .warps = 8 },
    .{ .halves = 14, .codebook = .mul1, .warps = 4 },
    .{ .halves = 14, .codebook = .mul1, .warps = 8 },
    .{ .halves = 16, .codebook = .mul1, .warps = 4 },
    .{ .halves = 16, .codebook = .mul1, .warps = 8 },
    .{ .halves = 9, .codebook = .mul1, .warps = 4 },
    .{ .halves = 9, .codebook = .mul1, .warps = 8 },
    .{ .halves = 11, .codebook = .mul1, .warps = 4 },
    .{ .halves = 11, .codebook = .mul1, .warps = 8 },
    .{ .halves = 8, .codebook = .mcg, .warps = 4 },
    .{ .halves = 8, .codebook = .mcg, .warps = 8 },
    .{ .halves = 8, .codebook = .inst3, .warps = 4 },
    .{ .halves = 8, .codebook = .inst3, .warps = 8 },
};

/// The kernel's extern "C" name for an instantiation (linear.cu's TF_EXL3_LINEAR macro).
pub fn symbol(comptime halves: u32, comptime codebook: Codebook, comptime warps: u32) [:0]const u8 {
    return std.fmt.comptimePrint("tf_exl3_linear_{d}_{d}_{d}", .{ halves, @intFromEnum(codebook), warps });
}

pub const rot_in_symbol = "tf_exl3_rot_in";

pub fn variantIndex(halves: u32, codebook: Codebook, warps: u32) ?usize {
    for (variants, 0..) |v, i| {
        if (v.halves == halves and v.codebook == codebook and v.warps == warps) return i;
    }
    return null;
}

/// What one shape launches, from the shape alone so a row's bits never depend on the row count (linear.py plan):
/// the K splits, and the warps a block runs. nb < 64 layers chase the block count first, wider ones use 8 warps;
/// then K must split into whole tiles with at least `min_tiles` per warp, else the splits or the warps shrink.
pub fn plan(k: usize, n: usize, blocks: usize, min_tiles: usize) struct { sk: u32, wk: u32 } {
    const kt = k / 16;
    const nb = n / 128;
    var sk: usize = 1;
    var wk: usize = if (nb >= 64) 8 else 4;
    if (nb < 64) {
        while (nb * sk < blocks and sk < 64) sk *= 2;
    }
    while ((wk > 4 or sk > 1) and (kt % (sk * wk) != 0 or kt / (sk * wk) < min_tiles)) {
        if (wk > 4) wk = 4 else if (sk > 1) sk /= 2 else break;
    }
    return .{ .sk = @intCast(sk), .wk = @intCast(wk) };
}

/// Words between k tiles, and between 128-column blocks, of one layout: strips [N/128, K/16, 8, W] or the stored
/// [K/16, N/16, W] (whose 128-column blocks are 8 back-to-back tiles, so 8 * W apart).
pub fn strides(k: usize, n: usize, halves: usize, strips: bool) struct { k: usize, nb: usize } {
    const words = halves * 4;
    return if (strips)
        .{ .k = 8 * words, .nb = (k / 16) * 8 * words }
    else
        .{ .k = (n / 16) * words, .nb = 8 * words };
}

/// One layer's tensors on the device: the trellis words, the Hadamard scales, the optional bias, shape and layout.
pub const Weight = struct {
    trellis: u64, // u32 words, `stride_k` and `stride_nb` apart
    suh: u64, // fp16 [k]
    svh: u64, // fp16 [n]
    bias: u64 = 0, // fp16 [n], 0 when the layer has none
    k: u32,
    n: u32,
    halves: u32, // 2 * bits, odd for the mul1 half steps
    codebook: Codebook,
    stride_k: u64,
    stride_nb: u64,
};

/// The module and its kernels.
pub const Linear = struct {
    mod: Module,
    fns: [variants.len]Function,
    rot_in: Function,

    pub fn load(d: *const Driver) !Linear {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var l: Linear = .{ .mod = try Module.load(d, kernels.exl3_linear), .fns = undefined, .rot_in = undefined };
        errdefer l.mod.unload();
        l.rot_in = try l.mod.function(rot_in_symbol);
        inline for (0..variants.len) |i| {
            l.fns[i] = try l.mod.function(symbol(variants[i].halves, variants[i].codebook, variants[i].warps));
        }
        return l;
    }

    pub fn unload(l: *Linear) void {
        l.mod.unload();
    }

    /// Shared bytes one call needs: WK * min(m, 8) * 128 floats of warp sums, at most 32 KB.
    pub fn sharedBytes(w: Weight, m: usize) u32 {
        const p = plan(w.k, w.n, 192, 8);
        return @intCast(p.wk * @min(m, 8) * 128 * 4);
    }

    /// Split partials one call needs (fp32 [sk, m, n]) when the plan splits K, else 0.
    pub fn zBytes(w: Weight, m: usize) usize {
        const p = plan(w.k, w.n, 192, 8);
        return if (p.sk > 1) @as(usize, p.sk) * m * w.n * 4 else 0;
    }

    /// Counter bytes one call needs: per pass and block, zeroed (fill8(0), never zeros()).
    pub fn counterBytes(w: Weight) usize {
        return 8 * (w.n / 128) * 4;
    }

    /// y [m, n] = x [m, k] through the weight; `xh` is fp16 [m, k] scratch, `z` the split partials, `counters` zeroed.
    pub fn matmul(l: *const Linear, s: Stream, w: Weight, x: u64, x_dtype: DType, m: usize, y: u64, y_dtype: DType, xh: u64, z: u64, counters: u64) !void {
        std.debug.assert(m >= 1 and m <= 128);
        std.debug.assert(w.k % 16 == 0 and w.n % 128 == 0);
        const p = plan(w.k, w.n, 192, 8);
        const idx = variantIndex(w.halves, w.codebook, p.wk) orelse return error.UnsupportedExl3Width;
        std.debug.assert((w.k / 16) % (p.sk * p.wk) == 0);
        std.debug.assert(p.sk == 1 or z != 0);

        var r: launch_.Args = .{};
        r.add(x);
        r.add(w.suh);
        r.add(xh);
        r.add(@as(i32, @intFromEnum(x_dtype)));
        r.add(@as(i32, @intCast(m)));
        r.add(@as(i32, @intCast(w.k)));
        try launch_.launch(l.rot_in, .{
            .grid = .{ .x = @intCast((w.k / 128 + 3) / 4), .y = @intCast(m), .z = 1 },
            .block = .{ .x = 128, .y = 1, .z = 1 },
            .shared = 0,
        }, s, &r);

        var args: launch_.Args = .{};
        args.add(xh);
        args.add(w.trellis);
        args.add(@as(i64, @intCast(w.stride_k)));
        args.add(@as(i64, @intCast(w.stride_nb)));
        args.add(w.svh);
        args.add(w.bias);
        args.add(y);
        args.add(if (p.sk > 1) z else @as(u64, 0));
        args.add(counters);
        args.add(@as(i32, @intFromEnum(y_dtype)));
        args.add(@as(i32, @intCast(w.n)));
        args.add(@as(i32, @intCast(p.sk)));
        args.add(@as(i32, @intCast(m)));
        args.add(@as(i32, @intCast(w.k)));
        try launch_.launch(l.fns[idx], .{
            .grid = .{ .x = @intCast(w.n / 128 * p.sk), .y = 1, .z = 1 },
            .block = .{ .x = @intCast(p.wk * 32), .y = 1, .z = 1 },
            .shared = sharedBytes(w, m),
        }, s, &args);
    }
};

test "the plan matches linear.py's for known shapes" {
    try std.testing.expectEqual(@as(@TypeOf(plan(0, 0, 192, 8)), .{ .sk = 8, .wk = 4 }), plan(4096, 1024, 192, 8));
    try std.testing.expectEqual(@as(@TypeOf(plan(0, 0, 192, 8)), .{ .sk = 2, .wk = 4 }), plan(1024, 4096, 192, 8));
    try std.testing.expectEqual(@as(@TypeOf(plan(0, 0, 192, 8)), .{ .sk = 4, .wk = 4 }), plan(2048, 512, 192, 8));
    try std.testing.expectEqual(@as(@TypeOf(plan(0, 0, 192, 8)), .{ .sk = 1, .wk = 4 }), plan(512, 1024, 192, 8));
    try std.testing.expectEqual(@as(@TypeOf(plan(0, 0, 192, 8)), .{ .sk = 1, .wk = 4 }), plan(512, 512, 192, 8));
    try std.testing.expectEqual(@as(@TypeOf(plan(0, 0, 192, 8)), .{ .sk = 4, .wk = 4 }), plan(2048, 4096, 192, 8));
}

test "the stored and strips strides of one layout" {
    try std.testing.expectEqual(@as(@TypeOf(strides(0, 0, 0, false)), .{ .k = 4096, .nb = 256 }), strides(4096, 2048, 8, false));
    try std.testing.expectEqual(@as(@TypeOf(strides(0, 0, 0, true)), .{ .k = 256, .nb = 65536 }), strides(4096, 2048, 8, true));
}

test "every variant's symbol is its instantiation's name" {
    try std.testing.expectEqualStrings("tf_exl3_linear_8_2_4", symbol(8, .mul1, 4));
    try std.testing.expectEqualStrings("tf_exl3_linear_11_2_8", symbol(11, .mul1, 8));
    try std.testing.expectEqualStrings("tf_exl3_linear_8_0_4", symbol(8, .inst3, 4));
    try std.testing.expectEqual(@as(?usize, 4), variantIndex(12, .mul1, 4));
    try std.testing.expectEqual(@as(?usize, null), variantIndex(6, .mul1, 4));
}
