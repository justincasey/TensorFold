//! GLM's sparse latent attention (chunks and merge) on one GPU, bf16 cache against --kv-cache fp8's: a 2,048-row prompt
//! chunk and a 4-row decode window, each row on 2,048 random keys of a 32K cache, 32 heads (a rank at TP=2).
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;
const latent = glm.latent;

const heads = 32;
const lw = 512;
const keys = 32768;
const w = 2048;
const reps = 10;

pub fn run(gpu: Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var gm = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer gm.unload();
    const glue = try glm.glue.Kernels.load(gm);
    const set: glm.glue.Set = .{ .glue = &glue };
    var prng = std.Random.DefaultPrng.init(5);
    const rng = prng.random();
    var bufs: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer {
        for (bufs.items) |*b| b.free();
        bufs.deinit(a);
    }
    const rows_max = 2048;
    const host = try a.alloc(u16, @max(keys, rows_max * heads) * lw);
    defer a.free(host);
    for (host) |*v| v.* = @truncate(@as(u32, @bitCast(rng.float(f32) * 2 - 1)) >> 16);
    const tok = try a.alloc(i32, rows_max * w);
    defer a.free(tok);
    for (tok) |*t| t.* = @intCast(rng.uintLessThan(u32, keys));
    const cnt = try a.alloc(i32, rows_max);
    defer a.free(cnt);
    @memset(cnt, w);
    const nch = latent.chunksFor(w);
    var ptrs: [10]u64 = undefined;
    const sizes = [_]usize{ keys * lw * 2, keys * lw, keys * 4, rows_max * heads * lw * 2, rows_max * w * 4, rows_max * 4, nch * rows_max * heads * lw * 4, nch * rows_max * heads * 4, nch * rows_max * heads * 4, rows_max * heads * lw * 2 };
    for (sizes, &ptrs) |n, *p| {
        try bufs.append(a, try cuda.DeviceBuffer.alloc(gpu.d, n));
        p.* = bufs.items[bufs.items.len - 1].ptr;
    }
    const lc, const lc8, const ls, const qa, const dtok, const dcnt, const po, const pm, const pl, const out = ptrs;
    try bufs.items[0].upload(0, std.mem.sliceAsBytes(host[0 .. keys * lw]));
    try bufs.items[3].upload(0, std.mem.sliceAsBytes(host[0 .. rows_max * heads * lw]));
    try bufs.items[4].upload(0, std.mem.sliceAsBytes(tok));
    try bufs.items[5].upload(0, std.mem.sliceAsBytes(cnt));
    var zero = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&@as(i32, 0)));
    defer zero.free();
    try latent.latentWrite(set, s, lc, lw, lc8, ls, zero.ptr, keys, lw);
    var t0 = try cuda.Event.init(gpu.d, true);
    defer t0.deinit();
    var t1 = try cuda.Event.init(gpu.d, true);
    defer t1.deinit();
    const p: latent.Partials = .{ .po = po, .pm = pm, .pl = pl };
    for ([_]usize{ 2048, 4 }) |rows| {
        var ms: [2]f64 = undefined;
        for (0..2) |fp8| {
            const cache = if (fp8 == 1) lc8 else lc;
            const scales = if (fp8 == 1) ls else 0;
            for (0..2) |round| {
                try t0.record(s);
                for (0..reps) |_| try latent.sparseAttention(set, s, qa, cache, scales, dtok, dcnt, w, p, out, rows, heads, lw, 0.0625, latent.headBlock(rows));
                try t1.record(s);
                try t1.synchronize();
                if (round == 1) ms[fp8] = @as(f64, try cuda.Event.elapsedMs(t0, t1)) / reps;
            }
        }
        std.debug.print("RESULT glm-attn-bench {d} rows x {d} keys of {d}, {d} heads: bf16 {d:.3} ms, fp8 {d:.3} ms ({d:.2}x)\n", .{ rows, w, keys, heads, ms[0], ms[1], ms[0] / ms[1] });
    }
    check.pass("glm-attn-bench: sparse latent attention timed on both caches", .{});
}
