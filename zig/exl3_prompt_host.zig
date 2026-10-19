//! GLM's actual common EXL3 dispatch with recording driver calls, no device or image.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm");
const lanes = @import("lanes");
const ex = cuda.exl3_experts;
const Capture = struct {
    const Call = struct { id: usize, grid: [3]u32, block: u32, shared: u32, args: [20]u64 = @splat(0) };
    var calls: [256]Call = undefined;
    var count: usize = 0;
    var copies: [16]cuda.abi.Memcpy2D = undefined;
    var copied: usize = 0;
    fn launch(f: cuda.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, _: c_uint, _: c_uint, smem: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        calls[count] = .{ .id = @intFromPtr(f), .grid = .{ gx, gy, gz }, .block = bx, .shared = smem };
        if (@intFromPtr(f) >= 11 and @intFromPtr(f) <= 13) {
            const sizes: []const u8 = switch (@intFromPtr(f)) {
                11 => &.{ 8, 4, 8, 8, 4 },
                12 => &.{ 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 4 },
                else => &.{ 8, 8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4 },
            };
            for (sizes, 0..) |size, i| {
                const bytes: [*]const u8 = @ptrCast(args.?[i].?);
                calls[count].args[i] = if (size == 8) std.mem.readInt(u64, bytes[0..8], .little) else std.mem.readInt(u32, bytes[0..4], .little);
            }
        }
        count += 1;
        return cuda.abi.success;
    }
    fn copy(m: *const cuda.abi.Memcpy2D, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        copies[copied] = m.*;
        copied += 1;
        return cuda.abi.success;
    }
    fn reset() void {
        count = 0;
        copied = 0;
    }
    fn by(id: usize) []const Call {
        var first: usize = count;
        var last: usize = count;
        for (calls[0..count], 0..) |c, i| if (c.id == id) {
            if (first == count) first = i;
            last = i + 1;
        };
        return calls[first..last];
    }
};
fn function(d: *const cuda.Driver, id: usize) cuda.Function {
    return .{ .d = d, .handle = @ptrFromInt(id) };
}
fn run(rows: usize, prompt: bool, shared: bool) !void {
    return runAt(rows, prompt, shared, .mcg, 8, 8);
}
fn runAt(rows: usize, prompt: bool, shared: bool, book: ex.Codebook, uniform: u32, widest: u32) !void {
    Capture.reset();
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Capture.launch;
    d.api.cuMemcpy2DAsync_v2 = Capture.copy;
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    var k: ex.Experts = .{ .mod = undefined, .group = function(&d, 1), .rot_in = function(&d, 2), .grouped = @splat(function(&d, 3)), .gateup = function(&d, 4), .down_combine = function(&d, 5) };
    if (comptime @hasField(ex.Experts, "prompt")) k.prompt = .{ .rotate = function(&d, 11), .gateup = function(&d, 12), .down = function(&d, 13) };
    const ek: glm.experts.Kernels = .{ .nvfp4 = undefined, .router = .{ .plan_small = function(&d, 20), .plan_rank = function(&d, 21), .plan_offsets = function(&d, 22), .plan_scatter = function(&d, 23) } };
    const w: glm.exl3.Experts = .{ .gt = 100, .ut = 200, .dt = 300, .gk2 = 400, .uk2 = 410, .dk2 = 420, .suh_g = 500, .suh_u = 600, .svh_g = 700, .svh_u = 800, .suh_d = 900, .svh_d = 1000, .book = book, .uniform = uniform, .widest = widest, .shx = shared };
    const sc: glm.exl3.Scratch = .{ .picks = 0x100000000, .xg = 0, .xu = 0, .z = 0 };
    const plan: cuda.grouped.Plan = .{ .items = 10000, .members = 20000, .counts = 30000, .rank = 40000, .hist = 50000 };
    try glm.exl3.run(&k, &ek, s, w, sc, plan, 0x200000000, 520, 0x300000000, 9, rows, 9, 512, 256, 10, prompt, 0x400000000, 0);
}

test "exl3 prompt: dispatch rotates once per row only with verified shared scales, with fused full-K kernels" {
    for ([_]bool{ true, false }) |shared| {
        try run(40, true, shared);
        try std.testing.expect(Capture.by(12).len > 0);
        try std.testing.expect(Capture.by(13).len > 0);
        try std.testing.expectEqual(@as(usize, 0), Capture.by(3).len);
        const rot = Capture.by(if (shared) 11 else 2);
        try std.testing.expect(rot.len > 0);
        try std.testing.expectEqual(@as(u32, if (shared) 40 else 320), rot[0].grid[0]);
        try std.testing.expectEqual(@as(u32, if (shared) 1 else 2), rot[0].grid[2]);
        try std.testing.expectEqual(@as(u32, 256), Capture.by(12)[0].block);
        try std.testing.expectEqual(@as(u32, 14), Capture.by(12)[0].grid[1]);
        try std.testing.expectEqual(@as(u64, if (shared) 8 else 1), Capture.by(12)[0].args[15]);
        try std.testing.expectEqual(@as(u64, 420), Capture.by(13)[0].args[2]);
        try std.testing.expectEqual(ex.prompt.shared(8, if (shared) 64 else 128), Capture.by(12)[0].shared);
        try std.testing.expectEqual(@as(u64, 0x400000000), Capture.by(13)[0].args[7]);
        try std.testing.expectEqual(@as(u64, 8), Capture.by(13)[0].args[10]);
        try std.testing.expectEqual(@as(u64, 9), Capture.by(13)[0].args[11]);
        try std.testing.expectEqual(@as(usize, 1), Capture.copied);
        try std.testing.expectEqual(@as(usize, 8 * 4), Capture.copies[0].width);
        try std.testing.expectEqual(@as(usize, 9 * 4), Capture.copies[0].src_pitch);
    }
}

test "exl3 prompt: RED-SNOW's mul1 experts at mixed widths take the fused kernels with their width tables and stride" {
    for ([_]bool{ true, false }) |shared| {
        try runAt(40, true, shared, .mul1, 0, 12);
        try std.testing.expectEqual(@as(usize, 0), Capture.by(3).len);
        const gu = Capture.by(12)[0];
        const dn = Capture.by(13)[0];
        try std.testing.expectEqual(@as(u64, 400), gu.args[4]);
        try std.testing.expectEqual(@as(u64, 410), gu.args[5]);
        try std.testing.expectEqual(@as(u64, 2), gu.args[18]);
        try std.testing.expectEqual(@as(u64, 12), gu.args[19]);
        try std.testing.expectEqual(@as(u64, 2), dn.args[12]);
        try std.testing.expectEqual(@as(u64, 12), dn.args[13]);
        try std.testing.expectEqual(ex.prompt.shared(12, if (shared) 64 else 128), gu.shared);
        try std.testing.expectEqual(ex.prompt.shared(12, 64), dn.shared);
        try std.testing.expect(gu.shared <= 50 * 1024); // two blocks an SM at the widest RED-SNOW width
    }
}

test "exl3 prompt: the fused scratch fits in the decode scratch and shared rotation storage is independent of pick count" {
    const decode = glm.exl3.scratchBytes(1024, 512, 256, false, false, .mcg, 8)[0];
    for ([_]bool{ true, false }) |shx| for ([_]ex.Codebook{ .mcg, .mul1 }) |cb| {
        try std.testing.expect(glm.exl3.scratchBytes(1024, 512, 256, true, shx, cb, if (cb == .mcg) 8 else 0)[0] <= decode);
    };
    if (comptime @hasDecl(ex, "prompt")) {
        var l: ex.Layer = .{ .gate = 0, .up = 0, .down = 0, .gate_k2 = 0, .up_k2 = 0, .down_k2 = 0, .suh_g = 0, .suh_u = 0, .svh_g = 0, .svh_u = 0, .suh_d = 0, .svh_d = 0, .dims = 512, .width = 256, .experts = 9, .codebook = .mcg, .uniform_k2 = 8, .shared_input = false };
        const separate = try ex.prompt.bytes(l, 1024, 8);
        l.shared_input = true;
        const bytes = try ex.prompt.bytes(l, 1024, 8);
        try std.testing.expect(bytes < separate and separate < decode);
        const sc = try ex.prompt.scratch(0x1000, l, 1024, 8);
        try std.testing.expectEqual(@as(u64, sc.xg), sc.xu);
        try std.testing.expectEqual(@as(u64, 0x1000 + bytes), sc.xd + 1024 * 8 * 256 * 2);
        try std.testing.expectError(error.Exl3Shape, ex.prompt.bytes(l, std.math.maxInt(usize), 8));
    } else try std.testing.expect(false);
}

test "exl3 prompt: chunked calls keep row, pick and output offsets on both sides of the boundary" {
    const n = glm.exl3.chunk_rows;
    try run(n + 6, true, true);
    try std.testing.expectEqual(@as(usize, 2), Capture.copied);
    try std.testing.expectEqual(@as(usize, n), Capture.copies[0].height);
    try std.testing.expectEqual(@as(usize, 6), Capture.copies[1].height);
    try std.testing.expectEqual(@as(u64, 0x300000000 + n * 9 * 4), Capture.copies[1].src);
    var rotations: usize = 0;
    var downs: usize = 0;
    for (Capture.calls[0..Capture.count]) |call| {
        const at: u64 = if (call.id == 11) rotations else if (call.id == 13) downs else continue;
        if (call.id == 11) {
            try std.testing.expectEqual(@as(u64, 0x200000000 + at * n * 520 * 2), call.args[0]);
            rotations += 1;
        } else {
            try std.testing.expectEqual(@as(u64, 0x400000000 + at * n * 9 * 512 * 2), call.args[7]);
            downs += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 2), rotations);
    try std.testing.expectEqual(@as(usize, 2), downs);
}

test "exl3 prompt: decode retains the existing split-K launches" {
    try run(4, false, true);
    try std.testing.expectEqual(@as(usize, 0), Capture.by(12).len);
    try std.testing.expect(Capture.by(1).len > 0 and Capture.by(2).len > 0 and Capture.by(3).len > 0);
}

test "exl3 prompt: the common dispatch takes mcg and mul1 at widths 2 to 16 and keeps every fallback" {
    const P = ex.prompt;
    var l: ex.Layer = .{ .gate = 0, .up = 0, .down = 0, .gate_k2 = 0, .up_k2 = 0, .down_k2 = 0, .suh_g = 0, .suh_u = 0, .svh_g = 0, .svh_u = 0, .suh_d = 0, .svh_d = 0, .dims = 512, .width = 256, .experts = 9, .codebook = .mcg, .uniform_k2 = 8 };
    try std.testing.expect(P.eligible(l));
    l.uniform_k2 = 0;
    try std.testing.expect(P.eligible(l) and P.widest(l) == 16);
    l.max_k2 = 12;
    try std.testing.expect(P.eligible(l) and P.widest(l) == 12);
    l.max_k2 = 17;
    try std.testing.expect(!P.eligible(l));
    l.max_k2 = 0;
    l.uniform_k2 = 6;
    try std.testing.expect(P.eligible(l) and P.widest(l) == 6);
    l.uniform_k2 = 8;
    l.codebook = .mul1;
    try std.testing.expect(P.eligible(l));
    l.codebook = .inst3;
    try std.testing.expect(!P.eligible(l));
    l.codebook = .mcg;
    l.gu_stride = 16;
    try std.testing.expect(!P.eligible(l));
    l.gu_stride = 0;
    l.dims = 128;
    try std.testing.expect(!P.eligible(l));
    l.dims = 512;
    l.act_mode = 2;
    try std.testing.expect(!P.eligible(l));
    l.act_mode = 1;
    for ([_]usize{ 1, 7, 8, 15, 16, 17, 63, 64, 65, 1023, 1024 }) |rows| for ([_]usize{ 1, 3, 8, 32 }) |slots| for ([_]bool{ false, true }) |shared| {
        l.shared_input = shared;
        const size = try P.bytes(l, rows, slots);
        const sc = try P.scratch(4096, l, rows, slots);
        try std.testing.expect(sc.xg + rows * (if (shared) @as(usize, 1) else slots) * l.dims * 2 <= sc.xd);
        try std.testing.expect(sc.xd + rows * slots * l.width * 2 <= 4096 + size);
        for ([_]u64{ sc.xg, sc.xu, sc.xd }) |p| try std.testing.expectEqual(@as(u64, 0), p % 256);
    };
    try std.testing.expectError(error.Exl3Shape, P.bytes(l, 1, 0));
    try std.testing.expectError(error.Exl3Shape, P.bytes(l, 1, 33));
    try std.testing.expectError(error.Exl3Shape, P.scratch(std.math.maxInt(u64) - 255, l, 1, 1));
}

test "exl3 prompt: the common entry takes as many experts as the grouped router counts, and refuses one more" {
    const P = ex.prompt;
    var l: ex.Layer = .{ .gate = 0, .up = 0, .down = 0, .gate_k2 = 0, .up_k2 = 0, .down_k2 = 0, .suh_g = 0, .suh_u = 0, .svh_g = 0, .svh_u = 0, .suh_d = 0, .svh_d = 0, .dims = 512, .width = 256, .experts = cuda.grouped.max_experts, .codebook = .mcg, .uniform_k2 = 8 };
    try std.testing.expectEqual(@as(usize, 1024), cuda.grouped.max_experts);
    for ([_]usize{ 1, 1024, 2048 }) |rows| {
        l.experts = cuda.grouped.max_experts;
        _ = try P.scratch(4096, l, rows, 8);
        l.experts = cuda.grouped.max_experts + 1;
        try std.testing.expectError(error.Exl3Shape, P.bytes(l, rows, 8));
        try std.testing.expectError(error.Exl3Shape, P.scratch(4096, l, rows, 8));
    }
}

fn reply(drafts: bool, neighbors: usize, seed: ?u64) ![32]u32 {
    const a = std.testing.allocator;
    var model: glm.engine.Engine = undefined;
    model.mtp = true;
    var adapter: glm.lanes.Lanes = .{ .x = .{ .e = &model, .link = undefined } };
    var cfg = try lanes.Config.init(a, adapter.facts(), glm.lanes.max_rows, glm.engine.max_drafts);
    defer cfg.deinit(a);
    cfg.timed = false;
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var engine = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var streams: [8]lanes.Stream = undefined;
    var made: usize = 0;
    defer for (streams[0..made]) |*s| s.deinit(a);
    for (streams[0 .. neighbors + 1], 0..) |*s, i| {
        s.* = try lanes.Stream.init(a, .{ .id = "GLM EXL3", .prompt = &.{ 3, 1, 4, 1, 5, 9, 2, 6 }, .max_new = 32, .drafts = drafts, .sampling = if (seed) |v| .{ .seed = v + i, .temperature = 0.8 } else null, .loop_guard = false });
        made += 1;
        try engine.addStream(s);
    }
    while (engine.activeCount() > 0) try engine.step();
    var out: [32]u32 = undefined;
    @memcpy(&out, streams[0].emitted());
    if (drafts) try std.testing.expect(streams[0].drafted > 0);
    return out;
}

test "exl3 prompt: GLM admission facts through shared lanes keep drafted plain and concurrent solo identities" {
    for ([_]?u64{ null, 17, 91 }) |seed| {
        const plain = try reply(false, 0, seed);
        for ([_]bool{ false, true }) |drafts| for ([_]usize{ 0, 1, 3, 7 }) |others| try std.testing.expectEqual(plain, try reply(drafts, others, seed));
    }
}
