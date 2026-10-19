//! GLM's EXL3 routed experts on one prompt chunk at a RED-SNOW rank's shapes: ms a call and an output digest.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

const E = 288; // routed experts (the shared one is id E, skipped)
const D = 4096; // dims
const N = 1024; // the rank's width of each expert
const rows = glm.exl3.chunk_rows;
const slots = glm.exl3.top_k + 1;
const reps = 10;
/// RED-SNOW's (gate, up, down) half-bit widths and how many expert layers take each (the five common triples).
const triples = [_][3]i32{ .{ 10, 10, 8 }, .{ 12, 12, 8 }, .{ 8, 8, 8 }, .{ 12, 12, 10 }, .{ 12, 12, 12 } };
const counts = [_]usize{ 5016, 2703, 2563, 1315, 622 };

fn put(gpu: Gpu, bufs: *std.ArrayList(cuda.DeviceBuffer), bytes: []const u8) !u64 {
    const b = try cuda.DeviceBuffer.fromHost(gpu.d, bytes);
    try bufs.append(gpu.gpa, b);
    return b.ptr;
}

/// Random trellis words for every expert at projection `p`'s widths, laid out as the loader does; the pointer table.
fn trellis(gpu: Gpu, bufs: *std.ArrayList(cuda.DeviceBuffer), rng: std.Random, k2s: []const i32, tiles: usize) !u64 {
    var total: usize = 0;
    for (k2s) |k2| total += tiles * 16 * @as(usize, @intCast(k2));
    const host = try gpu.gpa.alloc(u64, total / 8);
    defer gpu.gpa.free(host);
    var z = rng.int(u64);
    for (host) |*v| {
        z +%= 0x9e3779b97f4a7c15;
        var m = z;
        m = (m ^ (m >> 30)) *% 0xbf58476d1ce4e5b9;
        m = (m ^ (m >> 27)) *% 0x94d049bb133111eb;
        v.* = m ^ (m >> 31);
    }
    const base = try put(gpu, bufs, std.mem.sliceAsBytes(host));
    var ptrs: [E]u64 = undefined;
    var at: u64 = 0;
    for (k2s, 0..) |k2, e| {
        ptrs[e] = base + at;
        at += tiles * 16 * @as(u64, @intCast(k2));
    }
    return put(gpu, bufs, std.mem.sliceAsBytes(&ptrs));
}

fn signs(gpu: Gpu, bufs: *std.ArrayList(cuda.DeviceBuffer), rng: std.Random, n: usize, scale: f16) !u64 {
    const out = try gpu.gpa.alloc(f16, n);
    defer gpu.gpa.free(out);
    for (out) |*v| v.* = if (rng.boolean()) scale else -scale;
    return put(gpu, bufs, std.mem.sliceAsBytes(out));
}

/// Expert popularity: log-normal weights of spread `sigma`, 8 distinct picks a row by weight, the shared id last.
fn picks(a: std.mem.Allocator, rng: std.Random, sigma: f64) ![]i32 {
    var w: [E]f64 = undefined;
    var sum: f64 = 0;
    for (&w) |*v| {
        v.* = @exp(sigma * rng.floatNorm(f64));
        sum += v.*;
    }
    const out = try a.alloc(i32, rows * slots);
    for (0..rows) |row| {
        var seen: [E]bool = @splat(false);
        for (0..slots - 1) |j| {
            const e: usize = while (true) {
                var t = rng.float(f64) * sum;
                const c = for (w, 0..) |v, i| {
                    t -= v;
                    if (t <= 0) break i;
                } else E - 1;
                if (!seen[c]) break c;
            };
            seen[e] = true;
            out[row * slots + j] = @intCast(e);
        }
        out[row * slots + slots - 1] = E;
    }
    return out;
}

pub fn run(gpu: Gpu, sigma_arg: ?[]const u8) !void {
    const a = gpu.gpa;
    const sigma = if (sigma_arg) |s| try std.fmt.parseFloat(f64, s) else 1.0;
    var bufs: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer {
        for (bufs.items) |*b| b.free();
        bufs.deinit(a);
    }
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var pm = try cuda.Module.load(gpu.d, cuda.kernels.experts);
    defer pm.unload();
    var k = try glm.exl3.Kernels.load(gpu.d);
    defer k.unload();
    var ek = try glm.experts.Kernels.load(gpu.d, pm, @intCast(try gpu.ctx.attribute(.multiprocessor_count)));
    defer ek.unload();
    var prng = std.Random.DefaultPrng.init(5);
    const rng = prng.random();
    var ks: [3][E]i32 = undefined;
    var order: [E]usize = undefined;
    for (&order, 0..) |*o, i| o.* = i;
    rng.shuffle(usize, &order);
    var all: usize = 0;
    for (counts) |c| all += c;
    for (order, 0..) |e, i| {
        var t: usize = 0;
        var acc: usize = counts[0];
        while (i * all >= acc * E) : (acc += counts[t]) t += 1;
        for (0..3) |p| ks[p][e] = triples[t][p];
    }
    const w: glm.exl3.Experts = .{
        .gt = try trellis(gpu, &bufs, rng, &ks[0], D / 16 * (N / 16)),
        .ut = try trellis(gpu, &bufs, rng, &ks[1], D / 16 * (N / 16)),
        .dt = try trellis(gpu, &bufs, rng, &ks[2], N / 16 * (D / 16)),
        .suh_g = try signs(gpu, &bufs, rng, E * D, 1.0),
        .suh_u = try signs(gpu, &bufs, rng, E * D, 1.0),
        .svh_g = try signs(gpu, &bufs, rng, E * N, 0.02),
        .svh_u = try signs(gpu, &bufs, rng, E * N, 0.02),
        .suh_d = try signs(gpu, &bufs, rng, E * N, 1.0),
        .svh_d = try signs(gpu, &bufs, rng, E * D, 0.02),
        .gk2 = try put(gpu, &bufs, std.mem.sliceAsBytes(&ks[0])),
        .uk2 = try put(gpu, &bufs, std.mem.sliceAsBytes(&ks[1])),
        .dk2 = try put(gpu, &bufs, std.mem.sliceAsBytes(&ks[2])),
        .book = .mul1,
        .uniform = 0,
        .widest = 12,
        .shx = false,
    };
    const hx = try a.alloc(u16, rows * D);
    defer a.free(hx);
    for (hx) |*v| v.* = @truncate(@as(u32, @bitCast(@as(f32, @floatCast(rng.floatNorm(f64))))) >> 16);
    const x = try put(gpu, &bufs, std.mem.sliceAsBytes(hx));
    const pk = try picks(a, rng, sigma);
    defer a.free(pk);
    var per: [E]usize = @splat(0);
    for (0..rows) |row| for (0..slots - 1) |j| {
        per[@intCast(pk[row * slots + j])] += 1;
    };
    var items: usize = 0;
    var most: usize = 0;
    var used: usize = 0;
    for (per) |c| {
        items += (c + glm.experts.staged_tile - 1) / glm.experts.staged_tile;
        most = @max(most, c);
        used += @intFromBool(c > 0);
    }
    std.debug.print("RESULT glm-exl3-bench routing sigma {d:.2}: {d} experts used, most {d} pairs, {d} 64-pair items\n", .{ sigma, used, most, items });
    const P = rows * glm.exl3.top_k;
    const tile: usize = glm.experts.staged_tile;
    const zeros = struct {
        fn f(g: Gpu, b: *std.ArrayList(cuda.DeviceBuffer), n: usize) !u64 {
            const d = try cuda.DeviceBuffer.alloc(g.d, @max(n, 16));
            try b.append(g.gpa, d);
            return d.ptr;
        }
    }.f;
    const plan: cuda.grouped.Plan = .{ .members = try zeros(gpu, &bufs, P * 4), .items = try zeros(gpu, &bufs, cuda.grouped.maxItems(P, E, tile) * 12), .counts = try zeros(gpu, &bufs, 8), .rank = try zeros(gpu, &bufs, P * 4), .hist = try zeros(gpu, &bufs, (P + 1023) / 1024 * E * 4) };
    const sz = glm.exl3.scratchBytes(rows, D, N, true, false, .mul1, 0);
    const sc: glm.exl3.Scratch = .{ .picks = try zeros(gpu, &bufs, sz[0]), .xg = try zeros(gpu, &bufs, sz[1]), .xu = try zeros(gpu, &bufs, sz[2]), .z = try zeros(gpu, &bufs, sz[3]) };
    const ey_at = bufs.items.len;
    const ey = try zeros(gpu, &bufs, rows * slots * D * 4);
    const xd = try zeros(gpu, &bufs, P * N * 2);
    const dpk = try put(gpu, &bufs, std.mem.sliceAsBytes(pk));
    var t0 = try cuda.Event.init(gpu.d, true);
    defer t0.deinit();
    var t1 = try cuda.Event.init(gpu.d, true);
    defer t1.deinit();
    for (0..2) |_| try glm.exl3.run(&k, &ek, s, w, sc, plan, x, D, dpk, slots, rows, E, D, N, 7.0, true, ey, xd);
    try t0.record(s);
    for (0..reps) |_| try glm.exl3.run(&k, &ek, s, w, sc, plan, x, D, dpk, slots, rows, E, D, N, 7.0, true, ey, xd);
    try t1.record(s);
    try t1.synchronize();
    const ms = @as(f64, try cuda.Event.elapsedMs(t0, t1)) / reps;
    const out = try a.alloc(u8, rows * slots * D * 4);
    defer a.free(out);
    try bufs.items[ey_at].download(0, out);
    var h = std.hash.Wyhash.init(0);
    const eb = glm.exl3.slotBytes(w, E, D, N, true);
    for (0..rows) |row| h.update(out[row * slots * D * eb ..][0 .. (slots - 1) * D * eb]);
    std.debug.print("RESULT glm-exl3-bench {d} rows, {d} experts, D {d}, N {d}: {d:.3} ms a call, routed slots digest {x:0>16}\n", .{ rows, E, D, N, ms, h.final() });
}
