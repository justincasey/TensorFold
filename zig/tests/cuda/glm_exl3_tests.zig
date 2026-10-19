//! GLM's EXL3 routed experts (cuda_exl3.run, decode and prompt) against float64 matrices from core's EXL3 decoder.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const qr = @import("quant_ref_tests.zig");
const Gpu = check.Gpu;

const fmt = core.exl3_format;
const h = qr.helpers;
const Rig = qr.RigT;

const E = 12; // routed experts (the shared one is id E, skipped)
const D = 512; // dims
const N = 256; // the rank's width of each expert
const slots = glm.exl3.top_k + 1;
const limit: f32 = 7.0;

/// Each expert's gate, up and down half-bit widths.
const Widths = [E][3]i32;
const mcg4: Widths = @splat(.{ 8, 8, 8 });
/// RED-SNOW's ten width triples (gate and up apart in one), two of them twice.
const redsnow: Widths = .{ .{ 10, 10, 8 }, .{ 12, 12, 8 }, .{ 8, 8, 8 }, .{ 12, 12, 10 }, .{ 12, 12, 12 }, .{ 10, 10, 10 }, .{ 10, 10, 12 }, .{ 8, 8, 10 }, .{ 8, 8, 12 }, .{ 12, 10, 8 }, .{ 10, 10, 8 }, .{ 12, 12, 8 } };
/// Every width from 2 to 16 but 15, which the shared lane decoder misses in decode and prompt alike.
const every: Widths = .{ .{ 16, 16, 16 }, .{ 2, 3, 5 }, .{ 7, 9, 11 }, .{ 13, 16, 14 }, .{ 4, 6, 9 }, .{ 3, 3, 16 }, .{ 11, 12, 13 }, .{ 5, 5, 5 }, .{ 16, 8, 2 }, .{ 9, 9, 9 }, .{ 6, 14, 7 }, .{ 14, 4, 10 } };

/// One weight set: every expert's trellis words and scales as the loader lays them out, and their float64 matrices.
const Set = struct {
    gt: u64,
    ut: u64,
    dt: u64,
    suh_g: u64,
    suh_u: u64,
    svh_g: u64,
    svh_u: u64,
    suh_d: u64,
    svh_d: u64,
    gk2: u64,
    uk2: u64,
    dk2: u64,
    book: cuda.exl3.Codebook,
    uniform: u32,
    widest: u32,
    shx: bool,
    wg: []f64, // [E][D][N]
    wu: []f64,
    wd: []f64, // [E][N][D]

    fn experts(s: Set) glm.exl3.Experts {
        return .{ .gk2 = s.gk2, .uk2 = s.uk2, .dk2 = s.dk2, .gt = s.gt, .ut = s.ut, .dt = s.dt, .suh_g = s.suh_g, .suh_u = s.suh_u, .svh_g = s.svh_g, .svh_u = s.svh_u, .suh_d = s.suh_d, .svh_d = s.svh_d, .book = s.book, .uniform = s.uniform, .widest = s.widest, .shx = s.shx };
    }
};

fn signs(r: *Rig, n: usize, scale: f16) ![]f16 {
    const out = try r.a.alloc(f16, n);
    for (out) |*v| v.* = if (r.rng.boolean()) scale else -scale;
    return out;
}

/// `E` random K x N matrices at widths `ws[e][p]`: trellis words, suh (`shared_suh` if given), svh; device and float64.
fn matrices(r: *Rig, k: usize, n: usize, book: cuda.exl3.Codebook, ws: Widths, p: usize, shared_suh: ?[]const f16, out_suh: *u64, out_svh: *u64, w: []f64) !u64 {
    const tiles = k / 16 * (n / 16);
    var total: usize = 0;
    for (ws) |t| total += tiles * 8 * @as(usize, @intCast(t[p]));
    const tr = try r.a.alloc(i16, total);
    for (tr) |*v| v.* = r.rng.int(i16);
    const suh = try r.a.alloc(f16, E * k);
    const svh = try r.a.alloc(f16, E * n);
    const scratch = try r.a.alloc(f64, 2 * k * n);
    const offs = try r.a.alloc(usize, E);
    var at: usize = 0;
    for (0..E) |e| {
        const halves: u16 = @intCast(ws[e][p]);
        const words = tiles * 8 * @as(usize, halves);
        offs[e] = at;
        if (shared_suh) |s| @memcpy(suh[e * k ..][0..k], s) else @memcpy(suh[e * k ..][0..k], try signs(r, k, 1.0));
        @memcpy(svh[e * n ..][0..n], try signs(r, n, 0.02));
        const cb: fmt.Codebook = if (book == .mul1) .mul1 else .mcg;
        fmt.dequantize(.{ .halves = halves }, cb, tr[at..][0..words], k / 16, n / 16, suh[e * k ..][0..k], svh[e * n ..][0..n], w[e * k * n ..][0 .. k * n], scratch);
        at += words;
    }
    out_suh.* = try r.dev(std.mem.sliceAsBytes(suh));
    out_svh.* = try r.dev(std.mem.sliceAsBytes(svh));
    const base = try r.dev(std.mem.sliceAsBytes(tr));
    const ptrs = try r.a.alloc(u64, E);
    for (ptrs, offs) |*ptr, o| ptr.* = base + o * 2;
    return r.dev(std.mem.sliceAsBytes(ptrs));
}

fn weights(r: *Rig, book: cuda.exl3.Codebook, ws: Widths, one: bool) !Set {
    var s: Set = undefined;
    s.shx = one;
    s.book = book;
    var cols: [3][E]i32 = undefined;
    var widest: i32 = 0;
    var uniform: ?i32 = ws[0][0];
    for (ws, 0..) |t, e| for (t, 0..) |k2, p| {
        cols[p][e] = k2;
        widest = @max(widest, k2);
        if (uniform != null and uniform.? != k2) uniform = null;
    };
    s.widest = @intCast(widest);
    s.uniform = if (uniform) |u| @intCast(u) else 0;
    s.gk2 = try r.dev(std.mem.sliceAsBytes(&cols[0]));
    s.uk2 = try r.dev(std.mem.sliceAsBytes(&cols[1]));
    s.dk2 = try r.dev(std.mem.sliceAsBytes(&cols[2]));
    s.wg = try r.a.alloc(f64, E * D * N);
    s.wu = try r.a.alloc(f64, E * D * N);
    s.wd = try r.a.alloc(f64, E * N * D);
    const rot = if (one) try signs(r, D, 1.0) else null; // one input rotation for every gate and up
    s.gt = try matrices(r, D, N, book, ws, 0, rot, &s.suh_g, &s.svh_g, s.wg);
    s.ut = try matrices(r, D, N, book, ws, 1, rot, &s.suh_u, &s.svh_u, s.wu);
    s.dt = try matrices(r, N, D, book, ws, 2, null, &s.suh_d, &s.svh_d, s.wd);
    return s;
}

/// Each row's 8 distinct routed picks, then the shared expert's id.
fn picks(r: *Rig, rows: usize) ![]i32 {
    const out = try r.a.alloc(i32, rows * slots);
    for (0..rows) |row| {
        var seen: [E]bool = @splat(false);
        for (0..slots - 1) |j| {
            var e = r.rng.uintLessThan(usize, E);
            while (seen[e]) e = r.rng.uintLessThan(usize, E);
            seen[e] = true;
            out[row * slots + j] = @intCast(e);
        }
        out[row * slots + slots - 1] = E;
    }
    return out;
}

const Run = struct { ey: u64, rows: usize };

/// The family's dispatch on `x` (bf16 [rows][D]) and `pk`: ey [rows][slots][D] fp32, the shared slot left as filled.
fn run(r: *Rig, k: *const glm.exl3.Kernels, ek: *const glm.experts.Kernels, s: Set, x: []const u16, pk: []const i32, rows: usize, prompt: bool) !Run {
    const P = rows * glm.exl3.top_k;
    const tile: usize = if (prompt) glm.experts.staged_tile else glm.experts.tile;
    const wide = P > cuda.grouped.small;
    const plan: cuda.grouped.Plan = .{
        .members = try r.zeros(P * 4),
        .items = try r.zeros(cuda.grouped.maxItems(P, E, tile) * 12),
        .counts = try r.zeros(8),
        .rank = try r.zeros((if (wide) P else 1) * 4),
        .hist = try r.zeros((if (wide) (P + 1023) / 1024 * E else 1) * 4),
    };
    const sz = glm.exl3.scratchBytes(rows, D, N, prompt, s.shx, s.book, s.uniform);
    const sc: glm.exl3.Scratch = .{ .picks = try r.zeros(sz[0]), .xg = try r.zeros(sz[1]), .xu = try r.zeros(sz[2]), .z = try r.zeros(sz[3]) };
    const ey = try r.zeros(rows * slots * D * 4);
    const xd = try r.zeros(P * N * 2);
    try glm.exl3.run(k, ek, r.s, s.experts(), sc, plan, try r.dev(std.mem.sliceAsBytes(x)), D, try r.dev(std.mem.sliceAsBytes(pk)), slots, rows, E, D, N, limit, prompt, ey, xd);
    try r.s.synchronize();
    return .{ .ey = ey, .rows = rows };
}

/// The kernels' SwiGLU: gate and up rounded to bf16 and limited, silu to bf16, the product to bf16.
fn act(g: f64, u: f64) f64 {
    const gg = @min(h.bfOf(h.bf16Of(@floatCast(g))), limit);
    const uu = @max(@min(h.bfOf(h.bf16Of(@floatCast(u))), limit), -limit);
    return h.bfOf(h.bf16Of(@floatCast(h.bfOf(h.bf16Of(@floatCast(gg / (1 + @exp(-gg))))) * uu)));
}

/// Float64 slot rows for routed slots (the shared slot's want is NaN: checked apart) and their tolerances.
fn reference(r: *Rig, s: Set, x: []const u16, pk: []const i32, rows: usize, want: []f64, tol: []f64) !void {
    const a = try r.a.alloc(f64, N);
    for (0..rows) |row| for (0..slots - 1) |j| {
        const e: usize = @intCast(pk[row * slots + j]);
        for (0..N) |n| {
            var g: f64 = 0;
            var u: f64 = 0;
            for (0..D) |kk| {
                const xv = h.bfOf(x[row * D + kk]);
                g += xv * s.wg[(e * D + kk) * N + n];
                u += xv * s.wu[(e * D + kk) * N + n];
            }
            a[n] = act(g, u);
        }
        for (0..D) |n| {
            var y: f64 = 0;
            var mag: f64 = 0;
            for (0..N) |kk| {
                y += a[kk] * s.wd[(e * N + kk) * D + n];
                mag += @abs(a[kk] * s.wd[(e * N + kk) * D + n]);
            }
            const at = (row * slots + j) * D + n;
            want[at] = y;
            tol[at] = mag * (1.0 / 64.0) + 1e-6; // fp16 rotated inputs, bf16 activations, fp16 down inputs
        }
    };
}

/// Routed slots within tolerance; the shared slot untouched (the fill's bytes).
fn expect(r: *Rig, name: []const u8, out: Run, s: Set, x: []const u16, pk: []const i32) !void {
    const n = out.rows * slots * D;
    const want = try r.a.alloc(f64, n);
    const tol = try r.a.alloc(f64, n);
    try reference(r, s, x, pk, out.rows, want, tol);
    const raw = try r.back(out.ey, n * 4);
    var most: f64 = 0;
    var at: usize = 0;
    for (0..out.rows) |row| for (0..slots) |j| for (0..D) |c| {
        const i = (row * slots + j) * D + c;
        const bits = std.mem.readInt(u32, raw[i * 4 ..][0..4], .little);
        if (j == slots - 1) { // the shared slot: the fill's bytes
            try check.expect(bits == 0xffffffff, "{s}: the shared slot left alone", .{name});
            continue;
        }
        const g: f64 = @as(f32, @bitCast(bits));
        const q = @abs(g - want[i]) / tol[i];
        if (!(q <= most)) {
            most = q;
            at = i;
        }
    };
    r.worst = @max(r.worst, most);
    std.debug.print("RESULT {s} ({d} rows): largest difference {d:.3} of its tolerance (element {d})\n", .{ name, out.rows, most, at });
    try check.expect(most <= 1, "{s} within its float64 tolerance", .{name});
}

pub fn run_(gpu: Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var prng = std.Random.DefaultPrng.init(7);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = stream, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }

    var pm = try cuda.Module.load(gpu.d, cuda.kernels.experts);
    defer pm.unload();
    var k = try glm.exl3.Kernels.load(gpu.d);
    defer k.unload();
    var ek = try glm.experts.Kernels.load(gpu.d, pm, @intCast(try gpu.ctx.attribute(.multiprocessor_count)));
    defer ek.unload();
    const Case = struct { book: cuda.exl3.Codebook, ws: *const Widths, what: []const u8, prompt_rows: []const usize };
    const all_rows = [_]usize{ 1, 40, 200, glm.exl3.chunk_rows + 6 };
    const cases = [_]Case{
        .{ .book = .mcg, .ws = &mcg4, .what = "mcg 4-bit", .prompt_rows = &all_rows },
        .{ .book = .mul1, .ws = &redsnow, .what = "mul1 RED-SNOW widths", .prompt_rows = &all_rows },
        .{ .book = .mul1, .ws = &every, .what = "mul1 widths 2 to 16", .prompt_rows = &.{ 1, 40, 200 } },
    };
    for (cases) |cs| for ([_]bool{ true, false }) |one| {
        if (cs.ws == &every and one) continue;
        const s = try weights(&r, cs.book, cs.ws.*, one);
        const rot = if (one) "one rotation" else "a rotation an expert";
        for ([_]usize{ 1, 3, 8 }) |rows| { // decode windows
            const x = try r.bfs(rows * D, 1.0);
            const pk = try picks(&r, rows);
            const got = try run(&r, &k, &ek, s, x, pk, rows, false);
            try expect(&r, try std.fmt.allocPrint(r.a, "glm-exl3 decode, {s}, {s}", .{ cs.what, rot }), got, s, x, pk);
            if (rows > 1) { // every row's bits as alone
                const all = try r.back(got.ey, rows * slots * D * 4);
                for (0..rows) |row| {
                    const alone = try run(&r, &k, &ek, s, x[row * D ..][0..D], pk[row * slots ..][0..slots], 1, false);
                    const mine = try r.back(alone.ey, slots * D * 4);
                    try check.expect(std.mem.eql(u8, mine[0 .. (slots - 1) * D * 4], all[row * slots * D * 4 ..][0 .. (slots - 1) * D * 4]), "glm-exl3 decode, {s}: row {d} of {d} equals the row alone", .{ cs.what, row, rows });
                }
            }
        }
        for (cs.prompt_rows) |rows| { // prompt chunks, the last in two calls
            const x = try r.bfs(rows * D, 1.0);
            const pk = try picks(&r, rows);
            const got = try run(&r, &k, &ek, s, x, pk, rows, true);
            try expect(&r, try std.fmt.allocPrint(r.a, "glm-exl3 prompt, {s}, {s}", .{ cs.what, rot }), got, s, x, pk);
            if (rows == 1) continue;
            const all = try r.back(got.ey, rows * slots * D * 4);
            const again = try run(&r, &k, &ek, s, x, pk, rows, true);
            try check.expect(std.mem.eql(u8, all, try r.back(again.ey, rows * slots * D * 4)), "glm-exl3 prompt, {s}: {d} rows repeat their bytes", .{ cs.what, rows });
            // rows either side of the calls' boundary, and rows whose items other rows share, keep their bits as alone
            for ([_]usize{ 0, rows / 2, glm.exl3.chunk_rows - 1, glm.exl3.chunk_rows, rows - 1 }) |row| {
                if (row >= rows) continue;
                const alone = try run(&r, &k, &ek, s, x[row * D ..][0..D], pk[row * slots ..][0..slots], 1, true);
                const mine = try r.back(alone.ey, slots * D * 4);
                try check.expect(std.mem.eql(u8, mine[0 .. (slots - 1) * D * 4], all[row * slots * D * 4 ..][0 .. (slots - 1) * D * 4]), "glm-exl3 prompt, {s}: row {d} of {d} equals the row alone", .{ cs.what, row, rows });
            }
        }
    };
    { // a NaN in one row's input: that row's routed slots NaN, every other row finite
        const s = try weights(&r, .mul1, redsnow, true);
        const rows = 4;
        const x = try r.bfs(rows * D, 1.0);
        x[2 * D + 17] = 0x7fc0;
        const pk = try picks(&r, rows);
        for ([_]bool{ false, true }) |prompt| {
            const got = try run(&r, &k, &ek, s, x, pk, rows, prompt);
            const raw = try r.back(got.ey, rows * slots * D * 4);
            for (0..rows) |row| for (0..slots - 1) |j| for (0..D) |c| {
                const v: f32 = @bitCast(std.mem.readInt(u32, raw[((row * slots + j) * D + c) * 4 ..][0..4], .little));
                try check.expect(if (row == 2) std.math.isNan(v) else std.math.isFinite(v), "glm-exl3 {s}: a NaN stays in its row ({d})", .{ if (prompt) "prompt" else "decode", row });
            };
        }
    }
    check.pass("glm-exl3: decode and prompt slot rows within {d:.3} of their float64 tolerance, rows as alone, NaN kept to its row", .{r.worst});
}
