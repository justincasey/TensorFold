//! The EXL3 routed experts: float64 references, row and chunk invariance, and the Python engine's recorded bytes.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const quant_ref_tests = @import("quant_ref_tests.zig");
const exl3fmt = @import("core").exl3_format;
const exl3 = cuda.exl3;
const exl3e = cuda.exl3_experts;
const exl3t = @import("exl3_tests.zig");
const Gpu = check.Gpu;
const Rig = quant_ref_tests.RigT;
const bf = quant_ref_tests.helpers.bfOf;
const Run = exl3t.Run;

const E = 9; // experts of a synthetic layer
const D = 512; // model width
const I = 256; // expert width
const SLOTS = 8;
const LIMIT: f32 = 10.0; // the SwiGLU clamp
const CAP = 300; // rows of x, pick and wts the builders make (the f64 checks use the first 128)
const rows = [_]usize{ 1, 3, 8, 16, 17, 64, 128 }; // the f64 checks' row counts (the reference builds 128 rows)

const f64_ids = [_][]const u8{ "exl3.experts.f64.mul1.4", "exl3.experts.f64.mul1.5", "exl3.experts.f64.mul1.6" };

/// One synthetic layer: the device tensors, the host trellises the reference unpacks, and the seeded routing.
const Built = struct {
    layer: exl3e.Layer,
    bits: []const u32,
    gate: [E][]i16,
    up: [E][]i16,
    down: [E][]i16,
    suh_g: []u16, // fp16 [E, D]
    suh_u: []u16,
    svh_g: []u16, // fp16 [E, I]
    svh_u: []u16,
    suh_d: []u16, // fp16 [E, I]
    svh_d: []u16, // fp16 [E, D]
    dx: u64,
    dpick: u64,
    dwts: u64,
    x: []u16, // bf16 [CAP, D]
    pick: []i32, // [CAP, SLOTS]
    wts: []f32, // [CAP, SLOTS]
};

pub fn run(r: *Rig, state: *Run, dir: []const u8) !void {
    var exp = try exl3e.Experts.load(r.gpu.d);
    defer exp.unload();

    // every expert one width: 4, 5 and 6 bits (the mixed run folds into .6)
    for (f64_ids, 0..) |id, i| {
        var ok = true;
        f64Check(r, &exp, &allBits(@intCast(i + 4))) catch |e| {
            state.fail(e, id);
            ok = false;
        };
        if (ok) check.pass("{s}", .{id});
    }
    {
        var ok = true;
        f64Check(r, &exp, &.{ 4, 5, 6, 4, 5, 6, 4, 5, 6 }) catch |e| {
            state.fail(e, "exl3.experts.f64.mul1.6");
            ok = false;
        };
        if (ok) check.pass("{s}", .{f64_ids[2]});
    }
    {
        var ok = true;
        rowsInvariant(r, &exp) catch |e| {
            state.fail(e, "exl3.experts.rows");
            ok = false;
        };
        if (ok) check.pass("exl3.experts.rows", .{});
    }
    {
        var ok = true;
        prompt(r, &exp) catch |e| {
            state.fail(e, "exl3.prompt.experts.f64");
            ok = false;
        };
        if (ok) check.pass("exl3.prompt.experts.f64", .{});
    }
    {
        var ok = true;
        chunks(r, &exp) catch |e| {
            state.fail(e, "exl3.prompt.experts.chunks");
            ok = false;
        };
        if (ok) check.pass("exl3.prompt.experts.chunks", .{});
    }
    {
        var ok = true;
        python(r, &exp, dir, state) catch |e| {
            state.fail(e, "exl3.experts.python");
            ok = false;
        };
        if (ok) check.pass("exl3.experts.python", .{});
    }
}

fn allBits(bits: u32) [E]u32 {
    return .{ bits, bits, bits, bits, bits, bits, bits, bits, bits };
}

fn widthOf(bits: u32) exl3fmt.Width {
    return .{ .halves = @intCast(2 * bits) };
}

fn f16v(v: u16) f64 {
    return @floatCast(@as(f16, @bitCast(v)));
}

/// bf16 round-to-nearest-even of an f64 (the kernel rounds its fp32 value; the double round is within the tolerance).
fn bf16r(v: f64) f64 {
    const u: u32 = @bitCast(@as(f32, @floatCast(v)));
    return bf(@intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16));
}

fn buildLayer(r: *Rig, bits: []const u32) !Built {
    var b: Built = .{
        .layer = undefined,
        .bits = bits,
        .gate = undefined,
        .up = undefined,
        .down = undefined,
        .dx = 0,
        .dpick = 0,
        .dwts = 0,
        .suh_g = undefined,
        .suh_u = undefined,
        .svh_g = undefined,
        .svh_u = undefined,
        .suh_d = undefined,
        .svh_d = undefined,
        .x = undefined,
        .pick = undefined,
        .wts = undefined,
    };
    var gate_p: [E]u64 = undefined;
    var up_p: [E]u64 = undefined;
    var down_p: [E]u64 = undefined;
    var gk2: [E]i32 = undefined;
    var uk2: [E]i32 = undefined;
    var dk2: [E]i32 = undefined;
    const suh_g = try r.a.alloc(u16, E * D);
    const suh_u = try r.a.alloc(u16, E * D);
    const svh_g = try r.a.alloc(u16, E * I);
    const svh_u = try r.a.alloc(u16, E * I);
    const suh_d = try r.a.alloc(u16, E * I);
    const svh_d = try r.a.alloc(u16, E * D);
    for (0..E) |e| {
        b.gate[e] = try randI16(r, (D / 16) * (I / 16) * widthOf(bits[e]).tileWords());
        b.up[e] = try randI16(r, (D / 16) * (I / 16) * widthOf(bits[e]).tileWords());
        b.down[e] = try randI16(r, (I / 16) * (D / 16) * widthOf(bits[e]).tileWords());
        gate_p[e] = try r.dev(std.mem.sliceAsBytes(b.gate[e]));
        up_p[e] = try r.dev(std.mem.sliceAsBytes(b.up[e]));
        down_p[e] = try r.dev(std.mem.sliceAsBytes(b.down[e]));
        gk2[e] = @intCast(2 * bits[e]);
        uk2[e] = gk2[e];
        dk2[e] = gk2[e];
        randScale(r, suh_g[e * D ..][0..D]);
        randScale(r, suh_u[e * D ..][0..D]);
        randScale(r, svh_g[e * I ..][0..I]);
        randScale(r, svh_u[e * I ..][0..I]);
        randScale(r, suh_d[e * I ..][0..I]);
        randScale(r, svh_d[e * D ..][0..D]);
    }
    b.suh_g = suh_g;
    b.suh_u = suh_u;
    b.svh_g = svh_g;
    b.svh_u = svh_u;
    b.suh_d = suh_d;
    b.svh_d = svh_d;
    b.layer = .{
        .gate = try r.dev(std.mem.sliceAsBytes(gate_p[0..])),
        .up = try r.dev(std.mem.sliceAsBytes(up_p[0..])),
        .down = try r.dev(std.mem.sliceAsBytes(down_p[0..])),
        .gate_k2 = try r.dev(std.mem.sliceAsBytes(gk2[0..])),
        .up_k2 = try r.dev(std.mem.sliceAsBytes(uk2[0..])),
        .down_k2 = try r.dev(std.mem.sliceAsBytes(dk2[0..])),
        .suh_g = try r.dev(std.mem.sliceAsBytes(suh_g)),
        .suh_u = try r.dev(std.mem.sliceAsBytes(suh_u)),
        .svh_g = try r.dev(std.mem.sliceAsBytes(svh_g)),
        .svh_u = try r.dev(std.mem.sliceAsBytes(svh_u)),
        .suh_d = try r.dev(std.mem.sliceAsBytes(suh_d)),
        .svh_d = try r.dev(std.mem.sliceAsBytes(svh_d)),
        .experts = E,
        .dims = D,
        .width = I,
        .codebook = .mul1,
        .limit = LIMIT,
        .act_mode = 0,
    };
    b.x = try r.bfs(CAP * D, 1.0);
    b.pick = try picks(r, CAP);
    b.wts = try wtsOf(r, CAP);
    b.dx = try r.dev(std.mem.sliceAsBytes(b.x));
    b.dpick = try r.dev(std.mem.sliceAsBytes(b.pick));
    b.dwts = try r.dev(std.mem.sliceAsBytes(b.wts));
    return b;
}

fn randI16(r: *Rig, n: usize) ![]i16 {
    const out = try r.a.alloc(i16, n);
    for (out) |*v| v.* = r.rng.int(i16);
    return out;
}

fn randScale(r: *Rig, out: []u16) void {
    for (out) |*v| v.* = @bitCast(@as(f16, @floatCast(r.rng.float(f32) * 2 - 1)));
}

/// Distinct experts a row: a partial Fisher-Yates over the layer's experts, seeded.
fn picks(r: *Rig, cap: usize) ![]i32 {
    const out = try r.a.alloc(i32, cap * SLOTS);
    for (0..cap) |row| {
        var idx: [E]u32 = undefined;
        for (0..E, 0..) |_, i| idx[i] = @intCast(i);
        var n: usize = E;
        for (0..SLOTS) |k| {
            const j = r.rng.uintLessThan(usize, n);
            out[row * SLOTS + k] = @intCast(idx[j]);
            idx[j] = idx[n - 1];
            n -= 1;
        }
    }
    return out;
}

fn wtsOf(r: *Rig, cap: usize) ![]f32 {
    const out = try r.a.alloc(f32, cap * SLOTS);
    for (out) |*v| v.* = 0.1 + 0.9 * r.rng.float(f32);
    return out;
}

/// One routed call over `rows` rows starting at row 0 of the built routing; the scratch and out are fresh canaries.
fn runRows(r: *Rig, exp: *const exl3e.Experts, b: *const Built, rows_n: usize) !u64 {
    return (try runSlice(r, exp, b.layer, b.dx, b.dpick, b.dwts, rows_n)).out;
}

fn runSlice(r: *Rig, exp: *const exl3e.Experts, layer: exl3e.Layer, x: u64, pick: u64, wts: u64, rows_n: usize) !struct { out: u64, y: u64 } {
    const sb = try r.zeros(exl3e.scratchBytes(layer, rows_n, SLOTS));
    const sc = exl3e.scratch(sb, layer, rows_n, SLOTS);
    const out = try r.zeros(rows_n * D * 4);
    try exp.run(r.s, layer, x, D, .bf16, pick, wts, rows_n, out, sc);
    return .{ .out = out, .y = sc.y };
}

fn f64Check(r: *Rig, exp: *const exl3e.Experts, bits: []const u32) !void {
    const b = try buildLayer(r, bits);
    const want = try reference(r, &b, 128);
    for (rows) |m| try closeOut(r, b.bits, m, (try runRows(r, exp, &b, m)), want.want, want.tol);
}

/// fp32 outputs within the f64 reference's tolerance, with the exact FAIL line.
fn closeOut(r: *Rig, bits: []const u32, m: usize, ptr: u64, want: []const f64, tol: []const f64) !void {
    const raw = try r.back(ptr, m * D * 4);
    var most: f64 = 0;
    var at: usize = 0;
    for (want[0 .. m * D], tol[0 .. m * D], 0..) |wv, t, i| {
        const g: f64 = @floatCast(std.mem.bytesToValue(f32, raw[4 * i ..][0..4]));
        const q = @abs(g - wv) / t;
        if (q > most) {
            most = q;
            at = i;
        }
    }
    std.debug.print("RESULT exl3.experts.f64.mul1.{d} m={d}: largest difference {d:.3} of its tolerance (element {d})\n", .{ bits[0], m, most, at });
    if (most > 1) {
        std.debug.print("FAIL exl3.experts.f64.mul1.{d}: largest difference {d:.3} of its tolerance\n", .{ bits[0], most });
        return error.TestFailed;
    }
}

/// The f64 reference: fp16 xg and xd, the GEMMs and their |t| sums, the rotated errors, and the SwiGLU mirrored
/// (bf16 roundings, f64 exp); the tolerance covers fp32 accumulation, the bf16 activations, expf and fp16.
fn reference(r: *Rig, b: *const Built, rows_n: usize) !struct { want: []f64, tol: []f64 } {
    const p = rows_n * SLOTS;
    // xg: fp16((x * suh_g) @ H) per slot
    const xs = try r.a.alloc(f64, p * D);
    for (0..p) |s0| {
        const ex: usize = @intCast(b.pick[s0]);
        for (0..D) |i| xs[s0 * D + i] = bf(b.x[(s0 / SLOTS) * D + i]) * f16v(b.suh_g[ex * D + i]);
    }
    exl3fmt.rotate(xs, p, D, 1);
    const xg = try r.a.alloc(f16, p * D);
    for (xg, xs) |*o, v| o.* = @floatCast(v);

    const wq = try r.a.alloc(f16, @max(D, I) * @max(D, I));
    const zg = try r.a.alloc(f64, I * p); // [n][p]
    const zu = try r.a.alloc(f64, I * p);
    const s1g = try r.a.alloc(f64, I * p);
    const s1u = try r.a.alloc(f64, I * p);
    @memset(zg, 0);
    @memset(zu, 0);
    @memset(s1g, 0);
    @memset(s1u, 0);
    for (0..E) |ex| {
        exl3fmt.unpack(widthOf(b.bits[ex]), .mul1, b.gate[ex], D / 16, I / 16, wq[0 .. D * I]);
        for (0..p) |s0| {
            if (b.pick[s0] != @as(i32, @intCast(ex))) continue;
            gemmK(xg[s0 * D ..][0..D], wq, zg[s0..], s1g[s0..], D, I, p);
        }
        exl3fmt.unpack(widthOf(b.bits[ex]), .mul1, b.up[ex], D / 16, I / 16, wq[0 .. D * I]);
        for (0..p) |s0| {
            if (b.pick[s0] != @as(i32, @intCast(ex))) continue;
            gemmK(xg[s0 * D ..][0..D], wq, zu[s0..], s1u[s0..], D, I, p);
        }
    }
    exl3fmt.rotate(zg, I, p, 0);
    exl3fmt.rotate(zu, I, p, 0);
    rssRotate(s1g, I, p);
    rssRotate(s1u, I, p);

    // the SwiGLU (bf16 roundings, f64 exp) and the rotated errors
    const v = try r.a.alloc(f64, I * p);
    const verr = try r.a.alloc(f64, I * p);
    for (0..I) |n| {
        for (0..p) |s0| {
            const ex: usize = @intCast(b.pick[s0]);
            const gg = @min(bf16r(zg[n * p + s0] * f16v(b.svh_g[ex * I + n])), @as(f64, LIMIT));
            const uu = @min(@max(bf16r(zu[n * p + s0] * f16v(b.svh_u[ex * I + n])), -@as(f64, LIMIT)), @as(f64, LIMIT));
            const act = bf16r(bf16r(gg / (1.0 + std.math.exp(-gg))) * uu);
            v[n * p + s0] = act * f16v(b.suh_d[ex * I + n]);
            const eg = s1g[n * p + s0] / 2048;
            const eu = s1u[n * p + s0] / 2048;
            verr[n * p + s0] = (uu * eg + (@abs(gg) + 1) * eu + @abs(act) * 0.00000011920929) * @abs(f16v(b.suh_d[ex * I + n])) +
                @abs(v[n * p + s0]) * 0.000000059604645;
        }
    }
    exl3fmt.rotate(v, I, p, 0);
    rssRotate(verr, I, p);
    const xd = try r.a.alloc(f16, p * I);
    const xderr = try r.a.alloc(f64, p);
    for (0..p) |s0| {
        var most: f64 = 0;
        var mx: f64 = 0;
        for (0..I) |n| {
            xd[s0 * I + n] = @floatCast(v[n * p + s0]);
            mx = @max(mx, verr[n * p + s0]);
            most = @max(most, @abs(v[n * p + s0]));
        }
        xderr[s0] = mx + most / 2048; // the rotation mixes (RSS), fp16 rounds
    }

    // down: the GEMM, its |t| and |w| sums, the rotated errors
    const zd = try r.a.alloc(f64, D * p); // [d][p]
    const s1d = try r.a.alloc(f64, D * p);
    const s2d = try r.a.alloc(f64, D * p);
    @memset(zd, 0);
    @memset(s1d, 0);
    @memset(s2d, 0);
    for (0..E) |ex| {
        exl3fmt.unpack(widthOf(b.bits[ex]), .mul1, b.down[ex], I / 16, D / 16, wq[0 .. I * D]);
        for (0..p) |s0| {
            if (b.pick[s0] != @as(i32, @intCast(ex))) continue;
            for (0..I) |ki| {
                const t: f64 = @floatCast(xd[s0 * I + ki]);
                const ta = @abs(t);
                const row = wq[ki * D ..][0..D];
                for (0..D) |d| {
                    const wv: f64 = @floatCast(row[d]);
                    zd[d * p + s0] += t * wv;
                    s1d[d * p + s0] += ta * @abs(wv);
                    s2d[d * p + s0] += @abs(wv);
                }
            }
        }
    }
    exl3fmt.rotate(zd, D, p, 0);
    // the per-element down errors (the GEMM's accumulation and the input error), then the rotation's RSS mix
    const de = try r.a.alloc(f64, D * p);
    for (0..D) |d| {
        for (0..p) |s0| de[d * p + s0] = s1d[d * p + s0] / 2048 + s2d[d * p + s0] * xderr[s0];
    }
    rssRotate(de, D, p);
    // the per-slot outputs and the weighted combine, fp32 fma chains in slot order
    const want = try r.a.alloc(f64, rows_n * D);
    const tol = try r.a.alloc(f64, rows_n * D);
    for (0..rows_n) |row| {
        for (0..D) |d| {
            var acc: f32 = 0;
            var err: f64 = 0;
            for (0..SLOTS) |k| {
                const s0 = row * SLOTS + k;
                const ex: usize = @intCast(b.pick[s0]);
                const y: f32 = @floatCast(zd[d * p + s0] * f16v(b.svh_d[ex * D + d]));
                acc = @mulAdd(f32, b.wts[row * SLOTS + k], y, acc);
                const yerr = de[d * p + s0] + @abs(y) * 0.000000059604645;
                err += @abs(b.wts[row * SLOTS + k]) * yerr;
            }
            want[row * D + d] = acc;
            tol[row * D + d] = 8 * err + 1e-30;
        }
    }
    return .{ .want = want, .tol = tol };
}

/// One output column's GEMM over k: the sums into [n][p] and their |t| sums (the caller indexes the slot).
fn gemmK(xrow: []const f16, wq: []const f16, z: []f64, s1: []f64, k: usize, n: usize, p: usize) void {
    for (0..k) |ki| {
        const t: f64 = @floatCast(xrow[ki]);
        const ta = @abs(t);
        const wrow = wq[ki * n ..][0..n];
        for (0..n) |col| {
            const wv: f64 = @floatCast(wrow[col]);
            z[col * p] += t * wv;
            s1[col * p] += ta * @abs(wv);
        }
    }
}

/// Each 128-block along the row axis replaced by the RSS of its entries: the rotation mixes the errors, and the
/// mixed bound is the block's RMS norm (entries of magnitude 1/sqrt(128)).
fn rssRotate(x: []f64, nrows: usize, cols: usize) void {
    var r: usize = 0;
    while (r < nrows) : (r += 128) {
        for (0..cols) |c| {
            var s: f64 = 0;
            for (0..128) |n| {
                const v = x[(r + n) * cols + c];
                s += v * v;
            }
            s = @sqrt(s);
            for (0..128) |n| x[(r + n) * cols + c] = s;
        }
    }
}

/// A row's output bytes are its own: solo, with 3 rows, in 64-row batches, shared or not.
fn rowsInvariant(r: *Rig, exp: *const exl3e.Experts) !void {
    var bits: [E]u32 = .{ 6, 4, 5, 6, 4, 5, 6, 4, 5 };
    const b = try buildLayer(r, &bits);
    const solo = try r.back(try runRows(r, exp, &b, 1), D * 4);
    // 3 rows: two more rows on the same picks as row 0
    const p3 = try r.a.alloc(i32, 3 * SLOTS);
    @memcpy(p3[0 .. 2 * SLOTS], b.pick[0 .. 2 * SLOTS]);
    @memcpy(p3[2 * SLOTS ..], b.pick[0..SLOTS]);
    const w3 = try r.a.alloc(f32, 3 * SLOTS);
    @memcpy(w3, b.wts[0 .. 3 * SLOTS]);
    const got3 = try r.back((try runSlice(r, exp, b.layer, b.dx, try r.dev(std.mem.sliceAsBytes(p3)), try r.dev(std.mem.sliceAsBytes(w3)), 3)).out, D * 4);
    try check.sameBytes("exl3.experts.rows 3 rows", got3, solo);
    // 64 rows: the others share row 0's experts (the same picks), or run their own
    const p64 = try r.a.alloc(i32, 64 * SLOTS);
    const w64 = try r.a.alloc(f32, 64 * SLOTS);
    @memcpy(p64[0..SLOTS], b.pick[0..SLOTS]);
    @memcpy(w64[0..SLOTS], b.wts[0..SLOTS]);
    for (1..64) |row| {
        @memcpy(p64[row * SLOTS ..][0..SLOTS], b.pick[0..SLOTS]);
        @memcpy(w64[row * SLOTS ..][0..SLOTS], b.wts[0..SLOTS]);
    }
    const shared = try r.back((try runSlice(r, exp, b.layer, b.dx, try r.dev(std.mem.sliceAsBytes(p64)), try r.dev(std.mem.sliceAsBytes(w64)), 64)).out, D * 4);
    try check.sameBytes("exl3.experts.rows 64 rows shared", shared[0 .. D * 4], solo);
    const own = try r.back((try runSlice(r, exp, b.layer, b.dx, b.dpick, b.dwts, 64)).out, D * 4);
    try check.sameBytes("exl3.experts.rows 64 rows own", own[0 .. D * 4], solo);
    // and a 64-row batch whose other rows interleave with row 0's experts in both orders
    const p64b = try r.a.alloc(i32, 64 * SLOTS);
    const w64b = try r.a.alloc(f32, 64 * SLOTS);
    @memcpy(p64b[0..SLOTS], b.pick[0..SLOTS]);
    @memcpy(w64b[0..SLOTS], b.wts[0..SLOTS]);
    for (1..64) |row| {
        const src = if (row % 2 == 0) @as(usize, 0) else row % SLOTS;
        @memcpy(p64b[row * SLOTS ..][0..SLOTS], b.pick[src * SLOTS ..][0..SLOTS]);
        @memcpy(w64b[row * SLOTS ..][0..SLOTS], b.wts[src * SLOTS ..][0..SLOTS]);
    }
    const mixed = try r.back((try runSlice(r, exp, b.layer, b.dx, try r.dev(std.mem.sliceAsBytes(p64b)), try r.dev(std.mem.sliceAsBytes(w64b)), 64)).out, D * 4);
    try check.sameBytes("exl3.experts.rows 64 rows mixed", mixed[0 .. D * 4], solo);
}

/// A 300-row chunk against the f64 reference (the mixed-width layer).
fn prompt(r: *Rig, exp: *const exl3e.Experts) !void {
    var bits: [E]u32 = .{ 4, 5, 6, 4, 5, 6, 4, 5, 6 };
    const b = try buildLayer(r, &bits);
    const want = try reference(r, &b, 300);
    const res = try runRows(r, exp, &b, 300);
    const raw = try r.back(res, 300 * D * 4);
    var most: f64 = 0;
    var at: usize = 0;
    for (want.want, want.tol, 0..) |wv, t, i| {
        const g: f64 = @floatCast(std.mem.bytesToValue(f32, raw[4 * i ..][0..4]));
        const q = @abs(g - wv) / t;
        if (q > most) {
            most = q;
            at = i;
        }
    }
    std.debug.print("RESULT exl3.prompt.experts.f64: largest difference {d:.3} of its tolerance (element {d})\n", .{ most, at });
    if (most > 1) return error.TestFailed;
}

/// The same 300 rows as one call, split at 128 and at 37: a row's bytes never move.
fn chunks(r: *Rig, exp: *const exl3e.Experts) !void {
    var bits: [E]u32 = .{ 4, 5, 6, 4, 5, 6, 4, 5, 6 };
    const b = try buildLayer(r, &bits);
    const whole = try r.back(try runRows(r, exp, &b, 300), 300 * D * 4);
    var at: usize = 0;
    while (at < 300) : (at += 128) {
        const n: usize = @min(300 - at, 128);
        const res = try runSlice(r, exp, b.layer, b.dx + @as(u64, at) * D * 2, b.dpick + @as(u64, at) * SLOTS * 4, b.dwts + @as(u64, at) * SLOTS * 4, n);
        const got = try r.back(res.out, n * D * 4);
        try check.sameBytes("exl3.prompt.experts.chunks 128", got, whole[at * D * 4 ..][0 .. n * D * 4]);
    }
    at = 0;
    while (at < 300) : (at += 37) {
        const n: usize = @min(300 - at, 37);
        const res = try runSlice(r, exp, b.layer, b.dx + @as(u64, at) * D * 2, b.dpick + @as(u64, at) * SLOTS * 4, b.dwts + @as(u64, at) * SLOTS * 4, n);
        const got = try r.back(res.out, n * D * 4);
        try check.sameBytes("exl3.prompt.experts.chunks 37", got, whole[at * D * 4 ..][0 .. n * D * 4]);
    }
}

/// The Python engine's recorded y and out, bit for bit (exl3_oracle_experts.py's manifest and raw files).
fn python(r: *Rig, exp: *const exl3e.Experts, dir: []const u8, state: *Run) !void {
    if (std.mem.eql(u8, dir, "none")) {
        std.debug.print("SKIP exl3.experts.python\n", .{});
        return;
    }
    const manifest_path = try std.fs.path.join(r.a, &.{ dir, "manifest.json" });
    const text = std.Io.Dir.cwd().readFileAlloc(r.gpu.io, manifest_path, r.a, .limited(1 << 24)) catch |e| {
        std.debug.print("FAIL exl3.experts.python: {t}\n", .{e});
        state.failed = true;
        return;
    };
    var parsed = try std.json.parseFromSlice(std.json.Value, r.a, text, .{});
    const o = parsed.value.object.get("experts").?.object;
    const e_count: usize = @intCast(o.get("count").?.integer);
    const dims: usize = @intCast(o.get("dims").?.integer);
    const width: usize = @intCast(o.get("width").?.integer);
    const slots: usize = @intCast(o.get("slots").?.integer);
    const cb: exl3.Codebook = if (std.mem.eql(u8, o.get("codebook").?.string, "mcg"))
        .mcg
    else if (std.mem.eql(u8, o.get("codebook").?.string, "inst3"))
        .inst3
    else
        .mul1;
    const trellis = o.get("trellis").?.object;
    var gate_p = try r.a.alloc(u64, e_count);
    var up_p = try r.a.alloc(u64, e_count);
    var down_p = try r.a.alloc(u64, e_count);
    var gk2 = try r.a.alloc(i32, e_count);
    var uk2 = try r.a.alloc(i32, e_count);
    var dk2 = try r.a.alloc(i32, e_count);
    for (o.get("gate_k2").?.array.items, 0..) |v, i| gk2[i] = @intCast(v.integer);
    for (o.get("up_k2").?.array.items, 0..) |v, i| uk2[i] = @intCast(v.integer);
    for (o.get("down_k2").?.array.items, 0..) |v, i| dk2[i] = @intCast(v.integer);
    for (0..e_count) |i| {
        gate_p[i] = try r.dev(try readFile(r, dir, trellis.get("gate").?.array.items[i].string));
        up_p[i] = try r.dev(try readFile(r, dir, trellis.get("up").?.array.items[i].string));
        down_p[i] = try r.dev(try readFile(r, dir, trellis.get("down").?.array.items[i].string));
    }
    const layer: exl3e.Layer = .{
        .gate = try r.dev(std.mem.sliceAsBytes(gate_p)),
        .up = try r.dev(std.mem.sliceAsBytes(up_p)),
        .down = try r.dev(std.mem.sliceAsBytes(down_p)),
        .gate_k2 = try r.dev(std.mem.sliceAsBytes(gk2)),
        .up_k2 = try r.dev(std.mem.sliceAsBytes(uk2)),
        .down_k2 = try r.dev(std.mem.sliceAsBytes(dk2)),
        .suh_g = try r.dev(try readFile(r, dir, o.get("suh_g").?.string)),
        .suh_u = try r.dev(try readFile(r, dir, o.get("suh_u").?.string)),
        .svh_g = try r.dev(try readFile(r, dir, o.get("svh_g").?.string)),
        .svh_u = try r.dev(try readFile(r, dir, o.get("svh_u").?.string)),
        .suh_d = try r.dev(try readFile(r, dir, o.get("suh_d").?.string)),
        .svh_d = try r.dev(try readFile(r, dir, o.get("svh_d").?.string)),
        .experts = e_count,
        .dims = dims,
        .width = width,
        .codebook = cb,
        .limit = @floatCast(o.get("limit").?.float),
        .act_mode = @intCast(o.get("act_mode").?.integer),
    };
    const xmap = o.get("x").?.object;
    const pickmap = o.get("pick").?.object;
    const wtsmap = o.get("wts").?.object;
    const ymap = o.get("y").?.object;
    const outmap = o.get("out").?.object;
    for (o.get("rows").?.array.items) |rv| {
        const m: usize = @intCast(rv.integer);
        const key = try std.fmt.allocPrint(r.a, "{d}", .{m});
        const dx = try r.dev(try readFile(r, dir, xmap.get(key).?.string));
        const dpick = try r.dev(try readFile(r, dir, pickmap.get(key).?.string));
        const dwts = try r.dev(try readFile(r, dir, wtsmap.get(key).?.string));
        const sb = try r.zeros(exl3e.scratchBytes(layer, m, slots));
        const sc = exl3e.scratch(sb, layer, m, slots);
        const out = try r.zeros(m * dims * 4);
        try exp.run(r.s, layer, dx, dims, .bf16, dpick, dwts, m, out, sc);
        const got_y = try r.back(sc.y, m * slots * dims * 4);
        try check.sameBytes("exl3.experts.python y", got_y, try readFile(r, dir, ymap.get(key).?.string));
        const got_out = try r.back(out, m * dims * 4);
        try check.sameBytes("exl3.experts.python out", got_out, try readFile(r, dir, outmap.get(key).?.string));
    }
}

fn readFile(r: *Rig, dir: []const u8, name: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(r.gpu.io, try std.fs.path.join(r.a, &.{ dir, name }), r.a, .limited(1 << 32));
}

test "the ids are the contract's" {
    try std.testing.expectEqualStrings("exl3.experts.f64.mul1.4", f64_ids[0]);
    try std.testing.expectEqualStrings("exl3.experts.f64.mul1.6", f64_ids[2]);
}
