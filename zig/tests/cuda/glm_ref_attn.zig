//! GLM's MLA attention (chunked softmax, merge; dense, sparse) vs float64; 16/32-head blocks, 1-row windows same bits.
const std = @import("std");
const check = @import("check.zig");
const qr = @import("quant_ref_tests.zig");
const ref = @import("glm_ref_tests.zig");
const glm = @import("glm_cuda");
const Rig = qr.RigT;
const bf = qr.helpers.bfOf;
const latent = glm.latent;

const lw = 512;
const heads = 32;

fn tag(kv8: bool) []const u8 {
    return if (kv8) " (fp8 cache)" else "";
}

fn partials(r: *Rig, nch: usize, rows: usize) !latent.Partials {
    return .{ .po = try r.zeros(nch * rows * heads * lw * 4), .pm = try r.zeros(nch * rows * heads * 4), .pl = try r.zeros(nch * rows * heads * 4) };
}

/// float64 o = sum p c / sum p, p = exp(s - max); bound: bf16 out and p (2^-8 each), (512 + 8) 2^-24 scale sum |q c|.
fn reference(r: *Rig, q: []const u16, cd: []const f64, keys: []const []const u32, scale: f64) !struct { want: []f64, tol: []f64 } {
    const rows = keys.len;
    const want = try r.a.alloc(f64, rows * heads * lw);
    const tol = try r.a.alloc(f64, rows * heads * lw);
    const qd = try r.a.alloc(f64, lw);
    const o = try r.a.alloc(f64, lw);
    const spread = try r.a.alloc(f64, lw);
    var most_n: usize = 0;
    for (keys) |k| most_n = @max(most_n, k.len);
    const s = try r.a.alloc(f64, most_n);
    const ds = try r.a.alloc(f64, most_n);
    for (keys, 0..) |set, row| for (0..heads) |h| {
        const base = (row * heads + h) * lw;
        for (qd, q[base..][0..lw]) |*d, v| d.* = bf(v);
        var most = -std.math.inf(f64);
        for (set, 0..) |key, i| {
            var dot: f64 = 0;
            var abs: f64 = 0;
            for (qd, cd[key * lw ..][0..lw]) |a, b| {
                dot += a * b;
                abs += @abs(a * b);
            }
            s[i] = dot * scale;
            ds[i] = abs * scale * 520 * std.math.pow(f64, 2, -24);
            most = @max(most, s[i]);
        }
        @memset(o, 0);
        @memset(spread, 0);
        var l: f64 = 0;
        for (set, 0..) |key, i| {
            const p = @exp(s[i] - most);
            l += p;
            for (o, spread, cd[key * lw ..][0..lw]) |*od, *sd, v| {
                od.* += p * v;
                sd.* += p * @abs(v) * (std.math.pow(f64, 2, -8) + ds[i]);
            }
        }
        for (0..lw) |d| {
            want[base + d] = o[d] / l;
            tol[base + d] = @abs(o[d] / l) * std.math.pow(f64, 2, -8) + spread[d] / l + 1e-30;
        }
    };
    return .{ .want = want, .tol = tol };
}

/// A latent cache on the device: bf16 rows, or (kv8) e4m3 codes with fp32 row scales; f64s the values the kernels see.
const Cache = struct { dev: u64, scales: u64, f64s: []f64 };

/// e4m3 (fn) code b's value: subnormals m 2^-9, normals (1 + m / 8) 2^(e - 7); 0x7f / 0xff (NaN) never written.
fn e4m3(b: u8) f64 {
    const e: i32 = (b >> 3) & 15;
    const m: f64 = @floatFromInt(b & 7);
    const v = if (e == 0) m * std.math.pow(f64, 2, -9) else (1 + m / 8) * std.math.pow(f64, 2, @floatFromInt(e - 7));
    return if (b & 0x80 != 0) -v else v;
}

/// `keys` random bf16 rows, each scaled by 2^(key % 7 - 3) so the fp8 rows' scales differ; kv8: written by the
/// fp8 cache write, whose codes and scales are checked here (s = amax / 448, |x - code s| within half an e4m3 step).
fn cacheOf(r: *Rig, set: glm.glue.Set, keys: usize, kv8: bool) !Cache {
    const c = try r.bfs(keys * lw, 1.0);
    for (c, 0..) |*v, i| {
        const k: i32 = @as(i32, @intCast(i / lw % 7)) - 3;
        if (v.* & 0x7fff != 0) v.* = @intCast(@as(i32, v.*) + (k << 7));
    }
    const cd = try r.a.alloc(f64, keys * lw);
    for (cd, c) |*d, v| d.* = bf(v);
    const rows = try r.dev(std.mem.sliceAsBytes(c));
    if (!kv8) return .{ .dev = rows, .scales = 0, .f64s = cd };
    const codes = try r.zeros(keys * lw);
    const scales = try r.zeros(keys * 4);
    const zero: i32 = 0;
    try latent.latentWrite(set, r.s, rows, lw, codes, scales, try r.dev(std.mem.asBytes(&zero)), keys, lw);
    const cb = try r.back(codes, keys * lw);
    const sb = try r.back(scales, keys * 4);
    for (0..keys) |key| {
        const s: f64 = @as(f32, @bitCast(std.mem.readInt(u32, sb[key * 4 ..][0..4], .little)));
        var amax: f64 = 0;
        for (cd[key * lw ..][0..lw]) |x| amax = @max(amax, @abs(x));
        try check.expect(@abs(s - amax / 448) <= amax / 448 * std.math.pow(f64, 2, -21), "fp8 latent write: key {d}'s scale {e} is amax / 448 ({e})", .{ key, s, amax / 448 });
        for (cb[key * lw ..][0..lw], cd[key * lw ..][0..lw]) |b, *x| {
            try check.expect(b & 0x7f != 0x7f, "fp8 latent write: key {d} wrote a NaN code", .{key});
            const v = e4m3(b) * s;
            const step = @max(@abs(x.*) * std.math.pow(f64, 2, -4), s * std.math.pow(f64, 2, -10)) * (1 + std.math.pow(f64, 2, -20));
            try check.expect(@abs(v - x.*) <= step, "fp8 latent write: key {d}: {e} coded as {e} (half a step {e})", .{ key, x.*, v, step });
            x.* = v;
        }
    }
    return .{ .dev = codes, .scales = scales, .f64s = cd };
}

/// latent.attention of `rows` rows at `pos` (keys 0 .. pos + r each) with head block hb into a fresh output.
fn attend(r: *Rig, set: glm.glue.Set, q: u64, c: Cache, pos: usize, rows: usize, scale: f32, hb: usize) !u64 {
    const nch = latent.chunksFor(pos + rows);
    const out = try r.zeros(rows * heads * lw * 2);
    const p0: i32 = @intCast(pos);
    const at = try r.dev(std.mem.asBytes(&p0));
    try latent.attention(set, r.s, q, c.dev, c.scales, at, try partials(r, nch, rows), out, try r.zeros(4), rows, heads, lw, nch, scale, hb);
    return out;
}

/// Row `row` of a [rows][heads][lw] bf16 output alone equals that row of the wide call's bytes.
fn sameRow(r: *Rig, name: []const u8, wide: []const u8, one: u64, row: usize) !void {
    const n = heads * lw * 2;
    try check.sameBytes(name, try r.back(one, n), wide[row * n ..][0..n]);
}

/// Dense causal attention on 1,500 keys: 70 rows at 1,430 vs float64 (32-head), 16-head blocks' bytes, one-row windows;
/// kv8: on the fp8 cache (float64 on its codes times scales), and a second call's bytes the same.
pub fn dense(r: *Rig, set: glm.glue.Set, kv8: bool) !void {
    const pos = 1430;
    const rows = 70;
    const scale: f32 = 0.125;
    const q = try r.bfs(rows * heads * lw, 1.0);
    const dq = try r.dev(std.mem.sliceAsBytes(q));
    const c = try cacheOf(r, set, pos + rows, kv8);
    const wide = try attend(r, set, dq, c, pos, rows, scale, latent.heads_tile_wide);
    const keys = try r.a.alloc([]const u32, rows);
    for (keys, 0..) |*k, row| {
        const all = try r.a.alloc(u32, pos + row + 1);
        for (all, 0..) |*v, i| v.* = @intCast(i);
        k.* = all;
    }
    const ref64 = try reference(r, q, c.f64s, keys, scale);
    var name: [128]u8 = undefined;
    try ref.closeCols(r, try std.fmt.bufPrint(&name, "latent attention{s} {d} rows at {d}, {d} heads, 32-head blocks", .{ tag(kv8), rows, pos, heads }), wide, false, ref64.want, ref64.tol, lw);
    const narrow = try attend(r, set, dq, c, pos, rows, scale, latent.heads_tile);
    try ref.sameRows(r, "latent attention: 16-head blocks equal the 32-head blocks", wide, narrow, rows * heads * lw * 2);
    try ref.sameRows(r, "latent attention: a second call's bytes equal the first's", wide, try attend(r, set, dq, c, pos, rows, scale, latent.heads_tile_wide), rows * heads * lw * 2);
    const all = try r.back(wide, rows * heads * lw * 2);
    for ([_]usize{ 0, 37, 69 }) |row| {
        const one = try attend(r, set, dq + row * heads * lw * 2, c, pos + row, 1, scale, latent.headBlock(1));
        try sameRow(r, try std.fmt.bufPrint(&name, "latent attention: row {d} alone equals its row in the {d}-row call", .{ row, rows }), all, one, row);
    }
    check.pass("latent attention{s} over {d} keys: within float64 tolerance, both head blocks, one-row windows and repeats equal", .{ tag(kv8), pos + rows });
}

/// Sparse attention on indexer lists (<= 1,100 of 6,000 keys): 70 rows vs float64, 1-row windows, count 0 left alone.
pub fn sparse(r: *Rig, set: glm.glue.Set, kv8: bool) !void {
    const keys = 6000;
    const rows = 70;
    const w = 1100;
    const scale: f32 = 0.125;
    const q = try r.bfs(rows * heads * lw, 1.0);
    const dq = try r.dev(std.mem.sliceAsBytes(q));
    const c = try cacheOf(r, set, keys, kv8);
    const tok = try r.a.alloc(i32, rows * w);
    @memset(tok, -1);
    const cnt = try r.a.alloc(i32, rows);
    const sets = try r.a.alloc([]const u32, rows);
    for (sets, 0..) |*set_, row| {
        const pick = try r.a.alloc(u32, 1 + r.rng.uintLessThan(usize, w));
        for (pick) |*v| v.* = r.rng.uintLessThan(u32, keys);
        std.mem.sort(u32, pick, {}, std.sort.asc(u32));
        var n: usize = 0;
        for (pick) |v| if (n == 0 or pick[n - 1] != v) {
            pick[n] = v;
            n += 1;
        };
        set_.* = pick[0..n];
        cnt[row] = @intCast(n);
        for (pick[0..n], tok[row * w ..][0..n]) |v, *t| t.* = @intCast(v);
    }
    const dtok = try r.dev(std.mem.sliceAsBytes(tok));
    const dcnt = try r.dev(std.mem.sliceAsBytes(cnt));
    const out = try r.zeros(rows * heads * lw * 2);
    const nch = latent.chunksFor(w);
    try latent.sparseAttention(set, r.s, dq, c.dev, c.scales, dtok, dcnt, w, try partials(r, nch, rows), out, rows, heads, lw, scale);
    const ref64 = try reference(r, q, c.f64s, sets, scale);
    var name: [128]u8 = undefined;
    try ref.closeCols(r, try std.fmt.bufPrint(&name, "latent sparse attention{s} {d} rows, up to {d} of {d} keys", .{ tag(kv8), rows, w, keys }), out, false, ref64.want, ref64.tol, lw);
    const all = try r.back(out, rows * heads * lw * 2);
    for ([_]usize{ 0, 37, 69 }) |row| {
        const one = try r.zeros(heads * lw * 2);
        try latent.sparseAttention(set, r.s, dq + row * heads * lw * 2, c.dev, c.scales, dtok + row * w * 4, dcnt + row * 4, w, try partials(r, nch, 1), one, 1, heads, lw, scale);
        try sameRow(r, try std.fmt.bufPrint(&name, "latent sparse attention: row {d} alone equals its row in the {d}-row call", .{ row, rows }), all, one, row);
    }
    const none: i32 = 0;
    const left = try r.zeros(heads * lw * 2);
    try latent.sparseAttention(set, r.s, dq, c.dev, c.scales, dtok, try r.dev(std.mem.asBytes(&none)), w, try partials(r, nch, 1), left, 1, heads, lw, scale);
    for (try r.back(left, heads * lw * 2)) |b| try check.expect(b == 0xff, "latent sparse attention: a row with count 0 is left alone", .{});
    check.pass("latent sparse attention{s} over up to {d} tokens: within float64 tolerance, one-row windows equal, count 0 left alone", .{ tag(kv8), w });
}
