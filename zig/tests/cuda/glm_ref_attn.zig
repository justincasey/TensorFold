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

/// A device latent cache: bf16 rows or e4m3 codes + fp32 scales; f64s, `src` bf16 rows, `t` its page table.
const Cache = struct { dev: u64, scales: u64, f64s: []f64, src: u64, t: latent.Table = .{} };

/// e4m3 (fn) code b's value: subnormals m 2^-9, normals (1 + m / 8) 2^(e - 7); 0x7f / 0xff (NaN) never written.
fn e4m3(b: u8) f64 {
    const e: i32 = (b >> 3) & 15;
    const m: f64 = @floatFromInt(b & 7);
    const v = if (e == 0) m * std.math.pow(f64, 2, -9) else (1 + m / 8) * std.math.pow(f64, 2, @floatFromInt(e - 7));
    return if (b & 0x80 != 0) -v else v;
}

/// `keys` random bf16 rows scaled by 2^(key % 7 - 3); kv8: written and checked by the fp8 cache write.
fn cacheOf(r: *Rig, set: glm.glue.Set, keys: usize, kv8: bool) !Cache {
    const c = try r.bfs(keys * lw, 1.0);
    for (c, 0..) |*v, i| {
        const k: i32 = @as(i32, @intCast(i / lw % 7)) - 3;
        if (v.* & 0x7fff != 0) v.* = @intCast(@as(i32, v.*) + (k << 7));
    }
    const cd = try r.a.alloc(f64, keys * lw);
    for (cd, c) |*d, v| d.* = bf(v);
    const rows = try r.dev(std.mem.sliceAsBytes(c));
    if (!kv8) return .{ .dev = rows, .scales = 0, .f64s = cd, .src = rows };
    const codes = try r.zeros(keys * lw);
    const scales = try r.zeros(keys * 4);
    const zero: i32 = 0;
    try latent.latentWriteFormat(set, r.s, rows, lw, codes, scales, .{}, try r.dev(std.mem.asBytes(&zero)), keys, lw);
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
    return .{ .dev = codes, .scales = scales, .f64s = cd, .src = rows };
}

/// The same rows on pages in reverse order, 3 KiB apart beyond their rows, to catch in-order or overrun reads.
fn paged(r: *Rig, set: glm.glue.Set, c: Cache, keys: usize) !Cache {
    const kv8 = c.scales != 0;
    const row: usize = if (kv8) lw else lw * 2;
    const ps = 512 * row + (if (kv8) @as(usize, 512 * 4) else 0) + 3 * 1024;
    const n = latent.chunksFor(keys);
    const ids = try r.a.alloc(i32, n);
    for (ids, 0..) |*id, i| id.* = @intCast(n - 1 - i);
    const base = try r.zeros(n * ps);
    const t: latent.Table = .{ .pt = try r.dev(std.mem.sliceAsBytes(ids)), .ps = ps, .page_rows = ps / row };
    const scales: u64 = if (kv8) base + 512 * lw else 0;
    const zero: i32 = 0;
    try latent.latentWriteFormat(set, r.s, c.src, lw, base, scales, t, try r.dev(std.mem.asBytes(&zero)), keys, lw);
    return .{ .dev = base, .scales = scales, .f64s = c.f64s, .src = c.src, .t = t };
}

/// latent.attention of `rows` rows at `pos` (keys 0 .. pos + r each) with head block hb into a fresh output.
fn attend(r: *Rig, set: glm.glue.Set, q: u64, c: Cache, pos: usize, rows: usize, scale: f32, hb: usize) !u64 {
    const nch = latent.chunksFor(pos + rows);
    const out = try r.zeros(rows * heads * lw * 2);
    const p0: i32 = @intCast(pos);
    const at = try r.dev(std.mem.asBytes(&p0));
    try latent.attentionFormat(set, r.s, q, c.dev, c.scales, c.t, at, try partials(r, nch, rows), out, try r.zeros(4), rows, heads, lw, nch, scale, hb);
    return out;
}

/// Row `row` of a [rows][heads][lw] bf16 output alone equals that row of the wide call's bytes.
fn sameRow(r: *Rig, name: []const u8, wide: []const u8, one: u64, row: usize) !void {
    const n = heads * lw * 2;
    try check.sameBytes(name, try r.back(one, n), wide[row * n ..][0..n]);
}

/// Dense causal attention on 1,500 keys vs float64, 16-head blocks, one-row windows; kv8 on the fp8 cache.
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
    const on_pages = try paged(r, set, c, pos + rows);
    try ref.sameRows(r, "latent attention: on pages (written through the table) equals the window", wide, try attend(r, set, dq, on_pages, pos, rows, scale, latent.heads_tile_wide), rows * heads * lw * 2);
    try segments(r, set, dq, c, on_pages, scale);
    const all = try r.back(wide, rows * heads * lw * 2);
    for ([_]usize{ 0, 37, 69 }) |row| {
        const one = try attend(r, set, dq + row * heads * lw * 2, c, pos + row, 1, scale, latent.headBlock(1));
        try sameRow(r, try std.fmt.bufPrint(&name, "latent attention: row {d} alone equals its row in the {d}-row call", .{ row, rows }), all, one, row);
    }
    check.pass("latent attention{s} over {d} keys: within float64 tolerance, both head blocks, one-row windows, repeats and pages equal", .{ tag(kv8), pos + rows });
}

/// A shared window's dense pass: segments on their own caches and positions (one on pages), each as its call alone.
fn segments(r: *Rig, set: glm.glue.Set, dq: u64, c: Cache, on_pages: Cache, scale: f32) !void {
    const Seg = struct { c: Cache, pos: usize, rows: usize };
    const segs = [_]Seg{ .{ .c = c, .pos = 1000, .rows = 1 }, .{ .c = on_pages, .pos = 1200, .rows = 3 }, .{ .c = c, .pos = 400, .rows = 4 } };
    var s: latent.Segs = .{ .n = segs.len };
    var rows: usize = 0;
    var nch: usize = 0;
    for (segs, 0..) |g, i| {
        const p: i32 = @intCast(g.pos);
        s.row0[i] = @intCast(rows);
        s.pos[i] = try r.dev(std.mem.asBytes(&p));
        s.lc[i] = g.c.dev;
        s.ls[i] = g.c.scales;
        s.pt[i] = g.c.t.pt;
        if (g.c.t.pt != 0) s.pstep = @intCast(g.c.t.ps);
        rows += g.rows;
        nch = @max(nch, latent.chunksFor(g.pos + g.rows));
    }
    const out = try r.zeros(rows * heads * lw * 2);
    try latent.attentionSegs(set, r.s, &s, c.scales != 0, dq, try partials(r, nch, rows), out, try r.zeros(4), rows, heads, lw, nch, scale);
    const all = try r.back(out, rows * heads * lw * 2);
    var name: [128]u8 = undefined;
    var row0: usize = 0;
    for (segs, 0..) |g, i| {
        const n = g.rows * heads * lw * 2;
        const alone = try attend(r, set, dq + row0 * heads * lw * 2, g.c, g.pos, g.rows, scale, latent.heads_tile);
        try check.sameBytes(try std.fmt.bufPrint(&name, "latent attention: shared window segment {d} equals its call alone", .{i}), try r.back(alone, n), all[row0 * heads * lw * 2 ..][0..n]);
        row0 += g.rows;
    }
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
    try latent.sparseAttentionBlock(set, r.s, dq, c.dev, c.scales, .{}, dtok, dcnt, w, try partials(r, nch, rows), out, rows, heads, lw, scale, latent.headBlock(rows));
    const ref64 = try reference(r, q, c.f64s, sets, scale);
    var name: [128]u8 = undefined;
    try ref.closeCols(r, try std.fmt.bufPrint(&name, "latent sparse attention{s} {d} rows, up to {d} of {d} keys", .{ tag(kv8), rows, w, keys }), out, false, ref64.want, ref64.tol, lw);
    const all = try r.back(out, rows * heads * lw * 2);
    for ([_]usize{ 0, 37, 69 }) |row| {
        const one = try r.zeros(heads * lw * 2);
        try latent.sparseAttentionBlock(set, r.s, dq + row * heads * lw * 2, c.dev, c.scales, .{}, dtok + row * w * 4, dcnt + row * 4, w, try partials(r, nch, 1), one, 1, heads, lw, scale, latent.headBlock(rows));
        try sameRow(r, try std.fmt.bufPrint(&name, "latent sparse attention: row {d} alone equals its row in the {d}-row call", .{ row, rows }), all, one, row);
    }
    const on_pages = try paged(r, set, c, keys);
    const ids = try r.a.alloc(i32, latent.chunksFor(keys));
    for (ids, 0..) |*id, i| id.* = @intCast(ids.len - 1 - i);
    const mapped = try r.a.alloc(i32, tok.len); // the token list as tf_glm_dsa_tokens maps it onto the pages
    for (tok, mapped) |v, *m| m.* = if (v < 0) v else ids[@intCast(@divTrunc(v, 512))] * @as(i32, @intCast(on_pages.t.page_rows)) + @rem(v, 512);
    const paged_out = try r.zeros(rows * heads * lw * 2);
    try latent.sparseAttentionBlock(set, r.s, dq, on_pages.dev, on_pages.scales, on_pages.t, try r.dev(std.mem.sliceAsBytes(mapped)), dcnt, w, try partials(r, nch, rows), paged_out, rows, heads, lw, scale, latent.headBlock(rows));
    try ref.sameRows(r, "latent sparse attention: on pages (mapped tokens) equals the window", out, paged_out, rows * heads * lw * 2);
    const none: i32 = 0;
    const left = try r.zeros(heads * lw * 2);
    try latent.sparseAttentionBlock(set, r.s, dq, c.dev, c.scales, .{}, dtok, try r.dev(std.mem.asBytes(&none)), w, try partials(r, nch, 1), left, 1, heads, lw, scale, latent.headBlock(1));
    for (try r.back(left, heads * lw * 2)) |b| try check.expect(b == 0xff, "latent sparse attention: a row with count 0 is left alone", .{});
    check.pass("latent sparse attention{s} over up to {d} tokens: within float64 tolerance, one-row windows and pages equal, count 0 left alone", .{ tag(kv8), w });
}
