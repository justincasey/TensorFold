//! GLM-5.3-Flash's shared windows: several streams' verify windows or whole prompts in one forward, chains per segment.

const std = @import("std");
const cuda = @import("cuda");
const fw = @import("cuda_forward.zig");
const weights = @import("cuda_weights.zig");
const kda = @import("cuda_kda.zig");
const latent = @import("cuda_latent.zig");
const kda_block = @import("cuda_kda_block.zig");
const dsa_block = @import("cuda_dsa_block.zig");
const glue = @import("cuda_glue.zig");
const projection = @import("cuda_proj.zig");

/// Checks a shared window's segments (a prompt buffer's: whole prompts) against its `rows` rows; each holds its own.
pub fn begin(r: fw.Rank, rows: usize, stream: cuda.Stream) !void {
    const most: usize = if (r.b.prefill) fw.prefill_rows else fw.window_rows;
    if (r.b.prefill and r.segs.len > @import("cuda_state.zig").prompt_heads) return error.BadWindow;
    var at: usize = 0;
    for (r.segs) |g| {
        if (g.row0 != at or g.rows == 0 or g.rows > most) return error.BadWindow;
        try g.st.ready(stream, g.rows);
        at += g.rows;
    }
    if (at != rows) return error.BadWindow;
    for (r.segs) |g| {
        g.st.pending = g.rows;
        g.st.prompt = r.b.prefill;
    }
}

/// Forward.finish's head on a prompt buffer: each segment's last final-norm row, logits row k for segment k.
pub fn heads(f: *const fw.Forward, r: fw.Rank) !void {
    const D = r.w.config.hidden;
    const b = r.b;
    const V = r.w.share.vocab.len();
    for (r.segs, 0..) |g, k| try projection.matmul(f.set(), f.stream, r.w.head, b.fnormed + (g.row0 + g.rows - 1) * D * 2, D, b.logits + k * V * 2, b.kda.mm_part, 1, V, D, false);
}

/// Forward.mixer on a shared window: all rows' projections, each segment's chain or attention, o_proj (null: pieced).
pub fn mixer(f: *const fw.Forward, r: fw.Rank, l: weights.Layer, li: usize, di: usize, rows: usize, out: ?u64) !void {
    const c = &r.w.config;
    const b = r.b;
    switch (l.mixer) {
        .kda => |k| {
            var s = b.kda;
            s.p = b.kproj;
            if (b.prefill) try promptChains(f, r, k, li, s) else {
                try kda_block.project(f.set(), f.stream, k, s, b.normed, c.hidden, rows);
                if (f.kda_wide) try chainSegs(f, r, k, li, s, rows) else try chainEach(f, r, k, li, s);
            }
            if (out) |o| try kda_block.fill(f.set(), f.stream, k, s, 0, rows, o);
        },
        .mla => |a| if (b.prefill) try promptAttend(f, r, a, di, out) else {
            var sparse_any = false;
            for (r.segs) |g| sparse_any = sparse_any or dsa_block.sparseRows(a, g.st.pos, g.rows);
            try dsa_block.prepare(f.set(), f.stream, a, b.dsa, b.normed, c.hidden, rows, sparse_any);
            try attendSegs(f, r, a, di, rows);
            try latent.expandV(f.set(), f.stream, b.dsa.ol, a.wv, b.dsa.vn, rows, a.heads, a.dv, a.lw, a.wkt != 0);
            if (out) |o| try dsa_block.fill(f.set(), f.stream, a, b.dsa, 0, rows, o);
        },
    }
}

/// Each segment's rows into its caches, every dense segment's attention in one pass, then each sparse segment's.
fn attendSegs(f: *const fw.Forward, r: fw.Rank, a: dsa_block.Layer, di: usize, rows: usize) !void {
    const b = r.b;
    const k: dsa_block.Kernels = .{ .set = f.set(), .ops = f.m.ops() };
    var segs: latent.Segs = .{ .n = @intCast(r.segs.len) };
    if (r.segs.len > segs.row0.len) return error.BadWindow;
    var nch: usize = 0;
    var fp8 = false;
    for (r.segs, 0..) |g, i| {
        const cs = g.st.caches[di];
        try dsa_block.write(k, f.stream, a, cs, b.dsa, g.row0, g.rows, g.st.pos_dev);
        segs.row0[i] = @intCast(g.row0);
        if (g.st.pos >= a.dense_limit) continue;
        segs.pos[i] = g.st.pos_dev;
        segs.lc[i] = cs.lc;
        segs.ls[i] = cs.ls;
        segs.pt[i] = cs.pt;
        segs.pstep = @intCast(cs.ps);
        fp8 = cs.ls != 0;
        nch = @max(nch, @min(latent.chunksFor(g.st.pos + g.rows), b.nch));
    }
    if (nch > 0) try latent.attentionSegs(k.set, f.stream, &segs, fp8, b.dsa.qa, b.dsa.partials, b.dsa.ol, b.dsa.dummy, rows, a.heads, a.lw, nch, dsa_block.scaleOf(a));
    for (r.segs) |g| if (dsa_block.sparseRows(a, g.st.pos, g.rows)) try dsa_block.sparseAttend(k, f.stream, a, g.st.caches[di], b.dsa, g.row0, g.rows, g.st.pos_dev, g.st.pos, b.part_rows);
}

/// A prompt buffer's DSA layer: each whole prompt's projections, attention and value rows as its pass alone has them.
fn promptAttend(f: *const fw.Forward, r: fw.Rank, a: dsa_block.Layer, di: usize, out: ?u64) !void {
    const b = r.b;
    const D = r.w.config.hidden;
    const k: dsa_block.Kernels = .{ .set = f.set(), .ops = f.m.ops() };
    for (r.segs) |g| {
        const s = dsa_block.from(a, b.dsa, g.row0);
        try dsa_block.prepare(k.set, f.stream, a, s, b.normed + g.row0 * D * 2, D, g.rows, dsa_block.sparseRows(a, g.st.pos, g.rows));
        const nch = @min(latent.chunksFor(g.st.pos + g.rows), b.nch);
        try dsa_block.attend(k, f.stream, a, g.st.caches[di], b.dsa, g.row0, g.rows, g.st.pos_dev, g.st.pos, nch, b.part_rows);
        const vn = b.dsa.vn + g.row0 * a.heads * a.dv * 2;
        try latent.expandV(k.set, f.stream, s.ol, a.wv, vn, g.rows, a.heads, a.dv, a.lw, a.wkt != 0);
        if (out) |o| try projection.matmul(k.set, f.stream, a.o, vn, a.heads * a.dv, o + g.row0 * D * 4, b.dsa.mm_part, g.rows, a.dims, a.heads * a.dv, true);
    }
}

/// A prompt buffer's KDA layer: each whole prompt projected and chained as its pass alone would, committed now.
fn promptChains(f: *const fw.Forward, r: fw.Rank, k: kda_block.Layer, li: usize, s: kda_block.Scratch) !void {
    const n = k.heads * kda.dk;
    const w = k.width();
    const D = r.w.config.hidden;
    for (r.segs) |g| {
        const st = g.st;
        const cur = st.cur[li];
        var v = s;
        v.ka += g.row0 * n * 2;
        v.kg += g.row0 * n * 2;
        v.saves.out += g.row0 * n * 2;
        v.saves.state_out = st.recAt(1 - cur, li);
        const conv = st.conv + li * 3 * st.conv_c * 2;
        v.p = s.p + g.row0 * w * 2;
        try kda_block.project(f.set(), f.stream, k, v, r.b.normed + g.row0 * D * 2, D, g.rows);
        try kda_block.chainRows(&f.m.kda, f.stream, k, v, g.rows, conv, st.recAt(cur, li), false, true);
        try glue.convShift(f.set(), f.stream, conv, v.p, g.rows, g.rows * w, w, 1, 3 * n);
        st.cur[li] = 1 - cur;
    }
}

/// The KDA layer's chains, every segment's in one three-kernel pass (each on its own sequence: the same bits).
fn chainSegs(f: *const fw.Forward, r: fw.Rank, k: kda_block.Layer, li: usize, s: kda_block.Scratch, rows: usize) !void {
    var segs: kda.Segs = .{ .n = @intCast(r.segs.len) };
    if (r.segs.len > kda.max_segs) return error.BadWindow;
    for (r.segs, 0..) |g, i| {
        const st = g.st;
        const cur = st.cur[li];
        const kv = li * fw.window_rows * st.heads * kda.dk;
        segs.row0[i] = @intCast(g.row0);
        segs.rows[i] = @intCast(g.rows);
        segs.cs[i] = st.conv + li * 3 * st.conv_c * 2;
        segs.state_in[i] = st.recAt(cur, li);
        segs.state_out[i] = st.recAt(1 - cur, li);
        segs.keep[i] = st.proj + li * st.proj_layer * 2; // the conv channels stay with the layer until its commit
        segs.k[i] = st.save_k + kv * 4;
        segs.v[i] = st.save_v + kv * 2;
        segs.g[i] = st.save_g + kv * 4;
        segs.b[i] = st.save_b + li * fw.window_rows * st.heads * 4;
    }
    try kda_block.chainSegs(&f.m.kda, f.stream, k, s, &segs, rows);
}

/// The KDA layer's chains a segment at a time (TF_GLM_KDA_DECODE_WIDE=0: each on the fused kernel).
fn chainEach(f: *const fw.Forward, r: fw.Rank, k: kda_block.Layer, li: usize, s: kda_block.Scratch) !void {
    const n = k.heads * kda.dk;
    const w = k.width();
    for (r.segs) |g| {
        const st = g.st;
        const cur = st.cur[li];
        var v = s;
        v.p = st.proj + li * st.proj_layer * 2; // the window's rows stay with the layer until its commit
        try f.d.check(f.d.api.cuMemcpyDtoDAsync_v2(v.p, s.p + g.row0 * w * 2, g.rows * w * 2, f.stream.handle), "cuMemcpyDtoDAsync");
        v.ka += g.row0 * n * 2;
        v.kg += g.row0 * n * 2;
        v.saves.out += g.row0 * n * 2;
        v.saves.state_out = st.recAt(1 - cur, li);
        const kv = li * fw.window_rows * st.heads * kda.dk;
        v.saves.k = st.save_k + kv * 4;
        v.saves.v = st.save_v + kv * 2;
        v.saves.g = st.save_g + kv * 4;
        v.saves.b = st.save_b + li * fw.window_rows * st.heads * 4;
        try kda_block.chainRows(&f.m.kda, f.stream, k, v, g.rows, st.conv + li * 3 * st.conv_c * 2, st.recAt(cur, li), false, false);
    }
}
