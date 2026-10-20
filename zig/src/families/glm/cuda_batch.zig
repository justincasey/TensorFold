//! Several whole prompts in one prompt forward on bp: row-wise sites take every row, each prompt's mixer runs alone.

const std = @import("std");
const cuda = @import("cuda");
const engine = @import("cuda_engine.zig");
const fw = @import("cuda_forward.zig");
const glue = @import("cuda_glue.zig");
const b16 = @import("cuda_b16.zig");
const latent = @import("cuda_latent.zig");
const dflash = @import("dflash.zig");
const digest = @import("cuda_digest.zig");
const Engine = engine.Engine;

/// One prompt of a batched pass: its slot's sequence (begun), its ids from position 0; `first` its drawn token.
pub const Seg = struct { slot: u32, ids: []const u32, first: u32 = 0 };

/// A batched pass's prompts at most (the lane core's max_together).
pub const max_segs = 16;

/// Every segment's whole prompt (each one chunk from 0) in one forward; each segment's first token in `first`.
pub fn run(e: *Engine, segs: []Seg) !void {
    if (segs.len == 0 or segs.len > max_segs) return error.BadBatch;
    if (e.mtp and e.drafter == null) return error.BadBatch; // the MTP head's prompt rows run per chunk only
    const b = &e.bp;
    var R: usize = 0;
    for (segs) |g| {
        if (g.ids.len == 0 or g.ids.len > fw.prefill_rows) return error.BadBatch;
        R += g.ids.len;
    }
    if (R > b.rows) return error.BadBatch;
    for (segs) |g| {
        try e.bind(g.slot);
        if (e.cur.layer != 0 or try e.prefillStart(g.ids.len, null) != 0) return error.BadBatch;
    }
    try e.gate(null);
    const ids = try e.gpa.alloc(u32, R);
    defer e.gpa.free(ids);
    var row0: usize = 0;
    for (segs) |g| {
        @memcpy(ids[row0..][0..g.ids.len], g.ids);
        row0 += g.ids.len;
    }
    errdefer for (segs) |g| { // a failed pass leaves no rows pending: each stream's release resets its sequence
        e.seqs.items[g.slot].?.st.pending = 0;
    };
    for (segs) |g| {
        const st = &e.seqs.items[g.slot].?.st;
        try st.ready(e.stream, g.ids.len);
        st.pending = g.ids.len;
        st.prompt = true;
    }
    try cuda.copy_source.enqueue(e.d, b.ids, std.mem.sliceAsBytes(ids), e.stream);
    var one: [1]fw.Rank = undefined;
    var f = e.view(b, &one);
    f.images = null;
    try layers(e, &f, segs, R);
    const c = &e.w.config;
    const D = c.hidden;
    row0 = 0;
    for (segs) |*g| {
        const n = g.ids.len;
        try e.bind(g.slot);
        const taps = if (e.drafter) |dr| dr.taps_prefill + row0 * dflash.TAP_ROW else 0;
        if (e.drafter) |dr| try dr.promptTaps(e, g.slot, 0, n, n, taps);
        try b16.matmul(f.set(), f.stream, b.fnormed + (row0 + n - 1) * D * 2, D, e.w.head, b.logits, b.kda.mm_part, 1, e.w.share.vocab.len(), D, false);
        var t: [1]u32 = undefined;
        try e.draw(b.logits, 1, n, &t);
        g.first = t[0];
        var rk: [1]fw.Rank = .{.{ .w = &e.w, .st = &e.cur.st, .b = b }};
        var fc = f;
        fc.ranks = &rk;
        try fc.commit(n);
        if (digest.on()) try digest.seq(e, &e.cur.st, g.ids, taps, b.logits, "batch", t[0]);
        row0 += n;
    }
}

/// Every layer over `R` rows in each prompt's own solo pieces (their fills, exchanges and glue), each mixer alone.
fn layers(e: *Engine, f: *const fw.Forward, segs: []const Seg, R: usize) !void {
    const b = &e.bp;
    const c = &e.w.config;
    const D = c.hidden;
    const p = f.pieces orelse return error.BadBatch;
    if (!f.split) return error.BadBatch; // a one-piece chunk runs pieced only when split
    var cut: [fw.max_pieces][2]usize = undefined;
    var spans: [max_segs][2]usize = undefined;
    var q: usize = 0;
    var row0: usize = 0;
    for (segs, spans[0..segs.len]) |g, *s| {
        var own: [fw.max_pieces][2]usize = undefined;
        const k = fw.ranges(g.ids.len, p.count, &own);
        if (q + k > cut.len) return error.BadBatch;
        for (own[0..k], cut[q..][0..k]) |x, *y| y.* = .{ row0 + x[0], row0 + x[1] };
        s.* = .{ q, q + k };
        q += k;
        row0 += g.ids.len;
    }
    const r = f.ranks[0];
    try glue.embedB16(f.set(), f.stream, b.ids, e.w.embed, b.x, R, D, c.hc);
    try f.pre(r, e.w.layers[0].attn_hc, e.w.layers[0].in_norm, R);
    var li: usize = 0;
    var di: usize = 0;
    for (e.w.layers, 0..) |l, j| {
        row0 = 0;
        for (segs, spans[0..segs.len]) |g, s| {
            const st = &e.seqs.items[g.slot].?.st;
            const n = g.ids.len;
            var sb = b.*;
            sb.normed = b.normed + row0 * D * 2;
            const rs: fw.Rank = .{ .w = &e.w, .st = st, .b = &sb };
            try f.mixer(rs, l, li, di, n, @min(latent.chunksFor(st.pos + n), b.nch), null);
            try f.exchangePieces(p, j, .mixer, cut[s[0]..s[1]], s[0], row0);
            row0 += n;
        }
        try f.gluePieces(p, j, .mixer, cut[0..q]);
        try f.ffn(r, l, R, null);
        try f.exchangePieces(p, j, .ffn, cut[0..q], 0, 0);
        try f.gluePieces(p, j, .ffn, cut[0..q]);
        if (f.taps) |t| try t.at(f.stream, j, b.x, R);
        switch (l.mixer) {
            .kda => li += 1,
            .mla => di += 1,
        }
    }
    try glue.rmsnorm(f.set(), f.stream, b.hidden, D, e.w.norm, b.fnormed, D, b.fxs, c.eps, R, D);
}
