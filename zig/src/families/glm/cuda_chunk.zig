//! A prompt chunk's forward in layer groups: other sequences' steps may run between two groups (same kernels, bits).

const std = @import("std");
const engine = @import("cuda_engine.zig");
const fw = @import("cuda_forward.zig");
const digest = @import("cuda_digest.zig");
const Engine = engine.Engine;

const labels = [_][]const u8{ "prompt chunk to 64 rows", "prompt chunk to 256 rows", "prompt chunk to 1024 rows", "prompt chunk to 2048 rows" };

/// Layers lo..hi of the chunk at `start` (`part`: rows, next id) on bp or bq (`aside`): 0 readies, the last draws.
pub fn run(e: *Engine, part: []const u32, start: usize, len: usize, from: ?engine.Resume, cancel: ?engine.Cancel, marks: ?engine.Marks, lo: usize, hi: usize, aside: bool) !?u32 {
    const layers = e.w.layers.len;
    if (lo >= hi or hi > layers or lo != e.cur.layer) return error.BadChunk;
    if (lo == 0 and (from != null or start == 0) and try e.prefillStart(len, from) != start) return error.BadChunk;
    if (start >= len or start % fw.prefill_rows != 0) return error.BadChunk;
    const rows = @min(len - start, fw.prefill_rows);
    const b = if (aside) &e.bq else &e.bp;
    if (rows > b.rows) return error.BadChunk;
    if (part.len != @min(len - start, rows + 1)) return error.BadChunk;
    const st = &e.cur.st;
    if (lo > 0 and (!st.prompt or st.pending != rows or st.pos != start)) return error.BadChunk;
    errdefer { // a failed or cancelled group leaves no chunk in flight: the stream's release frees or resets its rows
        e.cur.layer = 0;
        st.pending = 0;
    }
    var one: [1]fw.Rank = undefined;
    var f = e.view(b, &one);
    const as_i32: []const i32 = @ptrCast(part);
    if (lo == 0) if (marks) |m| if (start > 0 and std.mem.indexOfScalar(u32, m.at, @intCast(start)) != null) m.call(m.ctx, @intCast(start));
    const size: usize = if (rows <= 64) 0 else if (rows <= 256) 1 else if (rows <= 1024) 2 else 3;
    f.prof = if (lo == 0 and hi == layers) e.prompt_profs[size] else null; // a grouped chunk's sections span other steps
    if (f.prof) |p| try p.begin(e.stream, labels[size]);
    defer if (f.prof) |p| p.end();
    try e.gate(cancel);
    if (f.prof) |p| p.mark(e.stream, .gate);
    if (lo == 0) try f.begin(as_i32[0..rows]);
    try f.layersOf(rows, lo, hi);
    if (hi < layers) {
        e.cur.layer = hi;
        return null;
    }
    e.cur.layer = 0;
    if (e.drafter) |dr| try dr.prompt(e, e.slot, start, rows, len, aside);
    const last = start + rows == len;
    var first: ?u32 = null;
    if (last) {
        var t: [1]u32 = undefined;
        try e.draw(b.logits, 1, len, &t);
        first = t[0];
    }
    if (f.prof) |p| p.mark(e.stream, .draw);
    if (e.mtp and e.drafter == null) { // DFlash2 drafts without the MTP head: its rows are never read
        if (last) try e.copy(e.hold.ptr, b.fnormed + (rows - 1) * e.c.hidden * 2, e.c.hidden * 2);
        const nxt = as_i32[1..];
        if (nxt.len > 0) { // decode._absorb_rows
            const mp = f.prof;
            f.prof = null; // the head's pass as one section
            try f.mtp(nxt, &.{b.fnormed});
            f.prof = mp;
            try st.setMtpLen(e.stream, st.mtp_len + nxt.len);
        }
    }
    if (f.prof) |p| p.mark(e.stream, .mtp);
    try f.commit(rows);
    if (f.prof) |p| p.mark(e.stream, .commit);
    if (last and start == 0 and digest.on()) try digest.seq(e, st, part[0..rows], if (e.drafter) |dr| (if (aside) dr.taps_aside else dr.taps_prefill) else 0, b.logits, "solo", first.?);
    return first;
}
