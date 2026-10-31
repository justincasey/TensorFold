//! A prompt chunk's forward in layer groups: other sequences' steps may run between two groups (same kernels, bits).

const std = @import("std");
const engine = @import("cuda_engine.zig");
const fw = @import("cuda_forward.zig");
const dflash = @import("dflash.zig");
const prof_mod = @import("cuda_prof.zig");
const prompt_lp = @import("cuda_prompt_logprobs.zig");
const snapshots = @import("cuda_snapshots.zig");
const Engine = engine.Engine;

const labels = [_][]const u8{ "prompt chunk to 64 rows", "prompt chunk to 256 rows", "prompt chunk to 1024 rows", "prompt chunk to 4096 rows" };

/// Layers lo..hi of the chunk at `start` (`part`: rows, next id) on bp or bq (`aside`): 0 gates, the last draws.
pub fn run(e: *Engine, part: []const u32, start: usize, len: usize, from: ?engine.Resume, cancel: ?engine.Cancel, marks: ?engine.Marks, lo: usize, hi: usize, aside: bool) !?u32 {
    const split = e.index_split; // this step's alone (every group of the chunk sends the same)
    e.index_split = false;
    const layers = e.w.layers.len;
    if (lo >= hi or hi > layers or lo != e.cur.layer) return error.BadChunk;
    if (lo == 0 and (from != null or start == 0) and try e.prefillStart(len, from) != start) return error.BadChunk;
    if (start >= len or start % fw.prefill_rows != 0) return error.BadChunk;
    const rows = @min(len - start, fw.prefill_rows);
    const b = if (aside) &e.bq else &e.bp;
    if (rows > b.rows or part.len != @min(len - start, rows + 1)) return error.BadChunk;
    const st = &e.cur.st;
    if (lo > 0 and (!st.prompt or st.pending != rows or st.pos != start)) return error.BadChunk;
    errdefer { // a failed group leaves no chunk in flight: the stream's release frees or resets its rows
        e.cur.layer = 0;
        st.pending = 0;
    }
    var one: [1]fw.Rank = undefined;
    var f = e.view(b, &one);
    f.index_split = split;
    const as_i32: []const i32 = @ptrCast(part);
    if (f.taps) |*t| t.keep = dflash.keepFrom(len) -| start; // a split piece's tap rows before it aren't swapped
    if (lo == 0) if (marks) |m| if (start > 0 and std.mem.indexOfScalar(u32, m.at, @intCast(start)) != null) m.call(m.ctx, @intCast(start));
    const size: usize = if (rows <= 64) 0 else if (rows <= 256) 1 else if (rows <= 1024) 2 else 3;
    f.prof = if (lo == 0 and hi == layers) e.prompt_profs[size] else null; // a grouped chunk's sections span other steps
    if (f.prof) |p| try p.beginPrompt(e.stream, labels[size], .{ .rank = e.share.rank, .slot = e.slot, .start = start, .rows = rows }, e.c.run, e.pieces.count);
    if (f.prof) |p| e.profiling_prompt = p;
    defer if (f.prof != null) {
        e.profiling_prompt = null;
    };
    defer if (f.prof) |p| p.end();
    errdefer if (f.prof) |p| {
        p.prompt.?.invalid = true;
    };
    if (lo == 0) { // the gate before any layer: a cancel past it would leave KDA states advanced past the kept rows
        var stamp = prof_mod.span(f.prof, e.stream, .gate);
        e.gate(cancel) catch |err| {
            if (err == error.Cancelled) try snapshots.interrupted(e, marks, @intCast(start));
            return err;
        };
        stamp.end();
        if (f.prof) |p| p.mark(e.stream, .gate);
        try f.begin(as_i32[0..rows]);
        e.index_splits += @intFromBool(split);
    }
    try f.layersOf(rows, lo, hi);
    if (hi < layers) {
        e.cur.layer = hi;
        return null;
    }
    e.cur.layer = 0;
    var stamp = prof_mod.span(f.prof, e.stream, .prompt_score);
    if (e.prompt_logprobs) |p| if (part.len > 1) try prompt_lp.run(e, p, part[1..]);
    stamp.end();
    stamp = prof_mod.span(f.prof, e.stream, .draw);
    const last = start + rows == len;
    var first: ?u32 = null;
    if (last) {
        var t: [1]u32 = undefined;
        try e.draw(b.logits, 1, len, 0, &t);
        first = t[0];
    }
    stamp.end();
    if (f.prof) |p| p.mark(e.stream, .draw);
    stamp = prof_mod.span(f.prof, e.stream, .drafter);
    if (e.drafter) |dr| try dr.prompt(e, e.slot, start, rows, len, aside); // after the draw: no first token waits on it
    stamp.end();
    if (last) try e.holdPrompt(b.fnormed + (rows - 1) * e.c.hidden * 2);
    stamp = prof_mod.span(f.prof, e.stream, .mtp);
    if (e.mtp and e.drafter == null) { // DFlash2 drafts without the MTP head: its rows are never read
        const nxt = as_i32[1..];
        if (nxt.len > 0) { // decode._absorb_rows
            const mp = f.prof;
            defer f.prof = mp;
            f.prof = null; // the head's pass as one section
            try f.mtp(nxt, &.{b.fnormed});
            f.prof = mp;
            try st.setMtpLen(e.stream, st.mtp_len + nxt.len);
        }
    }
    stamp.end();
    if (f.prof) |p| p.mark(e.stream, .mtp);
    stamp = prof_mod.span(f.prof, e.stream, .commit);
    try f.commit(rows);
    stamp.end();
    if (f.prof) |p| p.mark(e.stream, .commit);
    return first;
}
