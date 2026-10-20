//! Engine.verifyShared: several streams' windows in one forward on this rank, then every part's draws.

const std = @import("std");
const engine = @import("cuda_engine.zig");
const fw = @import("cuda_forward.zig");
const prof_mod = @import("cuda_prof.zig");

const Engine = engine.Engine;
const Part = engine.Part;

/// TF_GLM_PROF's shared windows, by streams: 2, 3 to 4, 5 to 8, 9 to 16.
pub const prof_buckets = 4;
const labels = [prof_buckets][]const u8{ "shared 2 streams", "shared 3-4 streams", "shared 5-8 streams", "shared 9-16 streams" };

fn bucket(streams: usize) usize {
    return if (streams <= 2) 0 else if (streams <= 4) 1 else if (streams <= 8) 2 else 3;
}

/// Several streams' windows in one forward: shared row ops, per-part attention and KDA.
pub fn verifyShared(e: *Engine, parts: []const Part) !void {
    const b = if (e.bs) |*x| x else return error.NoSharedWindows;
    if (parts.len == 0 or parts.len > engine.max_shared) return error.WindowTooWide;
    var segs: [engine.max_shared]fw.Seg = undefined;
    var ids: [engine.shared_rows]u32 = undefined;
    var rows: usize = 0;
    for (parts, 0..) |p, i| {
        if (p.ids.len == 0 or p.ids.len > fw.window_rows or rows + p.ids.len > engine.shared_rows) return error.WindowTooWide;
        for (parts[0..i]) |q| if (q.slot == p.slot) return error.SlotTwice;
        try e.bind(p.slot);
        segs[i] = .{ .st = &e.cur.st, .row0 = rows, .rows = p.ids.len };
        @memcpy(ids[rows..][0..p.ids.len], p.ids);
        rows += p.ids.len;
    }
    var one: [1]fw.Rank = .{.{ .w = &e.w, .st = segs[0].st, .b = b, .segs = segs[0..parts.len] }};
    var f: fw.Forward = .{ .d = e.d, .m = &e.m, .stream = e.stream, .ranks = &one, .exchange = e.exchange(), .pieces = null, .split = e.settings.split, .ring = e.settings.ring, .kda_wide = e.settings.kda_wide, .taps = if (e.drafter) |dr| dr.taps(.step) else null };
    f.prof = e.shared_profs[bucket(parts.len)];
    if (f.prof) |p| {
        try p.begin(e.stream, labels[bucket(parts.len)]);
        p.rows += rows;
    }
    defer if (f.prof) |p| p.end();
    try f.run(@ptrCast(ids[0..rows]));
    if (e.drafter) |dr| for (parts, segs[0..parts.len], 0..) |p, g, i| try dr.noteVerify(p.slot, g.row0, g.rows, g.st.pos, i == 0);
    for (parts, segs[0..parts.len]) |p, g| {
        try e.bind(p.slot);
        e.cur.rows_at = b.fnormed + g.row0 * e.c.hidden * 2;
        try e.draw(b.logits + g.row0 * e.share.vocab.len() * 2, g.rows, g.st.pos + 1, p.out);
    }
    if (f.prof) |p| p.mark(e.stream, .draw);
}

test "shared window profiles bucket by streams" {
    try std.testing.expectEqual(@as(usize, 0), bucket(2));
    try std.testing.expectEqual(@as(usize, 1), bucket(4));
    try std.testing.expectEqual(@as(usize, 2), bucket(5));
    try std.testing.expectEqual(@as(usize, 3), bucket(16));
}
