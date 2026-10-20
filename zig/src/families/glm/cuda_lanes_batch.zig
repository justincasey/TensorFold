//! Fresh short prompts in one prompt pass (cuda_batch; TF_GLM_BATCH_FILL=0: off), each stream's first token held.

const std = @import("std");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const batch = @import("cuda_batch.zig");
const glm_lanes = @import("cuda_lanes.zig");
const Lanes = glm_lanes.Lanes;

/// Prompts shorter than this fill alone: their solo pass takes the narrow kernels (other bits than a batch's).
pub const min_rows = 64;

/// A pass's rows at most while other streams hold lanes (they wait for the whole pass).
pub const busy_rows = 2048;

/// The streams' prompts in one pass; false (nothing done) when one resumes, has images or the rows do not fit.
pub fn prefillMany(self: *Lanes, ss: []const *lanes.Stream) !bool {
    if (ss.len < 2 or ss.len > batch.max_segs) return false;
    var it = self.lanes.valueIterator();
    while (it.next()) |l| if (l.layer > 0) return false; // a chunk between its layer groups holds bp
    const busy = self.lanes.count() > 0;
    var rows: usize = 0;
    for (ss) |s| {
        const n = s.prompt().len;
        if (n < min_rows or n > engine.prompt_rows or s.images.len > 0 or self.resumable(s, n) != 0) return false;
        if (n + s.max_new + glm_lanes.max_rows >= self.x.e.capacity or (busy and n > engine.aside_rows)) return false;
        rows += n;
    }
    if (rows > @as(usize, if (busy) busy_rows else engine.prompt_rows)) return false;
    var segs: [batch.max_segs]batch.Seg = undefined;
    for (ss, segs[0..ss.len]) |s, *g| {
        if (try self.opening(s) != null) return error.BadBatch;
        g.* = .{ .slot = self.lanes.getPtr(s).?.slot, .ids = s.prompt() };
    }
    try self.x.batch(segs[0..ss.len]);
    for (ss, segs[0..ss.len]) |s, g| self.lanes.getPtr(s).?.first = self.take(g.first);
    return true;
}
