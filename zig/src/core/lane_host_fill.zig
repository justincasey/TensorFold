//! Prefer foreground and short prompts, with bounded waiting for every admitted fill.
const LaneHost = @import("lane_host.zig").LaneHost;

pub const preference_rounds = 8;

/// After eight skipped choices, oldest wait wins; each fill runs within eight plus the admitted lane count.
pub fn next(h: *const LaneHost) usize {
    var at: usize = 0;
    for (h.filling.items, 0..) |j, i| {
        const best = h.filling.items[at];
        const aged = j.fill_wait >= preference_rounds;
        const best_aged = best.fill_wait >= preference_rounds;
        if (aged != best_aged) {
            if (aged) at = i;
        } else if (aged) {
            if (j.fill_wait > best.fill_wait) at = i;
        } else if (j.request.background != best.request.background) {
            if (!j.request.background) at = i;
        } else if (j.stream.prompt_len - j.stream.cached < best.stream.prompt_len - best.stream.cached) at = i;
    }
    for (h.filling.items, 0..) |j, i| j.fill_wait = if (i == at) 0 else j.fill_wait +| 1;
    return at;
}

/// The filling job whose chunk stands between layer groups (the backend's error.FillBusy: it goes on first), if any.
pub fn held(h: *const LaneHost) ?usize {
    for (h.filling.items, 0..) |j, i| if (j.stream.mid_chunk) return i;
    return null;
}

/// The next filling job's next prompt step: the other streams' rounds run between steps, not after the prompt.
pub fn one(h: *LaneHost) void {
    if (h.filling.items.len == 0) return;
    var at = next(h);
    var job = h.filling.items[at];
    var done = false;
    while (true) {
        const first = !job.fill_began;
        job.fill_began = true;
        done = h.core.fillStreamCommitted(&job.stream, first, .{ .ptr = job, .call = LaneHost.firstCommitted }) catch |e| {
            if (e == error.FillBusy) if (held(h)) |mid| if (mid != at) { // its chunk could not start beside another's
                job.fill_wait +|= 1;
                at = mid;
                job = h.filling.items[at];
                continue;
            };
            _ = switch (e) {
                error.Cancelled => h.cancel(job),
                error.LaneMemory => blk: {
                    unfill(h, job);
                    break :blk h.requeue(job);
                },
                else => h.drop(job, h.words(e)),
            };
            return;
        };
        break;
    }
    if (!done) return;
    _ = h.filling.orderedRemove(at);
    h.prefilled(job, job.began);
    _ = h.deliver(job);
}

/// Takes `job` off the filling list, wherever it is.
pub fn unfill(h: *LaneHost, job: anytype) void {
    for (h.filling.items, 0..) |j, i| if (j == job) {
        _ = h.filling.orderedRemove(i);
        return;
    };
}
