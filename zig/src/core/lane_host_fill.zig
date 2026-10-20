//! Prefer foreground and short prompts, with bounded waiting for every admitted fill.
const std = @import("std");
const lanes = @import("lanes");
const LaneHost = @import("lane_host.zig").LaneHost;
const Job = LaneHost.Job;

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
    if (together(h)) return;
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

fn shorter(_: void, a: *Job, b: *Job) bool {
    return a.stream.prompt_len < b.stream.prompt_len;
}

/// Fresh fills' prompts in one pass, shortest first, the longest left out until the backend takes them; false: none.
fn together(h: *LaneHost) bool {
    if (!h.core.batchesPrompts() or held(h) != null) return false;
    var jobs: [lanes.Engine.max_together]*Job = undefined;
    var n: usize = 0;
    for (h.filling.items) |j| if (!j.fill_began and !j.request.background and n < jobs.len) {
        jobs[n] = j;
        n += 1;
    };
    std.mem.sort(*Job, jobs[0..n], {}, shorter);
    var streams: [lanes.Engine.max_together]*lanes.Stream = undefined;
    var hooks: [lanes.Engine.max_together]?lanes.backend.Committed = undefined;
    var errs: [lanes.Engine.max_together]?anyerror = undefined;
    while (n >= 2) : (n -= 1) {
        for (jobs[0..n], 0..) |job, i| {
            streams[i] = &job.stream;
            hooks[i] = .{ .ptr = job, .call = LaneHost.firstCommitted };
        }
        const took = h.core.fillTogether(streams[0..n], hooks[0..n], errs[0..n]) catch |e| {
            for (jobs[0..n]) |job| {
                unfill(h, job);
                _ = h.drop(job, h.words(e));
            }
            return true;
        };
        if (!took) continue;
        for (jobs[0..n], errs[0..n]) |job, err| {
            job.fill_began = true;
            unfill(h, job);
            if (err) |e| {
                _ = h.failedPass(job, e);
                continue;
            }
            h.prefilled(job, job.began);
            _ = h.deliver(job);
        }
        return true;
    }
    return false;
}

/// Takes `job` off the filling list, wherever it is.
pub fn unfill(h: *LaneHost, job: anytype) void {
    for (h.filling.items, 0..) |j, i| if (j == job) {
        _ = h.filling.orderedRemove(i);
        return;
    };
}
