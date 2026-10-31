//! A verify window's second stream: the MoE's shared expert and KDA's bf16 projections run there beside the main one.
const std = @import("std");
const cuda = @import("cuda");

/// The stream and its fork and join events; a captured window records them as graph branches.
pub const Side = struct {
    s: cuda.Stream,
    forked: cuda.Event,
    done: cuda.Event,

    /// TF_GLM_SIDE unset or anything but 0 (0: every window on one stream, the same bits).
    pub fn wanted() bool {
        const v = std.c.getenv("TF_GLM_SIDE") orelse return true;
        return !std.mem.eql(u8, std.mem.span(v), "0");
    }

    pub fn init(d: *const cuda.Driver) !Side {
        var s = try cuda.Stream.init(d, true);
        errdefer s.deinit();
        var forked = try cuda.Event.init(d, false);
        errdefer forked.deinit();
        return .{ .s = s, .forked = forked, .done = try cuda.Event.init(d, false) };
    }

    pub fn deinit(sd: *Side) void {
        sd.done.deinit();
        sd.forked.deinit();
        sd.s.deinit();
    }

    /// The side stream's next work starts after `main`'s work so far.
    pub fn fork(sd: *const Side, main: cuda.Stream) !void {
        try sd.forked.record(main);
        try sd.s.wait(sd.forked);
    }

    /// `main`'s next work starts after the side stream's work so far.
    pub fn join(sd: *const Side, main: cuda.Stream) !void {
        try sd.done.record(sd.s);
        try main.wait(sd.done);
    }
};
