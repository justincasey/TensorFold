//! A prompt pass's steps on the GLM lanes: whole chunks, or a chunk's layer groups while other streams decode.

const std = @import("std");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const fw = @import("cuda_forward.zig");
const aside_rows = @import("cuda_state.zig").aside_rows;
const Lanes = @import("cuda_lanes.zig").Lanes;

/// A 4,096-row chunk's layers a group while other streams hold lanes (TF_GLM_FILL_LAYERS; 0: whole chunks).
pub const default_fill_layers = 1;

/// A prompt pass's step (rounds run between): a chunk, or its next layer group; true once the first token drew.
pub fn step(self: *Lanes, s: *lanes.Stream) !bool {
    const known = self.lanes.getPtr(s);
    const going: ?usize = if (known) |l| l.fill else null;
    const len = s.prompt().len;
    const mid = if (known) |l| l.layer > 0 else false;
    // decided before the pass opens: a refusal (FillBusy) leaves the stream as it was
    const start = going orelse if (s.prompt_logprobs != null) 0 else resumable(self, s, len);
    const aside = if (mid) known.?.aside else try room(self, @min(len - start, engine.prompt_rows));
    const from = if (going == null) try self.opening(s) else null;
    if (going == null) try self.images(s, from);
    const l = self.lanes.getPtr(s).?;
    if (going != null) try self.x.bind(l.slot);
    const at = going orelse if (from) |r| r.at else 0;
    errdefer {
        l.fill = null;
        l.layer = 0;
        l.aside = false;
        s.mid_chunk = false;
    }
    var sink: Lanes.Sink = .{ .gpa = self.x.e.gpa, .s = s };
    try sink.prepare();
    self.x.e.prompt_logprobs = sink.want();
    defer self.x.e.prompt_logprobs = null;
    const hi = groupEnd(self, l.layer, @min(len - at, engine.prompt_rows), aside);
    const got = try self.x.chunk(s.prompt(), at, from, .{ .ptr = s, .check = Lanes.cancelled }, Lanes.marksOf(s), l.layer, hi, aside);
    if (s.settle_step) try self.x.e.stream.synchronize(); // its tail would land in the next decode round's time
    if (hi < self.x.e.w.layers.len) { // the chunk goes on from here: its rows stay in its prompt buffers
        l.fill = at;
        l.layer = hi;
        l.aside = aside;
        s.mid_chunk = true;
        return false;
    }
    l.layer = 0;
    l.aside = false;
    s.mid_chunk = false;
    if (got) |first| {
        l.first = self.take(first);
        l.fill = null;
        return true;
    }
    l.fill = at + engine.prompt_rows;
    return false;
}

/// Whether a chunk of `rows` rows starts on the aside prompt buffers: short, while another's holds the main ones.
fn room(self: *const Lanes, rows: usize) !bool {
    var main = false;
    var side = false;
    var it = self.lanes.valueIterator();
    while (it.next()) |x| if (x.layer > 0) {
        if (x.aside) side = true else main = true;
    };
    if (!main) return false;
    if (self.aside and !side and rows <= aside_rows) return true;
    return error.FillBusy; // the host runs the chunk that holds them on first
}

/// A group's end from `lo`: whole alone or short (halves aside, beside a held one), else fill_layers' chunk time.
fn groupEnd(self: *const Lanes, lo: usize, rows: usize, aside: bool) usize {
    const layers = self.x.e.w.layers.len;
    if (self.fill_layers == 0 or (lo == 0 and self.lanes.count() < 2)) return layers;
    if (rows <= aside_rows) return if (aside) @min(layers, lo + (layers + 1) / 2) else layers;
    return @min(layers, lo + @max(1, self.fill_layers * engine.prompt_rows / rows));
}

/// Where the stream's prompt pass would start: its kept state's position if it resumes there, else 0.
pub fn resumable(self: *const Lanes, s: *const lanes.Stream, len: usize) usize {
    const saved = s.reuse.saved orelse return 0;
    const k: *const engine.Kept = @ptrCast(@alignCast(saved));
    if (k.at != s.reuse.at or k.at == 0 or k.at >= len or k.at % fw.prefill_rows != 0 or !self.x.e.holds(k.id)) return 0;
    return k.at;
}

/// A cancel mid pass: the state at the next chunk's start kept (a chunk between groups has none; its lane lets go).
pub fn cancel(self: *Lanes, s: *lanes.Stream) !void {
    const l = self.lanes.getPtr(s) orelse return;
    const at = l.fill orelse return;
    if (l.layer > 0) { // its KDA layers ran past `at`: nothing there to keep, and bp is free for the next chunk
        l.fill = null;
        l.layer = 0;
        l.aside = false;
        s.mid_chunk = false;
        return;
    }
    if (at <= s.cached or s.reuse.hook == null) return;
    try self.x.bind(l.slot);
    _ = self.x.chunk(s.prompt(), at, null, .{ .ptr = null, .check = cancelNow }, Lanes.marksOf(s), 0, self.x.e.w.layers.len, false) catch |err| {
        if (err == error.Cancelled) return;
        return err;
    };
    return error.CancellationIgnored;
}

fn cancelNow(_: ?*anyopaque) bool {
    return true;
}
