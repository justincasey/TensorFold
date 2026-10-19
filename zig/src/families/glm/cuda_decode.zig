//! Decode windows on CUDA: each replayed as a graph captured on first use, and their greedy picks in one exchange.
const std = @import("std");
const cuda = @import("cuda");
const fw = @import("cuda_forward.zig");
const latent = @import("cuda_latent.zig");
const sparse = @import("cuda_sparse.zig");
const ranks = @import("cuda_ranks.zig");

/// A greedy pick as it crosses ranks: (value, id) and padding, so the RoCE gather (16-byte packs) can carry it.
pub const pair_bytes = 16;

/// Graphs held at most; past it every one is dropped (each holds a whole forward's launches).
const max_graphs = 96;

/// What a window's launches take from the host beyond its buffers: the rest reads the position from pos_dev.
pub const Key = struct {
    st: usize, // the sequence's State (its caches' addresses)
    rows: u8,
    cur: u1, // the KDA layers' state copy (every layer the same between commits)
    nch: u16, // dense attention's 512-key chunks
    base: usize, // the index rings' base (their pointers move with it)
    dense: bool, // rows within the dense limit (dense attention runs)
    sparse: bool, // rows past it (selection and sparse attention run)
    np: u32, // with sparse rows: the pool bucket and the score blocks
    nb: u32,
};

pub const Graphs = struct {
    gpa: std.mem.Allocator,
    on: bool, // TF_GLM_GRAPHS=0 runs every window eagerly
    map: std.AutoHashMapUnmanaged(Key, cuda.graph.Exec) = .empty,
    seen: std.AutoHashMapUnmanaged(Key, void) = .empty, // keys whose first window ran (eagerly)
    grown: u64 = 0, // the EXL3 linears' scratch reallocations the graphs were captured after
    failed: bool = false, // a capture failed: windows run eagerly from then on

    /// TF_GLM_GRAPHS unset or anything but 0.
    pub fn wanted() bool {
        const v = std.c.getenv("TF_GLM_GRAPHS") orelse return true;
        return !std.mem.eql(u8, std.mem.span(v), "0");
    }

    pub fn deinit(g: *Graphs) void {
        g.clear();
        g.map.deinit(g.gpa);
        g.seen.deinit(g.gpa);
    }

    /// Drops every graph (a sequence freed: a later one may take its State's address with other caches).
    pub fn clear(g: *Graphs) void {
        var it = g.map.valueIterator();
        while (it.next()) |x| x.deinit();
        g.map.clearRetainingCapacity();
    }

    /// forward.run of a decode window: its graph replayed (captured the second time its key comes), else eager.
    pub fn run(g: *Graphs, f: *const fw.Forward, ids: []const i32) !void {
        try f.begin(ids);
        const k = (if (g.on and !g.failed) keyOf(f, ids.len) else null) orelse return f.body(ids.len);
        if (f.m.xl) |x| if (x.grown != g.grown) {
            g.clear();
            g.grown = x.grown;
        };
        if (g.map.get(k)) |x| return x.launchOn(f.stream);
        if (!g.seen.contains(k)) { // a key's first window runs eagerly: what it grows is in place before its capture
            try g.seen.put(g.gpa, k, {});
            return f.body(ids.len);
        }
        if (g.map.count() >= max_graphs) g.clear();
        const x = capture(f, ids.len) catch |err| {
            std.log.warn("decode graphs: capture failed ({s}), windows run eagerly from now on", .{@errorName(err)});
            g.failed = true;
            return f.body(ids.len);
        };
        g.map.put(g.gpa, k, x) catch |err| {
            var y = x;
            y.deinit();
            return err;
        };
        try x.launchOn(f.stream);
    }

    /// The window's launches recorded (nothing runs), instantiated and uploaded.
    fn capture(f: *const fw.Forward, rows: usize) !cuda.graph.Exec {
        try cuda.graph.beginCapture(f.stream, .thread_local);
        const recorded = f.body(rows);
        var graph = cuda.graph.endCapture(f.stream) catch |err| { // invalidated: the stream left capture, so the guard goes
            if ((cuda.graph.captureStatus(f.stream) catch .active) == .none) cuda.copy_source.captureEnd(f.stream);
            return err;
        };
        defer graph.deinit();
        try recorded;
        var x = try graph.instantiate();
        errdefer x.deinit();
        try x.upload(f.stream);
        return x;
    }
};

/// The key of a one-sequence decode window on this rank, or null when it runs eagerly (prompt, shared, profiled).
pub fn keyOf(f: *const fw.Forward, rows: usize) ?Key {
    if (f.ranks.len != 1 or f.prof != null or f.images != null or f.pieces != null) return null;
    const r = f.ranks[0];
    const st = r.st;
    if (r.segs.len != 0 or r.b.prefill or rows == 0 or rows > fw.window_rows) return null;
    const cur: u1 = if (st.cur.len > 0) st.cur[0] else 0;
    for (st.cur) |x| if (x != cur) return null;
    var k: Key = .{ .st = @intFromPtr(st), .rows = @intCast(rows), .cur = cur, .nch = @intCast(@min(latent.chunksFor(st.pos + rows), r.b.nch)), .base = st.base, .dense = false, .sparse = false, .np = 0, .nb = 0 };
    const limit = for (r.w.layers) |l| switch (l.mixer) {
        .mla => |a| break a.dense_limit,
        .kda => {},
    } else return k;
    k.dense = st.pos < limit;
    k.sparse = st.pos + rows - 1 >= limit;
    if (k.sparse) {
        const np = sparse.poolBucket(st.pos, rows, st.caches[0].pk_rows - 2);
        k.np = @intCast(np);
        k.nb = @intCast((sparse.scoreColumns(st.pos, rows, np) + 63) / 64);
    }
    return k;
}

/// decode.sample_rows' greedy picks: every row's top-1 on each rank, one gather, one wait; largest value, lowest id.
pub fn greedy(e: anytype, logits: u64, rows: usize, out: []u32) !void {
    if (rows == 0 or rows > fw.window_rows) return error.WindowTooWide;
    const world = e.share.world;
    const vocab = e.share.vocab.len();
    const bytes = rows * pair_bytes;
    const mine = e.pairs.ptr + e.share.rank * bytes;
    for (0..rows) |r| {
        var a: cuda.Args = .{};
        a.add(logits + r * vocab * 2);
        a.add(@as(i64, @intCast(vocab)));
        a.add(@as(u32, @intCast(e.share.vocab.lo)));
        a.add(mine + r * pair_bytes);
        try cuda.launch.launch(e.m.glue.top1, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, e.stream, &a);
    }
    if (e.group) |g| try g.allGather(e.stream, mine, e.pairs.ptr, bytes) else try e.lib.check(e.lib.api.ncclAllGather(mine, e.pairs.ptr, bytes, .u8, e.comm, e.stream.handle), "ncclAllGather");
    try e.stream.synchronize();
    if (e.group) |g| try g.check(); // a failed RoCE gather's rows hold nothing
    var all: [ranks.max_world * fw.window_rows][4]u32 = undefined; // Engine.init refuses a larger world
    try e.d.check(e.d.api.cuMemcpyDtoH_v2(&all, e.pairs.ptr, world * bytes), "cuMemcpyDtoH");
    for (out[0..rows], 0..) |*o, r| {
        var pick = all[r];
        for (1..world) |q| {
            const p = all[q * rows + r];
            const v: f32 = @bitCast(p[0]);
            const b: f32 = @bitCast(pick[0]);
            if (v > b or (v == b and p[1] < pick[1])) pick = p;
        }
        o.* = pick[1];
    }
}
