//! GLM-5.3-Flash's KDA on CUDA (the Python family's kda): fused and three-kernel chains match bits; replay rebuilds.

const std = @import("std");
const cuda = @import("cuda");

pub const dk = 128;
pub const dv = 128;
/// Windows of this many rows or more (prefill chunks) take the three-kernel chain (kda.WIDE_ROWS).
pub const wide_rows = 64;

/// glm_kda.cu's entries.
pub const Kernels = struct {
    chain: cuda.Function,
    replay: cuda.Function,
    replay_layers: cuda.Function,
    prep: cuda.Function,
    step: cuda.Function,
    out: cuda.Function,
    prep_segs: cuda.Function,
    step_segs: cuda.Function,
    chunk: ?Chunk = null, // prompt chunks in chunked (WY) form (Modules.load), else the three-kernel chain

    pub fn load(m: cuda.Module) !Kernels {
        var k: Kernels = undefined;
        inline for (.{ "chain", "replay", "replay_layers", "prep", "step", "out", "prep_segs", "step_segs" }) |name| @field(k, name) = try m.function("tf_glm_kda_" ++ name);
        k.chunk = null;
        return k;
    }
};

/// glm_kda_chunk.cu: a prompt chunk's KDA in chunked (WY) form, one 256-thread block a head, 32-row sub-chunks in
/// order on the state in registers. Prompt chunks start on multiples of 32 (the 2,048-row grid), so a row's bits
/// depend only on its sub-chunk and the entering state, never on where its prompt chunk started.
pub const Chunk = struct {
    mod: cuda.Module,
    f: cuda.Function,
    smem: u32,

    pub fn load(d: *const cuda.Driver) !Chunk {
        var m = try cuda.Module.load(d, cuda.kernels.glm_kda_chunk);
        errdefer m.unload();
        const g = try m.global("tf_glm_kda_chunk_smem_v");
        var buf = try cuda.DeviceBuffer.alloc(d, 8);
        defer buf.free();
        try buf.copyFrom(0, g.ptr, 8, null);
        var raw: [8]u8 = undefined;
        try buf.download(0, &raw);
        const smem: u32 = @intCast(std.mem.readInt(i64, &raw, .little));
        const f = try m.function("tf_glm_kda_chunk");
        try f.allowDynamicShared(smem);
        return .{ .mod = m, .f = f, .smem = smem };
    }

    pub fn unload(c: *Chunk) void {
        c.mod.unload();
    }

    /// `rows` prompt rows of chain `c` from state_in into s.out and s.state_out (no replay saves: a chunk commits whole).
    pub fn run(k: *const Chunk, stream: cuda.Stream, c: Chain, rows: usize, s: Saves) !void {
        var a: cuda.Args = .{};
        a.add(@as(c_int, @intCast(c.heads)));
        a.add(c.p);
        a.add(@as(c_int, @intCast(c.p_stride)));
        a.add(@as(c_int, @intCast(c.b_off)));
        a.add(c.a);
        a.add(@as(c_int, @intCast(c.a_stride)));
        a.add(c.g);
        a.add(@as(c_int, @intCast(c.g_stride)));
        for ([_]u64{ c.conv_state, c.conv_w, c.state_in, c.a_log, c.dt_bias, c.norm_w }) |v| a.add(v);
        a.add(c.eps);
        a.add(c.lower);
        a.add(@as(c_int, @intCast(rows)));
        a.add(@as(c_int, 0)); // off: the chunk starts on a sub-chunk boundary
        a.add(s.out);
        a.add(s.state_out);
        try cuda.launch.launch(k.f, .{ .grid = .{ .x = @intCast(c.heads) }, .block = .{ .x = 256 }, .shared = k.smem }, stream, &a);
    }
};

/// One layer's chain inputs, `heads` heads: p rows [q|k|v|f_a|g_a|...] (beta at `b_off`), decay and gate rows, states.
pub const Chain = struct {
    p: u64,
    p_stride: usize,
    b_off: usize,
    a: u64,
    a_stride: usize,
    g: u64,
    g_stride: usize,
    conv_state: u64,
    conv_w: u64,
    state_in: u64,
    a_log: u64,
    dt_bias: u64,
    norm_w: u64,
    eps: f32,
    lower: f32,
    heads: usize,
};

/// Where a chain writes: out bf16, state_out like state_in, and replay's rows: k fp32, v bf16, decay and beta fp32.
pub const Saves = struct { out: u64, state_out: u64, k: u64, v: u64, g: u64, b: u64 };

/// The three-kernel chain's scratch: q fp32 and the read-out bf16, [rows][heads][128] each.
pub const Wide = struct { q: u64, y: u64 };

/// kda.chain: `rows` rows through the fused kernel, or with `wide` through prep, step and out (WARPS 8, TR 16).
pub fn chain(ks: *const Kernels, stream: cuda.Stream, c: Chain, rows: usize, s: Saves, wide: ?Wide) !void {
    const h: c_int = @intCast(c.heads);
    if (wide) |w| {
        var a: cuda.Args = .{};
        a.add(h);
        a.add(c.p);
        a.add(@as(c_int, @intCast(c.p_stride)));
        a.add(@as(c_int, @intCast(c.b_off)));
        a.add(c.a);
        a.add(@as(c_int, @intCast(c.a_stride)));
        for ([_]u64{ c.conv_state, c.conv_w, c.a_log, c.dt_bias }) |v| a.add(v);
        a.add(c.lower);
        for ([_]u64{ w.q, s.k, s.v, s.g, s.b }) |v| a.add(v);
        const grid: cuda.launch.Dim3 = .{ .x = @intCast(rows), .y = @intCast(c.heads) };
        try cuda.launch.launch(ks.prep, .{ .grid = grid, .block = .{ .x = 512 } }, stream, &a);
        var b: cuda.Args = .{};
        b.add(h);
        for ([_]u64{ c.state_in, w.q, s.k, s.v, s.g, s.b }) |v| b.add(v);
        b.add(@as(c_int, @intCast(rows)));
        b.add(w.y);
        b.add(s.state_out);
        try cuda.launch.launch(ks.step, .{ .grid = .{ .x = @intCast(c.heads), .y = dv / 4 / 8 }, .block = .{ .x = 8 * 32 } }, stream, &b);
        var o: cuda.Args = .{};
        o.add(h);
        o.add(w.y);
        o.add(c.g);
        o.add(@as(c_int, @intCast(c.g_stride)));
        o.add(c.norm_w);
        o.add(c.eps);
        o.add(s.out);
        return cuda.launch.launch(ks.out, .{ .grid = grid, .block = .{ .x = dv } }, stream, &o);
    }
    var a: cuda.Args = .{};
    a.add(h);
    a.add(c.p);
    a.add(@as(c_int, @intCast(c.p_stride)));
    a.add(@as(c_int, @intCast(c.b_off)));
    a.add(c.a);
    a.add(@as(c_int, @intCast(c.a_stride)));
    a.add(c.g);
    a.add(@as(c_int, @intCast(c.g_stride)));
    for ([_]u64{ c.conv_state, c.conv_w, c.state_in, c.a_log, c.dt_bias, c.norm_w }) |v| a.add(v);
    a.add(c.eps);
    a.add(c.lower);
    a.add(@as(c_int, @intCast(rows)));
    for ([_]u64{ s.out, s.state_out, s.k, s.v, s.g, s.b }) |v| a.add(v);
    try cuda.launch.launch(ks.chain, .{ .grid = .{ .x = @intCast(c.heads) }, .block = .{ .x = 1024 } }, stream, &a);
}

/// The segments a shared window's chain takes at most (glm_kda.cu's MAX_SEGS).
pub const max_segs = 16;

/// glm_kda.cu's Segs: each segment's rows (one stream's) and its sequence's conv window, states, saves and kept rows.
pub const Segs = extern struct {
    n: c_int = 0,
    row0: [max_segs]c_int = @splat(0),
    rows: [max_segs]c_int = @splat(0),
    cs: [max_segs]u64 = @splat(0),
    state_in: [max_segs]u64 = @splat(0),
    state_out: [max_segs]u64 = @splat(0),
    keep: [max_segs]u64 = @splat(0),
    k: [max_segs]u64 = @splat(0),
    v: [max_segs]u64 = @splat(0),
    g: [max_segs]u64 = @splat(0),
    b: [max_segs]u64 = @splat(0),
};

/// chain's kernels over a shared window (c's p, a, g by window row; each segment's own conv window and states).
pub fn chainSegs(ks: *const Kernels, stream: cuda.Stream, c: Chain, segs: *const Segs, rows: usize, out: u64, w: Wide) !void {
    if (segs.n < 1 or segs.n > max_segs) return error.BadSegments;
    const h: c_int = @intCast(c.heads);
    var a: cuda.Args = .{};
    a.add(segs.*);
    a.add(h);
    a.add(c.p);
    a.add(@as(c_int, @intCast(c.p_stride)));
    a.add(@as(c_int, @intCast(c.b_off)));
    a.add(c.a);
    a.add(@as(c_int, @intCast(c.a_stride)));
    for ([_]u64{ c.conv_w, c.a_log, c.dt_bias }) |v| a.add(v);
    a.add(c.lower);
    a.add(w.q);
    const grid: cuda.launch.Dim3 = .{ .x = @intCast(rows), .y = @intCast(c.heads) };
    try cuda.launch.launch(ks.prep_segs, .{ .grid = grid, .block = .{ .x = 512 } }, stream, &a);
    var b: cuda.Args = .{};
    b.add(segs.*);
    b.add(h);
    b.add(w.q);
    b.add(w.y);
    try cuda.launch.launch(ks.step_segs, .{ .grid = .{ .x = @intCast(c.heads), .y = dv / 4 / 8, .z = @intCast(segs.n) }, .block = .{ .x = 8 * 32 } }, stream, &b);
    var o: cuda.Args = .{};
    o.add(h);
    o.add(w.y);
    o.add(c.g);
    o.add(@as(c_int, @intCast(c.g_stride)));
    o.add(c.norm_w);
    o.add(c.eps);
    o.add(out);
    try cuda.launch.launch(ks.out, .{ .grid = grid, .block = .{ .x = dv } }, stream, &o);
}

/// kda.replay: the state after the first `rows` saved rows, from state_in.
pub fn replay(ks: *const Kernels, stream: cuda.Stream, state_in: u64, k: u64, v: u64, g: u64, b: u64, rows: usize, heads: usize, state_out: u64) !void {
    var a: cuda.Args = .{};
    a.add(@as(c_int, @intCast(heads)));
    for ([_]u64{ state_in, k, v, g, b }) |x| a.add(x);
    a.add(@as(c_int, @intCast(rows)));
    a.add(state_out);
    try cuda.launch.launch(ks.replay, .{ .grid = .{ .x = @intCast(heads) }, .block = .{ .x = 1024 } }, stream, &a);
}

test "the three-kernel chain's step grid is the Python family's" {
    try std.testing.expectEqual(@as(u32, 4), dv / 4 / 8); // DV / 4 / WARPS value rows a block, 8 warps of 4 rows
}
