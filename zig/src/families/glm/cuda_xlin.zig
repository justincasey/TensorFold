//! RED-SNOW's EXL3 linears behind b16.matmul's weight handle. A handle names one linear [n][k] made of up to three
//! parts written side by side in its output columns (KDA's [q|k|v | f_a|g_a|b], DSA's [q_a | kv_a], [gate | up]):
//! an EXL3 group (exl3.Linear: the rows-invariant decode kernel, or prompt rows' dense kernel) or a bf16 matrix.
const std = @import("std");
const cuda = @import("cuda");
const exl3 = cuda.exl3;

pub const Part = union(enum) {
    q: exl3.Weight, // an EXL3 group, its rank's cut, in the strips layout
    b: u64, // bf16 [n][k] row-major, untiled
};

/// A part's kind: a window may run its EXL3 parts and its bf16 part on two streams.
pub const Kind = std.meta.Tag(Part);

pub const Lin = struct {
    id: u64,
    n: usize,
    k: usize,
    count: u8 = 0,
    parts: [3]Part = undefined,
    cols: [3]usize = undefined, // each part's first output column
    widths: [3]usize = undefined, // and its columns
};

/// Bf16 linears run through `b16` with a set that has no EXL3 table and no tiles (the caller supplies it).
pub const Bf16Fn = *const fn (ctx: *const anyopaque, stream: cuda.Stream, x: u64, x_stride: usize, w: u64, out: u64, part: u64, m: usize, n: usize, k: usize, wide: bool) anyerror!void;

const max_streams = 4;
const block_rows = 128; // exl3.Linear takes 1..128 rows; a row's bits never depend on the count

/// Calls of this many rows or more (prompt chunks; decode windows stay under) run exl3.Linear.dense.
pub const prompt_rows = 64;

/// One stream's scratch: the rotated inputs, split-K partials, counters, a contiguous copy of x and a part's output.
const Scratch = struct {
    handle: ?usize = null,
    xh: ?cuda.DeviceBuffer = null,
    z: ?cuda.DeviceBuffer = null,
    counters: ?cuda.DeviceBuffer = null,
    xc: ?cuda.DeviceBuffer = null,
    yt: ?cuda.DeviceBuffer = null,

    fn free(s: *Scratch) void {
        inline for (.{ "xh", "z", "counters", "xc", "yt" }) |f| if (@field(s, f)) |*b| b.free();
        s.* = .{};
    }
};

pub const Runner = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    lin: ?exl3.Linear = null,
    items: std.ArrayList(Lin) = .empty, // sorted by id once sealed
    sealed: bool = false,
    max_k: usize = 0,
    max_n: usize = 0,
    max_z: usize = 0,
    max_counters: usize = 0,
    streams: [max_streams]Scratch = @splat(.{}),
    grown: u64 = 0, // scratch reallocations: a captured decode graph holds the old addresses (cuda_decode)

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver) !*Runner {
        const r = try gpa.create(Runner);
        r.* = .{ .gpa = gpa, .d = d };
        return r;
    }

    pub fn deinit(r: *Runner) void {
        for (&r.streams) |*s| s.free();
        if (r.lin) |*l| l.unload();
        r.items.deinit(r.gpa);
        r.gpa.destroy(r);
    }

    /// Registers a linear; its id is its first EXL3 part's trellis address (unique, never a bf16 weight's).
    pub fn add(r: *Runner, parts: []const Part, widths: []const usize, k: usize) !u64 {
        if (parts.len == 0 or parts.len > 3 or parts.len != widths.len or r.sealed) return error.BadExl3Linear;
        var l: Lin = .{ .id = 0, .n = 0, .k = k, .count = @intCast(parts.len) };
        for (parts, widths, 0..) |p, w, i| {
            switch (p) {
                .q => |q| {
                    if (q.k != k or q.n != w) return error.BadExl3Linear;
                    if (l.id == 0) l.id = q.trellis;
                    const pl = exl3.plan(q.k, q.n, 192, 8);
                    if (pl.sk > 1) r.max_z = @max(r.max_z, @as(usize, pl.sk) * block_rows * q.n * 4);
                    r.max_counters = @max(r.max_counters, exl3.Linear.counterBytes(q));
                },
                .b => {},
            }
            l.parts[i] = p;
            l.cols[i] = l.n;
            l.widths[i] = w;
            l.n += w;
        }
        if (l.id == 0) return error.BadExl3Linear; // a linear with no EXL3 part stays a plain bf16 weight
        r.max_k = @max(r.max_k, k);
        r.max_n = @max(r.max_n, l.n);
        try r.items.append(r.gpa, l);
        return l.id;
    }

    /// Sorts the table and loads the kernels; no linear is added after.
    pub fn seal(r: *Runner) !void {
        std.mem.sortUnstable(Lin, r.items.items, {}, struct {
            fn lt(_: void, a: Lin, b: Lin) bool {
                return a.id < b.id;
            }
        }.lt);
        r.sealed = true;
        if (r.lin == null) r.lin = try exl3.Linear.load(r.d);
    }

    pub fn find(r: *const Runner, id: u64) ?*const Lin {
        if (!r.sealed) {
            for (r.items.items) |*l| if (l.id == id) return l;
            return null;
        }
        var lo: usize = 0;
        var hi: usize = r.items.items.len;
        while (lo < hi) {
            const mid = (lo + hi) / 2;
            const v = r.items.items[mid].id;
            if (v == id) return &r.items.items[mid];
            if (v < id) lo = mid + 1 else hi = mid;
        }
        return null;
    }

    fn scratchFor(r: *Runner, stream: cuda.Stream) !*Scratch {
        const h: usize = if (stream.handle) |p| @intFromPtr(p) else 0; // the legacy stream: 0
        for (&r.streams) |*s| if (s.handle == h) return s;
        for (&r.streams) |*s| if (s.handle == null) {
            s.handle = h;
            return s;
        };
        return error.TooManyExl3Streams;
    }

    fn grow(r: *Runner, slot: *?cuda.DeviceBuffer, bytes: usize) !u64 {
        if (slot.*) |b| if (b.len >= bytes) return b.ptr;
        const nb = try cuda.DeviceBuffer.alloc(r.d, @max(bytes, 256));
        if (slot.*) |*old| old.free();
        slot.* = nb;
        r.grown += 1;
        return nb.ptr;
    }

    /// The split-K counters, zeroed when allocated: each call's last block of a column block zeroes its own again.
    fn zeroedCounters(r: *Runner, s: *Scratch, stream: cuda.Stream, bytes: usize) !u64 {
        const before = r.grown;
        const p = try r.grow(&s.counters, bytes);
        if (r.grown != before) try stream.d.check(stream.d.api.cuMemsetD8Async(p, 0, s.counters.?.len, stream.handle), "EXL3 counters");
        return p;
    }

    /// Grows each stream's scratch for `rows` rows a call (prompt chunks) before any decode window runs.
    pub fn reserve(r: *Runner, stream: cuda.Stream, rows: usize) !void {
        const s = try r.scratchFor(stream);
        _ = try r.grow(&s.xh, @max(rows, block_rows) * r.max_k * 2);
        _ = try r.grow(&s.z, r.max_z);
        _ = try r.zeroedCounters(s, stream, r.max_counters);
        _ = try r.grow(&s.xc, rows * r.max_k * 2);
        _ = try r.grow(&s.yt, rows * r.max_n * 4);
    }

    fn copy2d(stream: cuda.Stream, dst: u64, dpitch: usize, src: u64, spitch: usize, width: usize, height: usize) !void {
        const m: cuda.abi.Memcpy2D = .{ .src = src, .src_pitch = spitch, .dst = dst, .dst_pitch = dpitch, .width = width, .height = height };
        try stream.d.check(stream.d.api.cuMemcpy2DAsync_v2(&m, stream.handle), "EXL3 linear rows");
    }

    /// out [m][n] (fp32 when `wide`, else bf16) = x [m][k] (rows `x_stride` apart) through `l`; `only`: that kind's parts.
    pub fn run(r: *Runner, bf: Bf16Fn, ctx: *const anyopaque, stream: cuda.Stream, l: *const Lin, only: ?Kind, x: u64, x_stride: usize, out: u64, part: u64, m: usize, n: usize, k: usize, wide: bool) !void {
        if (n != l.n or k != l.k) return error.Exl3LinearShape;
        const s = try r.scratchFor(stream);
        const ye: usize = if (wide) 4 else 2;
        const yd: exl3.DType = if (wide) .f32 else .bf16;
        var xc: ?u64 = if (x_stride == k) x else null; // contiguous rows, copied once when a part needs them
        for (l.parts[0..l.count], l.cols[0..l.count], l.widths[0..l.count]) |p, col, w| {
            if (only) |o| if (p != o) continue;
            if (p == .q and m >= prompt_rows and exl3.denseIndex(p.q) != null) { // in place at its columns
                const xh = try r.grow(&s.xh, m * k * 2);
                try (try r.linear()).dense(stream, p.q, x, x_stride, m, out + col * ye, n, yd, xh);
                continue;
            }
            const xx = xc orelse blk: {
                const c = try r.grow(&s.xc, m * k * 2);
                try copy2d(stream, c, k * 2, x, x_stride * 2, k * 2, m);
                xc = c;
                break :blk c;
            };
            const direct = l.count == 1;
            const y = if (direct) out else try r.grow(&s.yt, m * l.n * 4);
            switch (p) {
                .q => |q| try r.blocks(s, stream, q, xx, m, y, yd),
                .b => |b| try bf(ctx, stream, xx, k, b, y, part, m, w, k, wide),
            }
            if (!direct) try copy2d(stream, out + col * ye, n * ye, y, w * ye, w * ye, m);
        }
    }

    /// The kernels, loaded on first use: the loader's head pass runs before seal.
    fn linear(r: *Runner) !*const exl3.Linear {
        if (r.lin == null) r.lin = try exl3.Linear.load(r.d);
        return &r.lin.?;
    }

    /// exl3.Dense's 128-row blocks on this stream's scratch.
    fn blocks(r: *Runner, s: *Scratch, stream: cuda.Stream, w: exl3.Weight, x: u64, rows: usize, y: u64, yd: exl3.DType) !void {
        const lin = try r.linear();
        const p = exl3.plan(w.k, w.n, 192, 8);
        const z: u64 = if (p.sk > 1) try r.grow(&s.z, @as(usize, p.sk) * block_rows * w.n * 4) else 0;
        const counters = try r.zeroedCounters(s, stream, exl3.Linear.counterBytes(w));
        const xh = try r.grow(&s.xh, block_rows * w.k * 2);
        const ye: u64 = if (yd == .f32) 4 else 2;
        var at: usize = 0;
        while (at < rows) : (at += block_rows) {
            const mm = @min(rows - at, block_rows);
            try lin.matmul(stream, w, x + @as(u64, at) * w.k * 2, .bf16, mm, y + @as(u64, at) * w.n * ye, yd, xh, z, counters);
        }
    }
};

test "a linear's parts sit side by side and the table finds it by its first EXL3 part" {
    var r: Runner = .{ .gpa = std.testing.allocator, .d = undefined };
    defer r.items.deinit(r.gpa);
    const st = exl3.strides(4096, 1536, 10, true);
    const qa: exl3.Weight = .{ .trellis = 0x1000, .suh = 1, .svh = 2, .k = 4096, .n = 1536, .halves = 10, .codebook = .mul1, .stride_k = st.k, .stride_nb = st.nb };
    var kva = qa;
    kva.trellis = 0x2000;
    kva.n = 512;
    const id = try r.add(&.{ .{ .q = qa }, .{ .q = kva } }, &.{ 1536, 512 }, 4096);
    try std.testing.expectEqual(@as(u64, 0x1000), id);
    _ = try r.add(&.{ .{ .q = kva }, .{ .b = 0x9000 } }, &.{ 512, 288 }, 4096);
    std.mem.sortUnstable(Lin, r.items.items, {}, struct {
        fn lt(_: void, a: Lin, b: Lin) bool {
            return a.id < b.id;
        }
    }.lt);
    r.sealed = true;
    const l = r.find(0x1000).?;
    try std.testing.expectEqual(@as(usize, 2048), l.n);
    try std.testing.expectEqual(@as(usize, 1536), l.cols[1]);
    try std.testing.expectEqual(@as(usize, 800), r.find(0x2000).?.n);
    try std.testing.expect(r.find(0x9000) == null);
    try std.testing.expectError(error.BadExl3Linear, r.add(&.{.{ .b = 0x9000 }}, &.{288}, 4096));
}
