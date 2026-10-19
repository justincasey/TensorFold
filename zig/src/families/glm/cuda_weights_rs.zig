//! RED-SNOW's (codebook "mul1") non-expert weights on a TP rank: every attention, dense, shared-expert, head and
//! eh_proj linear is an EXL3 group (weights.py's Exl3Linear, split.py's EXL3 cuts), the small projections fp16.
//! Each EXL3 linear goes into the rank's `xlin.Runner` and the block keeps its handle where a bf16 matrix would be.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const exl3 = cuda.exl3;
const xlin = @import("cuda_xlin.zig");
const Span = @import("cuda_shard.zig").Span;
const kda_block = @import("cuda_kda_block.zig");
const dsa_block = @import("cuda_dsa_block.zig");
const mlp = @import("cuda_mlp.zig");
const Tensor = core.checkpoint.Tensor;

const prefix = "model.language_model.";

/// split.py's EXL3 rules on a rank's span: whole, output columns, a fused q|k|v's columns in each third, input rows.
pub const Cut = union(enum) { rep, out: Span, qkv: Span, in: Span };

fn tensor(ck: *core.Checkpoint, buf: []u8, base: []const u8, field: []const u8, dtype: core.checkpoint.DType) !Tensor {
    const name = try std.fmt.bufPrint(buf, "{s}.{s}", .{ base, field });
    const t = ck.get(name) catch |err| {
        std.log.err("RED-SNOW: {s} missing", .{name});
        return err;
    };
    if (t.dtype != dtype) {
        std.log.err("RED-SNOW: {s} is {t}, expected {t}", .{ name, t.dtype, dtype });
        return error.UnexpectedTensor;
    }
    return t;
}

/// Trellis words [K/16][N/16][W] -> strips [N/128][K/16][8][W] (linear.py): each 128-column block's tiles in k order.
pub fn strips(out: []u8, src: []const u8, kt: usize, nt: usize, wb: usize) !void {
    if (nt % 8 != 0 or src.len != kt * nt * wb or out.len != src.len) return error.Exl3Cut;
    for (0..nt / 8) |jb| for (0..kt) |i| {
        @memcpy(out[(jb * kt + i) * 8 * wb ..][0 .. 8 * wb], src[(i * nt + jb * 8) * wb ..][0 .. 8 * wb]);
    };
}

/// One EXL3 group `base` (full name) cut for the rank and uploaded in the strips layout.
pub fn group(L: anytype, ck: *core.Checkpoint, base: []const u8, cut: Cut) !exl3.Weight {
    var nb: [256]u8 = undefined;
    const mark = try tensor(ck, &nb, base, "mul1", .i32);
    if (mark.numel() != 1 or std.mem.readInt(u32, mark.bytes[0..4], .little) != core.exl3_load.markers[1].value) return error.NotMul1;
    const tr = try tensor(ck, &nb, base, "trellis", .i16);
    const suh_t = try tensor(ck, &nb, base, "suh", .f16);
    const svh_t = try tensor(ck, &nb, base, "svh", .f16);
    if (tr.rank != 3 or tr.dim(2) % 8 != 0) return error.Exl3Cut;
    const kt0 = tr.dim(0);
    const nt0 = tr.dim(1);
    const wb = tr.dim(2) * 2; // a 16 x 16 tile's bytes
    const halves: u32 = @intCast(tr.dim(2) / 8);
    if (suh_t.numel() != kt0 * 16 or svh_t.numel() != nt0 * 16) return error.Exl3Cut;
    var kt = kt0;
    var nt = nt0;
    var k_lo: usize = 0;
    // the rank's output column runs [lo, hi) in tiles: one run, or one in each third of a fused q|k|v
    var runs: [3][2]usize = undefined;
    var nruns: usize = 1;
    runs[0] = .{ 0, nt0 };
    switch (cut) {
        .rep => {},
        .out => |s| {
            if (s.lo % 128 != 0 or s.hi % 128 != 0 or s.hi > nt0 * 16) return error.Exl3Cut;
            runs[0] = .{ s.lo / 16, s.hi / 16 };
        },
        .qkv => |s| {
            if (nt0 % 3 != 0 or s.lo % 128 != 0 or s.hi % 128 != 0 or s.hi > nt0 / 3 * 16) return error.Exl3Cut;
            nruns = 3;
            for (0..3) |j| runs[j] = .{ j * nt0 / 3 + s.lo / 16, j * nt0 / 3 + s.hi / 16 };
        },
        .in => |s| {
            if (s.lo % 128 != 0 or s.hi % 128 != 0 or s.hi > kt0 * 16) return error.Exl3Cut;
            k_lo = s.lo / 16;
            kt = (s.hi - s.lo) / 16;
        },
    }
    nt = 0;
    for (runs[0..nruns]) |r| nt += r[1] - r[0];
    const words = try L.gpa.alloc(u8, kt * nt * wb);
    defer L.gpa.free(words);
    for (0..kt) |i| {
        var at: usize = 0;
        for (runs[0..nruns]) |r| {
            const len = (r[1] - r[0]) * wb;
            @memcpy(words[(i * nt * wb) + at ..][0..len], tr.bytes[((k_lo + i) * nt0 + r[0]) * wb ..][0..len]);
            at += len;
        }
    }
    try L.host.resize(L.gpa, words.len);
    try strips(L.host.items, words, kt, nt, wb);
    const trellis = try L.put("{s}.trellis", .{base});
    try L.host.appendSlice(L.gpa, suh_t.bytes[k_lo * 32 ..][0 .. kt * 32]);
    const suh = try L.put("{s}.suh", .{base});
    for (runs[0..nruns]) |r| try L.host.appendSlice(L.gpa, svh_t.bytes[r[0] * 32 .. r[1] * 32]);
    const svh = try L.put("{s}.svh", .{base});
    const k = kt * 16;
    const n = nt * 16;
    const st = exl3.strides(k, n, halves, true);
    return .{ .trellis = trellis, .suh = suh, .svh = svh, .k = @intCast(k), .n = @intCast(n), .halves = halves, .codebook = .mul1, .stride_k = st.k, .stride_nb = st.nb };
}

fn layerGroup(L: anytype, i: u32, comptime what: []const u8, cut: Cut) !exl3.Weight {
    var nb: [192]u8 = undefined;
    return group(L, L.ck, try std.fmt.bufPrint(&nb, prefix ++ "layers.{d}." ++ what, .{i}), cut);
}

fn one(L: anytype, w: exl3.Weight) !u64 {
    return L.w.xl.?.add(&.{.{ .q = w }}, &.{@as(usize, w.n)}, w.k);
}

fn heads(s: Span, per: usize) Span {
    return .{ .lo = s.lo * per, .hi = s.hi * per };
}

/// weights.load's kda(i): [q | k | v] (EXL3) beside [f_a | g_a | b] (fp16 -> bf16) as one handle, f_b/g_b by heads,
/// the fused q|k|v conv's rank rows, o by input rows.
pub fn kda(L: anytype, i: u32) !kda_block.Layer {
    const c = &L.w.config;
    const h = L.w.share.kda;
    const dk = c.kda_dim;
    const all = c.kda_heads * dk;
    const n = h.len() * dk;
    const qkv = try layerGroup(L, i, "self_attn.qkv_proj", .{ .qkv = heads(h, dk) });
    inline for (.{ "f_a", "g_a" }) |x| try L.whole(try L.get("layers.{d}.self_attn." ++ x ++ "_proj.weight", .{i}), .bf16);
    try L.rows(try L.get("layers.{d}.self_attn.b_proj.weight", .{i}), h.lo, h.hi, c.kda_heads, .bf16);
    const fab_rows = L.host.items.len / (2 * c.hidden);
    const fab = try L.put("layers.{d}.kda.fab", .{i});
    const proj = try L.w.xl.?.add(&.{ .{ .q = qkv }, .{ .b = fab } }, &.{ 3 * n, fab_rows }, c.hidden);
    try L.rows(try L.get("layers.{d}.self_attn.f_b_proj.weight", .{i}), h.lo * dk, h.hi * dk, all, .bf16);
    const fb = try L.put("layers.{d}.kda.fb", .{i});
    try L.rows(try L.get("layers.{d}.self_attn.g_b_proj.weight", .{i}), h.lo * dk, h.hi * dk, all, .bf16);
    const gb = try L.put("layers.{d}.kda.gb", .{i});
    const conv = try L.get("layers.{d}.self_attn.conv1d.weight", .{i}); // [q | k | v] rows
    for (0..3) |j| try L.rows(conv, j * all + h.lo * dk, j * all + h.hi * dk, 3 * all, .f32);
    const conv_w = try L.put("layers.{d}.kda.conv", .{i});
    try L.rows(try L.get("layers.{d}.self_attn.A_log", .{i}), h.lo, h.hi, c.kda_heads, .f32);
    const a_log = try L.put("layers.{d}.kda.a_log", .{i});
    try L.rows(try L.get("layers.{d}.self_attn.dt_bias", .{i}), h.lo * dk, h.hi * dk, all, .f32);
    const dt = try L.put("layers.{d}.kda.dt_bias", .{i});
    try L.whole(try L.get("layers.{d}.self_attn.o_norm.weight", .{i}), .bf16);
    const norm = try L.put("layers.{d}.kda.norm", .{i});
    const o = try one(L, try layerGroup(L, i, "self_attn.o_proj", .{ .in = heads(h, dk) }));
    return .{ .proj = proj, .fb = fb, .gb = gb, .o = o, .conv_w = conv_w, .a_log = a_log, .dt_bias = dt, .norm_w = norm, .eps = c.eps, .lower = c.lower_bound, .heads = h.len(), .dims = c.hidden };
}

/// weights.load's dsa(i): [q_a | kv_a] (EXL3, whole) as one handle, q_b by heads, kv_b (fp16) as wk/wv, o by inputs.
pub fn dsa(L: anytype, i: u32) !dsa_block.Layer {
    const c = &L.w.config;
    const h = L.w.share.mla;
    const qk = c.nope;
    const qa = try layerGroup(L, i, "self_attn.q_a_proj", .rep);
    const kva = try layerGroup(L, i, "self_attn.kv_a_proj_with_mqa", .rep);
    if (qa.n != c.q_lora or kva.n != c.kv_lora) return error.Exl3Cut;
    const proj = try L.w.xl.?.add(&.{ .{ .q = qa }, .{ .q = kva } }, &.{ @as(usize, qa.n), @as(usize, kva.n) }, c.hidden);
    try L.whole(try L.get("layers.{d}.self_attn.q_a_layernorm.weight", .{i}), .bf16);
    const q_norm = try L.put("layers.{d}.dsa.q_norm", .{i});
    try L.whole(try L.get("layers.{d}.self_attn.kv_a_layernorm.weight", .{i}), .bf16);
    const kv_norm = try L.put("layers.{d}.dsa.kv_norm", .{i});
    const q_b = try one(L, try layerGroup(L, i, "self_attn.q_b_proj", .{ .out = heads(h, qk) }));
    const kv_b = try L.get("layers.{d}.self_attn.kv_b_proj.weight", .{i});
    const per = qk + c.v_dim; // a head's kv_b rows: its keys, then its values
    const total = c.mla_heads * per;
    for (h.lo..h.hi) |head| try L.rows(kv_b, head * per, head * per + qk, total, .bf16);
    const wk = try L.put("layers.{d}.dsa.absorb.wk", .{i});
    // wk per head transposed ([heads][lw][qk]) for the prompt rows' tensor-core absorb: the same rows, rewritten
    for (h.lo..h.hi) |head| try L.rows(kv_b, head * per, head * per + qk, total, .bf16);
    {
        const nh = h.len();
        const lw: usize = c.kv_lora;
        const src = try L.gpa.dupe(u8, L.host.items);
        defer L.gpa.free(src);
        const s16: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, src));
        const d16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, L.host.items));
        for (0..nh) |hh| for (0..qk) |kk| for (0..lw) |nn| {
            d16[(hh * lw + nn) * qk + kk] = s16[(hh * qk + kk) * lw + nn];
        };
    }
    const wkt = try L.put("layers.{d}.dsa.absorb.wkt", .{i});
    for (h.lo..h.hi) |head| try L.rows(kv_b, head * per + qk, (head + 1) * per, total, .bf16);
    const wv = try L.put("layers.{d}.dsa.absorb.wv", .{i});
    const o = try one(L, try layerGroup(L, i, "self_attn.o_proj", .{ .in = heads(h, c.v_dim) }));
    inline for (.{ "wk", "weights_proj" }) |x| try L.whole(try L.get("layers.{d}.self_attn.indexer." ++ x ++ ".weight", .{i}), .bf16);
    const kw = try L.put("layers.{d}.dsa.index.kw", .{i});
    try L.whole(try L.get("layers.{d}.self_attn.indexer.wq_b.weight", .{i}), .bf16);
    const qb = try L.put("layers.{d}.dsa.index.qb", .{i});
    try L.whole(try L.get("layers.{d}.self_attn.indexer.k_norm.weight", .{i}), .bf16);
    const ln_w = try L.put("layers.{d}.dsa.index.ln_w", .{i});
    try L.whole(try L.get("layers.{d}.self_attn.indexer.k_norm.bias", .{i}), .bf16);
    const ln_b = try L.put("layers.{d}.dsa.index.ln_b", .{i});
    try L.whole(try L.get("layers.{d}.self_attn.indexer.index_kpool_compress_gate", .{i}), .bf16);
    const gate = try L.put("layers.{d}.dsa.index.gate", .{i});
    try L.whole(try L.get("layers.{d}.self_attn.indexer.index_kpool_compress_ape", .{i}), .bf16);
    const ape = try L.put("layers.{d}.dsa.index.ape", .{i});
    return .{
        .proj = proj,
        .q_norm = q_norm,
        .kv_norm = kv_norm,
        .q_b = q_b,
        .wk = wk,
        .wkt = wkt,
        .wv = wv,
        .o = o,
        .index = .{ .kw = kw, .gate = gate, .ln_w = ln_w, .ln_b = ln_b, .ape = ape, .qb = qb, .heads = c.i_heads, .d = c.i_dim },
        .heads = h.len(),
        .qk = qk,
        .dv = c.v_dim,
        .lw = c.kv_lora,
        .q_lora = c.q_lora,
        .dims = c.hidden,
        .eps = c.eps,
        .dense_limit = c.keyWidth(),
    };
}

/// [gate | up] by the span's output columns as one handle, down by its input rows: a dense MLP's or the shared expert's.
pub fn swiglu(L: anytype, i: u32, comptime base: []const u8, s: Span) !struct { gu: u64, down: u64 } {
    const g = try layerGroup(L, i, base ++ ".gate_proj", .{ .out = s });
    const u = try layerGroup(L, i, base ++ ".up_proj", .{ .out = s });
    const gu = try L.w.xl.?.add(&.{ .{ .q = g }, .{ .q = u } }, &.{ s.len(), s.len() }, L.w.config.hidden);
    const down = try one(L, try layerGroup(L, i, base ++ ".down_proj", .{ .in = s }));
    return .{ .gu = gu, .down = down };
}

pub fn dense(L: anytype, i: u32) !mlp.Layer {
    const c = &L.w.config;
    const s = L.w.share.dense;
    const w = try swiglu(L, i, "mlp", s);
    return .{ .w = .{ .bf16 = .{ .gu = w.gu, .down = w.down } }, .width = s.len(), .dims = c.hidden, .limit = c.swiglu_limit };
}

/// lm_head: the rank's vocabulary rows (output columns of the EXL3 group).
pub fn lmHead(L: anytype, head_ck: *core.Checkpoint) !u64 {
    return one(L, try group(L, head_ck, "lm_head", .{ .out = L.w.share.vocab }));
}

/// The MTP layer's eh_proj, whole: [hnorm(h) | enorm(e)] -> the layer's input.
pub fn eh(L: anytype) !u64 {
    var nb: [192]u8 = undefined;
    return one(L, try group(L, L.ck, try std.fmt.bufPrint(&nb, prefix ++ "layers.{d}.eh_proj", .{L.w.config.layers}), .rep));
}

fn noBf16(_: *const anyopaque, _: cuda.Stream, _: u64, _: usize, _: u64, _: u64, _: u64, _: usize, _: usize, _: usize, _: bool) anyerror!void {
    return error.UnexpectedBf16Part;
}

/// The head's rank rows as bf16 [n][k] for the 4-bit draft head: the identity's rows through the EXL3 linear
/// (fp32 W[k][n] in 32-row passes), transposed and rounded on the host. The caller frees the buffer.
pub fn headBf16(L: anytype, id: u64, n: usize, k: usize) !cuda.DeviceBuffer {
    const r = L.w.xl.?;
    const l = r.find(id) orelse return error.Exl3LinearMissing;
    const pass = 32; // under xlin.prompt_rows: the decode kernel's bits
    const host = try L.gpa.alloc(u16, n * k);
    defer L.gpa.free(host);
    const eye = try L.gpa.alloc(u16, pass * k);
    defer L.gpa.free(eye);
    const y = try L.gpa.alloc(f32, pass * n);
    defer L.gpa.free(y);
    var xb = try cuda.DeviceBuffer.alloc(L.d, pass * k * 2);
    defer xb.free();
    var yb = try cuda.DeviceBuffer.alloc(L.d, pass * n * 4);
    defer yb.free();
    var r0: usize = 0;
    while (r0 < k) : (r0 += pass) {
        @memset(eye, 0);
        for (0..pass) |j| eye[j * k + r0 + j] = 0x3F80; // bf16 1.0
        try xb.upload(0, std.mem.sliceAsBytes(eye));
        try r.run(noBf16, undefined, L.stream, l, null, xb.ptr, k, yb.ptr, 0, pass, n, k, true);
        try L.stream.synchronize();
        try yb.download(0, std.mem.sliceAsBytes(y));
        for (0..pass) |j| for (0..n) |col| {
            host[col * k + r0 + j] = bf16Of(y[j * n + col]);
        };
    }
    return cuda.DeviceBuffer.fromHost(L.d, std.mem.sliceAsBytes(host));
}

fn bf16Of(v: f32) u16 {
    const b: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @intCast((b >> 16) | 0x40);
    return @intCast((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

test "a fused q|k|v group's rank columns come from each third, in the strips order" {
    // kt 1, nt 48 one-byte tiles (three thirds of 16): rank 1 of 2 takes tiles 8..15 of each third
    var src: [48]u8 = undefined;
    for (&src, 0..) |*v, x| v.* = @intCast(x);
    var out: [16]u8 = undefined;
    try strips(&out, src[8..24], 1, 16, 1);
    try std.testing.expectEqualSlices(u8, src[8..24], &out); // one tile row: strips keep each 8-tile block in order
    var bad: [8]u8 = undefined;
    try std.testing.expectError(error.Exl3Cut, strips(&bad, src[0..8], 1, 4, 2));
}
