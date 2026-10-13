//! glm.layer.kda: a real RED-SNOW KDA MoE layer at TP=2 rank 0 vs the Python engine's layer_forward, bit for bit.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const quant_ref_tests = @import("quant_ref_tests.zig");
const exl3_tests = @import("exl3_tests.zig");
const ref = @import("glm_kda_ref.zig");
const lref = @import("glm_layer_ref.zig");
const lay = cuda.glm_layer;
const exl3 = cuda.exl3;
const exl3e = cuda.exl3_experts;

const Run = exl3_tests.Run;
const Rig = quant_ref_tests.RigT;
const Obj = std.json.ObjectMap;
const id = "glm.layer.kda";

pub fn run(r: *Rig, state: *Run, dir: []const u8) !void {
    kdaLayer(r, dir) catch |e| state.fail(e, id);
}

fn skip() void {
    std.debug.print("SKIP {s}\n", .{id});
}

fn file(r: *Rig, dir: []const u8, name: []const u8) ![]u8 {
    return ref.readFile(r.a, r.gpu.io, dir, name);
}

fn up(r: *Rig, dir: []const u8, name: []const u8) !u64 {
    return r.dev(try file(r, dir, name));
}

fn uint(o: Obj, key: []const u8) u32 {
    return @intCast(ref.int(o, key));
}

fn kdaLayer(r: *Rig, dir: []const u8) !void {
    if (std.mem.eql(u8, dir, "none")) return skip();
    var man = try ref.Manifest.open(r.gpu.gpa, r.gpu.io, dir);
    defer man.deinit();
    const o = (man.root().get("layer") orelse return skip()).object;
    const p = o.get("params").?.object;
    const c: lay.Config = .{
        .eps = @floatCast(ref.float(p, "eps")),
        .hc_eps = @floatCast(ref.float(p, "hc_eps")),
        .hc_iters = uint(p, "hc_iters"),
        .lower = @floatCast(ref.float(p, "lower")),
        .limit = @floatCast(ref.float(p, "limit")),
        .routed_scale = @floatCast(ref.float(p, "routed_scale")),
        .top_k = uint(p, "top_k"),
        .experts = uint(p, "experts"),
        .d = uint(p, "hidden"),
        .streams = uint(p, "streams"),
    };
    const host = try lref.Host.read(r.a, r.gpu.io, dir, o.get("weights").?.object);
    const w = try weights(r, &host, c);
    var k = try lay.Kernels.load(r.gpu.d);
    defer k.unload();
    var worst: f64 = 0;
    const cases = o.get("cases").?.object;
    var it = cases.iterator();
    while (it.next()) |kv| {
        const t0 = check.now(r.gpu.io);
        const tag = kv.key_ptr.*;
        const e = kv.value_ptr.object;
        try window(r, &k, dir, c, &w, tag, e);
        const q = try lref.layerF64(r.a, r.gpu.io, dir, c, &host, tag, e);
        worst = @max(worst, q);
        const dt = @as(f64, @floatFromInt(check.now(r.gpu.io) - t0)) / 1e9;
        std.debug.print("RESULT {s} {s}: bit-exact, f64 worst {d:.4} of tolerance ({d:.1} s)\n", .{ id, tag, q, dt });
    }
    std.debug.print("RESULT {s}: {d} windows bit-exact, f64 worst {d:.4} of tolerance\n", .{ id, cases.count(), worst });
    check.pass("{s}", .{id});
}

fn linW(r: *Rig, l: lref.Lin) !exl3.Weight {
    const st = exl3.strides(l.k, l.n, l.halves, true);
    return .{
        .trellis = try r.dev(l.words),
        .suh = try r.dev(l.suh),
        .svh = try r.dev(l.svh),
        .k = @intCast(l.k),
        .n = @intCast(l.n),
        .halves = l.halves,
        .codebook = .mul1,
        .stride_k = st.k,
        .stride_nb = st.nb,
    };
}

fn b16W(r: *Rig, b: lref.Mat) !cuda.glm_bmm.B16 {
    return .{ .w = try r.dev(b.w), .n = @intCast(b.n), .k = @intCast(b.k) };
}

/// Rank 0's tensors on the device as the engine holds them; the routed experts' trellises stay in one buffer each.
fn weights(r: *Rig, h: *const lref.Host, c: lay.Config) !lay.LayerW {
    const ex = &h.ex;
    var tables: [3]u64 = undefined;
    for (ex.proj, ex.offs, 0..) |bytes, offs, j| {
        const base = try r.dev(bytes);
        const ptrs = try r.a.alloc(u64, ex.count);
        for (offs, 0..) |off, i| ptrs[i] = base + off;
        tables[j] = try r.dev(std.mem.sliceAsBytes(ptrs));
    }
    const experts: exl3e.Layer = .{
        .gate = tables[0],
        .up = tables[1],
        .down = tables[2],
        .gate_k2 = try r.dev(std.mem.sliceAsBytes(ex.k2[0])),
        .up_k2 = try r.dev(std.mem.sliceAsBytes(ex.k2[1])),
        .down_k2 = try r.dev(std.mem.sliceAsBytes(ex.k2[2])),
        .suh_g = try r.dev(ex.suh_g),
        .suh_u = try r.dev(ex.suh_u),
        .svh_g = try r.dev(ex.svh_g),
        .svh_u = try r.dev(ex.svh_u),
        .suh_d = try r.dev(ex.suh_d),
        .svh_d = try r.dev(ex.svh_d),
        .experts = ex.count,
        .dims = ex.dims,
        .width = ex.width,
        .codebook = .mul1,
        .limit = c.limit,
        .act_mode = 0,
    };
    const kd = &h.kda;
    return .{
        .attn_hc = .{ .fn_w = try r.dev(h.attn_hc[0]), .base = try r.dev(h.attn_hc[1]), .scale = try r.dev(h.attn_hc[2]) },
        .ffn_hc = .{ .fn_w = try r.dev(h.ffn_hc[0]), .base = try r.dev(h.ffn_hc[1]), .scale = try r.dev(h.ffn_hc[2]) },
        .in_norm = try r.dev(h.in_norm),
        .post_norm = try r.dev(h.post_norm),
        .kda = .{
            .qkv = try linW(r, kd.qkv),
            .fa = try b16W(r, kd.fa),
            .ga = try b16W(r, kd.ga),
            .b = try b16W(r, kd.b),
            .fb = try b16W(r, kd.fb),
            .gb = try b16W(r, kd.gb),
            .o = try linW(r, kd.o),
            .conv = try r.dev(kd.conv),
            .a_log = try r.dev(kd.a_log),
            .dt_bias = try r.dev(kd.dt_bias),
            .norm = try r.dev(kd.norm),
            .heads = kd.heads,
            .fa_off = kd.fa_off,
            .ga_off = kd.ga_off,
            .b_off = kd.b_off,
            .p_stride = kd.p_stride,
        },
        .moe = .{
            .router = try r.dev(h.router),
            .bias = try r.dev(h.bias),
            .experts = experts,
            .gate = try linW(r, h.sh[0]),
            .up = try linW(r, h.sh[1]),
            .down = try linW(r, h.sh[2]),
        },
    };
}

fn same(r: *Rig, dir: []const u8, tag: []const u8, what: []const u8, ptr: u64, name: []const u8) !void {
    const want = try file(r, dir, name);
    const label = try std.fmt.allocPrint(r.a, "{s} {s} {s}", .{ id, tag, what });
    try check.sameBytes(label, try r.back(ptr, want.len), want);
}

const Stage = struct { []const u8, u64 };

/// The test's all-gather: own part in slot 0, the fixture's rank-1 partial in slot 1; then the site's stage bits.
const Ctx = struct {
    r: *Rig,
    dir: []const u8,
    tag: []const u8,
    e: Obj,
    sc: *const lay.Scratch,
    x: u64,
    d: usize,

    fn gather(p: *anyopaque, s: cuda.Stream, site: lay.Site, part: u64, gath: u64, rows: usize) anyerror!void {
        const c: *Ctx = @ptrCast(@alignCast(p));
        try s.synchronize();
        const n = rows * c.d * 4;
        const so = c.e.get(if (site == .attn) "s0" else "s1").?.object;
        const g: cuda.DeviceBuffer = .{ .d = c.r.gpu.d, .ptr = gath, .len = 2 * n, .borrowed = true };
        try g.copyFrom(0, part, n, s.handle);
        try g.upload(n, try file(c.r, c.dir, ref.str(so, "peer")));
        const sc = c.sc;
        const stages = if (site == .attn) [_]Stage{
            .{ "normed", sc.normed }, .{ "post", sc.post }, .{ "comb", sc.comb }, .{ "p", sc.p },
            .{ "ka", sc.ka },         .{ "kg", sc.kg },     .{ "kout", sc.kout }, .{ "sk", sc.kda.k },
            .{ "sv", sc.kda.v },      .{ "sg", sc.kda.g },  .{ "sb", sc.kda.b },  .{ "part", part },
            .{ "", 0 },
        } else [_]Stage{
            .{ "x", c.x },        .{ "normed", sc.normed }, .{ "post", sc.post }, .{ "comb", sc.comb },
            .{ "mlog", sc.mlog }, .{ "pick", sc.pick },     .{ "wts", sc.wts },   .{ "ey", sc.ey },
            .{ "smg", sc.smg },   .{ "smu", sc.smu },       .{ "sact", sc.sact }, .{ "sy", sc.sy },
            .{ "part", part },
        };
        for (stages) |st| {
            const name = (so.get(st[0]) orelse continue).string;
            const what = try std.fmt.allocPrint(c.r.a, "{s} {s}", .{ @tagName(site), st[0] });
            try same(c.r, c.dir, c.tag, what, st[1], name);
        }
    }
};

/// One window from the fixture's state: forward (+ commit for decode), every stage's bits, x and the KDA state.
fn window(r: *Rig, k: *lay.Kernels, dir: []const u8, c: lay.Config, w: *const lay.LayerW, tag: []const u8, e: Obj) !void {
    const rows: usize = @intCast(ref.int(e, "rows"));
    const prefill = e.get("prefill").?.bool;
    var sc = try lay.Scratch.init(r.gpu.d, r.s, c, w, rows, 2);
    defer sc.deinit();
    const x = try up(r, dir, ref.str(e, "x0"));
    const rec0 = try file(r, dir, ref.str(e, "rec0"));
    const st: lay.KdaState = .{ .conv = try up(r, dir, ref.str(e, "conv0")), .rec_in = try r.dev(rec0), .rec_out = try r.zeros(rec0.len) };
    var ctx: Ctx = .{ .r = r, .dir = dir, .tag = tag, .e = e, .sc = &sc, .x = x, .d = c.d };
    const g: lay.Gather = .{ .ctx = &ctx, .world = 2, .run = Ctx.gather };
    const mode: lay.Mode = if (prefill) .{ .prompt = @intCast(ref.int(e, "pos")) } else .decode;
    try lay.forward(k, &r.s, c, w, st, &sc, x, rows, mode, g);
    if (!prefill) try lay.commit(k, &r.s, &w.kda, st, &sc, rows, rows);
    try same(r, dir, tag, "x", x, ref.str(e, "x"));
    try same(r, dir, tag, "rec", st.rec_out, ref.str(e, "rec"));
    try same(r, dir, tag, "conv", st.conv, ref.str(e, "conv"));
    // the oracle ran the chunk row-split too (HC_SPLIT=1, PREFILL_OVERLAP=2): own rows and both states equal unsplit
    if (prefill) try check.expect(if (e.get("split_equal")) |v| v.bool else false, "{s} {s}: no split check", .{ id, tag });
    if (e.get("keep")) |kv| {
        const ko = kv.object;
        const kst: lay.KdaState = .{ .conv = try up(r, dir, ref.str(e, "conv0")), .rec_in = st.rec_in, .rec_out = try r.zeros(rec0.len) };
        try lay.commit(k, &r.s, &w.kda, kst, &sc, rows, @intCast(ref.int(ko, "keep")));
        try same(r, dir, tag, "keep rec", kst.rec_out, ref.str(ko, "rec"));
        try same(r, dir, tag, "keep conv", kst.conv, ref.str(ko, "conv"));
    }
}
