//! The EXL3 routed experts: the f64 checks, row and chunk invariance, and the Python engine's recorded bytes.
const std = @import("std");
const check = @import("check.zig");
const quant_ref_tests = @import("quant_ref_tests.zig");
const cuda = @import("cuda");
const exl3 = cuda.exl3;
const exl3e = cuda.exl3_experts;
const exl3t = @import("exl3_tests.zig");
const ref = @import("exl3_experts_f64_tests.zig");
const Rig = quant_ref_tests.RigT;
const Run = exl3t.Run;

pub fn run(r: *Rig, state: *Run, dir: []const u8) !void {
    var exp = try exl3e.Experts.load(r.gpu.d);
    defer exp.unload();

    // every expert one width: 4, 5 and 6 bits (the mixed run folds into .6)
    ref.checks(r, &exp, state);
    {
        var ok = true;
        rowsInvariant(r, &exp) catch |e| {
            state.fail(e, "exl3.experts.rows");
            ok = false;
        };
        if (ok) check.pass("exl3.experts.rows", .{});
    }
    {
        var ok = true;
        ref.prompt(r, &exp) catch |e| {
            state.fail(e, "exl3.prompt.experts.f64");
            ok = false;
        };
        if (ok) check.pass("exl3.prompt.experts.f64", .{});
    }
    {
        var ok = true;
        chunks(r, &exp) catch |e| {
            state.fail(e, "exl3.prompt.experts.chunks");
            ok = false;
        };
        if (ok) check.pass("exl3.prompt.experts.chunks", .{});
    }
    {
        var ok = true;
        fused(r, &exp) catch |e| {
            state.fail(e, "exl3.prompt.fused");
            ok = false;
        };
        if (ok) check.pass("exl3.prompt.fused: mul1 at mixed widths (gate and up apart) within f64, chunk and repeat bytes equal", .{});
    }
    {
        var ok = true;
        python(r, &exp, dir, state) catch |e| {
            state.fail(e, "exl3.experts.python");
            ok = false;
        };
        if (ok) check.pass("exl3.experts.python", .{});
    }
}

/// A row's output bytes are its own: solo, with 3 rows, in 64-row batches, shared or not.
fn rowsInvariant(r: *Rig, exp: *const exl3e.Experts) !void {
    var bits: [ref.E]u32 = .{ 6, 4, 5, 6, 4, 5, 6, 4, 5 };
    const b = try ref.buildLayer(r, &bits);
    const solo = try r.back(try ref.runRows(r, exp, &b, 1), ref.D * 4);
    // 3 rows: two more rows on the same picks as row 0
    const p3 = try r.a.alloc(i32, 3 * ref.SLOTS);
    @memcpy(p3[0 .. 2 * ref.SLOTS], b.pick[0 .. 2 * ref.SLOTS]);
    @memcpy(p3[2 * ref.SLOTS ..], b.pick[0..ref.SLOTS]);
    const w3 = try r.a.alloc(f32, 3 * ref.SLOTS);
    @memcpy(w3, b.wts[0 .. 3 * ref.SLOTS]);
    const got3 = try r.back((try ref.runSlice(r, exp, b.layer, b.dx, try r.dev(std.mem.sliceAsBytes(p3)), try r.dev(std.mem.sliceAsBytes(w3)), 3)).out, ref.D * 4);
    try check.sameBytes("exl3.experts.rows 3 rows", got3, solo);
    // 64 rows: the others share row 0's experts (the same picks), or run their own
    const p64 = try r.a.alloc(i32, 64 * ref.SLOTS);
    const w64 = try r.a.alloc(f32, 64 * ref.SLOTS);
    @memcpy(p64[0..ref.SLOTS], b.pick[0..ref.SLOTS]);
    @memcpy(w64[0..ref.SLOTS], b.wts[0..ref.SLOTS]);
    for (1..64) |row| {
        @memcpy(p64[row * ref.SLOTS ..][0..ref.SLOTS], b.pick[0..ref.SLOTS]);
        @memcpy(w64[row * ref.SLOTS ..][0..ref.SLOTS], b.wts[0..ref.SLOTS]);
    }
    const shared = try r.back((try ref.runSlice(r, exp, b.layer, b.dx, try r.dev(std.mem.sliceAsBytes(p64)), try r.dev(std.mem.sliceAsBytes(w64)), 64)).out, ref.D * 4);
    try check.sameBytes("exl3.experts.rows 64 rows shared", shared[0 .. ref.D * 4], solo);
    const own = try r.back((try ref.runSlice(r, exp, b.layer, b.dx, b.dpick, b.dwts, 64)).out, ref.D * 4);
    try check.sameBytes("exl3.experts.rows 64 rows own", own[0 .. ref.D * 4], solo);
    // and a 64-row batch whose other rows interleave with row 0's experts in both orders
    const p64b = try r.a.alloc(i32, 64 * ref.SLOTS);
    const w64b = try r.a.alloc(f32, 64 * ref.SLOTS);
    @memcpy(p64b[0..ref.SLOTS], b.pick[0..ref.SLOTS]);
    @memcpy(w64b[0..ref.SLOTS], b.wts[0..ref.SLOTS]);
    for (1..64) |row| {
        const src = if (row % 2 == 0) @as(usize, 0) else row % ref.SLOTS;
        @memcpy(p64b[row * ref.SLOTS ..][0..ref.SLOTS], b.pick[src * ref.SLOTS ..][0..ref.SLOTS]);
        @memcpy(w64b[row * ref.SLOTS ..][0..ref.SLOTS], b.wts[src * ref.SLOTS ..][0..ref.SLOTS]);
    }
    const mixed = try r.back((try ref.runSlice(r, exp, b.layer, b.dx, try r.dev(std.mem.sliceAsBytes(p64b)), try r.dev(std.mem.sliceAsBytes(w64b)), 64)).out, ref.D * 4);
    try check.sameBytes("exl3.experts.rows 64 rows mixed", mixed[0 .. ref.D * 4], solo);
}

/// The same 300 rows as one call, split at 128 and at 37: a row's bytes never move.
fn chunks(r: *Rig, exp: *const exl3e.Experts) !void {
    var bits: [ref.E]u32 = .{ 4, 5, 6, 4, 5, 6, 4, 5, 6 };
    const b = try ref.buildLayer(r, &bits);
    const whole = try r.back(try ref.runRows(r, exp, &b, 300), 300 * ref.D * 4);
    var at: usize = 0;
    while (at < 300) : (at += 128) {
        const n: usize = @min(300 - at, 128);
        const res = try ref.runSlice(r, exp, b.layer, b.dx + @as(u64, at) * ref.D * 2, b.dpick + @as(u64, at) * ref.SLOTS * 4, b.dwts + @as(u64, at) * ref.SLOTS * 4, n);
        const got = try r.back(res.out, n * ref.D * 4);
        try check.sameBytes("exl3.prompt.experts.chunks 128", got, whole[at * ref.D * 4 ..][0 .. n * ref.D * 4]);
    }
    at = 0;
    while (at < 300) : (at += 37) {
        const n: usize = @min(300 - at, 37);
        const res = try ref.runSlice(r, exp, b.layer, b.dx + @as(u64, at) * ref.D * 2, b.dpick + @as(u64, at) * ref.SLOTS * 4, b.dwts + @as(u64, at) * ref.SLOTS * 4, n);
        const got = try r.back(res.out, n * ref.D * 4);
        try check.sameBytes("exl3.prompt.experts.chunks 37", got, whole[at * ref.D * 4 ..][0 .. n * ref.D * 4]);
    }
}

/// The fused prompt kernels' per-slot rows of `n` rows from `at`, over a fresh plan and scratch.
fn fusedRows(r: *Rig, exp: *const exl3e.Experts, router: cuda.grouped.Router, b: *const ref.Built, at: usize, n: usize) ![]u8 {
    const P = n * ref.SLOTS;
    const wide = P > cuda.grouped.small;
    const plan: cuda.grouped.Plan = .{
        .members = try r.zeros(P * 4),
        .items = try r.zeros(cuda.grouped.maxItems(P, ref.E, exl3e.prompt.tile) * 12),
        .counts = try r.zeros(16),
        .rank = try r.zeros((if (wide) P else 1) * 4),
        .hist = try r.zeros((if (wide) (P + 1023) / 1024 * ref.E else 1) * 4),
    };
    const sc = try exl3e.prompt.scratch(try r.zeros(try exl3e.prompt.bytes(b.layer, n, ref.SLOTS)), b.layer, n, ref.SLOTS);
    const y = try r.zeros(P * ref.D * 4);
    try exp.prompt.run(exp, router, r.s, b.layer, b.dx + @as(u64, at) * ref.D * 2, ref.D, b.dpick + @as(u64, at) * ref.SLOTS * 4, n, sc, plan, y, ref.SLOTS);
    try r.s.synchronize();
    return r.back(y, P * ref.D * 4);
}

/// The fused path at RED-SNOW's widths, gate and up apart in one expert: f64, repeat, and chunks of 128 and 37 rows.
fn fused(r: *Rig, exp: *const exl3e.Experts) !void {
    const gbits = [ref.E]u32{ 5, 6, 4, 6, 6, 5, 5, 4, 6 };
    const ubits = [ref.E]u32{ 5, 6, 4, 6, 6, 5, 5, 4, 5 };
    const dbits = [ref.E]u32{ 4, 4, 4, 5, 6, 5, 6, 5, 4 };
    var b = try ref.buildWidths(r, &gbits, &ubits, &dbits);
    b.layer.max_k2 = 12;
    if (!exl3e.prompt.eligible(b.layer)) return error.TestFailed;
    var pm = try cuda.Module.load(r.gpu.d, cuda.kernels.experts);
    defer pm.unload();
    const router = try cuda.grouped.Router.resolve(pm);
    const rows = 300;
    const y = try fusedRows(r, exp, router, &b, 0, rows);
    const want = try ref.reference(r, &b, rows);
    var most: f64 = 0;
    for (0..rows) |row| for (0..ref.D) |d| { // the reference's combine: fp32 fmas in slot order
        var acc: f32 = 0;
        for (0..ref.SLOTS) |k| acc = @mulAdd(f32, b.wts[row * ref.SLOTS + k], std.mem.bytesToValue(f32, y[((row * ref.SLOTS + k) * ref.D + d) * 4 ..][0..4]), acc);
        most = @max(most, @abs(@as(f64, acc) - want.want[row * ref.D + d]) / want.tol[row * ref.D + d]);
    };
    std.debug.print("RESULT exl3.prompt.fused.f64: largest difference {d:.3} of its tolerance\n", .{most});
    if (!(most <= 1)) return error.TestFailed;
    try check.sameBytes("exl3.prompt.fused repeat", try fusedRows(r, exp, router, &b, 0, rows), y);
    for ([_]usize{ 128, 37 }) |step| {
        var at: usize = 0;
        while (at < rows) : (at += step) {
            const n: usize = @min(rows - at, step);
            const per = ref.SLOTS * ref.D * 4;
            try check.sameBytes("exl3.prompt.fused chunks", try fusedRows(r, exp, router, &b, at, n), y[at * per ..][0 .. n * per]);
        }
    }
}

/// The Python engine's recorded y and out, bit for bit (exl3_oracle_experts.py's manifest and raw files).
fn python(r: *Rig, exp: *const exl3e.Experts, dir: []const u8, state: *Run) !void {
    if (std.mem.eql(u8, dir, "none")) {
        std.debug.print("SKIP exl3.experts.python\n", .{});
        return;
    }
    const manifest_path = try std.fs.path.join(r.a, &.{ dir, "manifest.json" });
    const text = std.Io.Dir.cwd().readFileAlloc(r.gpu.io, manifest_path, r.a, .limited(1 << 24)) catch |e| {
        std.debug.print("FAIL exl3.experts.python: {t}\n", .{e});
        state.failed = true;
        return;
    };
    var parsed = try std.json.parseFromSlice(std.json.Value, r.a, text, .{});
    const o = parsed.value.object.get("experts").?.object;
    const e_count: usize = @intCast(o.get("count").?.integer);
    const dims: usize = @intCast(o.get("dims").?.integer);
    const width: usize = @intCast(o.get("width").?.integer);
    const slots: usize = @intCast(o.get("slots").?.integer);
    const cb: exl3.Codebook = if (std.mem.eql(u8, o.get("codebook").?.string, "mcg"))
        .mcg
    else if (std.mem.eql(u8, o.get("codebook").?.string, "inst3"))
        .inst3
    else
        .mul1;
    const trellis = o.get("trellis").?.object;
    var gate_p = try r.a.alloc(u64, e_count);
    var up_p = try r.a.alloc(u64, e_count);
    var down_p = try r.a.alloc(u64, e_count);
    var gk2 = try r.a.alloc(i32, e_count);
    var uk2 = try r.a.alloc(i32, e_count);
    var dk2 = try r.a.alloc(i32, e_count);
    for (o.get("gate_k2").?.array.items, 0..) |v, i| gk2[i] = @intCast(v.integer);
    for (o.get("up_k2").?.array.items, 0..) |v, i| uk2[i] = @intCast(v.integer);
    for (o.get("down_k2").?.array.items, 0..) |v, i| dk2[i] = @intCast(v.integer);
    for (0..e_count) |i| {
        gate_p[i] = try r.dev(try readFile(r, dir, trellis.get("gate").?.array.items[i].string));
        up_p[i] = try r.dev(try readFile(r, dir, trellis.get("up").?.array.items[i].string));
        down_p[i] = try r.dev(try readFile(r, dir, trellis.get("down").?.array.items[i].string));
    }
    const layer: exl3e.Layer = .{
        .gate = try r.dev(std.mem.sliceAsBytes(gate_p)),
        .up = try r.dev(std.mem.sliceAsBytes(up_p)),
        .down = try r.dev(std.mem.sliceAsBytes(down_p)),
        .gate_k2 = try r.dev(std.mem.sliceAsBytes(gk2)),
        .up_k2 = try r.dev(std.mem.sliceAsBytes(uk2)),
        .down_k2 = try r.dev(std.mem.sliceAsBytes(dk2)),
        .suh_g = try r.dev(try readFile(r, dir, o.get("suh_g").?.string)),
        .suh_u = try r.dev(try readFile(r, dir, o.get("suh_u").?.string)),
        .svh_g = try r.dev(try readFile(r, dir, o.get("svh_g").?.string)),
        .svh_u = try r.dev(try readFile(r, dir, o.get("svh_u").?.string)),
        .suh_d = try r.dev(try readFile(r, dir, o.get("suh_d").?.string)),
        .svh_d = try r.dev(try readFile(r, dir, o.get("svh_d").?.string)),
        .experts = e_count,
        .dims = dims,
        .width = width,
        .codebook = cb,
        .limit = @floatCast(o.get("limit").?.float),
        .act_mode = @intCast(o.get("act_mode").?.integer),
    };
    const xmap = o.get("x").?.object;
    const pickmap = o.get("pick").?.object;
    const wtsmap = o.get("wts").?.object;
    const ymap = o.get("y").?.object;
    const outmap = o.get("out").?.object;
    for (o.get("rows").?.array.items) |rv| {
        const m: usize = @intCast(rv.integer);
        const key = try std.fmt.allocPrint(r.a, "{d}", .{m});
        const dx = try r.dev(try readFile(r, dir, xmap.get(key).?.string));
        const dpick = try r.dev(try readFile(r, dir, pickmap.get(key).?.string));
        const dwts = try r.dev(try readFile(r, dir, wtsmap.get(key).?.string));
        const sb = try r.zeros(try exl3e.scratchBytes(layer, m, slots));
        const sc = try exl3e.scratch(sb, layer, m, slots);
        const out = try r.zeros(m * dims * 4);
        try exp.run(r.s, layer, dx, dims, .bf16, dpick, dwts, m, out, sc);
        const got_y = try r.back(sc.y, m * slots * dims * 4);
        try check.sameBytes("exl3.experts.python y", got_y, try readFile(r, dir, ymap.get(key).?.string));
        const got_out = try r.back(out, m * dims * 4);
        try check.sameBytes("exl3.experts.python out", got_out, try readFile(r, dir, outmap.get(key).?.string));
    }
}

fn readFile(r: *Rig, dir: []const u8, name: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(r.gpu.io, try std.fs.path.join(r.a, &.{ dir, name }), r.a, .limited(1 << 32));
}
