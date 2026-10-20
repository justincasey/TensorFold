//! TF_GLM_DIGEST=1: a whole prompt pass's sequence state as sha256 digests on stderr (checks of prompt rows' bits).

const std = @import("std");
const engine = @import("cuda_engine.zig");
const fw = @import("cuda_forward.zig");
const dflash = @import("dflash.zig");

const Hash = std.crypto.hash.sha2.Sha256;

/// Whether TF_GLM_DIGEST=1 asks for them.
pub fn on() bool {
    const v = std.c.getenv("TF_GLM_DIGEST") orelse return false;
    return v[0] == '1';
}

fn add(e: *engine.Engine, h: *Hash, ptr: u64, bytes: usize) !void {
    if (bytes == 0) return;
    const buf = try e.gpa.alloc(u8, bytes);
    defer e.gpa.free(buf);
    try e.stream.synchronize();
    try e.d.check(e.d.api.cuMemcpyDtoH_v2(buf.ptr, ptr, bytes), "cuMemcpyDtoH");
    h.update(buf);
}

fn hex(h: *Hash) [16]u8 {
    var out: [32]u8 = undefined;
    h.final(&out);
    return std.fmt.bytesToHex(out[0..8].*, .lower);
}

/// Sequence `st` after a whole prompt of `ids` from 0: KDA states and conv rows, DSA caches, its taps and logits row.
pub fn seq(e: *engine.Engine, st: *fw.State, ids: []const u32, taps: u64, logits: u64, path: []const u8, first: u32) !void {
    const n = ids.len;
    var hk: Hash = .init(.{});
    var hc: Hash = .init(.{});
    var hl: Hash = .init(.{});
    var hx: Hash = .init(.{});
    var ht: Hash = .init(.{});
    var hg: Hash = .init(.{});
    var hi: Hash = .init(.{});
    var li: usize = 0;
    var di: usize = 0;
    for (e.w.layers) |l| switch (l.mixer) {
        .kda => {
            try add(e, &hk, st.recAt(st.cur[li], li), st.rec_layer);
            try add(e, &hc, st.conv + li * 3 * st.conv_c * 2, 3 * st.conv_c * 2);
            li += 1;
        },
        .mla => |a| {
            const cs = st.caches[di];
            try add(e, &hl, cs.lc, n * (if (cs.ls != 0) a.lw else a.lw * 2));
            if (cs.ls != 0) try add(e, &hl, cs.ls, n * 4);
            try add(e, &hx, cs.ik, n * 256);
            try add(e, &hx, cs.ig, n * 256);
            try add(e, &hx, cs.pk, n / 4 * 256);
            di += 1;
        },
    };
    if (taps != 0) try add(e, &ht, taps, n * dflash.TAP_ROW);
    try add(e, &hg, logits, e.w.share.vocab.len() * 2);
    hi.update(std.mem.sliceAsBytes(ids));
    std.debug.print("DIGEST rank={d} path={s} len={d} ids={s} kda={s} conv={s} lat={s} idx={s} taps={s} logits={s} first={d}\n", .{ e.share.rank, path, n, hex(&hi), hex(&hk), hex(&hc), hex(&hl), hex(&hx), hex(&ht), hex(&hg), first });
}
