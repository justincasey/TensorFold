//! The MTP layer's BF16 routed experts quantized to NVFP4 on the GPU (a weights.Loader method moved out).
const std = @import("std");
const cuda = @import("cuda");
const quant = @import("cuda_quant.zig");
const X = cuda.experts.Nvfp4Experts;

/// moe_nvfp4 of the MTP layer: BF16 experts cut to the rank, GPU-quantized 8 at a time, scale as weight_scale_2.
pub fn quantizedExperts(L: anytype, i: u32, up: []u32, down: []u32, up_scale: []f32, down_scale: []f32) !void {
    const qk = L.tools.quant orelse return error.NoQuantizer;
    const c = &L.w.config;
    const d: usize = c.hidden;
    const cut = L.w.share.moe;
    const ni = cut.len();
    const up_len = X.bytes(1, ni, d, 2) / 4;
    const down_len = X.bytes(1, d, ni, 1) / 4;
    const chunk = 8;
    var wb = try L.scratch(chunk * ni * d * 2);
    defer wb.free();
    var cb = try L.scratch(chunk * ni * d / 2);
    defer cb.free();
    var sb = try L.scratch(chunk * ni * d / 16);
    defer sb.free();
    var gb = try L.scratch(chunk * 4);
    defer gb.free();
    const codes = try L.gpa.alloc(u8, 3 * chunk * ni * d / 2); // gate, up and down apart: packGateUp takes both
    defer L.gpa.free(codes);
    const scales = try L.gpa.alloc(u8, 3 * chunk * ni * d / 16);
    defer L.gpa.free(scales);
    var g: [3][chunk]f32 = undefined;
    var e0: usize = 0;
    while (e0 < c.experts) : (e0 += chunk) {
        const ec = @min(chunk, c.experts - e0);
        inline for (.{ "gate_proj", "up_proj", "down_proj" }, 0..) |proj, m| {
            const n = if (m < 2) ni else d; // gate and up: the rank's rows; down: every row, the rank's columns
            const k = if (m < 2) d else ni;
            for (e0..e0 + ec) |e| {
                try L.format("layers.{d}.mlp.experts.{d}." ++ proj, .{ i, e }, .bf16);
                const t = try L.get("layers.{d}.mlp.experts.{d}." ++ proj ++ ".weight", .{ i, e });
                if (m < 2) try L.rows(t, cut.lo, cut.hi, c.moe_inter, .bf16) else try L.cols(t, cut.lo, cut.hi, .bf16);
            }
            try wb.upload(0, L.host.items);
            L.host.clearRetainingCapacity();
            try quant.nvfp4(qk, L.stream, wb.ptr, ec, n, k, cb.ptr, sb.ptr, gb.ptr);
            try L.stream.synchronize();
            try cb.download(0, codes[m * chunk * n * k / 2 ..][0 .. ec * n * k / 2]);
            try sb.download(0, scales[m * chunk * n * k / 16 ..][0 .. ec * n * k / 16]);
            try gb.download(0, std.mem.sliceAsBytes(g[m][0..ec]));
        }
        const mc = chunk * ni * d / 2; // one matrix's codes for the chunk; scales an eighth of that
        for (0..ec) |j| {
            const e = e0 + j;
            const one = ni * d;
            var cs: [3][]const u8 = undefined;
            var ss: [3][]const u8 = undefined;
            for (0..3) |m| {
                cs[m] = codes[m * mc + j * one / 2 ..][0 .. one / 2];
                ss[m] = scales[m * mc / 8 + j * one / 16 ..][0 .. one / 16];
            }
            X.packGateUp(up[e * up_len ..][0..up_len], .{ cs[0], ss[0] }, .{ cs[1], ss[1] }, ni, d);
            X.packOne(down[e * down_len ..][0..down_len], cs[2], ss[2], d, ni);
            up_scale[e * 2] = g[0][j];
            up_scale[e * 2 + 1] = g[1][j];
            down_scale[e] = g[2][j];
        }
    }
}
