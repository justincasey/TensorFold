//! GLM E4M3 latent cache and shared-window attention resources from nvcc 13.0; numerical device gates remain separate.
const registry = @import("registry.zig");
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "glm_glue tf_glm_lat_write8", .symbol = "tf_glm_lat_write8", .image = "glm_glue", .threads = 128, .min_sm = 80 }, .usage = .{ .{ .regs = 24, .static = 16, .bound = 128 }, .{ .regs = 24, .static = 16, .bound = 128 }, .{ .regs = 23, .static = 16, .bound = 128 }, .{ .regs = 24, .static = 16, .bound = 128 }, .{ .regs = 24, .static = 16, .bound = 128 } }, .later_pending = true },
    .{ .launch = .{ .name = "glm_glue tf_glm_chunks8_16", .symbol = "tf_glm_chunks8_16", .image = "glm_glue", .threads = 256, .dynamic = 83584, .opt_in = true, .min_sm = 80 }, .usage = .{ .{ .regs = 71, .static = 0, .bound = 256 }, .{ .regs = 71, .static = 0, .bound = 256 }, .{ .regs = 71, .static = 0, .bound = 256 }, .{ .regs = 118, .static = 0, .bound = 256 }, .{ .regs = 118, .static = 0, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "glm_glue tf_glm_chunks8_32", .symbol = "tf_glm_chunks8_32", .image = "glm_glue", .threads = 256, .dynamic = 101376, .opt_in = true, .min_sm = 80 }, .usage = .{ .{ .regs = 128, .static = 0, .bound = 256 }, .{ .regs = 126, .static = 0, .bound = 256 }, .{ .regs = 126, .static = 0, .bound = 256 }, .{ .regs = 128, .static = 0, .bound = 256 }, .{ .regs = 128, .static = 0, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "glm_glue tf_glm_chunks_16s", .symbol = "tf_glm_chunks_16s", .image = "glm_glue", .threads = 256, .dynamic = 83328, .opt_in = true, .min_sm = 80 }, .usage = .{ .{ .regs = 64, .static = 0, .bound = 256 }, .{ .regs = 64, .static = 0, .bound = 256 }, .{ .regs = 64, .static = 0, .bound = 256 }, .{ .regs = 72, .static = 0, .bound = 256 }, .{ .regs = 72, .static = 0, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "glm_glue tf_glm_chunks8_16s", .symbol = "tf_glm_chunks8_16s", .image = "glm_glue", .threads = 256, .dynamic = 83584, .opt_in = true, .min_sm = 80 }, .usage = .{ .{ .regs = 71, .static = 0, .bound = 256 }, .{ .regs = 71, .static = 0, .bound = 256 }, .{ .regs = 71, .static = 0, .bound = 256 }, .{ .regs = 122, .static = 0, .bound = 256 }, .{ .regs = 122, .static = 0, .bound = 256 } }, .later_pending = true },
};
test "GLM FP8 KV: measured 13.0 launches fit their physical resource limits" {
    const t = @import("std").testing;
    for (records) |r| {
        for (r.usage) |u| try t.expect(u.regs > 0);
        try t.expectEqual(@as(usize, 0), try registry.checkCuda(t.allocator, &.{r}, true));
    }
}
