//! DFlash2's taps on the target's forward: after each tapped layer, its rows' 4-stream mean into one tap slot.
const cuda = @import("cuda");
const relay = @import("cuda_relay.zig");
const Pieces = @import("cuda_pieces.zig").Pieces;

/// The target layers DFlash2 reads (incoai/GLM-5.3-Flash-DFlash2's target_layer_ids), in tap-slot order.
pub const layers = [_]usize{ 5, 14, 24, 33, 42 };

/// The last tapped layer: a split prompt piece's tap rows are exchanged after it (every slot of a row then complete).
pub const last = layers[layers.len - 1];

/// A tap row's bf16 values (every slot).
pub const row_dims = layers.len * 4096;

/// Where a forward's taps go: rows [R][5 * hidden] bf16 at `out`, through glm_dflash's df_tap.
pub const Taps = struct {
    f: cuda.Function,
    out: u64,
    keep: usize = 0, // a prompt chunk's first row the drafter ingests (Drafter.keepFrom): rows before it aren't swapped

    /// After layer `j` of an `R`-row pass on residual `x` (bf16 [R][4][4096]): its tap slot, if `j` is tapped.
    pub fn at(t: Taps, s: cuda.Stream, j: usize, x: u64, R: usize) !void {
        return t.rows(s, j, x, 0, R);
    }

    /// `at` for rows lo..hi only (a split piece's glued rows; the others' residual rows are stale on this rank).
    pub fn rows(t: Taps, s: cuda.Stream, j: usize, x: u64, lo: usize, hi: usize) !void {
        const slot: i32 = for (layers, 0..) |l, i| {
            if (l == j) break @intCast(i);
        } else return;
        if (hi <= lo) return;
        var a: cuda.Args = .{};
        a.add(x + lo * 4 * 4096 * 2);
        a.add(t.out + lo * row_dims * 2);
        a.add(slot);
        a.add(@as(i32, 0)); // the mean of the 4 streams (df_tap's mode 0)
        try cuda.launch.launch(t.f, .{ .grid = .{ .x = @intCast(hi - lo), .y = 4096 / 1024 }, .block = .{ .x = 256 } }, s, &a);
    }

    /// hcsplit.glue's swap of a split piece's tap rows lo..lo + n (every slot) with the other ranks, on `side`.
    pub fn swap(t: Taps, exchange: anytype, side: cuda.Stream, plan: relay.Relay, lo: usize, ring: bool) !void {
        switch (exchange) {
            .local => {}, // in-process ranks tapped their own rows of the one buffer
            .nccl => |x| if (ring) try relay.glued(x.lib, x.comm, side, plan, t.out, lo, row_dims) else try relay.splitGlued(x.lib, x.comm, side, plan, t.out, lo, row_dims),
        }
    }
};

/// The swapped tap rows in (Pieces.tapped) on `s` before a pass of layers lo..hi ends, if it ran the last tapped layer.
pub fn swapped(taps: ?Taps, s: cuda.Stream, p: *const Pieces, lo: usize, hi: usize) !void {
    if (taps != null and lo <= last and last < hi) try s.wait(p.tapped);
}
