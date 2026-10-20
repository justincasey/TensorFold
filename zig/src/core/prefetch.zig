//! Parallel read-ahead of file byte spans into the page cache (Linux only; elsewhere a no-op).
//! A loader that page-faults a mapped checkpoint reads one fault-readahead window (read_ahead_kb, 128 KiB on DGX OS)
//! at a time, ~0.43 GB/s on a GB10's NVMe; several threads' preads keep the drive's queue full instead. Only a hint:
//! the bytes the loader reads come from the mapping either way.

const std = @import("std");
const builtin = @import("builtin");

pub const Span = struct { fd: std.posix.fd_t, lo: u64, hi: u64 };

pub const Prefetch = struct {
    gpa: std.mem.Allocator,
    spans: []Span,
    blocks: u64, // every span's blocks, counted from the first span's
    next: std.atomic.Value(u64) = .init(0),
    threads: [max_threads]?std.Thread = @splat(null),

    pub const max_threads = 8; // 16 measured slower (108 s vs 96-102 s a boot): the loader, not the drive, is the pace
    const block: u64 = 4 << 20;

    /// Starts `threads` readers over `spans` (copied); `join` waits for them and frees it. Null: nothing to read.
    pub fn start(gpa: std.mem.Allocator, spans: []const Span, threads: usize) !?*Prefetch {
        if (builtin.os.tag != .linux) return null;
        var blocks: u64 = 0;
        for (spans) |s| blocks += std.math.divCeil(u64, s.hi - s.lo, block) catch unreachable;
        if (blocks == 0) return null;
        const p = try gpa.create(Prefetch);
        errdefer gpa.destroy(p);
        p.* = .{ .gpa = gpa, .spans = try gpa.dupe(Span, spans), .blocks = blocks };
        for (p.threads[0..@min(threads, max_threads)]) |*t| t.* = std.Thread.spawn(.{}, run, .{p}) catch null;
        return p;
    }

    /// Waits for every reader, then frees the prefetch.
    pub fn join(p: *Prefetch) void {
        for (&p.threads) |*t| if (t.*) |th| th.join();
        p.gpa.free(p.spans);
        p.gpa.destroy(p);
    }

    /// Takes block indices from the shared cursor until every block is read (a failed read is dropped: a hint).
    fn run(p: *Prefetch) void {
        const buf = std.heap.page_allocator.alloc(u8, block) catch return;
        defer std.heap.page_allocator.free(buf);
        while (true) {
            var b = p.next.fetchAdd(1, .monotonic);
            if (b >= p.blocks) return;
            for (p.spans) |s| {
                const n = s.hi - s.lo;
                const in = std.math.divCeil(u64, n, block) catch unreachable;
                if (b >= in) {
                    b -= in;
                    continue;
                }
                const len: usize = @intCast(@min(block, n - b * block));
                _ = std.os.linux.pread(s.fd, buf.ptr, len, @intCast(s.lo + b * block));
                break;
            }
        }
    }
};
