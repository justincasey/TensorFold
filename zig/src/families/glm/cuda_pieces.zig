//! A prompt chunk's pieced exchanges (the Python family's overlap.Overlap): a second stream and each piece's events.
const cuda = @import("cuda");

/// Pieces' events at most: a chunk's (overlap.PIECES_MAX, 16) or a batched pass's (each prompt's own pieces).
pub const max_pieces = 32;

/// The second stream and events; `count`: TF_GLM_OVERLAP_PIECES (3).
pub const Pieces = struct {
    count: usize = 4,
    side: cuda.Stream,
    filled: [max_pieces]cuda.Event,
    gathered: [max_pieces]cuda.Event,
    glues: [max_pieces]cuda.Event, // split: a piece's own rows glued
    shared: [max_pieces]cuda.Event, // split: every rank's glued rows of a piece in

    pub fn init(d: *const cuda.Driver, count: usize) !Pieces {
        if (count == 0 or count > max_pieces) return error.BadPieces;
        var p: Pieces = .{ .count = count, .side = try cuda.Stream.init(d, false), .filled = undefined, .gathered = undefined, .glues = undefined, .shared = undefined };
        for (&p.filled, &p.gathered, &p.glues, &p.shared) |*a, *b, *g, *s| {
            a.* = try cuda.Event.init(d, false);
            b.* = try cuda.Event.init(d, false);
            g.* = try cuda.Event.init(d, false);
            s.* = try cuda.Event.init(d, false);
        }
        return p;
    }

    pub fn deinit(p: *Pieces) void {
        for (&p.filled, &p.gathered, &p.glues, &p.shared) |*a, *b, *g, *s| {
            a.deinit();
            b.deinit();
            g.deinit();
            s.deinit();
        }
        p.side.deinit();
        p.* = undefined;
    }
};
