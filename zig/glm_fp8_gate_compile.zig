//! Force compilation of the latent attention device checks (bf16 and fp8, windows and pages) without running GPU code.
comptime {
    _ = &@import("gate").dense;
    _ = &@import("gate").sparse;
}
