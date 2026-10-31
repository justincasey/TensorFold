// EXL3 prompt building blocks: tiles decoded from shared memory, cp.async stages, ldmatrix (exl3_prompt.cuh's).
#pragma once
#include "exl3_decode.cuh"
namespace tf_exl3_prompt {
using namespace tf_exl3;
// One tile's lane values at a compile-time width from shared memory (the tile's 4 K2 words from `tile` on).
template <int K2, int CB>
__device__ __forceinline__ void decode_tile_k(const uint32_t* tile, int lane, uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    uint32_t w[lane_words<K2>()];
    int word, offset;
    lane_start<K2>(lane, word, offset);
#pragma unroll
    for (int i = 0; i < lane_words<K2>(); ++i) w[i] = tile[(word + i) % tile_words<K2>()];
    decode_lane<K2, CB>(w, lane, b0, b1);
}
// A warp's N tiles (`stride` words apart) at its warp-uniform width k2 (2..16) as mma A fragments.
template <int CB, int N>
__device__ __forceinline__ void decode_tiles_p(const uint32_t* tile, int stride, int k2, int lane, uint32_t (&a)[N][4]) {
    uint32_t b0[N][2], b1[N][2];
    switch (k2) {
#define TF_EXL3P_CASE(K2_)                                                                \
    case K2_:                                                                             \
        _Pragma("unroll") for (int i = 0; i < N; ++i) decode_tile_k<K2_, CB>(tile + i * stride, lane, b0[i], b1[i]); \
        break;
        TF_EXL3P_CASE(2) TF_EXL3P_CASE(3) TF_EXL3P_CASE(4) TF_EXL3P_CASE(5) TF_EXL3P_CASE(6) TF_EXL3P_CASE(7)
        TF_EXL3P_CASE(8) TF_EXL3P_CASE(9) TF_EXL3P_CASE(10) TF_EXL3P_CASE(11) TF_EXL3P_CASE(12) TF_EXL3P_CASE(13)
        TF_EXL3P_CASE(14) TF_EXL3P_CASE(15) TF_EXL3P_CASE(16)
#undef TF_EXL3P_CASE
    default:
        for (int i = 0; i < N; ++i) b0[i][0] = b0[i][1] = b1[i][0] = b1[i][1] = 0u;
    }
#pragma unroll
    for (int i = 0; i < N; ++i) {
        a[i][0] = b0[i][0]; a[i][1] = b1[i][0]; a[i][2] = b0[i][1]; a[i][3] = b1[i][1];
    }
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool valid) {
    const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    const int n = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(n));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&a)[4], const half* p) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(s));
}
}
