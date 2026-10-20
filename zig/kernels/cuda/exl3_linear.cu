// EXL3 decode-row linear: any codebook, width, 1-128 rows, a row's bits only its own; port of tensorfold's linear.cu.
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "exl3_decode.cuh"
#include "exl3_stage.cuh"

namespace tf_exl3 {

enum DType : int { F16 = 0, BF16 = 1, F32 = 2 };

__device__ __forceinline__ int imin(int a, int b) { return a < b ? a : b; }

struct LinearArgs {
    const void* xh;          // fp16 [m, k], the hadamard-rotated inputs
    const uint32_t* t;       // trellis words
    long long stride_k;      // words between k tiles
    long long stride_nb;     // words between 128-column blocks
    const half* svh;         // fp16 [n], output hadamard scales
    const half* bias;        // fp16 [n] or null
    void* y;                 // output
    float* z;                // fp32 split partials [sk, m, n] or null
    int* counters;           // per (pass, block) arrival counts, zeroed before the launch
    int y_dtype;
    int n, sk, m, k;
};

struct RotInArgs {
    const void* x;           // f16 / bf16 / f32 [m, k]
    const half* suh;         // fp16 [k], input hadamard scales
    half* xh;                // fp16 [m, k] out
    int x_dtype;
    int m, k;
};

// Prompt rows: x's rows `x_stride` apart (bf16) rotated as tf_exl3_rot_in rotates them into contiguous xh.
struct RotRowsArgs {
    const __nv_bfloat16* x;
    long long x_stride;
    const half* suh;
    half* xh;
    int m, k;
};

struct DenseArgs {
    const half* xh;          // fp16 [m, k], the hadamard-rotated inputs
    const uint32_t* t;       // trellis words
    long long stride_k;      // words between k tiles
    long long stride_nb;     // words between 128-column blocks
    const half* svh;         // fp16 [n], output hadamard scales
    const half* bias;        // fp16 [n] or null
    void* y;                 // output rows, y_stride elements apart
    long long y_stride;
    int y_dtype;
    int n, m, k;
    int group;               // row blocks a column block sweeps before the next column block starts
};

__device__ __forceinline__ void load4(const void* p, int dtype, size_t i, float (&v)[4]) {
    if (dtype == F32) {
        const float4 u = *reinterpret_cast<const float4*>(static_cast<const float*>(p) + i);
        v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
    } else if (dtype == BF16) {
        const uint2 u = *reinterpret_cast<const uint2*>(static_cast<const __nv_bfloat16*>(p) + i);
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
        const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
        v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
    } else {
        const uint2 u = *reinterpret_cast<const uint2*>(static_cast<const half*>(p) + i);
        const float2 a = __half22float2(*reinterpret_cast<const half2*>(&u.x));
        const float2 b = __half22float2(*reinterpret_cast<const half2*>(&u.y));
        v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
    }
}

__device__ __forceinline__ void store4(void* p, int dtype, size_t i, const float (&v)[4]) {
    if (dtype == F32) {
        *reinterpret_cast<float4*>(static_cast<float*>(p) + i) = make_float4(v[0], v[1], v[2], v[3]);
    } else if (dtype == BF16) {
        __nv_bfloat162 a = __floats2bfloat162_rn(v[0], v[1]), b = __floats2bfloat162_rn(v[2], v[3]);
        uint2 u;
        u.x = *reinterpret_cast<uint32_t*>(&a);
        u.y = *reinterpret_cast<uint32_t*>(&b);
        *reinterpret_cast<uint2*>(static_cast<__nv_bfloat16*>(p) + i) = u;
    } else {
        half2 a = __floats2half2_rn(v[0], v[1]), b = __floats2half2_rn(v[2], v[3]);
        uint2 u;
        u.x = *reinterpret_cast<uint32_t*>(&a);
        u.y = *reinterpret_cast<uint32_t*>(&b);
        *reinterpret_cast<uint2*>(static_cast<half*>(p) + i) = u;
    }
}

// The finished outputs of one row's 128 columns from their fp32 sums (4 a lane): H / sqrt(128), * svh, + bias.
__device__ __forceinline__ void finish(float (&v)[4], int lane, const half* svh, const half* bias, int col) {
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j] = v[j] * HAD_SCALE * __half2float(__ldg(svh + col + j));
        // Speculative loads must have a valid address even when the split-K result has no bias.
        const half bv = __ldg((bias ? bias : svh) + col + j);
        if (bias) v[j] += __half2float(bv);
    }
}

// A k step's words for one lane: the step is 8 tiles back to back, each lane takes the same slice of every tile.
template <int K2>
__device__ __forceinline__ void load_step(const uint32_t* step, int lane, uint32_t (&raw)[8 * lane_words<K2>()]) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    int word, offset;
    lane_start<K2>(lane, word, offset);
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int q = 0; q < LW; ++q) raw[j * LW + q] = __ldg(step + j * TW + (word + q) % TW);
}

// 16 bytes of trellis words, read-only and not kept in L1.
__device__ __forceinline__ uint4 ldg_nc_v4(const uint32_t* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

// Tile j's lane words from a loaded step.
template <int K2>
__device__ __forceinline__ void step_lane_words(const uint32_t (&raw)[8 * lane_words<K2>()], int j,
                                                uint32_t (&w)[lane_words<K2>()]) {
#pragma unroll
    for (int q = 0; q < lane_words<K2>(); ++q) w[q] = raw[j * lane_words<K2>() + q];
}

// xh = fp16(((x * suh) @ H) / sqrt(128)): one warp per 128 columns, one block of 4 warps per 512.
extern "C" __global__ void __launch_bounds__(128) tf_exl3_rot_in(RotInArgs a) {
    const int blk = blockIdx.x * 4 + (threadIdx.x >> 5), row = blockIdx.y, lane = threadIdx.x & 31;
    if (blk * 128 >= a.k) return;
    const int k = blk * 128 + 4 * lane;
    float v[4], s[4];
    load4(a.x, a.x_dtype, (size_t)row * a.k + k, v);
    load4(a.suh, F16, k, s);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= s[j];
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= HAD_SCALE;
    store4(a.xh, F16, (size_t)row * a.k + k, v);
}

extern "C" __global__ void __launch_bounds__(128) tf_exl3_rot_rows(RotRowsArgs a) {
    const int blk = blockIdx.x * 4 + (threadIdx.x >> 5), row = blockIdx.y, lane = threadIdx.x & 31;
    if (blk * 128 >= a.k) return;
    const int k = blk * 128 + 4 * lane;
    float v[4], s[4];
    load4(a.x + (size_t)row * a.x_stride, BF16, k, v);
    load4(a.suh, F16, k, s);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= s[j];
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= HAD_SCALE;
    store4(a.xh, F16, (size_t)row * a.k + k, v);
}

constexpr int DBK = 32, DKT = DBK / 16, DXS = DBK + 8, DSTAGES = 3, DES = 132;   // dense stages; DES: sums' row stride

// Prompt rows (mul1): warp j sums column tile j over all K in k order, each tile decoded once per BM rows.
template <int K2, int BM>
__device__ __forceinline__ void dense_body(const DenseArgs& a) {
    using namespace tf_exl3_prompt;
    constexpr int TW = tile_words<K2>(), MT = BM / 16;
    constexpr int XB = BM * DXS * 2, SB = XB + DKT * 8 * TW * 4;   // a stage's x bytes, then its weight tiles
    extern __shared__ __align__(16) unsigned char dsm[];
    const int NB = a.n >> 7, RB = (a.m + BM - 1) / BM, K = a.k;
    const int span = a.group * NB, first = (int)blockIdx.x / span * a.group, inner = (int)blockIdx.x % span;
    const int rows = imin(a.group, RB - first);
    const int rb = first + inner % rows, nb = inner / rows;
    const int row0 = rb * BM, MB = imin(BM, a.m - row0);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const half* xh = a.xh + (size_t)row0 * K;
    const uint32_t* T = a.t + nb * a.stride_nb;
    const int chunks = K / DBK;
    auto load = [&](int c) {
        unsigned char* st = dsm + (c % DSTAGES) * SB;
        half* xs = reinterpret_cast<half*>(st);
        for (int i = threadIdx.x; i < BM * (DBK / 8); i += 256) {
            const int r = i / (DBK / 8), j = i % (DBK / 8);
            cp_async16(xs + r * DXS + j * 8, xh + (size_t)(r < MB ? r : 0) * K + c * DBK + j * 8, r < MB);
        }
        uint32_t* ws = reinterpret_cast<uint32_t*>(st + XB);
        for (int i = threadIdx.x; i < DKT * 2 * TW; i += 256) {   // a k tile's 8 tiles: 8 TW words in 2 TW pieces
            const int kk = i / (2 * TW), p = i % (2 * TW);
            cp_async16(ws + kk * 8 * TW + p * 4, T + (size_t)(c * DKT + kk) * a.stride_k + p * 4, true);
        }
    };
    float acc[MT][2][4];
#pragma unroll
    for (int q = 0; q < MT; ++q)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[q][h][e] = 0.f;
#pragma unroll
    for (int c = 0; c < DSTAGES - 1; ++c) {
        if (c < chunks) load(c);
        cp_async_commit();
    }
#pragma unroll 1
    for (int c = 0; c < chunks; ++c) {
        cp_async_wait<DSTAGES - 2>();
        __syncthreads();
        if (c + DSTAGES - 1 < chunks) load(c + DSTAGES - 1);
        cp_async_commit();
        const unsigned char* st = dsm + (c % DSTAGES) * SB;
        const half* xs = reinterpret_cast<const half*>(st);
        const uint32_t* ws = reinterpret_cast<const uint32_t*>(st + XB);
#pragma unroll
        for (int kk = 0; kk < DKT; ++kk) {
            uint32_t b0[2], b1[2];
            decode_tile_k<K2, CB_MUL1>(ws + (kk * 8 + warp) * TW, lane, b0, b1);
#pragma unroll
            for (int q = 0; q < MT; ++q) {
                if (q * 16 < MB) {
                    uint32_t u[4];
                    ldmatrix_x4(u, xs + (q * 16 + (lane & 15)) * DXS + kk * 16 + (lane >> 4) * 8);
                    mma16816(acc[q][0], u, b0);
                    mma16816(acc[q][1], u, b1);
                }
            }
        }
    }
    cp_async_wait<0>();
    __syncthreads();   // the stages become the epilogue's sums
    float* E = reinterpret_cast<float*>(dsm);
    const int col0 = nb * 128;
#pragma unroll
    for (int hf = 0; hf < BM / 64; ++hf) {
        if (hf * 64 < MB) {
            if (hf) __syncthreads();   // the previous half's rows are read
#pragma unroll
            for (int q = 0; q < 4; ++q)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    float* p = E + (q * 16 + g) * DES + warp * 16 + h * 8 + 2 * t;
                    *reinterpret_cast<float2*>(p) = make_float2(acc[hf * 4 + q][h][0], acc[hf * 4 + q][h][1]);
                    *reinterpret_cast<float2*>(p + 8 * DES) = make_float2(acc[hf * 4 + q][h][2], acc[hf * 4 + q][h][3]);
                }
            __syncthreads();
            for (int r = warp; r < 64 && hf * 64 + r < MB; r += 8) {
                const float4 u = *reinterpret_cast<const float4*>(E + r * DES + 4 * lane);
                float v[4] = {u.x, u.y, u.z, u.w};
                finish(v, lane, a.svh, a.bias, col0 + 4 * lane);
                store4(a.y, a.y_dtype, (size_t)(row0 + hf * 64 + r) * a.y_stride + col0 + 4 * lane, v);
            }
        }
    }
}


// One program: (N / 128) * SK blocks, split-major like a (N / 128, SK) grid, WK warps each; MT 16-row m tiles a pass
// (each weight tile decoded once a pass; a row's mma, warp order and split order are the same at either MT).
template <int K2, int CB, int WK, int MT>
__device__ __forceinline__ void linear_body(LinearArgs a) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    constexpr bool ST = TW % 8 == 0;                          // even widths: staged 16-byte loads, PD steps in flight
    constexpr int PD = 2;
    constexpr bool PF = !ST && K2 <= 12;                      // odd widths: the next step's lane words in registers
    constexpr int SR = 8 * LW;
    extern __shared__ __align__(16) float red[];              // WK * RH * 128 floats; even widths stage steps here
    __shared__ int last;
    const int M = a.m, K = a.k, N = a.n, SK = a.sk, NB = N >> 7;
    const int RH = imin(M, 8);                                // rows of red a warp
    const int nb = (int)blockIdx.x % NB, split = (int)blockIdx.x / NB;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int per_warp = (K >> 4) / SK / WK;
    const int kt0 = split * (per_warp * WK) + warp * per_warp;
    const uint32_t* tiles = a.t + nb * a.stride_nb;
    const int col0 = nb * 128;

    for (int m0 = 0, pass = 0; m0 < M; m0 += 16 * MT, ++pass) {
        const int R = imin(16 * MT, M - m0);

        float acc[MT][8][2][4];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int c = 0; c < 4; ++c) acc[mt][i][h][c] = 0.f;

        // the walk's rows for the mma, clamped so rows past the pass read inside the buffer (their outputs are dropped)
        const half* x0[MT];
        const half* x1[MT];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const int g0 = 16 * mt + g, g1 = 16 * mt + g + 8;
            x0[mt] = static_cast<const half*>(a.xh) + (size_t)(m0 + (g0 < R ? g0 : R - 1)) * K;
            x1[mt] = static_cast<const half*>(a.xh) + (size_t)(m0 + (g1 < R ? g1 : R - 1)) * K;
        }
        const uint32_t* tile = tiles + (size_t)kt0 * a.stride_k;
        uint32_t cur[PF ? SR : 1], nxt[PF ? SR : 1];
        if constexpr (PF) load_step<K2>(tile, lane, cur);     // the weights do not wait for the previous kernel
        constexpr int NV = (8 * TW + 127) / 128;              // a staged step's 16-byte loads a lane
        uint4 rv[ST ? PD : 1][ST ? NV : 1];
        if constexpr (ST) {
#pragma unroll
            for (int d = 0; d < PD; ++d)
                if (d < per_warp)
#pragma unroll
                    for (int v = 0; v < NV; ++v)
                        if (128 * v + 4 * lane < 8 * TW) rv[d][v] = ldg_nc_v4(tile + d * a.stride_k + 128 * v + 4 * lane);
        }

        // x's fragments of k step kt: rows g (and g + 8) of each m tile: [0] [2] k 2t.., 2t + 8..; [1] [3] row g + 8's
        auto load_x = [&](int kt, uint32_t (&u)[MT][4]) {
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                u[mt][0] = __ldg(reinterpret_cast<const uint32_t*>(x0[mt] + kt * 16 + 2 * t));
                u[mt][2] = __ldg(reinterpret_cast<const uint32_t*>(x0[mt] + kt * 16 + 2 * t + 8));
                u[mt][1] = __ldg(reinterpret_cast<const uint32_t*>(x1[mt] + kt * 16 + 2 * t));
                u[mt][3] = __ldg(reinterpret_cast<const uint32_t*>(x1[mt] + kt * 16 + 2 * t + 8));
            }
        };
        if constexpr (ST) {
            uint32_t* stage = reinterpret_cast<uint32_t*>(red) + warp * 8 * 64;   // a warp's step: 8 tiles, 64 words most
            int word, offset;
            lane_start<K2>(lane, word, offset);
#pragma unroll 1
            for (int ib = 0; ib < per_warp; ib += PD) {
#pragma unroll
                for (int d = 0; d < PD; ++d) {
                    const int i = ib + d;
                    if (i >= per_warp) break;
                    __syncwarp();   // every lane has read the previous step back
#pragma unroll
                    for (int v = 0; v < NV; ++v)
                        if (128 * v + 4 * lane < 8 * TW) *reinterpret_cast<uint4*>(stage + 128 * v + 4 * lane) = rv[d][v];
                    if (i + PD < per_warp)
#pragma unroll
                        for (int v = 0; v < NV; ++v)
                            if (128 * v + 4 * lane < 8 * TW)
                                rv[d][v] = ldg_nc_v4(tile + (size_t)(i + PD) * a.stride_k + 128 * v + 4 * lane);
                    __syncwarp();   // the step is staged before any lane reads it
                    uint32_t u[MT][4];
                    load_x(kt0 + i, u);
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        uint32_t w[LW];
#pragma unroll
                        for (int q = 0; q < LW; ++q) w[q] = stage[j * TW + (word + q) % TW];
                        uint32_t b0[2], b1[2];
                        decode_lane<K2, CB>(w, lane, b0, b1);
#pragma unroll
                        for (int mt = 0; mt < MT; ++mt) {
                            mma16816(acc[mt][j][0], u[mt], b0);
                            mma16816(acc[mt][j][1], u[mt], b1);
                        }
                    }
                }
            }
        }
#pragma unroll 1
        for (int i = 0; i < (ST ? 0 : per_warp); ++i) {
            uint32_t u[MT][4];
            load_x(kt0 + i, u);
            if constexpr (PF) {
                if (i + 1 < per_warp) load_step<K2>(tile + (size_t)(i + 1) * a.stride_k, lane, nxt);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                uint32_t w[LW];
                if constexpr (PF) step_lane_words<K2>(cur, j, w);
                else ldg_lane_words<K2>(tile + (size_t)i * a.stride_k + j * TW, lane, w);
                uint32_t b0[2], b1[2];
                decode_lane<K2, CB>(w, lane, b0, b1);
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    mma16816(acc[mt][j][0], u[mt], b0);
                    mma16816(acc[mt][j][1], u[mt], b1);
                }
            }
            if constexpr (PF) {
#pragma unroll
                for (int q = 0; q < SR; ++q) cur[q] = nxt[q];
            }
        }

        // the warps' sums, added in warp order, 8 rows of the pass at a time (an m tile's rows 0-7, then 8-15)
#pragma unroll
        for (int q = 0; q < 2 * MT; ++q) {
            const int rlo = 8 * q;
            if (rlo >= R) break;
            const int rn = imin(R - rlo, 8);
            __syncthreads();                         // red is reused by every half and pass
            if (g < RH) {
#pragma unroll
                for (int i = 0; i < 8; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = i * 16 + h * 8 + 2 * t;
                        *reinterpret_cast<float2*>(red + (warp * RH + g) * 128 + col) =
                            q & 1 ? make_float2(acc[q >> 1][i][h][2], acc[q >> 1][i][h][3])
                                  : make_float2(acc[q >> 1][i][h][0], acc[q >> 1][i][h][1]);
                    }
            }
            __syncthreads();

            if (SK == 1) {
                for (int r = warp; r < rn; r += WK) {
                    float v[4];
                    const float4 u = *reinterpret_cast<const float4*>(red + r * 128 + 4 * lane);
                    v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
#pragma unroll
                    for (int w = 1; w < WK; ++w) {
                        const float4 q = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + 4 * lane);
                        v[0] += q.x; v[1] += q.y; v[2] += q.z; v[3] += q.w;
                    }
                    finish(v, lane, a.svh, a.bias, col0 + 4 * lane);
                    store4(a.y, a.y_dtype, (size_t)(m0 + rlo + r) * N + col0 + 4 * lane, v);
                }
            } else {
                for (int idx = threadIdx.x; idx < rn * 32; idx += WK * 32) {
                    const int r = idx >> 5, c = 4 * (idx & 31);
                    float4 s = *reinterpret_cast<const float4*>(red + r * 128 + c);
#pragma unroll
                    for (int w = 1; w < WK; ++w) {
                        const float4 q = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + c);
                        s.x += q.x; s.y += q.y; s.z += q.z; s.w += q.w;
                    }
                    *reinterpret_cast<float4*>(a.z + ((size_t)split * M + m0 + rlo + r) * N + col0 + c) = s;
                }
            }
        }
        if (SK > 1) {
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0) last = atomicAdd(a.counters + pass * NB + nb, 1) == SK - 1;
            __syncthreads();
            if (last) {
                __threadfence();
                for (int r = warp; r < R; r += WK) {
                    const size_t at = ((size_t)m0 + r) * N + col0 + 4 * lane;
                    float4 s = __ldcg(reinterpret_cast<const float4*>(a.z + at));
                    for (int q = 1; q < SK; ++q) {
                        const float4 u = __ldcg(reinterpret_cast<const float4*>(a.z + (size_t)q * M * N + at));
                        s.x += u.x; s.y += u.y; s.z += u.z; s.w += u.w;
                    }
                    float v[4] = {s.x, s.y, s.z, s.w};
                    finish(v, lane, a.svh, a.bias, col0 + 4 * lane);
                    store4(a.y, a.y_dtype, (size_t)(m0 + r) * N + col0 + 4 * lane, v);
                }
                if (threadIdx.x == 0) a.counters[pass * NB + nb] = 0;   // every program of the block has arrived
            }
        }
        __syncthreads();                             // red is reused in the next pass
    }
}

// One extern "C" kernel per instantiation: the fatbin carries __global__ functions, the host resolves them by name.
#define TF_EXL3_LINEAR(K2, CB, WK)                                                       \
    extern "C" __global__ void __launch_bounds__((WK) * 32)                              \
        tf_exl3_linear_##K2##_##CB##_##WK(tf_exl3::LinearArgs a) {                       \
        tf_exl3::linear_body<K2, CB, WK, 1>(a);                                          \
    }
// mul1's 32-row passes (two m tiles: each weight tile decoded once for 32 rows)
#define TF_EXL3_LINEAR32(K2, WK)                                                         \
    extern "C" __global__ void __launch_bounds__((WK) * 32)                              \
        tf_exl3_linear32_##K2##_##WK(tf_exl3::LinearArgs a) {                            \
        tf_exl3::linear_body<K2, 2, WK, 2>(a);                                           \
    }

// Every width the format serves (linear.cu TF_EXL3_ALL plus 4.5-6.5): whole bits, all codebooks; 7.5 is not a width.
#define TF_EXL3_WIDTHS(K2, CB) TF_EXL3_LINEAR(K2, CB, 4) TF_EXL3_LINEAR(K2, CB, 8)
#define TF_EXL3_CODEBOOK(CB)                                                                                \
    TF_EXL3_WIDTHS(2, CB) TF_EXL3_WIDTHS(4, CB) TF_EXL3_WIDTHS(6, CB) TF_EXL3_WIDTHS(8, CB)                 \
    TF_EXL3_WIDTHS(10, CB) TF_EXL3_WIDTHS(12, CB) TF_EXL3_WIDTHS(14, CB) TF_EXL3_WIDTHS(16, CB)
TF_EXL3_CODEBOOK(2) // mul1
TF_EXL3_WIDTHS(3, 2) TF_EXL3_WIDTHS(5, 2) TF_EXL3_WIDTHS(7, 2) TF_EXL3_WIDTHS(9, 2)
TF_EXL3_WIDTHS(11, 2) TF_EXL3_WIDTHS(13, 2)
TF_EXL3_CODEBOOK(1) // mcg
TF_EXL3_CODEBOOK(0) // inst3
#undef TF_EXL3_CODEBOOK
#undef TF_EXL3_WIDTHS
#define TF_EXL3_WIDTHS32(K2) TF_EXL3_LINEAR32(K2, 4) TF_EXL3_LINEAR32(K2, 8)
TF_EXL3_WIDTHS32(2) TF_EXL3_WIDTHS32(3) TF_EXL3_WIDTHS32(4) TF_EXL3_WIDTHS32(5) TF_EXL3_WIDTHS32(6) TF_EXL3_WIDTHS32(7)
TF_EXL3_WIDTHS32(8) TF_EXL3_WIDTHS32(9) TF_EXL3_WIDTHS32(10) TF_EXL3_WIDTHS32(11) TF_EXL3_WIDTHS32(12)
TF_EXL3_WIDTHS32(13) TF_EXL3_WIDTHS32(14) TF_EXL3_WIDTHS32(16)
#undef TF_EXL3_WIDTHS32

// The prompt kernels: mul1, every width the decode kernels serve but 15 halves, 64- and 128-row blocks of 8 warps.
#define TF_EXL3_DENSE(K2)                                                                                            \
    extern "C" __global__ void __launch_bounds__(256, 2) tf_exl3_dense_##K2##_64(tf_exl3::DenseArgs a) {             \
        tf_exl3::dense_body<K2, 64>(a);                                                                              \
    }                                                                                                                \
    extern "C" __global__ void __launch_bounds__(256, 2) tf_exl3_dense_##K2##_128(tf_exl3::DenseArgs a) {            \
        tf_exl3::dense_body<K2, 128>(a);                                                                             \
    }
TF_EXL3_DENSE(2) TF_EXL3_DENSE(3) TF_EXL3_DENSE(4) TF_EXL3_DENSE(5) TF_EXL3_DENSE(6) TF_EXL3_DENSE(7)
TF_EXL3_DENSE(8) TF_EXL3_DENSE(9) TF_EXL3_DENSE(10) TF_EXL3_DENSE(11) TF_EXL3_DENSE(12) TF_EXL3_DENSE(13)
TF_EXL3_DENSE(14) TF_EXL3_DENSE(16)
#undef TF_EXL3_DENSE
}  // namespace tf_exl3
