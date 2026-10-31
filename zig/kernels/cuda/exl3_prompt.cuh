// Common EXL3 prompt helpers, after @BobClawblaw (#976) and MiaAI-Lab (Apache-2.0, patches 0004 and 0009).
#pragma once
#include "exl3_stage.cuh"
namespace tf_exl3_prompt {

constexpr int TPM = 64, TNW = 8, TMT = 16 / TNW, TNB = TPM / 8, TT = TNW * 32, TPH = TPM / 2;
constexpr int TBK = 32, TKT = TBK / 16, TSTAGES = 3, TXS = TBK + 8;
constexpr int GU_ES = 128 + 4, DN_ES = 256 + 4;                        // the epilogue sums' row strides (floats)
// A stage's halves: XR rows of TXS, then 16 TKT weight tiles of ws words (4 times the layer's widest k2).
__device__ __forceinline__ int tstage_halves(int XR, int ws) { return XR * TXS + 2 * 16 * TKT * ws; }

// Which (matrix, weight tile) warp-tile q (0..15) of column block cb is.
template <int MATS>
__device__ __forceinline__ void ttile(int q, int cb, int& mat, int& nt) {
    mat = MATS == 2 ? q >> 3 : 0;
    nt = MATS == 2 ? cb * 8 + (q & 7) : cb * 16 + q;
}

// Main loop: acc[i][nb] = warp tile i x members nb * 8.. over K at widths k2a, k2b (2..16); NB 16-member blocks in use.
template <int MATS, int XR, int CB, int NB>
__device__ __forceinline__ void tloop(const half* __restrict__ X0, const half* __restrict__ X1, const uint64_t* __restrict__ tp0,
        const uint64_t* __restrict__ tp1, int k2a, int k2b, int kmax, const int* rows_sh, int e, int K, int NTILES,
        int cb, int xdiv, half* stage, float (&acc)[TMT][TNB][4]) {
    const int ws = 4 * kmax, SH = tstage_halves(XR, ws);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int mat, nt_unused;
    ttile<MATS>(warp * TMT, cb, mat, nt_unused);
    const int k2w = mat ? k2b : k2a;   // this warp's tiles' width
    const int chunks = K / TBK;
#pragma unroll
    for (int i = 0; i < TMT; ++i)
#pragma unroll
        for (int j = 0; j < TNB; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
    constexpr int XP = XR * (TBK / 8) / TT, WP = 16 * TKT * 16 / TT;   // this thread's x and weight pieces a stage
    static_assert(XP * TT == XR * (TBK / 8) && WP * TT == 16 * TKT * 16, "a stage's pieces split evenly over the threads");
    const half* xsrc[XP];
    int xdst[XP];
    bool xon[XP], xval[XP];
#pragma unroll
    for (int p = 0; p < XP; ++p) {
        const int idx = threadIdx.x + p * TT, m = idx / (TPM * (TBK / 8)), rem = idx % (TPM * (TBK / 8));
        const int i = rem / (TBK / 8), j = rem % (TBK / 8);
        xon[p] = i < NB * 16;   // a 16-member block without a member: never read
        const int r = xon[p] ? rows_sh[i] : -1;
        xval[p] = r >= 0;
        xsrc[p] = (m ? X1 : X0) + (size_t)(r < 0 ? 0 : (r / xdiv)) * K + j * 8;
        xdst[p] = (m * TPM + i) * TXS + j * 8;
    }
    const uint32_t* wsrc[WP];
    int wdst[WP], wstep[WP];
    bool won[WP];
#pragma unroll
    for (int p = 0; p < WP; ++p) {   // a tile's 16-byte pieces: k2 of at most 16
        const int idx = threadIdx.x + p * TT, q = idx / (TKT * 16), kk = (idx / 16) % TKT, part = idx % 16;
        int wm, nt;
        ttile<MATS>(q, cb, wm, nt);
        const int k2 = wm ? k2b : k2a;
        won[p] = part < k2;
        wsrc[p] = reinterpret_cast<const uint32_t*>((wm ? tp1 : tp0)[e]) + ((size_t)kk * NTILES + nt) * (4 * k2) + part * 4;
        wstep[p] = TKT * NTILES * 4 * k2;
        wdst[p] = XR * TXS / 2 + (q * TKT + kk) * ws + part * 4;
    }
    auto load = [&](int chunk, half* dst) {
#pragma unroll
        for (int p = 0; p < XP; ++p)
            if (xon[p]) cp_async16(dst + xdst[p], xsrc[p] + chunk * TBK, xval[p]);
#pragma unroll
        for (int p = 0; p < WP; ++p)
            if (won[p]) cp_async16(reinterpret_cast<uint32_t*>(dst) + wdst[p], wsrc[p] + (size_t)chunk * wstep[p], true);
    };
#pragma unroll
    for (int c = 0; c < TSTAGES - 1; ++c) {
        if (c < chunks) load(c, stage + c * SH);
        cp_async_commit();
    }
    int cs = 0, ls = TSTAGES - 1;   // the stages chunk c and chunk c + TSTAGES - 1 use
    for (int c = 0; c < chunks; ++c) {
        cp_async_wait<TSTAGES - 2>();
        __syncthreads();
        if (c + TSTAGES - 1 < chunks) load(c + TSTAGES - 1, stage + ls * SH);
        cp_async_commit();
        const half* base = stage + cs * SH;
        cs = cs == TSTAGES - 1 ? 0 : cs + 1;
        ls = ls == TSTAGES - 1 ? 0 : ls + 1;
        const half* xs = base + (size_t)(XR == TPM ? 0 : mat) * TPM * TXS;
        const uint32_t* wsh = reinterpret_cast<const uint32_t*>(base + (size_t)XR * TXS);
        uint32_t a[TMT * TKT][4];   // tile i's kk at i * TKT + kk: the warp's tiles are ws words apart
        decode_tiles_p<CB, TMT * TKT>(wsh + (size_t)warp * TMT * TKT * ws, ws, k2w, lane, a);
#pragma unroll
        for (int kk = 0; kk < TKT; ++kk) {
#pragma unroll
            for (int jb = 0; jb < NB; ++jb) {
                uint32_t r[4];
                const int row = jb * 16 + (lane & 7) + ((lane >> 4) & 1) * 8;
                ldmatrix_x4(r, xs + (size_t)row * TXS + kk * 16 + ((lane >> 3) & 1) * 8);
                const uint32_t bl[2] = {r[0], r[1]}, bh[2] = {r[2], r[3]};
#pragma unroll
                for (int i = 0; i < TMT; ++i) {
                    mma16816(acc[i][2 * jb], a[i * TKT + kk], bl);
                    mma16816(acc[i][2 * jb + 1], a[i * TKT + kk], bh);
                }
            }
        }
    }
    cp_async_wait<0>();
    __syncthreads();   // the stages become the epilogue's sums
}

// tloop at the item's 16-member block count (cnt of 1..TPM members).
template <int MATS, int XR, int CB>
__device__ __forceinline__ void tmain(const half* __restrict__ X0, const half* __restrict__ X1, const uint64_t* __restrict__ tp0,
        const uint64_t* __restrict__ tp1, int k2a, int k2b, int kmax, const int* rows_sh, int cnt, int e, int K, int NTILES,
        int cb, int xdiv, half* stage, float (&acc)[TMT][TNB][4]) {
    switch ((cnt + 15) >> 4) {
#define TF_TLOOP(NB_) tloop<MATS, XR, CB, NB_>(X0, X1, tp0, tp1, k2a, k2b, kmax, rows_sh, e, K, NTILES, cb, xdiv, stage, acc)
    case 1: TF_TLOOP(1); break;
    case 2: TF_TLOOP(2); break;
    case 3: TF_TLOOP(3); break;
    default: TF_TLOOP(4);
#undef TF_TLOOP
    }
}

// acc (members TPH * hf .. + TPH - 1) -> E[member - TPH * hf][col] fp32, row stride ES; col0: first tile's column.
__device__ __forceinline__ void tspill(float* E, int ES, int col0, const float (&acc)[TMT][TNB][4], int hf) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int i = 0; i < TMT; ++i)
#pragma unroll
        for (int q = 0; q < TPM / 16; ++q) {
            const int nb = hf * (TPM / 16) + q;
            const int tok = q * 8 + 2 * t, col = col0 + i * 16 + g;
            E[(size_t)tok * ES + col] = acc[i][nb][0];
            E[(size_t)(tok + 1) * ES + col] = acc[i][nb][1];
            E[(size_t)tok * ES + col + 8] = acc[i][nb][2];
            E[(size_t)(tok + 1) * ES + col + 8] = acc[i][nb][3];
        }
}

// Block (128 expert-width columns, item): xd[pair][:] = the rotated SwiGLU of the item's pairs; kmax the widest k2.
template <int XR, int CB>
__device__ __forceinline__ void tgateup(const half* __restrict__ Xg, const half* __restrict__ Xu, const uint64_t* __restrict__ Tg,
        const uint64_t* __restrict__ Tu, const int* __restrict__ gk2, const int* __restrict__ uk2, const int* __restrict__ items,
        const int* __restrict__ counts, const int* __restrict__ members, const half* __restrict__ svh_g, const half* __restrict__ svh_u,
        const half* __restrict__ suh_d, half* __restrict__ xd, int K, int N, int xdiv, float limit, int act_mode, int kmax,
        unsigned char* smem_raw, int* rows_sh) {
    const int cb = blockIdx.x, item = blockIdx.y;
    if (item >= counts[0]) return;
    const int e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    if (threadIdx.x < TPM) rows_sh[threadIdx.x] = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float acc[TMT][TNB][4];
    tmain<2, XR, CB>(Xg, Xu, Tg, Tu, gk2[e], uk2[e], kmax, rows_sh, cnt, e, K, N >> 4, cb, xdiv, reinterpret_cast<half*>(smem_raw), acc);
    float* Eg = reinterpret_cast<float*>(smem_raw);
    int mat, nt;
    ttile<2>(warp * TMT, cb, mat, nt);
    const int col0 = (nt - cb * 8) * 16;
    const int n = cb * 128 + 4 * lane;
    for (int hf = 0; hf < 2 && hf * TPH < cnt; ++hf) {
        if (hf) __syncthreads();   // the first half's rows are read
        tspill(Eg + (size_t)mat * TPH * GU_ES, GU_ES, col0, acc, hf);
        __syncthreads();
        for (int i = warp; i < TPH && hf * TPH + i < cnt; i += TNW) {
            const int p = rows_sh[hf * TPH + i];
            float gv[4], uv[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                gv[j] = Eg[(size_t)i * GU_ES + 4 * lane + j];
                uv[j] = Eg[(size_t)(TPH + i) * GU_ES + 4 * lane + j];
            }
            fwht128(gv, lane);
            fwht128(uv, lane);
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float gg = gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j]);
                float uu = uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j]);
                if (act_mode == 0) { gg = bf16r(gg); uu = bf16r(uu); }
                if (limit > 0) { gg = upTo(gg, limit); uu = within(uu, limit); }
                float act = gg / (1.f + expf(-gg));
                if (act_mode == 0) act = bf16r(act);
                act *= uu;
                if (act_mode == 0) act = bf16r(act);
                v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
            }
            fwht128(v, lane);
            __align__(8) half o[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
            *reinterpret_cast<uint2*>(xd + (size_t)p * N + n) = *reinterpret_cast<const uint2*>(o);
        }
    }
}

// Block (256 model-width columns, item): y[pair / ps][slot pair % ps of ys][:] = its pairs' down epilogue, bf16 (rne).
template <int CB>
__device__ __forceinline__ void tdown(const half* __restrict__ Xd, const uint64_t* __restrict__ Td, const int* __restrict__ dk2,
        const int* __restrict__ items, const int* __restrict__ counts, const int* __restrict__ members, const half* __restrict__ svh_d,
        __nv_bfloat16* __restrict__ y, int K, int D, int ps, int ys, int kmax, unsigned char* smem_raw, int* rows_sh) {
    const int cb = blockIdx.x, item = blockIdx.y;
    if (item >= counts[0]) return;
    const int e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    if (threadIdx.x < TPM) rows_sh[threadIdx.x] = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float acc[TMT][TNB][4];
    tmain<1, TPM, CB>(Xd, Xd, Td, Td, dk2[e], dk2[e], kmax, rows_sh, cnt, e, K, D >> 4, cb, 1, reinterpret_cast<half*>(smem_raw), acc);
    float* Ed = reinterpret_cast<float*>(smem_raw);
    for (int hf = 0; hf < 2 && hf * TPH < cnt; ++hf) {
        if (hf) __syncthreads();
        tspill(Ed, DN_ES, warp * TMT * 16, acc, hf);
        __syncthreads();
        for (int i = warp; i < TPH && hf * TPH + i < cnt; i += TNW) {
            const int p = rows_sh[hf * TPH + i];
#pragma unroll
            for (int blk = 0; blk < 2; ++blk) {
                const int n = cb * 256 + blk * 128 + 4 * lane;
                float v[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) v[j] = Ed[(size_t)i * DN_ES + blk * 128 + 4 * lane + j];
                fwht128(v, lane);
                __align__(8) __nv_bfloat16 o[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) o[j] = __float2bfloat16_rn(v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * D + n + j]));
                *reinterpret_cast<uint2*>(y + ((size_t)(p / ps) * ys + p % ps) * D + n) = *reinterpret_cast<const uint2*>(o);
            }
        }
    }
}

}  // namespace tf_exl3_prompt
