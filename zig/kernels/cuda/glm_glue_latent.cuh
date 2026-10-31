// latent.py's BF16 absorb, expand and cache write by hand (the MLA layers' 512-wide latent, 256-wide heads).
#pragma once

#include <cuda_fp8.h>

#include "glm_triton.cuh"

using namespace glm_triton;

// latent._absorb_q(_rows), block (head, 32 latent columns, row block) of 128 threads; rows per block change no bits.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_absorb_q(const __nv_bfloat16* q, const __nv_bfloat16* wk, __nv_bfloat16* qa, int64_t rows, int64_t heads,
                    int64_t rb) {
  constexpr int D = 256, LW = 512;
  __shared__ float smem[128];
  const int t = threadIdx.x, k0 = t >> 2, nq = t & 3, warp = t >> 5;
  const int64_t h = blockIdx.x, n0 = (int64_t)blockIdx.y * 32 + nq * 8, r0 = blockIdx.z * rb;
  float w[8][8];
  for (int rep = 0; rep < 8; ++rep)
    for (int j = 0; j < 8; ++j) w[rep][j] = bf(wk[(h * D + k0 + 32 * rep) * LW + n0 + j]);
  for (int64_t r = r0; r < r0 + rb && r < rows; ++r) {
    float qv[8];
    for (int rep = 0; rep < 8; ++rep) qv[rep] = bf(q[(r * heads + h) * D + k0 + 32 * rep]);
    for (int j = 0; j < 8; ++j) {
      float a = fmul(w[1][j], qv[1]);
      a = __fmaf_rn(w[0][j], qv[0], a);
      for (int rep = 2; rep < 8; ++rep) a = __fmaf_rn(w[rep][j], qv[rep], a);
      for (int o = 16; o >= 4; o >>= 1) a = a + __shfl_xor_sync(0xffffffffu, a, o);
      if ((t & 28) == 0) smem[(nq * 8 + j) * 4 + warp] = a;
    }
    __syncthreads();
    float v = smem[t];
    v = v + __shfl_xor_sync(0xffffffffu, v, 2);
    v = v + __shfl_xor_sync(0xffffffffu, v, 1);
    if ((t & 3) == 0) qa[(r * heads + h) * LW + (int64_t)blockIdx.y * 32 + t / 4] = __float2bfloat16_rn(v);
    __syncthreads();
  }
}

// latent._expand_v(_rows), block (head, 16 outputs, row block) of 128 threads: o . wv by fma, lanes xor 16 .. 1.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_expand_v(const __nv_bfloat16* ol, const __nv_bfloat16* wv, __nv_bfloat16* out, int64_t rows,
                    int64_t heads, int64_t rb) {
  constexpr int DV = 256, LW = 512;
  __shared__ float smem[32];
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5, kb = (warp & 1) * 256 + lane * 8, nb = warp >> 1;
  const int64_t h = blockIdx.x, n0 = (int64_t)blockIdx.y * 16, r0 = blockIdx.z * rb;
  float w[8][8];
  for (int rep = 0; rep < 8; ++rep)
    for (int j = 0; j < 8; ++j) w[rep][j] = bf(wv[(h * DV + n0 + nb + 2 * rep) * LW + kb + j]);
  for (int64_t r = r0; r < r0 + rb && r < rows; ++r) {
    float o[8];
    for (int j = 0; j < 8; ++j) o[j] = bf(ol[(r * heads + h) * LW + kb + j]);
    for (int rep = 0; rep < 8; ++rep) {
      float a = fmul(w[rep][1], o[1]);
      a = __fmaf_rn(w[rep][0], o[0], a);
      for (int j = 2; j < 8; ++j) a = __fmaf_rn(w[rep][j], o[j], a);
      for (int s = 16; s > 0; s >>= 1) a = a + __shfl_xor_sync(0xffffffffu, a, s);
      if (lane == 0) smem[(nb + 2 * rep) * 2 + (warp & 1)] = a;
    }
    __syncthreads();
    if (t < 32) {
      float v = smem[t];
      v = v + __shfl_xor_sync(0xffffffffu, v, 1);
      if ((t & 1) == 0) out[(r * heads + h) * DV + n0 + t / 2] = __float2bfloat16_rn(v);
    }
    __syncthreads();
  }
}

namespace glm_pages {

// Row r's first element in a cache of `width`-element rows: in order (pt null), or row r % P of page pt[r / P].
template <int P>
__device__ __forceinline__ int64_t at(const int* pt, int64_t page_elems, int64_t r, int64_t width) {
  return pt ? (int64_t)pt[r / P] * page_elems + (r % P) * width : r * width;
}

}  // namespace glm_pages

// latent._lat_write (BF16 cache): row r of `lat` into cache slot pos + r (on pages, through the stream's table).
extern "C" __global__ void tf_glm_lat_write(const __nv_bfloat16* lat, int64_t lat_stride, __nv_bfloat16* cache,
                                            const int* pos, int64_t lw, const int* pt, int64_t ps) {
  const int64_t r = blockIdx.x, at = glm_pages::at<512>(pt, ps / 2, (int64_t)pos[0] + r, lw);
  for (int64_t k = threadIdx.x; k < lw; k += blockDim.x) cache[at + k] = lat[r * lat_stride + k];
}

// The fp8 latent cache's write (--kv-cache fp8), 128 threads of 4 values (lw 512): e4m3 codes of x / s, s = amax / 448.
// Slot p's codes and scale: row p of the window, or row p % 512 of page pt[p / 512] (codes ps bytes, scales ps / 4 apart).
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_lat_write8(const __nv_bfloat16* lat, int64_t lat_stride, uint8_t* cache, float* scales, const int* pos,
                      const int* pt, int64_t ps) {
  constexpr int LW = 512;
  __shared__ float wmax[4];
  const int t = threadIdx.x;
  const int64_t r = blockIdx.x, p = (int64_t)pos[0] + r;
  const int64_t at = glm_pages::at<512>(pt, ps, p, LW), sat = glm_pages::at<512>(pt, ps / 4, p, 1);
  const uint2 raw = *reinterpret_cast<const uint2*>(lat + r * lat_stride + 4 * t);
  const float2 lo = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.x));
  const float2 hi = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.y));
  float amax = fmaxf(fmaxf(fabsf(lo.x), fabsf(lo.y)), fmaxf(fabsf(hi.x), fabsf(hi.y)));
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  if ((t & 31) == 0) wmax[t >> 5] = amax;
  __syncthreads();
  amax = fmaxf(fmaxf(wmax[0], wmax[2]), fmaxf(wmax[1], wmax[3]));
  const float s = amax > 0.0f ? glm_triton::div_full(amax, 448.0f) : 1.0f;
  if (t == 0) scales[sat] = s;
  // e4m3x2 round-to-nearest, saturating: the cvt instruction from sm_89, cuda_fp8.h's same rounding before it
  const unsigned c0 = __nv_cvt_float2_to_fp8x2(make_float2(glm_triton::div_full(lo.x, s), glm_triton::div_full(lo.y, s)), __NV_SATFINITE, __NV_E4M3);
  const unsigned c1 = __nv_cvt_float2_to_fp8x2(make_float2(glm_triton::div_full(hi.x, s), glm_triton::div_full(hi.y, s)), __NV_SATFINITE, __NV_E4M3);
  *reinterpret_cast<unsigned*>(cache + at + 4 * t) = c0 | (c1 << 16);
}
