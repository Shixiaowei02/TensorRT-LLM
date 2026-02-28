/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "mhcKernels.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::mhc
{

// ===================================================================
// Kernel 1: big_fuse — one CTA per token, 256 threads (8 warps)
//
//  Phase 1a (warp 0, lanes 0-3): RMS norm + sigmoid → s_pre_mix, post_mix
//  ── __syncthreads ──
//  Phase 1b (warp 0, lanes 0-3) ‖ Phase 2 (warps 1-7) — overlapped
//    1b: parallel Sinkhorn (4 lanes, __shfl_xor col normalize) → comb_mix
//     2: stream residual × pre_mix → layer_input
// ===================================================================

__launch_bounds__(256) __global__ void mhcBigFuseKernel(float const* __restrict__ y_acc,
    float const* __restrict__ r_acc, __nv_bfloat16 const* __restrict__ residual, float const* __restrict__ hc_scale,
    float const* __restrict__ hc_base, float* __restrict__ post_mix, float* __restrict__ comb_mix,
    __nv_bfloat16* __restrict__ layer_input, int M, int K, int hidden_size, float rms_eps, float hc_pre_eps,
    float hc_sinkhorn_eps, float hc_post_mult_value, int sinkhorn_repeat)
{
    constexpr int n = 4;
    constexpr int n2 = 16;
    constexpr int hc_mult3 = 24;

    int const token = blockIdx.x;
    int const tid = threadIdx.x;
    int const warp_id = tid / 32;
    int const lane = tid % 32;

    __shared__ float s_pre_mix[n];

    float cm[n];

    if (warp_id == 0 && lane < n)
    {
        float const rstd = rsqrtf(r_acc[token] / static_cast<float>(K) + rms_eps);
        float const* y_row = y_acc + token * hc_mult3;
        float const s0 = hc_scale[0], s1 = hc_scale[1], s2 = hc_scale[2];

        float v = y_row[lane] * rstd * s0 + hc_base[lane];
        s_pre_mix[lane] = 1.0f / (1.0f + expf(-v)) + hc_pre_eps;

        v = y_row[n + lane] * rstd * s1 + hc_base[n + lane];
        post_mix[token * n + lane] = 1.0f / (1.0f + expf(-v)) * hc_post_mult_value;

#pragma unroll
        for (int k = 0; k < n; k++)
            cm[k] = y_row[2 * n + lane * n + k] * rstd * s2 + hc_base[2 * n + lane * n + k];
    }

    __syncthreads();

    if (warp_id == 0 && lane < n)
    {
        constexpr unsigned MASK = 0xf;

#pragma unroll
        for (int k = 0; k < n; k++)
            cm[k] = expf(cm[k]);
        float rs = cm[0] + cm[1] + cm[2] + cm[3];
#pragma unroll
        for (int k = 0; k < n; k++)
            cm[k] = cm[k] / rs + hc_sinkhorn_eps;

#pragma unroll
        for (int k = 0; k < n; k++)
        {
            float cs = cm[k];
            cs += __shfl_xor_sync(MASK, cs, 1);
            cs += __shfl_xor_sync(MASK, cs, 2);
            cm[k] /= (cs + hc_sinkhorn_eps);
        }

        for (int it = 1; it < sinkhorn_repeat; it++)
        {
            rs = cm[0] + cm[1] + cm[2] + cm[3] + hc_sinkhorn_eps;
#pragma unroll
            for (int k = 0; k < n; k++)
                cm[k] /= rs;

#pragma unroll
            for (int k = 0; k < n; k++)
            {
                float cs = cm[k];
                cs += __shfl_xor_sync(MASK, cs, 1);
                cs += __shfl_xor_sync(MASK, cs, 2);
                cm[k] /= (cs + hc_sinkhorn_eps);
            }
        }

        float* cm_out = comb_mix + token * n2;
#pragma unroll
        for (int k = 0; k < n; k++)
            cm_out[lane * n + k] = cm[k];
    }

    if (warp_id > 0)
    {
        float const pm0 = s_pre_mix[0], pm1 = s_pre_mix[1];
        float const pm2 = s_pre_mix[2], pm3 = s_pre_mix[3];

        __nv_bfloat16 const* rbase = residual + static_cast<long long>(token) * n * hidden_size;
        __nv_bfloat16* obase = layer_input + static_cast<long long>(token) * hidden_size;

        int const p2_tid = tid - 32;
        int const p2_threads = blockDim.x - 32;

        for (int h = p2_tid * 8; h < hidden_size; h += p2_threads * 8)
        {
            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
            float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

#pragma unroll
            for (int j = 0; j < n; j++)
            {
                float const pm = (j == 0) ? pm0 : (j == 1) ? pm1 : (j == 2) ? pm2 : pm3;

                uint4 raw = *reinterpret_cast<uint4 const*>(&rbase[j * hidden_size + h]);
                __nv_bfloat162 const* pairs = reinterpret_cast<__nv_bfloat162 const*>(&raw);

                float2 f0 = __bfloat1622float2(pairs[0]);
                float2 f1 = __bfloat1622float2(pairs[1]);
                float2 f2 = __bfloat1622float2(pairs[2]);
                float2 f3 = __bfloat1622float2(pairs[3]);

                acc0 += pm * f0.x;
                acc1 += pm * f0.y;
                acc2 += pm * f1.x;
                acc3 += pm * f1.y;
                acc4 += pm * f2.x;
                acc5 += pm * f2.y;
                acc6 += pm * f3.x;
                acc7 += pm * f3.y;
            }

            uint4 out_raw;
            __nv_bfloat162* opairs = reinterpret_cast<__nv_bfloat162*>(&out_raw);
            opairs[0] = __float22bfloat162_rn(make_float2(acc0, acc1));
            opairs[1] = __float22bfloat162_rn(make_float2(acc2, acc3));
            opairs[2] = __float22bfloat162_rn(make_float2(acc4, acc5));
            opairs[3] = __float22bfloat162_rn(make_float2(acc6, acc7));
            *reinterpret_cast<uint4*>(&obase[h]) = out_raw;
        }
    }
}

// ===================================================================
// Kernel 2: sqrsum — row-wise sum of squares
// ===================================================================

__global__ void mhcSqrsumKernel(__nv_bfloat16 const* __restrict__ X, float* __restrict__ R, int K)
{
    int const row = blockIdx.x;
    int const tid = threadIdx.x;
    float sum = 0.0f;

    __nv_bfloat16 const* row_ptr = X + static_cast<long long>(row) * K;
    for (int k = tid * 8; k < K; k += blockDim.x * 8)
    {
        uint4 raw = *reinterpret_cast<uint4 const*>(&row_ptr[k]);
        __nv_bfloat162 const* pairs = reinterpret_cast<__nv_bfloat162 const*>(&raw);

#pragma unroll
        for (int p = 0; p < 4; p++)
        {
            float2 f = __bfloat1622float2(pairs[p]);
            sum += f.x * f.x + f.y * f.y;
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, offset);

    __shared__ float warp_sums[32];
    int const warp_id = tid / 32;
    int const lane = tid % 32;

    if (lane == 0)
        warp_sums[warp_id] = sum;
    __syncthreads();

    if (tid == 0)
    {
        int const num_warps = blockDim.x / 32;
        float total = 0.0f;
        for (int w = 0; w < num_warps; w++)
            total += warp_sums[w];
        R[row] = total;
    }
}

// ===================================================================
// Kernel 3: gemm_sqrsum_fma — FP32 CUDA-core GEMM + sqrsum
//
//  Grid(M, num_k_blocks), Block(256).
//  N-tiled in passes of N_TILE=8 to limit register pressure.
//  Split-K with atomic reduction when num_k_blocks > 1.
// ===================================================================

__launch_bounds__(256) __global__
    void mhcGemmSqrsumFmaKernel(__nv_bfloat16 const* __restrict__ X, float const* __restrict__ W_T,
        float* __restrict__ Y, float* __restrict__ R, int M, int N, int K, int num_k_blocks, int k_chunk)
{
    constexpr int N_TILE = 8;
    constexpr int K_STEP = 1024;

    int const tid = threadIdx.x;
    int const warp_id = tid / 32;
    int const lane = tid % 32;

    int const row_idx = blockIdx.x;

    int const k_start = blockIdx.y * k_chunk;
    int k_end = k_start + k_chunk;
    if (k_end > K)
        k_end = K;

    if (k_start >= K)
        return;

    bool const single_k = (num_k_blocks == 1);
    __nv_bfloat16 const* x_row = X + static_cast<long long>(row_idx) * K;
    float* y_row = Y + static_cast<long long>(row_idx) * N;

    __shared__ float s_warp[8][9];

    float sqr = 0.0f;

    for (int n_base = 0; n_base < N; n_base += N_TILE)
    {
        float acc[N_TILE];
#pragma unroll
        for (int i = 0; i < N_TILE; i++)
            acc[i] = 0.0f;

        if (row_idx < M)
        {
            for (int k_base = k_start; k_base + K_STEP <= k_end; k_base += K_STEP)
            {
                int const my_k = k_base + tid * 4;

                unsigned xp0, xp1;
                asm volatile("ld.global.cs.v2.b32 {%0, %1}, [%2];" : "=r"(xp0), "=r"(xp1) : "l"(x_row + my_k));
                float xv0, xv1, xv2, xv3;
                asm volatile(
                    "{ .reg .b16 lo, hi;\n\t"
                    "  mov.b32 {lo, hi}, %4;\n\t"
                    "  cvt.f32.bf16 %0, lo;\n\t"
                    "  cvt.f32.bf16 %1, hi;\n\t"
                    "  mov.b32 {lo, hi}, %5;\n\t"
                    "  cvt.f32.bf16 %2, lo;\n\t"
                    "  cvt.f32.bf16 %3, hi; }"
                    : "=f"(xv0), "=f"(xv1), "=f"(xv2), "=f"(xv3)
                    : "r"(xp0), "r"(xp1));

                if (n_base == 0)
                {
                    sqr = fmaf(xv0, xv0, sqr);
                    sqr = fmaf(xv1, xv1, sqr);
                    sqr = fmaf(xv2, xv2, sqr);
                    sqr = fmaf(xv3, xv3, sqr);
                }

#pragma unroll
                for (int n = 0; n < N_TILE; n++)
                {
                    if (n_base + n < N)
                    {
                        float w0, w1, w2, w3;
                        asm volatile("ld.global.L1::evict_last.v4.f32 {%0, %1, %2, %3}, [%4];"
                                     : "=f"(w0), "=f"(w1), "=f"(w2), "=f"(w3)
                                     : "l"(W_T + static_cast<long long>(n_base + n) * K + my_k));
                        acc[n] = fmaf(xv0, w0, acc[n]);
                        acc[n] = fmaf(xv1, w1, acc[n]);
                        acc[n] = fmaf(xv2, w2, acc[n]);
                        acc[n] = fmaf(xv3, w3, acc[n]);
                    }
                }
            }

            {
                int const tail_start = k_end - ((k_end - k_start) % K_STEP);
                for (int kk = tail_start + tid; kk < k_end; kk += 256)
                {
                    float xv;
                    asm volatile(
                        "{ .reg .b16 tmp;\n\t"
                        "  ld.global.cs.b16 tmp, [%1];\n\t"
                        "  cvt.f32.bf16 %0, tmp; }"
                        : "=f"(xv)
                        : "l"(x_row + kk));
                    if (n_base == 0)
                        sqr = fmaf(xv, xv, sqr);
#pragma unroll
                    for (int n = 0; n < N_TILE; n++)
                    {
                        if (n_base + n < N)
                        {
                            float wv = W_T[static_cast<long long>(n_base + n) * K + kk];
                            acc[n] = fmaf(xv, wv, acc[n]);
                        }
                    }
                }
            }
        }

#pragma unroll
        for (int n = 0; n < N_TILE; n++)
        {
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                acc[n] += __shfl_xor_sync(0xffffffff, acc[n], offset);
        }
        if (n_base == 0)
        {
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                sqr += __shfl_xor_sync(0xffffffff, sqr, offset);
        }

        if (lane == 0)
        {
#pragma unroll
            for (int n = 0; n < N_TILE; n++)
                s_warp[warp_id][n] = acc[n];
            if (n_base == 0)
                s_warp[warp_id][N_TILE] = sqr;
        }
        __syncthreads();

        if (lane == 0 && row_idx < M && (n_base + warp_id) < N)
        {
            float val = s_warp[0][warp_id] + s_warp[1][warp_id] + s_warp[2][warp_id] + s_warp[3][warp_id]
                + s_warp[4][warp_id] + s_warp[5][warp_id] + s_warp[6][warp_id] + s_warp[7][warp_id];
            if (single_k)
                y_row[n_base + warp_id] = val;
            else
                asm volatile("red.global.add.f32 [%0], %1;" ::"l"(y_row + n_base + warp_id), "f"(val));

            if (n_base == 0 && warp_id == 0)
            {
                float sq = s_warp[0][N_TILE] + s_warp[1][N_TILE] + s_warp[2][N_TILE] + s_warp[3][N_TILE]
                    + s_warp[4][N_TILE] + s_warp[5][N_TILE] + s_warp[6][N_TILE] + s_warp[7][N_TILE];
                if (single_k)
                    R[row_idx] = sq;
                else
                    asm volatile("red.global.add.f32 [%0], %1;" ::"l"(R + row_idx), "f"(sq));
            }
        }
        __syncthreads();
    }
}

// ===================================================================
// Kernel 4: post_mapping — one CTA per token, 256 threads
//
//  out[b][j][h] = post[j] * x[h] + sum_k comb[k][j] * residual[k][h]
// ===================================================================

__launch_bounds__(256) __global__ void mhcPostMappingKernel(__nv_bfloat16 const* __restrict__ residual,
    __nv_bfloat16 const* __restrict__ x, float const* __restrict__ post_mix, float const* __restrict__ comb_mix,
    __nv_bfloat16* __restrict__ out, int hidden_size)
{
    constexpr int n = 4;

    int const token = blockIdx.x;
    int const tid = threadIdx.x;

    __shared__ float s_post[n];
    __shared__ float s_comb[n][n];

    if (tid < n)
    {
        s_post[tid] = post_mix[token * n + tid];
    }
    if (tid < n * n)
    {
        int const r = tid / n;
        int const c = tid % n;
        s_comb[r][c] = comb_mix[token * n * n + r * n + c];
    }
    __syncthreads();

    float const pm0 = s_post[0], pm1 = s_post[1];
    float const pm2 = s_post[2], pm3 = s_post[3];

    float const c00 = s_comb[0][0], c01 = s_comb[0][1];
    float const c02 = s_comb[0][2], c03 = s_comb[0][3];
    float const c10 = s_comb[1][0], c11 = s_comb[1][1];
    float const c12 = s_comb[1][2], c13 = s_comb[1][3];
    float const c20 = s_comb[2][0], c21 = s_comb[2][1];
    float const c22 = s_comb[2][2], c23 = s_comb[2][3];
    float const c30 = s_comb[3][0], c31 = s_comb[3][1];
    float const c32 = s_comb[3][2], c33 = s_comb[3][3];

    long long const tok_res = static_cast<long long>(token) * n * hidden_size;
    long long const tok_x = static_cast<long long>(token) * hidden_size;

    for (int h = tid * 8; h < hidden_size; h += 256 * 8)
    {
        uint4 x_raw = *reinterpret_cast<uint4 const*>(&x[tok_x + h]);
        __nv_bfloat162 const* xp = reinterpret_cast<__nv_bfloat162 const*>(&x_raw);
        float2 xf0 = __bfloat1622float2(xp[0]);
        float2 xf1 = __bfloat1622float2(xp[1]);
        float2 xf2 = __bfloat1622float2(xp[2]);
        float2 xf3 = __bfloat1622float2(xp[3]);

        float a0[8], a1[8], a2[8], a3[8];
        a0[0] = pm0 * xf0.x;
        a0[1] = pm0 * xf0.y;
        a0[2] = pm0 * xf1.x;
        a0[3] = pm0 * xf1.y;
        a0[4] = pm0 * xf2.x;
        a0[5] = pm0 * xf2.y;
        a0[6] = pm0 * xf3.x;
        a0[7] = pm0 * xf3.y;

        a1[0] = pm1 * xf0.x;
        a1[1] = pm1 * xf0.y;
        a1[2] = pm1 * xf1.x;
        a1[3] = pm1 * xf1.y;
        a1[4] = pm1 * xf2.x;
        a1[5] = pm1 * xf2.y;
        a1[6] = pm1 * xf3.x;
        a1[7] = pm1 * xf3.y;

        a2[0] = pm2 * xf0.x;
        a2[1] = pm2 * xf0.y;
        a2[2] = pm2 * xf1.x;
        a2[3] = pm2 * xf1.y;
        a2[4] = pm2 * xf2.x;
        a2[5] = pm2 * xf2.y;
        a2[6] = pm2 * xf3.x;
        a2[7] = pm2 * xf3.y;

        a3[0] = pm3 * xf0.x;
        a3[1] = pm3 * xf0.y;
        a3[2] = pm3 * xf1.x;
        a3[3] = pm3 * xf1.y;
        a3[4] = pm3 * xf2.x;
        a3[5] = pm3 * xf2.y;
        a3[6] = pm3 * xf3.x;
        a3[7] = pm3 * xf3.y;

#pragma unroll
        for (int k = 0; k < n; k++)
        {
            uint4 r_raw = *reinterpret_cast<uint4 const*>(&residual[tok_res + k * hidden_size + h]);
            __nv_bfloat162 const* rp = reinterpret_cast<__nv_bfloat162 const*>(&r_raw);
            float2 rf0 = __bfloat1622float2(rp[0]);
            float2 rf1 = __bfloat1622float2(rp[1]);
            float2 rf2 = __bfloat1622float2(rp[2]);
            float2 rf3 = __bfloat1622float2(rp[3]);

            float const ck0 = (k == 0) ? c00 : (k == 1) ? c10 : (k == 2) ? c20 : c30;
            float const ck1 = (k == 0) ? c01 : (k == 1) ? c11 : (k == 2) ? c21 : c31;
            float const ck2 = (k == 0) ? c02 : (k == 1) ? c12 : (k == 2) ? c22 : c32;
            float const ck3 = (k == 0) ? c03 : (k == 1) ? c13 : (k == 2) ? c23 : c33;

            a0[0] = fmaf(ck0, rf0.x, a0[0]);
            a0[1] = fmaf(ck0, rf0.y, a0[1]);
            a0[2] = fmaf(ck0, rf1.x, a0[2]);
            a0[3] = fmaf(ck0, rf1.y, a0[3]);
            a0[4] = fmaf(ck0, rf2.x, a0[4]);
            a0[5] = fmaf(ck0, rf2.y, a0[5]);
            a0[6] = fmaf(ck0, rf3.x, a0[6]);
            a0[7] = fmaf(ck0, rf3.y, a0[7]);

            a1[0] = fmaf(ck1, rf0.x, a1[0]);
            a1[1] = fmaf(ck1, rf0.y, a1[1]);
            a1[2] = fmaf(ck1, rf1.x, a1[2]);
            a1[3] = fmaf(ck1, rf1.y, a1[3]);
            a1[4] = fmaf(ck1, rf2.x, a1[4]);
            a1[5] = fmaf(ck1, rf2.y, a1[5]);
            a1[6] = fmaf(ck1, rf3.x, a1[6]);
            a1[7] = fmaf(ck1, rf3.y, a1[7]);

            a2[0] = fmaf(ck2, rf0.x, a2[0]);
            a2[1] = fmaf(ck2, rf0.y, a2[1]);
            a2[2] = fmaf(ck2, rf1.x, a2[2]);
            a2[3] = fmaf(ck2, rf1.y, a2[3]);
            a2[4] = fmaf(ck2, rf2.x, a2[4]);
            a2[5] = fmaf(ck2, rf2.y, a2[5]);
            a2[6] = fmaf(ck2, rf3.x, a2[6]);
            a2[7] = fmaf(ck2, rf3.y, a2[7]);

            a3[0] = fmaf(ck3, rf0.x, a3[0]);
            a3[1] = fmaf(ck3, rf0.y, a3[1]);
            a3[2] = fmaf(ck3, rf1.x, a3[2]);
            a3[3] = fmaf(ck3, rf1.y, a3[3]);
            a3[4] = fmaf(ck3, rf2.x, a3[4]);
            a3[5] = fmaf(ck3, rf2.y, a3[5]);
            a3[6] = fmaf(ck3, rf3.x, a3[6]);
            a3[7] = fmaf(ck3, rf3.y, a3[7]);
        }

#pragma unroll
        for (int j = 0; j < n; j++)
        {
            float* aj = (j == 0) ? a0 : (j == 1) ? a1 : (j == 2) ? a2 : a3;
            uint4 o_raw;
            __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&o_raw);
            op[0] = __float22bfloat162_rn(make_float2(aj[0], aj[1]));
            op[1] = __float22bfloat162_rn(make_float2(aj[2], aj[3]));
            op[2] = __float22bfloat162_rn(make_float2(aj[4], aj[5]));
            op[3] = __float22bfloat162_rn(make_float2(aj[6], aj[7]));
            *reinterpret_cast<uint4*>(&out[tok_res + j * hidden_size + h]) = o_raw;
        }
    }
}

// ===================================================================
// Kernel 5: hc_head_apply — RMS norm → sigmoid → weighted sum
// ===================================================================

__launch_bounds__(256) __global__
    void mhcHcHeadApplyKernel(float const* __restrict__ mixes, float const* __restrict__ sqrsum,
        __nv_bfloat16 const* __restrict__ x, __nv_bfloat16* __restrict__ out, float const* __restrict__ scale,
        float const* __restrict__ base, int mult, int hidden_size, int K, float norm_eps, float eps)
{
    int const token = blockIdx.x;
    int const tid = threadIdx.x;

    __shared__ float s_pre[8];

    if (tid < mult)
    {
        float sq = sqrsum[token];
        float rstd = rsqrtf(sq / static_cast<float>(K) + norm_eps);
        float m = mixes[token * mult + tid];
        float val = m * rstd * scale[0] + base[tid];
        float sig = 1.0f / (1.0f + expf(-val));
        s_pre[tid] = sig + eps;
    }
    __syncthreads();

    __nv_bfloat16 const* xbase = x + static_cast<long long>(token) * mult * hidden_size;
    __nv_bfloat16* obase = out + static_cast<long long>(token) * hidden_size;

    for (int h = tid * 8; h < hidden_size; h += 256 * 8)
    {
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

        for (int j = 0; j < mult; j++)
        {
            float pm = s_pre[j];

            uint4 raw = *reinterpret_cast<uint4 const*>(&xbase[j * hidden_size + h]);
            __nv_bfloat162 const* pairs = reinterpret_cast<__nv_bfloat162 const*>(&raw);

            float2 f0 = __bfloat1622float2(pairs[0]);
            float2 f1 = __bfloat1622float2(pairs[1]);
            float2 f2 = __bfloat1622float2(pairs[2]);
            float2 f3 = __bfloat1622float2(pairs[3]);

            acc0 += pm * f0.x;
            acc1 += pm * f0.y;
            acc2 += pm * f1.x;
            acc3 += pm * f1.y;
            acc4 += pm * f2.x;
            acc5 += pm * f2.y;
            acc6 += pm * f3.x;
            acc7 += pm * f3.y;
        }

        uint4 out_raw;
        __nv_bfloat162* opairs = reinterpret_cast<__nv_bfloat162*>(&out_raw);
        opairs[0] = __float22bfloat162_rn(make_float2(acc0, acc1));
        opairs[1] = __float22bfloat162_rn(make_float2(acc2, acc3));
        opairs[2] = __float22bfloat162_rn(make_float2(acc4, acc5));
        opairs[3] = __float22bfloat162_rn(make_float2(acc6, acc7));
        *reinterpret_cast<uint4*>(&obase[h]) = out_raw;
    }
}

// ===================================================================
// Launch wrappers (raw pointer + cudaStream_t)
// ===================================================================

void mhcBigFuseLaunch(float const* y_acc, float const* r_acc, __nv_bfloat16 const* residual, float const* hc_scale,
    float const* hc_base, float* post_mix, float* comb_mix, __nv_bfloat16* layer_input, int M, int K, int hidden_size,
    float rms_eps, float hc_pre_eps, float hc_sinkhorn_eps, float hc_post_mult_value, int sinkhorn_repeat,
    cudaStream_t stream)
{
    dim3 grid(static_cast<unsigned int>(M));
    dim3 block(256);

    mhcBigFuseKernel<<<grid, block, 0, stream>>>(y_acc, r_acc, residual, hc_scale, hc_base, post_mix, comb_mix,
        layer_input, M, K, hidden_size, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat);
}

void mhcSqrsumLaunch(__nv_bfloat16 const* x, float* r, int M, int K, cudaStream_t stream)
{
    dim3 grid(static_cast<unsigned int>(M));
    dim3 block(256);

    mhcSqrsumKernel<<<grid, block, 0, stream>>>(x, r, K);
}

void mhcGemmSqrsumFmaLaunch(__nv_bfloat16 const* x, float const* w_t, float* y, float* r, int M, int N, int K,
    int num_k_blocks, int k_chunk, bool zero_outputs, cudaStream_t stream)
{
    dim3 grid(static_cast<unsigned int>(M), static_cast<unsigned int>(num_k_blocks));
    dim3 block(256);

    if (zero_outputs)
    {
        cudaMemsetAsync(y, 0, static_cast<size_t>(M) * N * sizeof(float), stream);
        cudaMemsetAsync(r, 0, static_cast<size_t>(M) * sizeof(float), stream);
    }

    mhcGemmSqrsumFmaKernel<<<grid, block, 0, stream>>>(x, w_t, y, r, M, N, K, num_k_blocks, k_chunk);
}

void mhcHcHeadApplyLaunch(float const* mixes, float const* sqrsum, __nv_bfloat16 const* x, __nv_bfloat16* out,
    float const* scale, float const* base, int M, int mult, int hidden_size, int K, float norm_eps, float eps,
    cudaStream_t stream)
{
    dim3 grid(static_cast<unsigned int>(M));
    dim3 block(256);

    mhcHcHeadApplyKernel<<<grid, block, 0, stream>>>(
        mixes, sqrsum, x, out, scale, base, mult, hidden_size, K, norm_eps, eps);
}

void mhcPostMappingLaunch(__nv_bfloat16 const* residual, __nv_bfloat16 const* x, float const* post_mix,
    float const* comb_mix, __nv_bfloat16* out, int B, int hidden_size, cudaStream_t stream)
{
    dim3 grid(static_cast<unsigned int>(B));
    dim3 block(256);

    mhcPostMappingKernel<<<grid, block, 0, stream>>>(residual, x, post_mix, comb_mix, out, hidden_size);
}

} // namespace kernels::mhc

TRTLLM_NAMESPACE_END
