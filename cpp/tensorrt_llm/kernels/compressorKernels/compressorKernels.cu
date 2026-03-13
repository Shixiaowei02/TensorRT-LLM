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

#include "tensorrt_llm/kernels/compressorKernels/compressorKernels.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <type_traits>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::compressor
{

// ============================================================================
// Helper functions
// ============================================================================

__device__ inline float warpReduceSum(float val)
{
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
    return val;
}

__device__ inline float warpReduceMax(float val)
{
    for (int mask = 16; mask > 0; mask >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, mask));
    return val;
}

__device__ inline float loadAsFloat(void const* base, int64_t offset, int elem_bytes)
{
    if (elem_bytes == 2)
        return __bfloat162float(reinterpret_cast<__nv_bfloat16 const*>(base)[offset]);
    else
        return reinterpret_cast<float const*>(base)[offset];
}

__device__ inline void storeFromFloat(void* base, int64_t offset, float val, int elem_bytes)
{
    if (elem_bytes == 2)
        reinterpret_cast<__nv_bfloat16*>(base)[offset] = __float2bfloat16_rn(val);
    else
        reinterpret_cast<float*>(base)[offset] = val;
}

// Vectorized load/store types: maps byte-width to CUDA vector type.
template <int V> struct VecType;
template <> struct VecType<4>  { using type = unsigned int; };  //  32-bit: 2 bf16 or 1 fp32
template <> struct VecType<8>  { using type = uint2; };         //  64-bit: 4 bf16 or 2 fp32
template <> struct VecType<16> { using type = uint4; };         // 128-bit: 8 bf16 or 4 fp32

// ============================================================================
// Decode Kernel: pagedKvCompressKernel
//
// Template: <HEAD_DIM, IO_ELEM_BYTES, NEXT_N>
// Grid: (batch_size)
// Block: (NTHRD) where NTHRD = HEAD_DIM / VEC
// Each thread handles VEC elements of head_dim using 128-bit vectorized
// loads/stores. For overlap mode, Phase 1 iterates over 2 column halves.
// ============================================================================

// Helper: vectorized online softmax step reading from paged cache.
template <int HEAD_DIM, int IO_ELEM_BYTES>
__device__ __forceinline__ void decodeSoftmaxVec(
    void const* __restrict__ paged_kv_raw,
    void const* __restrict__ paged_score_raw,
    float const* __restrict__ ape,
    int64_t page_sd,    // page_size * state_dim (in elements)
    int state_dim,
    int phys_kv,        // physical page index for kv
    int phys_sc,        // physical page index for score
    int blk_off,        // offset within page
    int kv_col_off,     // column offset (0 or HEAD_DIM)
    int ape_row,        // r * state_dim + col_off
    int tid,
    float* __restrict__ rmax,
    float* __restrict__ rsum,
    float* __restrict__ rwsum)
{
    using IoElemT = typename std::conditional<IO_ELEM_BYTES == 2, __nv_bfloat16, float>::type;
    constexpr int MAX_VEC = 16 / IO_ELEM_BYTES;
    constexpr int VEC     = (HEAD_DIM / MAX_VEC >= 32) ? MAX_VEC : (HEAD_DIM / 32);
    using IoVecT = typename VecType<VEC * IO_ELEM_BYTES>::type;

    auto const* kv = reinterpret_cast<IoElemT const*>(paged_kv_raw);
    auto const* sc = reinterpret_cast<IoElemT const*>(paged_score_raw);

    int64_t base_kv = static_cast<int64_t>(phys_kv) * page_sd + blk_off * state_dim + kv_col_off;
    int64_t base_sc = static_cast<int64_t>(phys_sc) * page_sd + blk_off * state_dim + kv_col_off;

    IoVecT k_raw = reinterpret_cast<IoVecT const*>(&kv[base_kv])[tid];
    IoVecT s_raw = reinterpret_cast<IoVecT const*>(&sc[base_sc])[tid];
    IoElemT const* ke = reinterpret_cast<IoElemT const*>(&k_raw);
    IoElemT const* se = reinterpret_cast<IoElemT const*>(&s_raw);

#pragma unroll
    for (int i = 0; i < VEC; i += 4)
    {
        float4 av = *reinterpret_cast<float4 const*>(&ape[ape_row + tid * VEC + i]);
        float kf[4] = {static_cast<float>(ke[i]),   static_cast<float>(ke[i + 1]),
                        static_cast<float>(ke[i + 2]), static_cast<float>(ke[i + 3])};
        float sf[4] = {static_cast<float>(se[i])   + av.x, static_cast<float>(se[i + 1]) + av.y,
                        static_cast<float>(se[i + 2]) + av.z, static_cast<float>(se[i + 3]) + av.w};
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            float nm = fmaxf(rmax[i + j], sf[j]);
            float sc_f = expf(rmax[i + j] - nm);
            float tm = expf(sf[j] - nm);
            rsum[i + j]  = rsum[i + j] * sc_f + tm;
            rwsum[i + j] = rwsum[i + j] * sc_f + kf[j] * tm;
            rmax[i + j]  = nm;
        }
    }
}

template <int HEAD_DIM, int IO_ELEM_BYTES, int NEXT_N>
__global__ void pagedKvCompressKernel(
    void const* __restrict__ kv_score_raw,
    float const* __restrict__ ape,
    void* __restrict__ paged_kv_raw,
    void* __restrict__ paged_score_raw,
    int32_t const* __restrict__ block_table_kv,
    int32_t const* __restrict__ block_table_score,
    void* __restrict__ output_raw,
    int32_t const* __restrict__ kv_lens,
    int32_t const* __restrict__ start_pos_arr,
    int32_t const* __restrict__ cu_seq_lens,
    int32_t const* __restrict__ cu_kv_comp,
    bool* __restrict__ compressed_mask,
    int page_size,
    int state_dim,
    int compress_ratio,
    bool is_overlap,
    int max_blocks,
    int out_elem_bytes)
{
    using IoElemT = typename std::conditional<IO_ELEM_BYTES == 2, __nv_bfloat16, float>::type;
    constexpr int MAX_VEC = 16 / IO_ELEM_BYTES;
    constexpr int VEC     = (HEAD_DIM / MAX_VEC >= 32) ? MAX_VEC : (HEAD_DIM / 32);
    constexpr int NTHRD   = HEAD_DIM / VEC;
    using IoVecT = typename VecType<VEC * IO_ELEM_BYTES>::type;
    static_assert(VEC >= 4, "VEC must be >= 4 for float4 ape loads");

    int const tid = threadIdx.x;
    int const batch_idx = blockIdx.x;

    int const sp = start_pos_arr[batch_idx];
    int const kv_len = kv_lens[batch_idx];
    int const in_off = cu_seq_lens[batch_idx];
    int const out_off = cu_kv_comp[batch_idx];
    int64_t const two_sd = 2 * state_dim;
    int64_t const page_sd = static_cast<int64_t>(page_size) * state_dim;
    int const coff = is_overlap ? 2 : 1;

    auto const* kv_score = reinterpret_cast<IoElemT const*>(kv_score_raw);
    auto* paged_kv = reinterpret_cast<IoElemT*>(paged_kv_raw);
    auto* paged_score = reinterpret_cast<IoElemT*>(paged_score_raw);

    // ================================================================
    // Phase 1: Write NEXT_N tokens to paged cache (vectorized)
    // Each token has state_dim elements split into coff * HEAD_DIM columns.
    // ================================================================
#pragma unroll
    for (int t = 0; t < NEXT_N; t++)
    {
        int token_idx = sp + t;
        if (token_idx < kv_len)
        {
            int ape_idx = token_idx % compress_ratio;
            int log_blk = token_idx / page_size;
            int blk_off = token_idx % page_size;
            int phys_kv = block_table_kv[batch_idx * max_blocks + log_blk];
            int phys_sc = block_table_score[batch_idx * max_blocks + log_blk];

            for (int col_idx = 0; col_idx < coff; col_idx++)
            {
                int const col = col_idx * HEAD_DIM;
                int64_t const src = static_cast<int64_t>(in_off + t) * two_sd + col;
                int64_t const dkv = static_cast<int64_t>(phys_kv) * page_sd + blk_off * state_dim + col;
                int64_t const dsc = static_cast<int64_t>(phys_sc) * page_sd + blk_off * state_dim + col;

                // Load kv and score vectors from kv_score
                IoVecT kv_raw = reinterpret_cast<IoVecT const*>(&kv_score[src])[tid];
                IoVecT sc_raw = reinterpret_cast<IoVecT const*>(&kv_score[src + state_dim])[tid];

                // Store kv directly
                reinterpret_cast<IoVecT*>(&paged_kv[dkv])[tid] = kv_raw;

                // Add ape to score, then store
                IoElemT const* sc_e = reinterpret_cast<IoElemT const*>(&sc_raw);
                IoVecT sc_out;
                IoElemT* sc_o = reinterpret_cast<IoElemT*>(&sc_out);
#pragma unroll
                for (int i = 0; i < VEC; i += 4)
                {
                    float4 av = *reinterpret_cast<float4 const*>(
                        &ape[ape_idx * state_dim + col + tid * VEC + i]);
                    sc_o[i]     = static_cast<IoElemT>(static_cast<float>(sc_e[i])     + av.x);
                    sc_o[i + 1] = static_cast<IoElemT>(static_cast<float>(sc_e[i + 1]) + av.y);
                    sc_o[i + 2] = static_cast<IoElemT>(static_cast<float>(sc_e[i + 2]) + av.z);
                    sc_o[i + 3] = static_cast<IoElemT>(static_cast<float>(sc_e[i + 3]) + av.w);
                }
                reinterpret_cast<IoVecT*>(&paged_score[dsc])[tid] = sc_out;
            }
        }
    }

    // ================================================================
    // Phase 2: Count compressions, store mask
    // ================================================================
    int last_token_idx = sp + NEXT_N - 1;
    int num_compressions = (last_token_idx + 1) / compress_ratio - sp / compress_ratio;
    if (tid == 0)
        compressed_mask[batch_idx] = (num_compressions > 0);

    // ================================================================
    // Phase 3: Online softmax reduction (vectorized from paged cache)
    // ================================================================
    for (int c = 0; c < NEXT_N; c++)
    {
        if (c >= num_compressions)
            break;

        int compress_idx = sp / compress_ratio + c;
        int curr_chunk_start = compress_idx * compress_ratio;

        float rmax[VEC], rsum[VEC], rwsum[VEC];
#pragma unroll
        for (int i = 0; i < VEC; i++)
        {
            rmax[i]  = -INFINITY;
            rsum[i]  = 0.0f;
            rwsum[i] = 0.0f;
        }

        if (is_overlap)
        {
            // Previous chunk: first head_dim features (col_off=0)
            int prev_start = curr_chunk_start - compress_ratio;
            for (int r = 0; r < compress_ratio; r++)
            {
                int pos = prev_start + r;
                int log_blk = pos / page_size;
                int blk_off = pos % page_size;
                int phys_kv = block_table_kv[batch_idx * max_blocks + log_blk];
                int phys_sc = block_table_score[batch_idx * max_blocks + log_blk];

                decodeSoftmaxVec<HEAD_DIM, IO_ELEM_BYTES>(
                    paged_kv_raw, paged_score_raw, ape, page_sd, state_dim,
                    phys_kv, phys_sc, blk_off, 0, r * state_dim,
                    tid, rmax, rsum, rwsum);
            }

            // Current chunk: second head_dim features (col_off=HEAD_DIM)
            for (int r = 0; r < compress_ratio; r++)
            {
                int pos = curr_chunk_start + r;
                int log_blk = pos / page_size;
                int blk_off = pos % page_size;
                int phys_kv = block_table_kv[batch_idx * max_blocks + log_blk];
                int phys_sc = block_table_score[batch_idx * max_blocks + log_blk];

                decodeSoftmaxVec<HEAD_DIM, IO_ELEM_BYTES>(
                    paged_kv_raw, paged_score_raw, ape, page_sd, state_dim,
                    phys_kv, phys_sc, blk_off, HEAD_DIM, r * state_dim + HEAD_DIM,
                    tid, rmax, rsum, rwsum);
            }
        }
        else
        {
            for (int r = 0; r < compress_ratio; r++)
            {
                int pos = curr_chunk_start + r;
                int log_blk = pos / page_size;
                int blk_off = pos % page_size;
                int phys_kv = block_table_kv[batch_idx * max_blocks + log_blk];
                int phys_sc = block_table_score[batch_idx * max_blocks + log_blk];

                decodeSoftmaxVec<HEAD_DIM, IO_ELEM_BYTES>(
                    paged_kv_raw, paged_score_raw, ape, page_sd, state_dim,
                    phys_kv, phys_sc, blk_off, 0, r * state_dim,
                    tid, rmax, rsum, rwsum);
            }
        }

        // Store output (vectorized)
        int64_t const out_base = static_cast<int64_t>(out_off + c) * HEAD_DIM + tid * VEC;
        if (out_elem_bytes == 2)
        {
            __nv_bfloat16 packed[VEC];
#pragma unroll
            for (int i = 0; i < VEC; i++)
                packed[i] = __float2bfloat16_rn(rwsum[i] / rsum[i]);
            using OutVecT = typename VecType<VEC * 2>::type;
            *reinterpret_cast<OutVecT*>(
                &reinterpret_cast<__nv_bfloat16*>(output_raw)[out_base]) =
                *reinterpret_cast<OutVecT const*>(packed);
        }
        else
        {
            float result[VEC];
#pragma unroll
            for (int i = 0; i < VEC; i++)
                result[i] = rwsum[i] / rsum[i];
#pragma unroll
            for (int i = 0; i < VEC; i += 4)
                *reinterpret_cast<float4*>(
                    &reinterpret_cast<float*>(output_raw)[out_base + i]) =
                    *reinterpret_cast<float4 const*>(&result[i]);
        }
    }
}

// Explicit instantiations for decode kernel
#define INST_DECODE(HD, EB, NN) \
    template __global__ void pagedKvCompressKernel<HD, EB, NN>( \
        void const*, float const*, void*, void*, \
        int32_t const*, int32_t const*, void*, int32_t const*, int32_t const*, \
        int32_t const*, int32_t const*, bool*, \
        int, int, int, bool, int, int);

#define INST_DECODE_NN(HD, EB) \
    INST_DECODE(HD, EB, 1) INST_DECODE(HD, EB, 2) INST_DECODE(HD, EB, 3) INST_DECODE(HD, EB, 4)

INST_DECODE_NN(128, 2)
INST_DECODE_NN(128, 4)
INST_DECODE_NN(512, 2)
INST_DECODE_NN(512, 4)
#undef INST_DECODE_NN
#undef INST_DECODE


// ============================================================================
// Decode Launch Wrapper
// ============================================================================

// Forward declaration (defined in prefill section below).
static inline int prefillNthreads(int head_dim, int io_elem_bytes);

void pagedKvCompressLaunch(
    void const* kv_score,
    float const* ape,
    void* paged_kv,
    void* paged_score,
    int32_t const* block_table_kv,
    int32_t const* block_table_score,
    void* output,
    int32_t const* kv_lens,
    int32_t const* start_pos,
    int32_t const* cu_seq_lens,
    int32_t const* cu_kv_comp,
    bool* compressed_mask,
    int batch_size,
    int page_size,
    int max_blocks,
    int head_dim,
    int compress_ratio,
    bool is_overlap,
    int next_n,
    int io_elem_bytes,
    int out_elem_bytes,
    cudaStream_t stream)
{
    int const coff = is_overlap ? 2 : 1;
    int const state_dim = coff * head_dim;
    int const nthreads = prefillNthreads(head_dim, io_elem_bytes);
    dim3 grid(batch_size);

#define LAUNCH_DECODE(HD, EB, NN)                                                   \
    pagedKvCompressKernel<HD, EB, NN><<<grid, nthreads, 0, stream>>>(               \
        kv_score, ape, paged_kv, paged_score,                                       \
        block_table_kv, block_table_score, output,                                  \
        kv_lens, start_pos, cu_seq_lens, cu_kv_comp, compressed_mask,               \
        page_size, state_dim, compress_ratio, is_overlap, max_blocks,               \
        out_elem_bytes)

#define DISPATCH_NN(HD, EB)                          \
    switch (next_n) {                                \
    case 1: LAUNCH_DECODE(HD, EB, 1); break;         \
    case 2: LAUNCH_DECODE(HD, EB, 2); break;         \
    case 3: LAUNCH_DECODE(HD, EB, 3); break;         \
    default: LAUNCH_DECODE(HD, EB, 4); break;        \
    }

    if (io_elem_bytes == 4)
    {
        if (head_dim == 512) { DISPATCH_NN(512, 4); }
        else                 { DISPATCH_NN(128, 4); }
    }
    else
    {
        if (head_dim == 512) { DISPATCH_NN(512, 2); }
        else                 { DISPATCH_NN(128, 2); }
    }

#undef DISPATCH_NN
#undef LAUNCH_DECODE
}


// ============================================================================
// Prefill Kernel: prefillReductionKernel
//
// Grid: (batch_size, max_outputs_per_batch)
// Block: (NTHRD) where NTHRD = HEAD_DIM / VEC
// Each block computes one compressed output for the full head_dim using
// vectorized 128-bit loads (float4 for fp32, 8×bf16 for bf16).
// ============================================================================

// Per-element online softmax step on VEC elements via 128-bit vectorized loads.
template <int HEAD_DIM, int IO_ELEM_BYTES>
__device__ __forceinline__ void prefillSoftmaxVec(
    void const* __restrict__ kv_score_raw,
    float const* __restrict__ ape,
    int64_t row_elem,     // (input_offset + row_idx) * two_sd
    int kv_col_off,       // column offset into kv_score row (0 or HEAD_DIM)
    int ape_base,         // r * state_dim + ape_col_off
    int state_dim,
    int tid,
    float* __restrict__ rmax,
    float* __restrict__ rsum,
    float* __restrict__ rwsum)
{
    using IoElemT = typename std::conditional<IO_ELEM_BYTES == 2, __nv_bfloat16, float>::type;
    constexpr int MAX_VEC = 16 / IO_ELEM_BYTES;
    constexpr int VEC     = (HEAD_DIM / MAX_VEC >= 32) ? MAX_VEC : (HEAD_DIM / 32);
    using IoVecT = typename VecType<VEC * IO_ELEM_BYTES>::type;

    auto const* kv = reinterpret_cast<IoElemT const*>(kv_score_raw);

    IoVecT k_raw = reinterpret_cast<IoVecT const*>(&kv[row_elem + kv_col_off])[tid];
    IoVecT s_raw = reinterpret_cast<IoVecT const*>(&kv[row_elem + state_dim + kv_col_off])[tid];
    IoElemT const* ke = reinterpret_cast<IoElemT const*>(&k_raw);
    IoElemT const* se = reinterpret_cast<IoElemT const*>(&s_raw);

#pragma unroll
    for (int i = 0; i < VEC; i += 4)
    {
        float4 av = *reinterpret_cast<float4 const*>(&ape[ape_base + tid * VEC + i]);
        float kf[4] = {static_cast<float>(ke[i]),   static_cast<float>(ke[i + 1]),
                        static_cast<float>(ke[i + 2]), static_cast<float>(ke[i + 3])};
        float sf[4] = {static_cast<float>(se[i])   + av.x, static_cast<float>(se[i + 1]) + av.y,
                        static_cast<float>(se[i + 2]) + av.z, static_cast<float>(se[i + 3]) + av.w};
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            float nm = fmaxf(rmax[i + j], sf[j]);
            float sc = expf(rmax[i + j] - nm);
            float tm = expf(sf[j] - nm);
            rsum[i + j]  = rsum[i + j] * sc + tm;
            rwsum[i + j] = rwsum[i + j] * sc + kf[j] * tm;
            rmax[i + j]  = nm;
        }
    }
}

template <int HEAD_DIM, int IO_ELEM_BYTES>
__global__ void prefillReductionKernel(
    void const* __restrict__ kv_score_raw,
    float const* __restrict__ ape,
    void* __restrict__ paged_kv_raw,
    void* __restrict__ paged_score_raw,
    int32_t const* __restrict__ block_table_kv,
    int32_t const* __restrict__ block_table_score,
    void* __restrict__ output_raw,
    int32_t const* __restrict__ kv_lens,
    int32_t const* __restrict__ start_pos_arr,
    int32_t const* __restrict__ cu_seq_lens,
    int32_t const* __restrict__ cu_kv_comp,
    bool* __restrict__ compressed_mask,
    int page_size,
    int state_dim,
    int compress_ratio,
    bool is_overlap,
    int max_blocks,
    int out_elem_bytes)
{
    using IoElemT = typename std::conditional<IO_ELEM_BYTES == 2, __nv_bfloat16, float>::type;

    constexpr int MAX_VEC = 16 / IO_ELEM_BYTES;
    constexpr int VEC     = (HEAD_DIM / MAX_VEC >= 32) ? MAX_VEC : (HEAD_DIM / 32);
    constexpr int NTHRD   = HEAD_DIM / VEC;
    using IoVecT = typename VecType<VEC * IO_ELEM_BYTES>::type;
    static_assert(VEC >= 4, "VEC must be >= 4 for float4 ape loads");

    int const tid = threadIdx.x;
    int const batch_idx = blockIdx.x;
    int const local_output_idx = blockIdx.y;

    int const sp = start_pos_arr[batch_idx];
    int const kv_len = kv_lens[batch_idx];
    int const input_offset = cu_seq_lens[batch_idx];
    int const output_offset = cu_kv_comp[batch_idx];

    int const seqlen = kv_len - sp;
    int const num_outputs = max(seqlen / compress_ratio, 1);

    if (local_output_idx >= num_outputs)
        return;

    int const coff = is_overlap ? 2 : 1;
    int const actual_num_outputs = seqlen / compress_ratio;
    bool const should_compress = (local_output_idx < actual_num_outputs);

    if (local_output_idx == 0 && tid == 0)
        compressed_mask[batch_idx] = (actual_num_outputs > 0);

    auto const* kv_score = reinterpret_cast<IoElemT const*>(kv_score_raw);
    auto* paged_kv = reinterpret_cast<IoElemT*>(paged_kv_raw);
    auto* paged_score = reinterpret_cast<IoElemT*>(paged_score_raw);

    int64_t const two_sd = 2 * state_dim;
    int64_t const page_sd = static_cast<int64_t>(page_size) * state_dim;

    // ================================================================
    // Phase 1: State Update (last output block only)
    // Vectorized 128-bit loads/stores for paged KV and score state.
    // ================================================================
    if (local_output_idx == num_outputs - 1)
    {
        int remainder = seqlen % compress_ratio;
        int cutoff = seqlen - remainder;

        // 1a. Last full chunk (overlap only)
        if (is_overlap && cutoff >= compress_ratio)
        {
            for (int r = 0; r < compress_ratio; r++)
            {
                int write_pos = sp + cutoff - compress_ratio + r;
                int log_blk = write_pos / page_size;
                int blk_off = write_pos % page_size;
                int phys_kv = block_table_kv[batch_idx * max_blocks + log_blk];
                int phys_sc = block_table_score[batch_idx * max_blocks + log_blk];
                int base_row = cutoff - compress_ratio + r;

                for (int col_idx = 0; col_idx < 2; col_idx++)
                {
                    int const col = col_idx * HEAD_DIM;
                    int64_t const src = static_cast<int64_t>(input_offset + base_row) * two_sd + col;
                    int64_t const dkv = static_cast<int64_t>(phys_kv) * page_sd + blk_off * state_dim + col;
                    int64_t const dsc = static_cast<int64_t>(phys_sc) * page_sd + blk_off * state_dim + col;

                    IoVecT kv_raw = reinterpret_cast<IoVecT const*>(&kv_score[src])[tid];
                    IoVecT sc_raw = reinterpret_cast<IoVecT const*>(&kv_score[src + state_dim])[tid];
                    reinterpret_cast<IoVecT*>(&paged_kv[dkv])[tid] = kv_raw;

                    IoElemT const* sc_e = reinterpret_cast<IoElemT const*>(&sc_raw);
                    IoVecT sc_out;
                    IoElemT* sc_o = reinterpret_cast<IoElemT*>(&sc_out);
#pragma unroll
                    for (int i = 0; i < VEC; i += 4)
                    {
                        float4 av = *reinterpret_cast<float4 const*>(
                            &ape[r * state_dim + col + tid * VEC + i]);
                        sc_o[i]     = static_cast<IoElemT>(static_cast<float>(sc_e[i])     + av.x);
                        sc_o[i + 1] = static_cast<IoElemT>(static_cast<float>(sc_e[i + 1]) + av.y);
                        sc_o[i + 2] = static_cast<IoElemT>(static_cast<float>(sc_e[i + 2]) + av.z);
                        sc_o[i + 3] = static_cast<IoElemT>(static_cast<float>(sc_e[i + 3]) + av.w);
                    }
                    reinterpret_cast<IoVecT*>(&paged_score[dsc])[tid] = sc_out;
                }
            }
        }

        // 1b. Remainder tokens
        for (int r = 0; r < remainder; r++)
        {
            int write_pos = sp + cutoff + r;
            int log_blk = write_pos / page_size;
            int blk_off = write_pos % page_size;
            int phys_kv = block_table_kv[batch_idx * max_blocks + log_blk];
            int phys_sc = block_table_score[batch_idx * max_blocks + log_blk];
            int base_row = cutoff + r;

            for (int col_idx = 0; col_idx < coff; col_idx++)
            {
                int const col = col_idx * HEAD_DIM;
                int64_t const src = static_cast<int64_t>(input_offset + base_row) * two_sd + col;
                int64_t const dkv = static_cast<int64_t>(phys_kv) * page_sd + blk_off * state_dim + col;
                int64_t const dsc = static_cast<int64_t>(phys_sc) * page_sd + blk_off * state_dim + col;

                IoVecT kv_raw = reinterpret_cast<IoVecT const*>(&kv_score[src])[tid];
                IoVecT sc_raw = reinterpret_cast<IoVecT const*>(&kv_score[src + state_dim])[tid];
                reinterpret_cast<IoVecT*>(&paged_kv[dkv])[tid] = kv_raw;

                IoElemT const* sc_e = reinterpret_cast<IoElemT const*>(&sc_raw);
                IoVecT sc_out;
                IoElemT* sc_o = reinterpret_cast<IoElemT*>(&sc_out);
#pragma unroll
                for (int i = 0; i < VEC; i += 4)
                {
                    float4 av = *reinterpret_cast<float4 const*>(
                        &ape[r * state_dim + col + tid * VEC + i]);
                    sc_o[i]     = static_cast<IoElemT>(static_cast<float>(sc_e[i])     + av.x);
                    sc_o[i + 1] = static_cast<IoElemT>(static_cast<float>(sc_e[i + 1]) + av.y);
                    sc_o[i + 2] = static_cast<IoElemT>(static_cast<float>(sc_e[i + 2]) + av.z);
                    sc_o[i + 3] = static_cast<IoElemT>(static_cast<float>(sc_e[i + 3]) + av.w);
                }
                reinterpret_cast<IoVecT*>(&paged_score[dsc])[tid] = sc_out;
            }
        }
    }

    // ================================================================
    // Phase 2: Online softmax reduction (vectorized)
    // ================================================================
    if (!should_compress)
        return;

    float rmax[VEC], rsum[VEC], rwsum[VEC];
#pragma unroll
    for (int i = 0; i < VEC; i++)
    {
        rmax[i]  = -INFINITY;
        rsum[i]  = 0.0f;
        rwsum[i] = 0.0f;
    }

    if (is_overlap)
    {
        // Previous segment (first head_dim features)
        if (local_output_idx > 0)
        {
            int input_start = (local_output_idx - 1) * compress_ratio;
            for (int r = 0; r < compress_ratio; r++)
            {
                int64_t row = static_cast<int64_t>(input_offset + input_start + r) * two_sd;
                prefillSoftmaxVec<HEAD_DIM, IO_ELEM_BYTES>(
                    kv_score_raw, ape, row, 0, r * state_dim,
                    state_dim, tid, rmax, rsum, rwsum);
            }
        }

        // Current segment (second head_dim features)
        int cur_start = local_output_idx * compress_ratio;
        for (int r = 0; r < compress_ratio; r++)
        {
            int64_t row = static_cast<int64_t>(input_offset + cur_start + r) * two_sd;
            prefillSoftmaxVec<HEAD_DIM, IO_ELEM_BYTES>(
                kv_score_raw, ape, row, HEAD_DIM, r * state_dim + HEAD_DIM,
                state_dim, tid, rmax, rsum, rwsum);
        }
    }
    else
    {
        int input_start = local_output_idx * compress_ratio;
        for (int r = 0; r < compress_ratio; r++)
        {
            int64_t row = static_cast<int64_t>(input_offset + input_start + r) * two_sd;
            prefillSoftmaxVec<HEAD_DIM, IO_ELEM_BYTES>(
                kv_score_raw, ape, row, 0, r * state_dim,
                state_dim, tid, rmax, rsum, rwsum);
        }
    }

    // ================================================================
    // Store output (vectorized)
    // ================================================================
    int64_t const out_base = static_cast<int64_t>(output_offset + local_output_idx) * HEAD_DIM + tid * VEC;

    if (out_elem_bytes == 2)
    {
        __nv_bfloat16 packed[VEC];
#pragma unroll
        for (int i = 0; i < VEC; i++)
            packed[i] = __float2bfloat16_rn(rwsum[i] / rsum[i]);
        // VEC * 2 bytes: VEC=4 → 8B (uint2), VEC=8 → 16B (uint4)
        using OutVecT = typename VecType<VEC * 2>::type;
        *reinterpret_cast<OutVecT*>(
            &reinterpret_cast<__nv_bfloat16*>(output_raw)[out_base]) =
            *reinterpret_cast<OutVecT const*>(packed);
    }
    else
    {
        float result[VEC];
#pragma unroll
        for (int i = 0; i < VEC; i++)
            result[i] = rwsum[i] / rsum[i];
        // VEC * 4 bytes: VEC=4 → 16B (uint4/float4), VEC=8 → 32B (2×float4)
#pragma unroll
        for (int i = 0; i < VEC; i += 4)
            *reinterpret_cast<float4*>(
                &reinterpret_cast<float*>(output_raw)[out_base + i]) =
                *reinterpret_cast<float4 const*>(&result[i]);
    }
}

// Explicit instantiations
#define INST_PREFILL(HD, EB) \
    template __global__ void prefillReductionKernel<HD, EB>( \
        void const*, float const*, void*, void*, \
        int32_t const*, int32_t const*, void*, int32_t const*, int32_t const*, \
        int32_t const*, int32_t const*, bool*, \
        int, int, int, bool, int, int);

INST_PREFILL(128, 2)
INST_PREFILL(128, 4)
INST_PREFILL(512, 2)
INST_PREFILL(512, 4)
#undef INST_PREFILL


// ============================================================================
// Prefill Launch Wrapper
// ============================================================================

static inline int prefillNthreads(int head_dim, int io_elem_bytes)
{
    int max_vec = 16 / io_elem_bytes;
    int vec = (head_dim / max_vec >= 32) ? max_vec : (head_dim / 32);
    return head_dim / vec;
}

void prefillReductionLaunch(
    void const* kv_score,
    float const* ape,
    void* paged_kv,
    void* paged_score,
    int32_t const* block_table_kv,
    int32_t const* block_table_score,
    void* output,
    int32_t const* kv_lens,
    int32_t const* start_pos,
    int32_t const* cu_seq_lens,
    int32_t const* cu_kv_comp,
    bool* compressed_mask,
    int batch_size,
    int page_size,
    int max_blocks,
    int head_dim,
    int compress_ratio,
    bool is_overlap,
    int max_outputs,
    int io_elem_bytes,
    int out_elem_bytes,
    cudaStream_t stream)
{
    int const nthreads = prefillNthreads(head_dim, io_elem_bytes);
    int const coff = is_overlap ? 2 : 1;
    int const state_dim = coff * head_dim;
    dim3 grid(batch_size, max(max_outputs, 1));

#define LAUNCH_PREFILL(HD, EB)                                                      \
    prefillReductionKernel<HD, EB><<<grid, nthreads, 0, stream>>>(                  \
        kv_score, ape, paged_kv, paged_score,                                       \
        block_table_kv, block_table_score, output,                                  \
        kv_lens, start_pos, cu_seq_lens, cu_kv_comp, compressed_mask,               \
        page_size, state_dim, compress_ratio, is_overlap, max_blocks,               \
        out_elem_bytes)

    if (io_elem_bytes == 4)
    {
        if (head_dim == 512) { LAUNCH_PREFILL(512, 4); }
        else                 { LAUNCH_PREFILL(128, 4); }
    }
    else
    {
        if (head_dim == 512) { LAUNCH_PREFILL(512, 2); }
        else                 { LAUNCH_PREFILL(128, 2); }
    }

#undef LAUNCH_PREFILL
}


// ============================================================================
// Fused PostProcess + Scatter Kernel
//
// Fuses RMSNorm + RoPE + Hadamard + Scatter into a single kernel.
// Uses adaptive vectorization to ensure at least 32 threads (one full warp).
//
// Grid: (total_tokens) — one block per compressed token
// Block: (HEAD_DIM / VEC), always >= 32
// ============================================================================

// Adaptive VEC: ensure at least 32 threads (one full warp).
// VEC = min(16/ELEM_BYTES, HEAD_DIM/32).
//   HD128 bf16: VEC=4 (uint2, 32 threads)   HD512 bf16: VEC=8 (uint4, 64 threads)
//   HD128 fp32: VEC=4 (uint4, 32 threads)   HD512 fp32: VEC=4 (uint4, 128 threads)
template <int HEAD_DIM, int ELEM_BYTES>
__global__ void fusedPostProcessScatterKernel(
    void const* __restrict__ kv_comp,           // [total_tokens, head_dim]
    void const* __restrict__ rms_weight,        // [head_dim]
    float rms_eps,
    float const* __restrict__ cos_sin_table,    // [max_pos, 2, rope_dim/2]
    int32_t const* __restrict__ position_ids,   // [total_tokens]
    int nope_dim,
    int rope_dim,
    void* __restrict__ kv_cache,                // [num_blocks, block_elems]
    int32_t const* __restrict__ num_outputs_arr,
    int32_t const* __restrict__ cu_kv_comp,
    int32_t const* __restrict__ start_pos_arr,
    int32_t const* __restrict__ block_offsets,
    int batch_size,
    int tokens_per_block,
    int max_blocks,
    int cache_stride_blk,
    int total_tokens)
{
    using ElemT = typename std::conditional<ELEM_BYTES == 2, __nv_bfloat16, float>::type;
    constexpr int MAX_VEC = 16 / ELEM_BYTES;
    constexpr int VEC     = (HEAD_DIM / MAX_VEC >= 32) ? MAX_VEC : (HEAD_DIM / 32);
    constexpr int NTHRD   = HEAD_DIM / VEC;
    constexpr int VEC_BYTES = VEC * ELEM_BYTES;
    using VecT = typename VecType<VEC_BYTES>::type;

    int const token_idx = blockIdx.x;
    if (token_idx >= total_tokens)
        return;

    int const tid = threadIdx.x;

    extern __shared__ float smem[];

    // ================================================================
    // Step 0: Batch index via binary search on cu_kv_comp
    // ================================================================
    int batch_idx, local_output_idx;
    if (batch_size <= 1)
    {
        batch_idx = 0;
        local_output_idx = token_idx;
    }
    else
    {
        int lo = 0, hi = batch_size;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (cu_kv_comp[mid + 1] <= token_idx)
                lo = mid + 1;
            else
                hi = mid;
        }
        batch_idx = lo;
        if (batch_idx >= batch_size)
            return;
        local_output_idx = token_idx - cu_kv_comp[batch_idx];
    }

    if (local_output_idx >= num_outputs_arr[batch_idx])
        return;

    // ================================================================
    // Step 1: Vectorized load — VEC_BYTES per thread, unpack to float regs
    // ================================================================
    auto const* src = reinterpret_cast<VecT const*>(
        reinterpret_cast<ElemT const*>(kv_comp) + static_cast<int64_t>(token_idx) * HEAD_DIM);
    VecT raw_in = src[tid];
    ElemT const* in_elems = reinterpret_cast<ElemT const*>(&raw_in);

    float v[VEC];
#pragma unroll
    for (int i = 0; i < VEC; i++)
        v[i] = static_cast<float>(in_elems[i]);

    // ================================================================
    // Step 2: RMSNorm — per-thread partial sq sum, warp + smem reduce
    // ================================================================
    float local_sq = 0.f;
#pragma unroll
    for (int i = 0; i < VEC; i++)
        local_sq += v[i] * v[i];

    float warp_sum = warpReduceSum(local_sq);

    constexpr int NUM_WARPS = (NTHRD + 31) / 32;
    int const warp_id = tid / 32;
    int const lane_id = tid % 32;

    if (lane_id == 0)
        smem[warp_id] = warp_sum;
    __syncthreads();

    if (warp_id == 0)
    {
        float s = (lane_id < NUM_WARPS) ? smem[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            s += __shfl_xor_sync(0xFFFFFFFF, s, offset);
        if (lane_id == 0)
            smem[0] = s;
    }
    __syncthreads();
    float const rms_scale = rsqrtf(smem[0] / static_cast<float>(HEAD_DIM) + rms_eps);

    // ================================================================
    // Step 3: Load weight vectorized, apply RMSNorm + weight, truncate
    // ================================================================
    auto const* wt_src = reinterpret_cast<VecT const*>(
        reinterpret_cast<ElemT const*>(rms_weight));
    VecT raw_w = wt_src[tid];
    ElemT const* w_elems = reinterpret_cast<ElemT const*>(&raw_w);

    // P4: keep fp32 through entire pipeline
#pragma unroll
    for (int i = 0; i < VEC; i++)
        v[i] = v[i] * rms_scale * static_cast<float>(w_elems[i]);

    // ================================================================
    // Step 4: RoPE — register-only (P0: no shared memory needed)
    // With is_neox=False, even/odd pairs are interleaved and always
    // reside in the same thread's registers when VEC >= 2.
    // ================================================================
    int const half_rope = rope_dim / 2;
    int const pos_id = position_ids[token_idx];

#pragma unroll
    for (int i = 0; i < VEC; i += 2)
    {
        int const elem_idx = tid * VEC + i;
        if (elem_idx >= nope_dim)
        {
            int const rope_idx = elem_idx - nope_dim;
            int const d = rope_idx >> 1;
            float const cos_v = cos_sin_table[pos_id * rope_dim + d];
            float const sin_v = cos_sin_table[pos_id * rope_dim + half_rope + d];
            float const x_even = v[i];
            float const x_odd  = v[i + 1];
            v[i]     = x_even * cos_v - x_odd * sin_v;
            v[i + 1] = x_odd  * cos_v + x_even * sin_v;
        }
    }

    // ================================================================
    // Step 5: Hadamard butterfly (fp32 precision throughout)
    //
    // Phase A — local (stride < VEC): register-only, no communication
    // Phase B — shuffle (VEC..31*VEC): __shfl_xor per element
    // Phase C — smem   (>=32*VEC):    XOR-swizzled shared memory (P1)
    // ================================================================

    // Phase A: local butterfly
#pragma unroll
    for (int stride = 1; stride < VEC; stride <<= 1)
    {
#pragma unroll
        for (int i = 0; i < VEC; i++)
        {
            if ((i & stride) == 0)
            {
                float a = v[i], b = v[i ^ stride];
                v[i]          = a + b;
                v[i ^ stride] = a - b;
            }
        }
    }

    // Phase B: warp shuffle butterfly
    if constexpr (NTHRD > 1)
    {
        constexpr int SHFL_END = (NTHRD <= 32) ? NTHRD : 32;
#pragma unroll
        for (int ts = 1; ts < SHFL_END; ts <<= 1)
        {
            int const stride = ts * VEC;
#pragma unroll
            for (int i = 0; i < VEC; i++)
            {
                float partner = __shfl_xor_sync(0xFFFFFFFF, v[i], ts);
                int const elem_idx = tid * VEC + i;
                v[i] = (elem_idx & stride)
                    ? (partner - v[i])
                    : (v[i] + partner);
            }
        }
    }

    // Phase C: cross-warp via shared memory (P1: XOR-swizzled for bank-conflict-free access)
    if constexpr (NTHRD > 32)
    {
#pragma unroll
        for (int i = 0; i < VEC; i++)
        {
            int const idx = tid * VEC + i;
            smem[idx ^ ((idx >> 3) & 0x1F)] = v[i];
        }
        __syncthreads();

        for (int stride = 32 * VEC; stride < HEAD_DIM; stride <<= 1)
        {
#pragma unroll
            for (int i = 0; i < VEC; i++)
            {
                int const idx = tid * VEC + i;
                int const partner_idx = idx ^ stride;
                float const a = smem[idx ^ ((idx >> 3) & 0x1F)];
                float const b = smem[partner_idx ^ ((partner_idx >> 3) & 0x1F)];
                v[i] = (idx & stride) ? (b - a) : (a + b);
            }
            __syncthreads();
#pragma unroll
            for (int i = 0; i < VEC; i++)
            {
                int const idx = tid * VEC + i;
                smem[idx ^ ((idx >> 3) & 0x1F)] = v[i];
            }
            __syncthreads();
        }
    }

    // Scale fused into final store conversion
    float const had_scale = rsqrtf(static_cast<float>(HEAD_DIM));

    // ================================================================
    // Step 6: Vectorized store — pack floats → ElemT, wide write
    // ================================================================
    int const start_pos = start_pos_arr[batch_idx];
    int const cache_pos = start_pos + local_output_idx;
    int const logical_block = cache_pos / tokens_per_block;
    int const token_offset = cache_pos % tokens_per_block;
    int const phys_block = block_offsets[batch_idx * max_blocks + logical_block];

    VecT raw_out;
    ElemT* out_elems = reinterpret_cast<ElemT*>(&raw_out);
#pragma unroll
    for (int i = 0; i < VEC; i++)
        out_elems[i] = static_cast<ElemT>(v[i] * had_scale);

    int const vec_col = token_offset * (HEAD_DIM / VEC) + tid;
    reinterpret_cast<VecT*>(
        reinterpret_cast<ElemT*>(kv_cache) + static_cast<int64_t>(phys_block) * cache_stride_blk
    )[vec_col] = raw_out;
}

// Explicit instantiations
#define INST_FUSED(HD, EB) \
    template __global__ void fusedPostProcessScatterKernel<HD, EB>( \
        void const*, void const*, float, \
        float const*, int32_t const*, int, int, \
        void*, int32_t const*, int32_t const*, int32_t const*, int32_t const*, \
        int, int, int, int, int);

INST_FUSED(128, 2)
INST_FUSED(128, 4)
INST_FUSED(512, 2)
INST_FUSED(512, 4)
#undef INST_FUSED


// ============================================================================
// Launch Wrapper: Fused PostProcess + Scatter
// ============================================================================

static inline int fusedNthreads(int head_dim, int elem_bytes)
{
    int max_vec = 16 / elem_bytes;
    int vec = (head_dim / max_vec >= 32) ? max_vec : (head_dim / 32);
    return head_dim / vec;
}

void fusedPostProcessScatterLaunch(
    void const* kv_comp,
    void const* rms_weight,
    float rms_eps,
    float const* cos_sin_table,
    int32_t const* position_ids,
    int nope_dim,
    int rope_dim,
    void* kv_cache,
    int32_t const* num_outputs,
    int32_t const* cu_kv_comp,
    int32_t const* start_pos,
    int32_t const* block_offsets,
    int batch_size,
    int tokens_per_block,
    int head_dim,
    int max_blocks_per_seq,
    int cache_stride_blk,
    int elem_bytes,
    int total_tokens,
    cudaStream_t stream)
{
    if (total_tokens <= 0)
        return;

    int const smem_bytes = head_dim * sizeof(float);
    int const nthreads = fusedNthreads(head_dim, elem_bytes);

#define LAUNCH_FUSED(HD, EB)                                                                    \
    fusedPostProcessScatterKernel<HD, EB><<<total_tokens, nthreads, smem_bytes, stream>>>(      \
        kv_comp, rms_weight, rms_eps,                                                           \
        cos_sin_table, position_ids, nope_dim, rope_dim,                                        \
        kv_cache, num_outputs, cu_kv_comp, start_pos, block_offsets,                            \
        batch_size, tokens_per_block, max_blocks_per_seq, cache_stride_blk,                     \
        total_tokens)

    if (elem_bytes == 4)
    {
        switch (head_dim)
        {
        case 128: LAUNCH_FUSED(128, 4); break;
        default:  LAUNCH_FUSED(512, 4); break;
        }
    }
    else
    {
        switch (head_dim)
        {
        case 128: LAUNCH_FUSED(128, 2); break;
        default:  LAUNCH_FUSED(512, 2); break;
        }
    }

#undef LAUNCH_FUSED
}

} // namespace kernels::compressor

TRTLLM_NAMESPACE_END
