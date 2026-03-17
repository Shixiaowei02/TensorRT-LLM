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

// ============================================================================
// Compressor Kernels — Mewtwo KV Cache Compression
// ============================================================================
//
// This file implements CUDA kernels for KV cache compression in the Mewtwo
// sparse attention system. The compressor reduces sequences of input tokens
// into fewer compressed tokens via learned weighted averaging (online softmax),
// then post-processes and scatters results into a paged KV cache.
//
// Three kernels are provided:
//
//   1. pagedKvCompressKernel  — Decode path (single/few new tokens per batch).
//      Loads prior compressor state from paged memory, performs online softmax
//      with the new token(s), writes updated state back, and emits a compressed
//      output token when compress_ratio tokens have been accumulated.
//
//   2. prefillReductionKernel — Prefill path (many tokens per batch).
//      Processes full chunks of compress_ratio tokens in one shot via online
//      softmax reduction over the input sequence. Also saves compressor state
//      for any remainder tokens that don't form a complete chunk.
//
//   3. postProcessScatterKernel — Fused post-processing + paged cache write.
//      Takes compressed output tokens and applies: RMSNorm → RoPE → Hadamard
//      transform → scatter to paged KV cache. Supports three cache modes:
//      bf16/fp32 (default), FP8 per-tensor, and FP8 per-128-element blockwise.
//      Keeps all intermediate values in float32 registers to avoid extra DRAM
//      round-trips.
//
// Vectorization strategy:
//   All kernels use 128-bit vectorized loads/stores (float4 / 8×bf16).
//   VEC = number of elements per thread, chosen so that NTHRD = HEAD_DIM/VEC >= 32.
//   For HEAD_DIM=128, bf16: VEC=4, NTHRD=32.  For HEAD_DIM=512, bf16: VEC=8, NTHRD=64.
//
// Overlap mode (compress_ratio=4):
//   When enabled, state_dim = 2*head_dim and the compressor uses overlapping
//   windows: each compressed output is derived from both the previous and current
//   chunk of compress_ratio tokens (previous chunk → first head_dim features,
//   current chunk → second head_dim features). This doubles the state stored
//   per position but improves compression quality.
//
// Template parameters:
//   HEAD_DIM   — Head dimension (128 or 512)
//   ELEM_BYTES — Element size in bytes (2=bf16, 4=fp32)
//   CACHE_MODE — Output quantization for postProcessScatterKernel
// ============================================================================

#include "tensorrt_llm/kernels/compressorKernels/compressorKernels.h"

#include "tensorrt_llm/common/assert.h"
#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <type_traits>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::compressor
{

// ============================================================================
// Helper functions
// ============================================================================

// Full-warp butterfly reductions via __shfl_xor_sync (all 32 lanes participate).
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

// Runtime-dispatched element load (bf16 or fp32 → float). Used in the decode
// kernel where elem_bytes is a runtime parameter from paged state buffers.
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
template <int V>
struct VecType;

template <>
struct VecType<4>
{
    using type = unsigned int;
}; //  32-bit: 2 bf16 or 1 fp32

template <>
struct VecType<8>
{
    using type = uint2;
}; //  64-bit: 4 bf16 or 2 fp32

template <>
struct VecType<16>
{
    using type = uint4;
}; // 128-bit: 8 bf16 or 4 fp32

// Cache output mode for postProcessScatterKernel.
enum class CacheMode
{
    kDefault = 0,      // bf16/fp32 store (same dtype as input)
    kFP8PerTensor = 1, // FP8 E4M3 with static scale=1.0
    kFP8Blockwise = 2  // FP8 E4M3 with per-128-element blockwise scales
};

// ============================================================================
// Decode Kernel: pagedKvCompressKernel
//
// Template: <HEAD_DIM, IO_ELEM_BYTES, NEXT_N>
//   NEXT_N: number of new tokens per sequence in this decode step (1-4)
//
// Grid:  (batch_size) — one block per batch element
// Block: (NTHRD) where NTHRD = HEAD_DIM / VEC (>= 32 threads)
//
// Algorithm per batch element:
//   For each new token in the decode step:
//     1. Load existing compressor state (partial kv/score) from paged cache
//     2. Perform online softmax: accumulate new token's contribution using
//        the numerically stable running max + weighted sum formulation
//     3. Write updated state back to paged cache
//     4. If compress_ratio tokens accumulated → emit compressed output,
//        reset state for next compression window
//
// Each thread handles VEC contiguous elements of head_dim. In overlap mode
// (state_dim = 2*head_dim), Phase 1 iterates over 2 column halves.
//
// Memory layout:
//   kv_score:   [total_tokens, 2 * state_dim] — interleaved KV and score projections
//   paged_kv:   paged cache for compressor KV state
//   paged_score: paged cache for compressor score state (with APE bias)
//   output:     [total_comp_tokens, head_dim] — compressed output tokens
// ============================================================================

// Helper: vectorized online softmax step reading from paged KV/score state.
// Loads one position's KV and score from paged memory, adds the learnable APE
// (absolute positional encoding) bias to the score, and updates the running
// online softmax accumulators (rmax, rsum, rwsum) per element.
template <int HEAD_DIM, int IO_ELEM_BYTES>
__device__ __forceinline__ void decodeSoftmaxVec(void const* __restrict__ paged_kv_raw,
    void const* __restrict__ paged_score_raw, float const* __restrict__ ape,
    int64_t page_sd, // page_size * state_dim (in elements)
    int state_dim,
    int phys_kv,     // physical page index for kv
    int phys_sc,     // physical page index for score
    int blk_off,     // offset within page
    int kv_col_off,  // column offset (0 or HEAD_DIM)
    int ape_row,     // r * state_dim + col_off
    int tid, float* __restrict__ rmax, float* __restrict__ rsum, float* __restrict__ rwsum)
{
    using IoElemT = typename std::conditional<IO_ELEM_BYTES == 2, __nv_bfloat16, float>::type;
    constexpr int MAX_VEC = 16 / IO_ELEM_BYTES;
    constexpr int VEC = (HEAD_DIM / MAX_VEC >= 32) ? MAX_VEC : (HEAD_DIM / 32);
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
        float kf[4] = {static_cast<float>(ke[i]), static_cast<float>(ke[i + 1]), static_cast<float>(ke[i + 2]),
            static_cast<float>(ke[i + 3])};
        // score + APE bias
        float sf[4] = {static_cast<float>(se[i]) + av.x, static_cast<float>(se[i + 1]) + av.y,
            static_cast<float>(se[i + 2]) + av.z, static_cast<float>(se[i + 3]) + av.w};
        // Online softmax: maintain running (max, sum_exp, weighted_sum) per element.
        // nm = new max, sc_f = rescale factor for old accumulators, tm = exp(score - new_max).
        // Final output: rwsum / rsum = weighted average of KV values.
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            float nm = fmaxf(rmax[i + j], sf[j]);
            float sc_f = expf(rmax[i + j] - nm);
            float tm = expf(sf[j] - nm);
            rsum[i + j] = rsum[i + j] * sc_f + tm;
            rwsum[i + j] = rwsum[i + j] * sc_f + kf[j] * tm;
            rmax[i + j] = nm;
        }
    }
}

template <int HEAD_DIM, int IO_ELEM_BYTES, int NEXT_N>
__global__ void pagedKvCompressKernel(void const* __restrict__ kv_score_raw, float const* __restrict__ ape,
    void* __restrict__ paged_kv_raw, void* __restrict__ paged_score_raw, int32_t const* __restrict__ block_table_kv,
    int32_t const* __restrict__ block_table_score, void* __restrict__ output_raw, int32_t const* __restrict__ kv_lens,
    int32_t const* __restrict__ start_pos_arr, int32_t const* __restrict__ cu_seq_lens,
    int32_t const* __restrict__ cu_kv_comp, bool* __restrict__ compressed_mask, int page_size, int state_dim,
    int compress_ratio, bool is_overlap, int max_blocks, int out_elem_bytes)
{
    using IoElemT = typename std::conditional<IO_ELEM_BYTES == 2, __nv_bfloat16, float>::type;
    constexpr int MAX_VEC = 16 / IO_ELEM_BYTES;
    constexpr int VEC = (HEAD_DIM / MAX_VEC >= 32) ? MAX_VEC : (HEAD_DIM / 32);
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
    // Phase 1: Write NEXT_N new tokens' KV and score state to paged cache.
    //
    // Input layout: kv_score[token, 2*state_dim] where:
    //   kv_score[token, 0:state_dim]           = KV projection
    //   kv_score[token, state_dim:2*state_dim]  = score projection
    // Score gets the learnable APE bias added before storing. KV is stored as-is.
    // In overlap mode (coff=2), iterates over 2 column halves of state_dim.
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
                    float4 av = *reinterpret_cast<float4 const*>(&ape[ape_idx * state_dim + col + tid * VEC + i]);
                    sc_o[i] = static_cast<IoElemT>(static_cast<float>(sc_e[i]) + av.x);
                    sc_o[i + 1] = static_cast<IoElemT>(static_cast<float>(sc_e[i + 1]) + av.y);
                    sc_o[i + 2] = static_cast<IoElemT>(static_cast<float>(sc_e[i + 2]) + av.z);
                    sc_o[i + 3] = static_cast<IoElemT>(static_cast<float>(sc_e[i + 3]) + av.w);
                }
                reinterpret_cast<IoVecT*>(&paged_score[dsc])[tid] = sc_out;
            }
        }
    }

    // ================================================================
    // Phase 2: Count how many complete compression windows finished.
    // compressed_mask[batch] = true if at least one compressed token was produced.
    // ================================================================
    int last_token_idx = sp + NEXT_N - 1;
    int num_compressions = (last_token_idx + 1) / compress_ratio - sp / compress_ratio;
    if (tid == 0)
        compressed_mask[batch_idx] = (num_compressions > 0);

    // ================================================================
    // Phase 3: Online softmax reduction over each complete chunk.
    //
    // For each completed compression window, reads compress_ratio positions
    // from paged state and reduces them via online softmax:
    //   output[d] = sum_r( kv[r,d] * softmax(score[r,d]) )
    // where softmax is computed per-element (not across head_dim) using
    // the numerically stable running-max formulation.
    //
    // In overlap mode, the previous chunk's first-half features and the
    // current chunk's second-half features are combined into one output.
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
            rmax[i] = -INFINITY;
            rsum[i] = 0.0f;
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

                decodeSoftmaxVec<HEAD_DIM, IO_ELEM_BYTES>(paged_kv_raw, paged_score_raw, ape, page_sd, state_dim,
                    phys_kv, phys_sc, blk_off, 0, r * state_dim, tid, rmax, rsum, rwsum);
            }

            // Current chunk: second head_dim features (col_off=HEAD_DIM)
            for (int r = 0; r < compress_ratio; r++)
            {
                int pos = curr_chunk_start + r;
                int log_blk = pos / page_size;
                int blk_off = pos % page_size;
                int phys_kv = block_table_kv[batch_idx * max_blocks + log_blk];
                int phys_sc = block_table_score[batch_idx * max_blocks + log_blk];

                decodeSoftmaxVec<HEAD_DIM, IO_ELEM_BYTES>(paged_kv_raw, paged_score_raw, ape, page_sd, state_dim,
                    phys_kv, phys_sc, blk_off, HEAD_DIM, r * state_dim + HEAD_DIM, tid, rmax, rsum, rwsum);
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

                decodeSoftmaxVec<HEAD_DIM, IO_ELEM_BYTES>(paged_kv_raw, paged_score_raw, ape, page_sd, state_dim,
                    phys_kv, phys_sc, blk_off, 0, r * state_dim, tid, rmax, rsum, rwsum);
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
            *reinterpret_cast<OutVecT*>(&reinterpret_cast<__nv_bfloat16*>(output_raw)[out_base])
                = *reinterpret_cast<OutVecT const*>(packed);
        }
        else
        {
            float result[VEC];
#pragma unroll
            for (int i = 0; i < VEC; i++)
                result[i] = rwsum[i] / rsum[i];
#pragma unroll
            for (int i = 0; i < VEC; i += 4)
                *reinterpret_cast<float4*>(&reinterpret_cast<float*>(output_raw)[out_base + i])
                    = *reinterpret_cast<float4 const*>(&result[i]);
        }
    }
}

// Explicit instantiations for decode kernel
#define INST_DECODE(HD, EB, NN)                                                                                        \
    template __global__ void pagedKvCompressKernel<HD, EB, NN>(void const*, float const*, void*, void*,                \
        int32_t const*, int32_t const*, void*, int32_t const*, int32_t const*, int32_t const*, int32_t const*, bool*,  \
        int, int, int, bool, int, int);

#define INST_DECODE_NN(HD, EB)                                                                                         \
    INST_DECODE(HD, EB, 1) INST_DECODE(HD, EB, 2) INST_DECODE(HD, EB, 3) INST_DECODE(HD, EB, 4)

INST_DECODE_NN(128, 2)
INST_DECODE_NN(128, 4)
INST_DECODE_NN(512, 2)
INST_DECODE_NN(512, 4)
#undef INST_DECODE_NN
#undef INST_DECODE

// ============================================================================
// Decode Launch Wrapper
//
// Dispatches to the correct template instantiation based on head_dim, elem_bytes,
// and next_n (number of new tokens per decode step, capped at 4).
// Reuses prefillNthreads for thread count calculation (same VEC/NTHRD logic).
// ============================================================================

// Forward declaration (defined in prefill section below).
static inline int prefillNthreads(int head_dim, int io_elem_bytes);

void pagedKvCompressLaunch(void const* kv_score, float const* ape, void* paged_kv, void* paged_score,
    int32_t const* block_table_kv, int32_t const* block_table_score, void* output, int32_t const* kv_lens,
    int32_t const* start_pos, int32_t const* cu_seq_lens, int32_t const* cu_kv_comp, bool* compressed_mask,
    int batch_size, int page_size, int max_blocks, int head_dim, int compress_ratio, bool is_overlap, int next_n,
    int io_elem_bytes, int out_elem_bytes, cudaStream_t stream)
{
    int const coff = is_overlap ? 2 : 1;
    int const state_dim = coff * head_dim;
    int const nthreads = prefillNthreads(head_dim, io_elem_bytes);
    dim3 grid(batch_size);

#define LAUNCH_DECODE(HD, EB, NN)                                                                                      \
    pagedKvCompressKernel<HD, EB, NN><<<grid, nthreads, 0, stream>>>(kv_score, ape, paged_kv, paged_score,             \
        block_table_kv, block_table_score, output, kv_lens, start_pos, cu_seq_lens, cu_kv_comp, compressed_mask,       \
        page_size, state_dim, compress_ratio, is_overlap, max_blocks, out_elem_bytes)

#define DISPATCH_NN(HD, EB)                                                                                            \
    switch (next_n)                                                                                                    \
    {                                                                                                                  \
    case 1: LAUNCH_DECODE(HD, EB, 1); break;                                                                           \
    case 2: LAUNCH_DECODE(HD, EB, 2); break;                                                                           \
    case 3: LAUNCH_DECODE(HD, EB, 3); break;                                                                           \
    default: LAUNCH_DECODE(HD, EB, 4); break;                                                                          \
    }

    if (io_elem_bytes == 4)
    {
        if (head_dim == 512)
        {
            DISPATCH_NN(512, 4);
        }
        else
        {
            DISPATCH_NN(128, 4);
        }
    }
    else
    {
        if (head_dim == 512)
        {
            DISPATCH_NN(512, 2);
        }
        else
        {
            DISPATCH_NN(128, 2);
        }
    }

#undef DISPATCH_NN
#undef LAUNCH_DECODE
}

// ============================================================================
// Prefill Kernel: prefillReductionKernel
//
// Template: <HEAD_DIM, IO_ELEM_BYTES>
//
// Grid:  (batch_size, max_outputs_per_batch) — one block per compressed output
// Block: (NTHRD) where NTHRD = HEAD_DIM / VEC (>= 32 threads)
//
// Unlike the decode kernel (which operates token-by-token from paged state),
// the prefill kernel processes the full input sequence at once. Each block
// reads compress_ratio consecutive input rows from kv_score and reduces them
// via online softmax to produce one compressed output.
//
// The last block (local_output_idx == num_outputs - 1) also handles saving
// compressor state for any remainder tokens that don't form a full chunk.
// This state is written to paged kv/score caches for use in future decode steps.
//
// Memory layout:
//   kv_score:    [total_tokens, 2*state_dim] — interleaved KV and score from linear projection
//   paged_kv:    paged cache for compressor state (remainder)
//   paged_score: paged cache for compressor score state (remainder, with APE)
//   output:      [total_comp_tokens, head_dim] — compressed output tokens
// ============================================================================

// Per-element online softmax step on VEC elements via 128-bit vectorized loads.
// Reads directly from the kv_score input buffer (not paged state) since prefill
// has the full sequence available.
template <int HEAD_DIM, int IO_ELEM_BYTES>
__device__ __forceinline__ void prefillSoftmaxVec(void const* __restrict__ kv_score_raw, float const* __restrict__ ape,
    int64_t row_elem, // (input_offset + row_idx) * two_sd
    int kv_col_off,   // column offset into kv_score row (0 or HEAD_DIM)
    int ape_base,     // r * state_dim + ape_col_off
    int state_dim, int tid, float* __restrict__ rmax, float* __restrict__ rsum, float* __restrict__ rwsum)
{
    using IoElemT = typename std::conditional<IO_ELEM_BYTES == 2, __nv_bfloat16, float>::type;
    constexpr int MAX_VEC = 16 / IO_ELEM_BYTES;
    constexpr int VEC = (HEAD_DIM / MAX_VEC >= 32) ? MAX_VEC : (HEAD_DIM / 32);
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
        float kf[4] = {static_cast<float>(ke[i]), static_cast<float>(ke[i + 1]), static_cast<float>(ke[i + 2]),
            static_cast<float>(ke[i + 3])};
        float sf[4] = {static_cast<float>(se[i]) + av.x, static_cast<float>(se[i + 1]) + av.y,
            static_cast<float>(se[i + 2]) + av.z, static_cast<float>(se[i + 3]) + av.w};
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            float nm = fmaxf(rmax[i + j], sf[j]);
            float sc = expf(rmax[i + j] - nm);
            float tm = expf(sf[j] - nm);
            rsum[i + j] = rsum[i + j] * sc + tm;
            rwsum[i + j] = rwsum[i + j] * sc + kf[j] * tm;
            rmax[i + j] = nm;
        }
    }
}

template <int HEAD_DIM, int IO_ELEM_BYTES>
__global__ void prefillReductionKernel(void const* __restrict__ kv_score_raw, float const* __restrict__ ape,
    void* __restrict__ paged_kv_raw, void* __restrict__ paged_score_raw, int32_t const* __restrict__ block_table_kv,
    int32_t const* __restrict__ block_table_score, void* __restrict__ output_raw, int32_t const* __restrict__ kv_lens,
    int32_t const* __restrict__ start_pos_arr, int32_t const* __restrict__ cu_seq_lens,
    int32_t const* __restrict__ cu_kv_comp, bool* __restrict__ compressed_mask, int page_size, int state_dim,
    int compress_ratio, bool is_overlap, int max_blocks, int out_elem_bytes)
{
    using IoElemT = typename std::conditional<IO_ELEM_BYTES == 2, __nv_bfloat16, float>::type;

    constexpr int MAX_VEC = 16 / IO_ELEM_BYTES;
    constexpr int VEC = (HEAD_DIM / MAX_VEC >= 32) ? MAX_VEC : (HEAD_DIM / 32);
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
    //
    // Save compressor state for tokens that don't complete a full chunk.
    // In overlap mode, also save the last full chunk (needed as the
    // "previous segment" for the next decode step's overlap window).
    // Uses vectorized 128-bit loads/stores for paged KV and score state.
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
                        float4 av = *reinterpret_cast<float4 const*>(&ape[r * state_dim + col + tid * VEC + i]);
                        sc_o[i] = static_cast<IoElemT>(static_cast<float>(sc_e[i]) + av.x);
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
                    float4 av = *reinterpret_cast<float4 const*>(&ape[r * state_dim + col + tid * VEC + i]);
                    sc_o[i] = static_cast<IoElemT>(static_cast<float>(sc_e[i]) + av.x);
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
    //
    // Each block reduces compress_ratio rows into one output via per-element
    // online softmax: output[d] = sum_r(kv[r,d] * softmax(score[r,d] + ape[r,d]))
    // In overlap mode, combines previous chunk's first-half and current chunk's
    // second-half features (same as decode kernel).
    // ================================================================
    if (!should_compress)
        return;

    float rmax[VEC], rsum[VEC], rwsum[VEC];
#pragma unroll
    for (int i = 0; i < VEC; i++)
    {
        rmax[i] = -INFINITY;
        rsum[i] = 0.0f;
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
                    kv_score_raw, ape, row, 0, r * state_dim, state_dim, tid, rmax, rsum, rwsum);
            }
        }

        // Current segment (second head_dim features)
        int cur_start = local_output_idx * compress_ratio;
        for (int r = 0; r < compress_ratio; r++)
        {
            int64_t row = static_cast<int64_t>(input_offset + cur_start + r) * two_sd;
            prefillSoftmaxVec<HEAD_DIM, IO_ELEM_BYTES>(
                kv_score_raw, ape, row, HEAD_DIM, r * state_dim + HEAD_DIM, state_dim, tid, rmax, rsum, rwsum);
        }
    }
    else
    {
        int input_start = local_output_idx * compress_ratio;
        for (int r = 0; r < compress_ratio; r++)
        {
            int64_t row = static_cast<int64_t>(input_offset + input_start + r) * two_sd;
            prefillSoftmaxVec<HEAD_DIM, IO_ELEM_BYTES>(
                kv_score_raw, ape, row, 0, r * state_dim, state_dim, tid, rmax, rsum, rwsum);
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
        *reinterpret_cast<OutVecT*>(&reinterpret_cast<__nv_bfloat16*>(output_raw)[out_base])
            = *reinterpret_cast<OutVecT const*>(packed);
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
            *reinterpret_cast<float4*>(&reinterpret_cast<float*>(output_raw)[out_base + i])
                = *reinterpret_cast<float4 const*>(&result[i]);
    }
}

// Explicit instantiations
#define INST_PREFILL(HD, EB)                                                                                           \
    template __global__ void prefillReductionKernel<HD, EB>(void const*, float const*, void*, void*, int32_t const*,   \
        int32_t const*, void*, int32_t const*, int32_t const*, int32_t const*, int32_t const*, bool*, int, int, int,   \
        bool, int, int);

INST_PREFILL(128, 2)
INST_PREFILL(128, 4)
INST_PREFILL(512, 2)
INST_PREFILL(512, 4)
#undef INST_PREFILL

// ============================================================================
// Prefill Launch Wrapper
//
// Grid is (batch_size, max(max_outputs, 1)). Blocks for local_output_idx >= num_outputs
// early-exit inside the kernel.
// ============================================================================

// Compute threads per block: mirrors compile-time NTHRD = HEAD_DIM / VEC.
static inline int prefillNthreads(int head_dim, int io_elem_bytes)
{
    int max_vec = 16 / io_elem_bytes;
    int vec = (head_dim / max_vec >= 32) ? max_vec : (head_dim / 32);
    return head_dim / vec;
}

void prefillReductionLaunch(void const* kv_score, float const* ape, void* paged_kv, void* paged_score,
    int32_t const* block_table_kv, int32_t const* block_table_score, void* output, int32_t const* kv_lens,
    int32_t const* start_pos, int32_t const* cu_seq_lens, int32_t const* cu_kv_comp, bool* compressed_mask,
    int batch_size, int page_size, int max_blocks, int head_dim, int compress_ratio, bool is_overlap, int max_outputs,
    int io_elem_bytes, int out_elem_bytes, cudaStream_t stream)
{
    int const nthreads = prefillNthreads(head_dim, io_elem_bytes);
    int const coff = is_overlap ? 2 : 1;
    int const state_dim = coff * head_dim;
    dim3 grid(batch_size, max(max_outputs, 1));

#define LAUNCH_PREFILL(HD, EB)                                                                                         \
    prefillReductionKernel<HD, EB><<<grid, nthreads, 0, stream>>>(kv_score, ape, paged_kv, paged_score,                \
        block_table_kv, block_table_score, output, kv_lens, start_pos, cu_seq_lens, cu_kv_comp, compressed_mask,       \
        page_size, state_dim, compress_ratio, is_overlap, max_blocks, out_elem_bytes)

    if (io_elem_bytes == 4)
    {
        if (head_dim == 512)
        {
            LAUNCH_PREFILL(512, 4);
        }
        else
        {
            LAUNCH_PREFILL(128, 4);
        }
    }
    else
    {
        if (head_dim == 512)
        {
            LAUNCH_PREFILL(512, 2);
        }
        else
        {
            LAUNCH_PREFILL(128, 2);
        }
    }

#undef LAUNCH_PREFILL
}

// ============================================================================
// Postprocess + Scatter Kernel: postProcessScatterKernel
//
// Template: <HEAD_DIM, ELEM_BYTES, CACHE_MODE>
//
// Grid:  (total_tokens) — one block per compressed token
// Block: (NTHRD = HEAD_DIM / VEC) threads, always >= 32
// Smem:  HEAD_DIM * sizeof(float) — used for cross-warp Hadamard butterfly
//
// This kernel fuses all post-compression processing with the paged cache write
// into a single kernel launch, keeping data in float32 registers throughout.
// This eliminates the DRAM round-trip that a split postprocess+scatter would need.
//
// Pipeline (10 steps, all in fp32 registers):
//   1. Vectorized load compressed token from kv_comp
//   2. RMSNorm: compute sum-of-squares → cross-warp reduce → rsqrt → scale
//   3. Apply RMSNorm weights
//   4. RoPE: interleaved even/odd rotation on rope_dim elements (skip nope_dim)
//   5. Hadamard butterfly transform (3 phases: local → warp shuffle → shared mem)
//   6. Scale by 1/sqrt(HEAD_DIM) (Hadamard normalization)
//   7. Optionally write postprocessed result to kv_out (for callers that need it)
//   8. Binary search cu_kv_comp to find batch_idx for this token
//   9. Compute paged cache destination (logical→physical block via block table)
//  10. Store to cache with CacheMode-specific quantization:
//      - kDefault:       float→bf16/fp32 vectorized store
//      - kFP8PerTensor:  float→fp8_e4m3 (scale=1.0) vectorized byte store
//      - kFP8Blockwise:  per-128-element amax reduction via warp shuffle →
//                         compute scale → quantize → store fp8 data + fp32 scales
//                         Optionally also writes to fp8_output/scale_output buffers
//                         (used by indexer compressor that returns FP8 data to caller)
// ============================================================================

template <int HEAD_DIM, int ELEM_BYTES, CacheMode CACHE_MODE = CacheMode::kDefault, bool ROTATE_ACTIVATION = true>
__global__ void postProcessScatterKernel(void const* __restrict__ kv_comp, // [total_tokens, head_dim] input
    void* __restrict__ kv_out,                // [total_tokens, head_dim] postprocessed output (may be nullptr)
    void const* __restrict__ rms_weight,      // [head_dim]
    float rms_eps,
    float const* __restrict__ cos_sin_table,  // [max_pos, 2, rope_dim/2]
    int32_t const* __restrict__ position_ids, // [total_tokens]
    int nope_dim, int rope_dim,
    // scatter params
    void* __restrict__ kv_cache, // paged cache buffer
    int32_t const* __restrict__ num_outputs_arr, int32_t const* __restrict__ cu_kv_comp,
    int32_t const* __restrict__ start_pos_arr, int32_t const* __restrict__ block_offsets, int batch_size,
    int tokens_per_block, int max_blocks, int cache_stride_blk_bytes, int total_tokens, int num_scale_blocks,
    void* __restrict__ fp8_output, float* __restrict__ scale_output)
{
    using ElemT = typename std::conditional<ELEM_BYTES == 2, __nv_bfloat16, float>::type;
    constexpr int MAX_VEC = 16 / ELEM_BYTES;
    constexpr int VEC = (HEAD_DIM / MAX_VEC >= 32) ? MAX_VEC : (HEAD_DIM / 32);
    constexpr int NTHRD = HEAD_DIM / VEC;
    constexpr int VEC_BYTES = VEC * ELEM_BYTES;
    using VecT = typename VecType<VEC_BYTES>::type;

    int const token_idx = blockIdx.x;
    if (token_idx >= total_tokens)
        return;

    int const tid = threadIdx.x;
    extern __shared__ float smem[];

    // ================================================================
    // Step 1: Vectorized load from kv_comp
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
    // Step 2: RMSNorm
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
    // Step 3: Load weight, apply RMSNorm
    // ================================================================
    auto const* wt_src = reinterpret_cast<VecT const*>(reinterpret_cast<ElemT const*>(rms_weight));
    VecT raw_w = wt_src[tid];
    ElemT const* w_elems = reinterpret_cast<ElemT const*>(&raw_w);

#pragma unroll
    for (int i = 0; i < VEC; i++)
        v[i] = v[i] * rms_scale * static_cast<float>(w_elems[i]);

    // ================================================================
    // Step 4: RoPE (Rotary Positional Embedding)
    //
    // Applied only to elements in [nope_dim, nope_dim+rope_dim).
    // Uses interleaved even/odd pairs: (x_even, x_odd) → rotated by (cos, sin).
    // cos_sin_table layout: [max_pos, rope_dim] where first half is cos, second is sin.
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
            float const x_odd = v[i + 1];
            v[i] = x_even * cos_v - x_odd * sin_v;
            v[i + 1] = x_odd * cos_v + x_even * sin_v;
        }
    }

    // ================================================================
    // Step 5: Hadamard butterfly transform (rotate activation)
    //
    // Implements the Walsh-Hadamard transform H_n * v via butterfly network.
    // H_n has the recursive structure: H_n = [[H_{n/2}, H_{n/2}], [H_{n/2}, -H_{n/2}]]
    // which decomposes into log2(HEAD_DIM) butterfly stages.
    //
    // Three phases handle increasing stride lengths:
    //   A) Local: strides < VEC — within each thread's register file
    //   B) Warp shuffle: strides VEC..32*VEC-1 — via __shfl_xor_sync
    //   C) Shared memory: strides >= 32*VEC — via XOR-swizzled smem
    //
    // The XOR swizzle pattern `idx ^ ((idx >> 3) & 0x1F)` ensures bank-conflict-
    // free access to shared memory across all butterfly stride patterns.
    //
    // Skipped entirely when ROTATE_ACTIVATION=false.
    // ================================================================
    if constexpr (ROTATE_ACTIVATION)
    {

        // Phase A: local butterfly (strides 1..VEC-1, within each thread's VEC registers)
#pragma unroll
        for (int stride = 1; stride < VEC; stride <<= 1)
        {
#pragma unroll
            for (int i = 0; i < VEC; i++)
            {
                if ((i & stride) == 0)
                {
                    float a = v[i], b = v[i ^ stride];
                    v[i] = a + b;
                    v[i ^ stride] = a - b;
                }
            }
        }

        // Phase B: warp shuffle butterfly (strides VEC..32*VEC-1, within a single warp)
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
                    v[i] = (elem_idx & stride) ? (partner - v[i]) : (v[i] + partner);
                }
            }
        }

        // Phase C: cross-warp butterfly via XOR-swizzled shared memory (strides >= 32*VEC)
        // Only needed when NTHRD > 32 (i.e., multiple warps, e.g., HEAD_DIM=512, VEC=8 → 64 threads)
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

        // ================================================================
        // Step 6: Scale by Hadamard factor
        // ================================================================
        float const had_scale = rsqrtf(static_cast<float>(HEAD_DIM));

#pragma unroll
        for (int i = 0; i < VEC; i++)
            v[i] *= had_scale;

    } // ROTATE_ACTIVATION

    // ================================================================
    // Step 7: Write postprocessed output to kv_out (if requested)
    // ================================================================
    if (kv_out != nullptr)
    {
        VecT raw_out;
        ElemT* out_elems = reinterpret_cast<ElemT*>(&raw_out);
#pragma unroll
        for (int i = 0; i < VEC; i++)
            out_elems[i] = static_cast<ElemT>(v[i]);

        auto* dst
            = reinterpret_cast<VecT*>(reinterpret_cast<ElemT*>(kv_out) + static_cast<int64_t>(token_idx) * HEAD_DIM);
        dst[tid] = raw_out;
    }

    // ================================================================
    // Step 8: Binary search cu_kv_comp to find which batch element this token belongs to.
    // cu_kv_comp[b] = cumulative count of compressed tokens for batches 0..b-1.
    // After search: batch_idx is the owning batch, local_output_idx is the
    // position within that batch's compressed outputs.
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
    // Step 9: Compute paged cache destination address.
    // Map (batch_idx, local_output_idx) → logical block → physical block
    // via the block table. block_base points to the start of the physical
    // page; token_offset is the slot within that page.
    // ================================================================
    int const start_pos = start_pos_arr[batch_idx];
    int const cache_pos = start_pos + local_output_idx;
    int const logical_block = cache_pos / tokens_per_block;
    int const token_offset = cache_pos % tokens_per_block;
    int const phys_block = block_offsets[batch_idx * max_blocks + logical_block];

    uint8_t* block_base
        = reinterpret_cast<uint8_t*>(kv_cache) + static_cast<int64_t>(phys_block) * cache_stride_blk_bytes;

    // ================================================================
    // Step 10: Store to cache (compile-time dispatch on CacheMode)
    //
    // Cache addressing is byte-based: block_base points to the start of
    // the physical block, cache_stride_blk_bytes is the total block size.
    // ================================================================
    if constexpr (CACHE_MODE == CacheMode::kDefault)
    {
        // Default mode: float→bf16/fp32 pack + vectorized store.
        // Cache layout per block: [tokens_per_block * HEAD_DIM] elements of ElemT.
        VecT raw_out;
        ElemT* out_elems = reinterpret_cast<ElemT*>(&raw_out);
#pragma unroll
        for (int i = 0; i < VEC; i++)
            out_elems[i] = static_cast<ElemT>(v[i]);

        ElemT* row_base = reinterpret_cast<ElemT*>(block_base) + token_offset * HEAD_DIM;
        reinterpret_cast<VecT*>(row_base)[tid] = raw_out;
    }
    else if constexpr (CACHE_MODE == CacheMode::kFP8PerTensor)
    {
        // FP8 per-tensor: direct float→fp8_e4m3fn cast (implicit scale=1.0).
        // Cache layout per block: [tokens_per_block * HEAD_DIM] bytes of fp8.
        uint8_t fp8_bytes[VEC];
#pragma unroll
        for (int i = 0; i < VEC; i++)
        {
            __nv_fp8_e4m3 fp8_val(v[i]);
            fp8_bytes[i] = *reinterpret_cast<uint8_t const*>(&fp8_val);
        }

        using Fp8VecT = typename VecType<VEC>::type;
        uint8_t* fp8_dst = block_base + token_offset * HEAD_DIM + tid * VEC;
        *reinterpret_cast<Fp8VecT*>(fp8_dst) = *reinterpret_cast<Fp8VecT const*>(fp8_bytes);
    }
    else // kFP8Blockwise
    {
        // FP8 blockwise: per-128-element quantization with explicit scales.
        // GROUP_SIZE = number of threads that share one scale factor.
        // For HD=512, VEC=8: GROUP_SIZE=16 threads → 128 elements per scale block.
        //
        // Cache layout per block:
        //   [fp8_data: tokens_per_block * HEAD_DIM bytes]
        //   [scales:   tokens_per_block * (HEAD_DIM/128) * 4 bytes]
        constexpr int GROUP_SIZE = 128 / VEC;

        // Step 10a: Compute per-group amax via warp shuffle reduction.
        // GROUP_SIZE <= 16 (< warp), so shuffle is sufficient (no smem needed).
        float local_amax = 0.f;
#pragma unroll
        for (int i = 0; i < VEC; i++)
            local_amax = fmaxf(local_amax, fabsf(v[i]));

#pragma unroll
        for (int offset = GROUP_SIZE / 2; offset > 0; offset >>= 1)
            local_amax = fmaxf(local_amax, __shfl_xor_sync(0xFFFFFFFF, local_amax, offset));

        // Step 10b: Compute scale and inverse scale for quantization.
        // 448.0 is the max representable value for fp8_e4m3fn.
        float const scale = local_amax / 448.0f;
        float const inv_scale = (local_amax > 0.f) ? (448.0f / local_amax) : 1.0f;

        // Step 10c: Quantize to FP8 and store data.
        uint8_t fp8_bytes[VEC];
#pragma unroll
        for (int i = 0; i < VEC; i++)
        {
            __nv_fp8_e4m3 fp8_val(v[i] * inv_scale);
            fp8_bytes[i] = *reinterpret_cast<uint8_t const*>(&fp8_val);
        }

        using Fp8VecT = typename VecType<VEC>::type;
        uint8_t* fp8_dst = block_base + token_offset * HEAD_DIM + tid * VEC;
        *reinterpret_cast<Fp8VecT*>(fp8_dst) = *reinterpret_cast<Fp8VecT const*>(fp8_bytes);

        // Step 10d: Store scale factor (one thread per 128-element group writes it).
        if (tid % GROUP_SIZE == 0)
        {
            int const scale_idx = tid / GROUP_SIZE;
            float* scale_dst = reinterpret_cast<float*>(block_base + tokens_per_block * HEAD_DIM
                + (token_offset * num_scale_blocks + scale_idx) * sizeof(float));
            *scale_dst = scale;
        }

        // Step 10e: Optionally write FP8 data and scales to output buffers.
        // Used by the indexer compressor which returns (fp8_data, scales) to Python
        // for downstream sparse attention indexing.
        if (fp8_output != nullptr)
        {
            uint8_t* fp8_out_dst
                = reinterpret_cast<uint8_t*>(fp8_output) + static_cast<int64_t>(token_idx) * HEAD_DIM + tid * VEC;
            *reinterpret_cast<Fp8VecT*>(fp8_out_dst) = *reinterpret_cast<Fp8VecT const*>(fp8_bytes);
        }
        if (scale_output != nullptr && tid % GROUP_SIZE == 0)
        {
            int const scale_idx = tid / GROUP_SIZE;
            scale_output[static_cast<int64_t>(token_idx) * num_scale_blocks + scale_idx] = scale;
        }
    }
}

// Explicit instantiations — fused postprocess+scatter.
// Default mode supports both bf16 (EB=2) and fp32 (EB=4) input types.
// FP8 modes only support bf16 input (EB=2) since the compressor output is always bf16.
// Each combination is instantiated with ROTATE_ACTIVATION=true and ROTATE_ACTIVATION=false.
#define INST_PPS(HD, EB, CM, AR)                                                                                       \
    template __global__ void postProcessScatterKernel<HD, EB, CM, AR>(void const*, void*, void const*, float,          \
        float const*, int32_t const*, int, int, void*, int32_t const*, int32_t const*, int32_t const*, int32_t const*, \
        int, int, int, int, int, int, void*, float*);

INST_PPS(128, 2, CacheMode::kDefault, true)
INST_PPS(128, 4, CacheMode::kDefault, true)
INST_PPS(512, 2, CacheMode::kDefault, true)
INST_PPS(512, 4, CacheMode::kDefault, true)
INST_PPS(128, 2, CacheMode::kFP8PerTensor, true)
INST_PPS(128, 2, CacheMode::kFP8Blockwise, true)
INST_PPS(512, 2, CacheMode::kFP8PerTensor, true)
INST_PPS(512, 2, CacheMode::kFP8Blockwise, true)
INST_PPS(128, 2, CacheMode::kDefault, false)
INST_PPS(128, 4, CacheMode::kDefault, false)
INST_PPS(512, 2, CacheMode::kDefault, false)
INST_PPS(512, 4, CacheMode::kDefault, false)
INST_PPS(128, 2, CacheMode::kFP8PerTensor, false)
INST_PPS(128, 2, CacheMode::kFP8Blockwise, false)
INST_PPS(512, 2, CacheMode::kFP8PerTensor, false)
INST_PPS(512, 2, CacheMode::kFP8Blockwise, false)
#undef INST_PPS

// ============================================================================
// Postprocess + Scatter Launch Wrapper
//
// Derives cache layout parameters (cache_stride_blk_bytes, num_scale_blocks)
// from cache_mode, then dispatches to the appropriate template instantiation.
// ============================================================================

// Compute number of threads per block, mirroring the compile-time VEC/NTHRD logic.
// Ensures NTHRD >= 32 by reducing VEC when HEAD_DIM is small.
static inline int compressorNthreads(int head_dim, int elem_bytes)
{
    int max_vec = 16 / elem_bytes;
    int vec = (head_dim / max_vec >= 32) ? max_vec : (head_dim / 32);
    return head_dim / vec;
}

void postProcessScatterLaunch(void const* kv_comp, void* kv_out, void const* rms_weight, float rms_eps,
    float const* cos_sin_table, int32_t const* position_ids, int nope_dim, int rope_dim, void* kv_cache,
    int32_t const* num_outputs, int32_t const* cu_kv_comp, int32_t const* start_pos, int32_t const* block_offsets,
    int batch_size, int tokens_per_block, int head_dim, int max_blocks_per_seq, int elem_bytes, int total_tokens,
    int cache_mode, bool rotate_activation, void* fp8_output, float* scale_output, cudaStream_t stream)
{
    if (total_tokens == 0)
    {
        return;
    }

    TLLM_CHECK_WITH_INFO(cache_mode >= 0 && cache_mode <= 2, "Invalid cache_mode: %d", cache_mode);

    int const nthreads = compressorNthreads(head_dim, elem_bytes);
    int const smem_bytes = head_dim * sizeof(float);

    // Derive cache block stride (in bytes) and scale block count from cache_mode.
    // Each physical cache block stores tokens_per_block tokens:
    //   kDefault:       tpb * HD * elem_bytes       (bf16 or fp32 elements)
    //   kFP8PerTensor:  tpb * HD                    (1 byte per element, no scales)
    //   kFP8Blockwise:  tpb * HD + tpb * (HD/128)*4 (fp8 data + fp32 scales)
    int num_scale_blocks = 0;
    int cache_stride_blk_bytes = 0;
    switch (cache_mode)
    {
    case 1: // kFP8PerTensor
        cache_stride_blk_bytes = tokens_per_block * head_dim;
        break;
    case 2: // kFP8Blockwise
        num_scale_blocks = head_dim / 128;
        cache_stride_blk_bytes = tokens_per_block * head_dim + tokens_per_block * num_scale_blocks * 4;
        break;
    default: // kDefault (bf16 or fp32)
        cache_stride_blk_bytes = tokens_per_block * head_dim * elem_bytes;
        break;
    }

#define LAUNCH_PPS(HD, EB, CM, AR)                                                                                     \
    postProcessScatterKernel<HD, EB, CM, AR><<<total_tokens, nthreads, smem_bytes, stream>>>(kv_comp, kv_out,          \
        rms_weight, rms_eps, cos_sin_table, position_ids, nope_dim, rope_dim, kv_cache, num_outputs, cu_kv_comp,       \
        start_pos, block_offsets, batch_size, tokens_per_block, max_blocks_per_seq, cache_stride_blk_bytes,            \
        total_tokens, num_scale_blocks, fp8_output, scale_output)

// Dispatch helper: for a given (HD, EB, CM), dispatch on rotate_activation
#define DISPATCH_ROTATE(HD, EB, CM)                                                                                    \
    if (rotate_activation)                                                                                             \
    {                                                                                                                  \
        LAUNCH_PPS(HD, EB, CM, true);                                                                                  \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        LAUNCH_PPS(HD, EB, CM, false);                                                                                 \
    }

    auto const cm = static_cast<CacheMode>(cache_mode);

    if (cm == CacheMode::kFP8PerTensor)
    {
        switch (head_dim)
        {
        case 128: DISPATCH_ROTATE(128, 2, CacheMode::kFP8PerTensor); break;
        default: DISPATCH_ROTATE(512, 2, CacheMode::kFP8PerTensor); break;
        }
    }
    else if (cm == CacheMode::kFP8Blockwise)
    {
        switch (head_dim)
        {
        case 128: DISPATCH_ROTATE(128, 2, CacheMode::kFP8Blockwise); break;
        default: DISPATCH_ROTATE(512, 2, CacheMode::kFP8Blockwise); break;
        }
    }
    else
    {
        if (elem_bytes == 4)
        {
            switch (head_dim)
            {
            case 128: DISPATCH_ROTATE(128, 4, CacheMode::kDefault); break;
            default: DISPATCH_ROTATE(512, 4, CacheMode::kDefault); break;
            }
        }
        else
        {
            switch (head_dim)
            {
            case 128: DISPATCH_ROTATE(128, 2, CacheMode::kDefault); break;
            default: DISPATCH_ROTATE(512, 2, CacheMode::kDefault); break;
            }
        }
    }

#undef DISPATCH_ROTATE
#undef LAUNCH_PPS
}

} // namespace kernels::compressor

TRTLLM_NAMESPACE_END
