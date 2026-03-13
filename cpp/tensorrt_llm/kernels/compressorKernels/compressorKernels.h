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
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "tensorrt_llm/common/config.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::compressor
{

// Decode kernel: write NEXT_N tokens to paged cache + conditional compression
// via online softmax. Supports both overlap (state_dim = 2*head_dim) and
// non-overlap (state_dim = head_dim) modes.
//
// Grid: (batch_size, cdiv(state_dim, block_size))
// One thread per state_dim element across all phases.
void pagedKvCompressLaunch(
    void const* kv_score,       // [m, 2*state_dim]  (bf16 or fp32)
    float const* ape,           // [compress_ratio, state_dim]
    void* paged_kv,             // [num_blocks, page_size, state_dim]
    void* paged_score,          // [num_blocks, page_size, state_dim]
    int32_t const* block_table_kv,    // [bsz, max_blocks]
    int32_t const* block_table_score, // [bsz, max_blocks]
    void* output,               // [total_outputs, head_dim]
    int32_t const* kv_lens,     // [bsz]
    int32_t const* start_pos,   // [bsz]
    int32_t const* cu_seq_lens, // [bsz+1]
    int32_t const* cu_kv_comp,  // [bsz+1]
    bool* compressed_mask,      // [bsz]
    int batch_size,
    int page_size,
    int max_blocks,
    int head_dim,
    int compress_ratio,
    bool is_overlap,
    int next_n,
    int io_elem_bytes,          // bytes per element for kv_score/paged (2=bf16, 4=fp32)
    int out_elem_bytes,         // bytes per element for output
    cudaStream_t stream);

// Prefill kernel: bulk compression with per-token gather/scatter + state update.
// Writes remainder tokens to paged cache, then performs online softmax reduction.
//
// Grid: (batch_size, max_outputs_per_batch, num_head_chunks)
// Each block computes one compressed output for one head_dim chunk.
void prefillReductionLaunch(
    void const* kv_score,       // [m, 2*state_dim]  (bf16 or fp32)
    float const* ape,           // [compress_ratio, state_dim]
    void* paged_kv,             // [num_blocks, page_size, state_dim]
    void* paged_score,          // [num_blocks, page_size, state_dim]
    int32_t const* block_table_kv,    // [bsz, max_blocks]
    int32_t const* block_table_score, // [bsz, max_blocks]
    void* output,               // [total_outputs, head_dim]
    int32_t const* kv_lens,     // [bsz]
    int32_t const* start_pos,   // [bsz]
    int32_t const* cu_seq_lens, // [bsz+1]
    int32_t const* cu_kv_comp,  // [bsz+1]
    bool* compressed_mask,      // [bsz]
    int batch_size,
    int page_size,
    int max_blocks,
    int head_dim,
    int compress_ratio,
    bool is_overlap,
    int max_outputs,
    int io_elem_bytes,
    int out_elem_bytes,
    cudaStream_t stream);

// Fused RMSNorm + RoPE + Hadamard + Scatter kernel with adaptive vectorization.
// Replaces the separate RMSNorm -> RoPE -> Hadamard -> Scatter pipeline with a
// single kernel, eliminating intermediate DRAM round-trips.
// VEC is chosen to ensure at least 32 threads (one full warp).
//
// Grid:  (total_tokens) -- one block per compressed token
// Block: (head_dim / VEC), always >= 32
void fusedPostProcessScatterLaunch(
    void const* kv_comp,            // [total_tokens, head_dim]
    void const* rms_weight,         // [head_dim] (same dtype as kv_comp)
    float rms_eps,
    float const* cos_sin_table,     // [max_pos, 2, rope_dim/2]  (contiguous fp32)
    int32_t const* position_ids,    // [total_tokens]
    int nope_dim,
    int rope_dim,
    void* kv_cache,                 // [num_blocks, block_elems]
    int32_t const* num_outputs,     // [bsz]
    int32_t const* cu_kv_comp,      // [bsz+1]
    int32_t const* start_pos,       // [bsz]
    int32_t const* block_offsets,   // [bsz, max_blocks]
    int batch_size,
    int tokens_per_block,
    int head_dim,
    int max_blocks_per_seq,
    int cache_stride_blk,
    int elem_bytes,                 // element size (2=bf16, 4=fp32)
    int total_tokens,
    cudaStream_t stream);

} // namespace kernels::compressor

TRTLLM_NAMESPACE_END
