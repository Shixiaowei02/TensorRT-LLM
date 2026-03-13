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
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace tk = tensorrt_llm::kernels::compressor;

namespace
{

// Decode kernel: write tokens to paged cache + conditional compression
void compressorPagedKvCompressOp(
    torch::Tensor kv_score,          // [m, 2*state_dim] bf16
    torch::Tensor ape,               // [compress_ratio, state_dim] fp32
    torch::Tensor paged_kv,          // [num_blocks, page_size, state_dim] bf16
    torch::Tensor paged_score,       // [num_blocks, page_size, state_dim] bf16
    torch::Tensor block_table_kv,    // [bsz, max_blocks] int32
    torch::Tensor block_table_score, // [bsz, max_blocks] int32
    torch::Tensor output,            // [total_outputs, head_dim] bf16
    torch::Tensor kv_lens,           // [bsz] int32
    torch::Tensor start_pos,         // [bsz] int32
    torch::Tensor cu_seq_lens,       // [bsz+1] int32
    torch::Tensor cu_kv_comp,        // [bsz+1] int32
    torch::Tensor compressed_mask,   // [bsz] bool
    int64_t batch_size,
    int64_t page_size,
    int64_t head_dim,
    int64_t compress_ratio,
    bool is_overlap,
    int64_t next_n)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    int io_eb = static_cast<int>(kv_score.element_size());
    int out_eb = static_cast<int>(output.element_size());

    tk::pagedKvCompressLaunch(
        kv_score.data_ptr(),
        ape.data_ptr<float>(),
        paged_kv.data_ptr(),
        paged_score.data_ptr(),
        block_table_kv.data_ptr<int32_t>(),
        block_table_score.data_ptr<int32_t>(),
        output.data_ptr(),
        kv_lens.data_ptr<int32_t>(),
        start_pos.data_ptr<int32_t>(),
        cu_seq_lens.data_ptr<int32_t>(),
        cu_kv_comp.data_ptr<int32_t>(),
        reinterpret_cast<bool*>(compressed_mask.data_ptr()),
        static_cast<int>(batch_size),
        static_cast<int>(page_size),
        static_cast<int>(block_table_kv.size(1)),
        static_cast<int>(head_dim),
        static_cast<int>(compress_ratio),
        is_overlap,
        static_cast<int>(next_n),
        io_eb,
        out_eb,
        stream);
}

// Prefill kernel: bulk compression with state update
void compressorPrefillReductionOp(
    torch::Tensor kv_score,
    torch::Tensor ape,
    torch::Tensor paged_kv,
    torch::Tensor paged_score,
    torch::Tensor block_table_kv,
    torch::Tensor block_table_score,
    torch::Tensor output,
    torch::Tensor kv_lens,
    torch::Tensor start_pos,
    torch::Tensor cu_seq_lens,
    torch::Tensor cu_kv_comp,
    torch::Tensor compressed_mask,
    int64_t batch_size,
    int64_t page_size,
    int64_t head_dim,
    int64_t compress_ratio,
    bool is_overlap,
    int64_t max_outputs)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    int io_eb = static_cast<int>(kv_score.element_size());
    int out_eb = static_cast<int>(output.element_size());

    tk::prefillReductionLaunch(
        kv_score.data_ptr(),
        ape.data_ptr<float>(),
        paged_kv.data_ptr(),
        paged_score.data_ptr(),
        block_table_kv.data_ptr<int32_t>(),
        block_table_score.data_ptr<int32_t>(),
        output.data_ptr(),
        kv_lens.data_ptr<int32_t>(),
        start_pos.data_ptr<int32_t>(),
        cu_seq_lens.data_ptr<int32_t>(),
        cu_kv_comp.data_ptr<int32_t>(),
        reinterpret_cast<bool*>(compressed_mask.data_ptr()),
        static_cast<int>(batch_size),
        static_cast<int>(page_size),
        static_cast<int>(block_table_kv.size(1)),
        static_cast<int>(head_dim),
        static_cast<int>(compress_ratio),
        is_overlap,
        static_cast<int>(max_outputs),
        io_eb,
        out_eb,
        stream);
}

// Fused RMSNorm + RoPE + Hadamard + Scatter
void compressorFusedPostProcessScatterOp(
    torch::Tensor kv_comp,          // [total_tokens, head_dim]
    torch::Tensor rms_weight,       // [head_dim]
    double rms_eps,
    torch::Tensor cos_sin_table,    // [max_pos, 2, rope_dim/2] fp32
    torch::Tensor position_ids,     // [total_tokens] int32
    int64_t nope_dim,
    int64_t rope_dim,
    torch::Tensor kv_cache,         // [num_blocks, ...] paged cache
    torch::Tensor num_outputs,      // [bsz] int32
    torch::Tensor cu_kv_comp,       // [bsz+1] int32
    torch::Tensor start_pos,        // [bsz] int32
    torch::Tensor block_offsets,    // [bsz, max_blocks] int32
    int64_t tokens_per_block,
    int64_t head_dim,
    int64_t total_tokens)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    int const hd = static_cast<int>(head_dim);
    int const tpb = static_cast<int>(tokens_per_block);

    tk::fusedPostProcessScatterLaunch(
        kv_comp.data_ptr(),
        rms_weight.data_ptr(),
        static_cast<float>(rms_eps),
        cos_sin_table.data_ptr<float>(),
        position_ids.data_ptr<int32_t>(),
        static_cast<int>(nope_dim),
        static_cast<int>(rope_dim),
        kv_cache.data_ptr(),
        num_outputs.data_ptr<int32_t>(),
        cu_kv_comp.data_ptr<int32_t>(),
        start_pos.data_ptr<int32_t>(),
        block_offsets.data_ptr<int32_t>(),
        static_cast<int>(num_outputs.size(0)),
        tpb,
        hd,
        static_cast<int>(block_offsets.size(1)),
        tpb * hd,
        static_cast<int>(kv_cache.element_size()),
        static_cast<int>(total_tokens),
        stream);
}

} // anonymous namespace

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "compressor_paged_kv_compress("
        "Tensor kv_score, Tensor ape, "
        "Tensor(a!) paged_kv, Tensor(b!) paged_score, "
        "Tensor block_table_kv, Tensor block_table_score, "
        "Tensor(c!) output, "
        "Tensor kv_lens, Tensor start_pos, "
        "Tensor cu_seq_lens, Tensor cu_kv_comp, "
        "Tensor(d!) compressed_mask, "
        "int batch_size, int page_size, "
        "int head_dim, int compress_ratio, "
        "bool is_overlap, int next_n) -> ()");

    m.def(
        "compressor_prefill_reduction("
        "Tensor kv_score, Tensor ape, "
        "Tensor(a!) paged_kv, Tensor(b!) paged_score, "
        "Tensor block_table_kv, Tensor block_table_score, "
        "Tensor(c!) output, "
        "Tensor kv_lens, Tensor start_pos, "
        "Tensor cu_seq_lens, Tensor cu_kv_comp, "
        "Tensor(d!) compressed_mask, "
        "int batch_size, int page_size, "
        "int head_dim, int compress_ratio, "
        "bool is_overlap, int max_outputs) -> ()");

    m.def(
        "compressor_fused_postprocess_scatter("
        "Tensor kv_comp, Tensor rms_weight, "
        "float rms_eps, "
        "Tensor cos_sin_table, Tensor position_ids, "
        "int nope_dim, int rope_dim, "
        "Tensor(a!) kv_cache, "
        "Tensor num_outputs, Tensor cu_kv_comp, "
        "Tensor start_pos, Tensor block_offsets, "
        "int tokens_per_block, "
        "int head_dim, int total_tokens) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("compressor_paged_kv_compress", &compressorPagedKvCompressOp);
    m.impl("compressor_prefill_reduction", &compressorPrefillReductionOp);
    m.impl("compressor_fused_postprocess_scatter", &compressorFusedPostProcessScatterOp);
}
