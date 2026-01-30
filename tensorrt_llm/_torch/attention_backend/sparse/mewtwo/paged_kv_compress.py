# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton kernels for KV compression with paged cache.

Kernels:
- paged_kv_compress_kernel: Decode - token update + conditional reduction (MTP support)
- prefill_reduction_kernel: Prefill - bulk compression with state update (TMA optimized)

Input: Fused kv_score [m, 2*state_dim] packed tensor with cu_seq_lens offsets.
"""

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
from typing import Tuple

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['head_dim', 'compress_ratio']
)
@triton.jit
def paged_kv_compress_kernel(
    kv_score_ptr, ape_ptr,
    kv_lens_ptr, start_pos_ptr, cu_seq_lens_ptr, cu_kv_comp_ptr,
    paged_kv_ptr, paged_score_ptr, block_table_kv_ptr, block_table_score_ptr,
    output_ptr, compressed_mask_ptr,
    compress_ratio, head_dim, page_size, max_blocks,
    stride_in_t, stride_in_h,
    stride_ape_r, stride_ape_h,
    stride_cache_blk, stride_cache_p, stride_cache_h,
    stride_bt_b, stride_bt_s,
    stride_out_c, stride_out_h,
    IS_OVERLAP: tl.constexpr,
    NEXT_N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Decode kernel: update paged state + conditional compression.

    Expected inputs:
    - kv_score_ptr: packed [m, 2*state_dim] where m = sum(next_n per batch).
    - kv_lens_ptr/start_pos_ptr: [bsz] total KV length and start position.
    - cu_seq_lens_ptr: [bsz+1] cumulative input offsets into kv_score.
    - cu_kv_comp_ptr: [bsz+1] cumulative output offsets.
    - paged_kv_ptr/paged_score_ptr: [num_blocks, page_size, state_dim].
    - block_table_kv_ptr/block_table_score_ptr: [bsz, max_blocks].
    - output_ptr: [total_outputs, head_dim] packed output buffer.

    Grid: (batch_size, cdiv(state_dim, BLOCK_SIZE))
    """
    batch_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    
    coff = 2 if IS_OVERLAP else 1
    state_dim = coff * head_dim
    total_positions = max_blocks * page_size
    
    state_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    state_mask = state_offsets < state_dim
    head_mask = state_offsets < head_dim
    
    start_pos = tl.load(start_pos_ptr + batch_idx)
    kv_len = tl.load(kv_lens_ptr + batch_idx)
    input_offset = tl.load(cu_seq_lens_ptr + batch_idx)
    output_offset = tl.load(cu_kv_comp_ptr + batch_idx)
    
    # Phase 1: Write tokens to paged cache
    for t in range(NEXT_N):
        token_idx = start_pos + t
        if token_idx < kv_len:
            ape_idx = token_idx % compress_ratio
            write_pos_linear = token_idx + (compress_ratio if IS_OVERLAP else 0)
            write_pos = write_pos_linear % total_positions
            logical_block = write_pos // page_size
            block_offset = write_pos % page_size
            
            # Use separate block tables for kv and score
            phys_block_kv = tl.load(block_table_kv_ptr + batch_idx * stride_bt_b + logical_block * stride_bt_s)
            phys_block_score = tl.load(block_table_score_ptr + batch_idx * stride_bt_b + logical_block * stride_bt_s)
            cache_base_kv = phys_block_kv * stride_cache_blk + block_offset * stride_cache_p
            cache_base_score = phys_block_score * stride_cache_blk + block_offset * stride_cache_p
            in_ptr = kv_score_ptr + (input_offset + t) * stride_in_t
            new_kv = tl.load(in_ptr + state_offsets * stride_in_h,
                             mask=state_mask, other=0.0).to(tl.float32)
            new_score = tl.load(in_ptr + (state_dim + state_offsets) * stride_in_h,
                                mask=state_mask, other=0.0).to(tl.float32)
            ape_val = tl.load(ape_ptr + ape_idx * stride_ape_r + state_offsets * stride_ape_h,
                              mask=state_mask, other=0.0).to(tl.float32)
            
            tl.store(paged_kv_ptr + cache_base_kv + state_offsets * stride_cache_h, new_kv, mask=state_mask)
            tl.store(paged_score_ptr + cache_base_score + state_offsets * stride_cache_h, new_score + ape_val, mask=state_mask)
    
    # Phase 2: Count compressions and store mask
    last_token_idx = start_pos + NEXT_N - 1
    num_compressions = (last_token_idx + 1) // compress_ratio - start_pos // compress_ratio
    if block_idx == 0:
        tl.store(compressed_mask_ptr + batch_idx, num_compressions > 0)
    
    # Phase 3: Perform reductions and write to packed output
    for c in range(NEXT_N):
        if c < num_compressions:
            compress_idx = start_pos // compress_ratio + c
            token_idx = (compress_idx + 1) * compress_ratio - 1  # Last token of this chunk
            
            write_pos_linear = token_idx + (compress_ratio if IS_OVERLAP else 0)
            
            running_max = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
            running_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            running_wsum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            
            if IS_OVERLAP:
                # Previous chunk (first half features)
                prev_start = write_pos_linear - 2 * compress_ratio + 1
                for r in range(compress_ratio):
                    pos = prev_start + r
                    wrapped = ((pos % total_positions) + total_positions) % total_positions
                    log_blk = wrapped // page_size
                    off = wrapped % page_size
                    pblk_kv = tl.load(block_table_kv_ptr + batch_idx * stride_bt_b + log_blk * stride_bt_s)
                    pblk_score = tl.load(block_table_score_ptr + batch_idx * stride_bt_b + log_blk * stride_bt_s)
                    base_kv = pblk_kv * stride_cache_blk + off * stride_cache_p
                    base_score = pblk_score * stride_cache_blk + off * stride_cache_p
                    
                    k = tl.load(paged_kv_ptr + base_kv + state_offsets * stride_cache_h,
                               mask=head_mask, other=0.0).to(tl.float32)
                    s = tl.load(paged_score_ptr + base_score + state_offsets * stride_cache_h,
                               mask=head_mask, other=float('-inf')).to(tl.float32)
                    
                    new_max = tl.maximum(running_max, s)
                    scale = tl.exp(running_max - new_max)
                    term = tl.exp(s - new_max)
                    running_sum = running_sum * scale + term
                    running_wsum = running_wsum * scale + k * term
                    running_max = new_max
                
                # Current chunk (second half features)
                curr_start = write_pos_linear - compress_ratio + 1
                for r in range(compress_ratio):
                    pos = curr_start + r
                    wrapped = ((pos % total_positions) + total_positions) % total_positions
                    log_blk = wrapped // page_size
                    off = wrapped % page_size
                    pblk_kv = tl.load(block_table_kv_ptr + batch_idx * stride_bt_b + log_blk * stride_bt_s)
                    pblk_score = tl.load(block_table_score_ptr + batch_idx * stride_bt_b + log_blk * stride_bt_s)
                    base_kv = pblk_kv * stride_cache_blk + off * stride_cache_p
                    base_score = pblk_score * stride_cache_blk + off * stride_cache_p
                    
                    k = tl.load(paged_kv_ptr + base_kv + (head_dim + state_offsets) * stride_cache_h,
                               mask=head_mask, other=0.0).to(tl.float32)
                    s = tl.load(paged_score_ptr + base_score + (head_dim + state_offsets) * stride_cache_h,
                               mask=head_mask, other=float('-inf')).to(tl.float32)
                    
                    new_max = tl.maximum(running_max, s)
                    scale = tl.exp(running_max - new_max)
                    term = tl.exp(s - new_max)
                    running_sum = running_sum * scale + term
                    running_wsum = running_wsum * scale + k * term
                    running_max = new_max
            else:
                curr_start = write_pos_linear - compress_ratio + 1
                for r in range(compress_ratio):
                    pos = curr_start + r
                    wrapped = ((pos % total_positions) + total_positions) % total_positions
                    log_blk = wrapped // page_size
                    off = wrapped % page_size
                    pblk_kv = tl.load(block_table_kv_ptr + batch_idx * stride_bt_b + log_blk * stride_bt_s)
                    pblk_score = tl.load(block_table_score_ptr + batch_idx * stride_bt_b + log_blk * stride_bt_s)
                    base_kv = pblk_kv * stride_cache_blk + off * stride_cache_p
                    base_score = pblk_score * stride_cache_blk + off * stride_cache_p
                    
                    k = tl.load(paged_kv_ptr + base_kv + state_offsets * stride_cache_h,
                               mask=head_mask, other=0.0).to(tl.float32)
                    s = tl.load(paged_score_ptr + base_score + state_offsets * stride_cache_h,
                               mask=head_mask, other=float('-inf')).to(tl.float32)
                    
                    new_max = tl.maximum(running_max, s)
                    scale = tl.exp(running_max - new_max)
                    term = tl.exp(s - new_max)
                    running_sum = running_sum * scale + term
                    running_wsum = running_wsum * scale + k * term
                    running_max = new_max
            
            # Write to packed output: [total_outputs, head_dim]
            out_ptr = output_ptr + (output_offset + c) * stride_out_c + state_offsets * stride_out_h
            tl.store(out_ptr, running_wsum / running_sum, mask=head_mask)


@triton.jit
def prefill_reduction_kernel(
    kv_score_desc_ptr, ape_desc_ptr,
    kv_lens_ptr, start_pos_ptr, cu_seq_lens_ptr, cu_kv_comp_ptr,
    output_ptr, compressed_mask_ptr,
    paged_kv_ptr, paged_score_ptr, block_table_kv_ptr, block_table_score_ptr,
    head_dim, page_size, max_blocks, state_dim,
    stride_out_c, stride_out_h,
    stride_cache_blk, stride_cache_p, stride_cache_h,
    stride_bt_b, stride_bt_s,
    IS_OVERLAP: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Prefill kernel: bulk compression with TMA-optimized block loads.

    Expected inputs:
    - kv_score_desc_ptr: descriptor of packed [m, 2*state_dim] input.
    - ape_desc_ptr: descriptor of [compress_ratio, state_dim] APE.
    - kv_lens_ptr/start_pos_ptr: [bsz] total KV length and start position.
    - cu_seq_lens_ptr: [bsz+1] cumulative input offsets into kv_score.
    - cu_kv_comp_ptr: [bsz+1] cumulative output offsets.
    - paged_kv_ptr/paged_score_ptr: [num_blocks, page_size, state_dim].
    - block_table_kv_ptr/block_table_score_ptr: [bsz, max_blocks].
    - output_ptr: [total_outputs, head_dim] packed output buffer.

    Grid: (batch_size, max_outputs_per_batch)
    Each block computes one compressed output and handles state updates.
    """
    batch_idx = tl.program_id(0)
    local_output_idx = tl.program_id(1)

    start_pos = tl.load(start_pos_ptr + batch_idx)
    kv_len = tl.load(kv_lens_ptr + batch_idx)
    input_offset = tl.load(cu_seq_lens_ptr + batch_idx)
    output_offset = tl.load(cu_kv_comp_ptr + batch_idx)
    
    seqlen = kv_len - start_pos
    num_outputs = tl.maximum(seqlen // COMPRESS_RATIO, 1)
    
    if local_output_idx >= num_outputs:
        return
    
    total_positions = max_blocks * page_size

    head_offsets = tl.arange(0, BLOCK_SIZE)
    head_mask = head_offsets < head_dim
    tl.multiple_of(head_offsets, 8)

    coff = 2 if IS_OVERLAP else 1
    actual_num_outputs = seqlen // COMPRESS_RATIO
    should_compress = local_output_idx < actual_num_outputs
    
    # Write compressed_mask (only from first output block per batch)
    if local_output_idx == 0:
        tl.store(compressed_mask_ptr + batch_idx, actual_num_outputs > 0)

    # State update (last output block only)
    if local_output_idx == num_outputs - 1:
        remainder = seqlen % COMPRESS_RATIO
        cutoff = seqlen - remainder
        offset = COMPRESS_RATIO if IS_OVERLAP else 0
        bt_base_kv = block_table_kv_ptr + batch_idx * stride_bt_b
        bt_base_score = block_table_score_ptr + batch_idx * stride_bt_b
        r_offsets = tl.arange(0, COMPRESS_RATIO)

        # Last full chunk (overlap mode only)
        if IS_OVERLAP and cutoff >= COMPRESS_RATIO:
            base_row = cutoff - COMPRESS_RATIO
            row_mask = r_offsets < COMPRESS_RATIO
            
            write_pos = (start_pos + base_row + r_offsets + offset) % total_positions
            log_blk = write_pos // page_size
            blk_off = write_pos % page_size
            pblk_kv = tl.load(bt_base_kv + log_blk * stride_bt_s, mask=row_mask, other=0)
            pblk_score = tl.load(bt_base_score + log_blk * stride_bt_s, mask=row_mask, other=0)
            
            for col_idx in tl.static_range(2):
                col_off = col_idx * head_dim
                kv = kv_score_desc_ptr.load([input_offset + base_row, col_off]).reshape(COMPRESS_RATIO, BLOCK_SIZE).to(tl.float32)
                sc = kv_score_desc_ptr.load([input_offset + base_row, state_dim + col_off]).reshape(COMPRESS_RATIO, BLOCK_SIZE).to(tl.float32)
                ape = ape_desc_ptr.load([0, col_off]).reshape(COMPRESS_RATIO, BLOCK_SIZE).to(tl.float32)
                cache_col = col_off + head_offsets
                ptrs_kv = paged_kv_ptr + pblk_kv[:, None] * stride_cache_blk + blk_off[:, None] * stride_cache_p + cache_col[None, :] * stride_cache_h
                ptrs_sc = paged_score_ptr + pblk_score[:, None] * stride_cache_blk + blk_off[:, None] * stride_cache_p + cache_col[None, :] * stride_cache_h
                tl.store(ptrs_kv, kv, mask=row_mask[:, None] & head_mask[None, :])
                tl.store(ptrs_sc, sc + ape, mask=row_mask[:, None] & head_mask[None, :])

        # Remainder tokens
        if remainder > 0:
            base_row = cutoff
            row_mask = r_offsets < remainder
            
            write_pos = (start_pos + base_row + r_offsets + offset) % total_positions
            log_blk = write_pos // page_size
            blk_off = write_pos % page_size
            pblk_kv = tl.load(bt_base_kv + log_blk * stride_bt_s, mask=row_mask, other=0)
            pblk_score = tl.load(bt_base_score + log_blk * stride_bt_s, mask=row_mask, other=0)
            
            for col_idx in tl.static_range(2):
                if col_idx < coff:
                    col_off = col_idx * head_dim
                    kv = kv_score_desc_ptr.load([input_offset + base_row, col_off]).reshape(COMPRESS_RATIO, BLOCK_SIZE).to(tl.float32)
                    sc = kv_score_desc_ptr.load([input_offset + base_row, state_dim + col_off]).reshape(COMPRESS_RATIO, BLOCK_SIZE).to(tl.float32)
                    ape = ape_desc_ptr.load([0, col_off]).reshape(COMPRESS_RATIO, BLOCK_SIZE).to(tl.float32)
                    
                    cache_col = col_off + head_offsets
                    ptrs_kv = paged_kv_ptr + pblk_kv[:, None] * stride_cache_blk + blk_off[:, None] * stride_cache_p + cache_col[None, :] * stride_cache_h
                    ptrs_sc = paged_score_ptr + pblk_score[:, None] * stride_cache_blk + blk_off[:, None] * stride_cache_p + cache_col[None, :] * stride_cache_h
                    tl.store(ptrs_kv, kv, mask=row_mask[:, None] & head_mask[None, :])
                    tl.store(ptrs_sc, sc + ape, mask=row_mask[:, None] & head_mask[None, :])

    # Reduction
    if not should_compress:
        return

    running_max = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    running_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    running_wsum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    if IS_OVERLAP:
        # Previous segment (col=0)
        if local_output_idx > 0:
            input_start = (local_output_idx - 1) * COMPRESS_RATIO
            k = kv_score_desc_ptr.load([input_offset + input_start, 0]).reshape(COMPRESS_RATIO, BLOCK_SIZE).to(tl.float32)
            s = kv_score_desc_ptr.load([input_offset + input_start, state_dim]).reshape(COMPRESS_RATIO, BLOCK_SIZE).to(tl.float32)
            s = s + ape_desc_ptr.load([0, 0]).reshape(COMPRESS_RATIO, BLOCK_SIZE).to(tl.float32)
            
            s_max = tl.max(s, axis=0)
            new_max = tl.maximum(running_max, s_max)
            scale = tl.exp(running_max - new_max)
            exp_s = tl.exp(s - new_max[None, :])
            running_sum = running_sum * scale + tl.sum(exp_s, axis=0)
            running_wsum = running_wsum * scale + tl.sum(k * exp_s, axis=0)
            running_max = new_max
        
        # Current segment (col=head_dim)
        cur_start = local_output_idx * COMPRESS_RATIO
        k = kv_score_desc_ptr.load([input_offset + cur_start, head_dim]).reshape(COMPRESS_RATIO, BLOCK_SIZE).to(tl.float32)
        s = kv_score_desc_ptr.load([input_offset + cur_start, state_dim + head_dim]).reshape(COMPRESS_RATIO, BLOCK_SIZE).to(tl.float32)
        s = s + ape_desc_ptr.load([0, head_dim]).reshape(COMPRESS_RATIO, BLOCK_SIZE).to(tl.float32)
        
        s_max = tl.max(s, axis=0)
        new_max = tl.maximum(running_max, s_max)
        scale = tl.exp(running_max - new_max)
        exp_s = tl.exp(s - new_max[None, :])
        running_sum = running_sum * scale + tl.sum(exp_s, axis=0)
        running_wsum = running_wsum * scale + tl.sum(k * exp_s, axis=0)
        running_max = new_max
    else:
        # Non-overlap: single segment (col=0)
        input_start = local_output_idx * COMPRESS_RATIO
        k = kv_score_desc_ptr.load([input_offset + input_start, 0]).reshape(COMPRESS_RATIO, BLOCK_SIZE).to(tl.float32)
        s = kv_score_desc_ptr.load([input_offset + input_start, state_dim]).reshape(COMPRESS_RATIO, BLOCK_SIZE).to(tl.float32)
        s = s + ape_desc_ptr.load([0, 0]).reshape(COMPRESS_RATIO, BLOCK_SIZE).to(tl.float32)
        
        s_max = tl.max(s, axis=0)
        new_max = tl.maximum(running_max, s_max)
        scale = tl.exp(running_max - new_max)
        exp_s = tl.exp(s - new_max[None, :])
        running_sum = running_sum * scale + tl.sum(exp_s, axis=0)
        running_wsum = running_wsum * scale + tl.sum(k * exp_s, axis=0)
        running_max = new_max

    out_ptr = output_ptr + (output_offset + local_output_idx) * stride_out_c + head_offsets * stride_out_h
    tl.store(out_ptr, running_wsum / running_sum, mask=head_mask)


def kv_compress_triton(
    kv_score: torch.Tensor,
    ape: torch.Tensor,
    kv_lens: torch.Tensor,
    start_pos: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    cu_kv_comp: torch.Tensor,
    paged_kv: torch.Tensor,
    paged_score: torch.Tensor,
    block_table_kv: torch.Tensor,
    block_table_score: torch.Tensor = None,
    compress_ratio: int = None,
    head_dim: int = None,
    overlap: bool = False,
    page_size: int = 32,
    next_n: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode kernel: write tokens to cache, perform conditional compression.
    
    Args:
        kv_score: [m, 2*state_dim] fused input (m = bsz * next_n)
        ape: [compress_ratio, state_dim] positional embedding
        kv_lens: [bsz] total KV length per batch (past + current tokens)
        start_pos: [bsz] starting position (past KV length). Can be None to auto-compute from kv_lens.
        cu_seq_lens: [bsz+1] cumulative input offsets into kv_score.
        cu_kv_comp: [bsz+1] cumulative output offsets.
        paged_kv/paged_score: [num_blocks, page_size, state_dim]
        block_table_kv: [bsz, max_blocks] block table for kv cache
        block_table_score: [bsz, max_blocks] block table for score cache (if None, uses block_table_kv)
        next_n: Tokens per request (1 for decode, >1 for MTP)
    
    Returns: (kv_comp, compressed_mask)
        kv_comp: [total_outputs, head_dim] packed format
        compressed_mask: [bsz] bool mask
    
    Note:
        kv_lens stores the total number of KV tokens (past + current).
        start_pos = kv_lens - next_n (auto-computed if start_pos is None).
    """
    assert block_table_kv is not None, "block_table_kv must not be None"
    assert block_table_score is not None, "block_table_score must not be None"
    
    batch_size = kv_lens.shape[0]
    
    # Auto-compute start_pos from kv_lens if not provided
    if start_pos is None:
        start_pos = kv_lens - next_n
    
    coff = 2 if overlap else 1
    state_dim = coff * head_dim
    
    def grid(meta):
        return (batch_size, triton.cdiv(state_dim, meta['BLOCK_SIZE']))
    
    total_outputs = cu_kv_comp[-1].item()
    kv_comp = torch.empty(total_outputs, head_dim, device=kv_lens.device, dtype=torch.float32)
    compressed_mask = torch.empty(batch_size, device=kv_lens.device, dtype=torch.bool)

    paged_kv_compress_kernel[grid](
        kv_score, ape,
        kv_lens, start_pos, cu_seq_lens, cu_kv_comp,
        paged_kv, paged_score, block_table_kv, block_table_score,
        kv_comp, compressed_mask,
        compress_ratio, head_dim, page_size, block_table_kv.shape[1],
        kv_score.stride(0), kv_score.stride(1),
        ape.stride(0), ape.stride(1),
        paged_kv.stride(0), paged_kv.stride(1), paged_kv.stride(2),
        block_table_kv.stride(0), block_table_kv.stride(1),
        kv_comp.stride(0), kv_comp.stride(1),
        IS_OVERLAP=overlap,
        NEXT_N=next_n,
    )
    
    return kv_comp, compressed_mask


def kv_compress_prefill_triton(
    kv_score: torch.Tensor,
    ape: torch.Tensor,
    kv_lens: torch.Tensor,
    start_pos: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    cu_kv_comp: torch.Tensor,
    paged_kv: torch.Tensor,
    paged_score: torch.Tensor,
    block_table_kv: torch.Tensor,
    block_table_score: torch.Tensor = None,
    compress_ratio: int = None,
    head_dim: int = None,
    overlap: bool = False,
    page_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prefill kernel: bulk compression with TMA-optimized loads.
    
    Args:
        kv_score: [m, 2*state_dim] fused packed input (m = total tokens across batch)
        ape: [compress_ratio, state_dim] positional embedding
        kv_lens: [bsz] total KV length per batch (past + current tokens)
        start_pos: [bsz] starting position (past KV length). Can be None for pure prefill (start_pos=0).
        cu_seq_lens: [bsz+1] cumulative input offsets into kv_score.
        cu_kv_comp: [bsz+1] cumulative output offsets.
        paged_kv/paged_score: [num_blocks, page_size, state_dim]
        block_table_kv: [bsz, max_blocks] block table for kv cache
        block_table_score: [bsz, max_blocks] block table for score cache (if None, uses block_table_kv)
    
    Returns: (kv_comp, compressed_mask)
        kv_comp: [total_outputs, head_dim] compressed results
        compressed_mask: [bsz] bool mask (True if batch had any compression)
    
    Note:
        kv_lens stores the total number of KV tokens (past + current).
        seq_lens = kv_lens - start_pos (current input length).
        For pure prefill, start_pos=0 so kv_lens == seq_lens.
    """
    assert block_table_kv is not None, "block_table_kv must not be None"
    assert block_table_score is not None, "block_table_score must not be None"
    
    batch_size = kv_lens.shape[0]
    
    # Auto-compute start_pos if not provided (assume pure prefill, start_pos=0)
    if start_pos is None:
        start_pos = torch.zeros(batch_size, device=kv_lens.device, dtype=torch.int32)
    
    coff = 2 if overlap else 1
    state_dim = coff * head_dim
    num_warps = 4 if head_dim <= 128 else 8
    
    seq_lens = kv_lens - start_pos
    num_outputs_per_batch = torch.clamp(seq_lens // compress_ratio, min=1)
    max_outputs = num_outputs_per_batch.max().item()
    
    kv_score_desc = TensorDescriptor.from_tensor(kv_score, [compress_ratio, head_dim])
    ape_desc = TensorDescriptor.from_tensor(ape, [compress_ratio, head_dim])

    total_outputs = cu_kv_comp[-1].item()
    kv_comp = torch.empty(total_outputs, head_dim, device=kv_lens.device, dtype=torch.float32)
    compressed_mask = torch.empty(batch_size, device=kv_lens.device, dtype=torch.bool)

    prefill_reduction_kernel[(batch_size, max_outputs)](
        kv_score_desc, ape_desc,
        kv_lens, start_pos, cu_seq_lens, cu_kv_comp,
        kv_comp, compressed_mask,
        paged_kv, paged_score, block_table_kv, block_table_score,
        head_dim, page_size, block_table_kv.shape[1], state_dim,
        kv_comp.stride(0), kv_comp.stride(1),
        paged_kv.stride(0), paged_kv.stride(1), paged_kv.stride(2),
        block_table_kv.stride(0), block_table_kv.stride(1),
        IS_OVERLAP=overlap,
        COMPRESS_RATIO=compress_ratio,
        BLOCK_SIZE=head_dim,
        num_warps=num_warps,
    )
    
    return kv_comp, compressed_mask

# ============================================================================
# Compressed KV Cache Scatter Kernel
# ============================================================================

@triton.jit
def compressed_kv_scatter_kernel(
    # Input: compressed KV [total_outputs, head_dim] packed format
    compressed_kv_ptr,
    # Metadata
    num_outputs_ptr,    # [bsz] number of outputs per batch
    cu_kv_comp_ptr,     # [bsz+1] cumulative output offsets
    start_pos_ptr,      # [bsz] position offset (past compressed KV length)
    # KV cache: [num_blocks, num_layers, kv_factor, tokens_per_block * head_dim]
    kv_cache_ptr,
    # Block offsets: [num_pools, batch_size, 2, max_blocks_per_seq]
    block_offsets_ptr,
    # Dimensions
    layer_idx,
    tokens_per_block,
    head_dim,
    # Strides for kv_cache
    stride_cache_blk,
    stride_cache_layer,
    stride_cache_token,
    # Strides for block_offsets
    stride_bo_batch,
    stride_bo_blk,
    # Strides for input
    stride_in_c,
    stride_in_h,
    BLOCK_SIZE: tl.constexpr,
):
    """Scatter compressed KV to paged cache.

    Expected inputs:
    - compressed_kv_ptr: [total_outputs, head_dim] packed format.
    - num_outputs_ptr: [bsz] number of outputs per batch.
    - cu_kv_comp_ptr: [bsz+1] cumulative output offsets.
    - start_pos_ptr: [bsz] starting position in compressed cache.
    - kv_cache_ptr: [num_blocks, num_layers, kv_factor, tokens_per_block * head_dim].
    - block_offsets_ptr: [num_pools, bsz, 2, max_blocks_per_seq].

    Grid: (batch_size, max_outputs_per_batch)
    """
    batch_idx = tl.program_id(0)
    local_output_idx = tl.program_id(1)
    
    num_outputs = tl.load(num_outputs_ptr + batch_idx)
    if local_output_idx >= num_outputs:
        return
    
    start_pos = tl.load(start_pos_ptr + batch_idx)
    output_offset = tl.load(cu_kv_comp_ptr + batch_idx)
    
    # Compute cache position
    cache_pos = start_pos + local_output_idx
    logical_block = cache_pos // tokens_per_block
    token_offset = cache_pos % tokens_per_block
    
    # Load physical block (k offset at index 0, same as v for MLA)
    phys_block = tl.load(
        block_offsets_ptr + 
        batch_idx * stride_bo_batch + 
        logical_block * stride_bo_blk
    )
    
    # Input offset (packed format)
    input_offset = (output_offset + local_output_idx) * stride_in_c
    
    # Cache offset: [num_blocks, num_layers, kv_factor, tokens * head_dim]
    cache_base = (
        phys_block * stride_cache_blk +
        layer_idx * stride_cache_layer +
        token_offset * head_dim * stride_cache_token
    )
    
    # Load and store
    h_offsets = tl.arange(0, BLOCK_SIZE)
    h_mask = h_offsets < head_dim
    
    data = tl.load(compressed_kv_ptr + input_offset + h_offsets * stride_in_h, mask=h_mask)
    tl.store(kv_cache_ptr + cache_base + h_offsets * stride_cache_token, data, mask=h_mask)


def compressed_kv_scatter(
    compressed_kv: torch.Tensor,    # [total_outputs, head_dim] packed
    num_outputs: torch.Tensor,      # [bsz] number of outputs per batch
    cu_kv_comp: torch.Tensor,       # [bsz+1] cumulative output offsets
    start_pos: torch.Tensor,        # [bsz] compressed cache position
    kv_cache: torch.Tensor,         # [num_blocks, num_layers, kv_factor, tokens_per_block * head_dim]
    block_offsets: torch.Tensor,    # [num_pools, batch_size, 2, max_blocks_per_seq]
    layer_idx: int,
    tokens_per_block: int,
    head_dim: int,
):
    """Scatter compressed KV to paged cache.
    
    Args:
        compressed_kv: [total_outputs, head_dim] packed format (from prefill or decode)
        num_outputs: [bsz] number of valid outputs per batch
        cu_kv_comp: [bsz+1] cumulative output offsets
        start_pos: [bsz] starting position in compressed cache
        kv_cache: [num_blocks, num_layers, kv_factor, tokens_per_block * head_dim]
        block_offsets: [num_pools, batch_size, 2, max_blocks_per_seq]
        layer_idx: Current layer index
        tokens_per_block: Tokens per cache block
        head_dim: Hidden dimension
    """
    if compressed_kv.numel() == 0:
        return
    
    batch_size = num_outputs.shape[0]
    max_outputs = num_outputs.max().item()
    
    if max_outputs == 0:
        return
    
    block_size = triton.next_power_of_2(head_dim)
    
    compressed_kv_scatter_kernel[(batch_size, max_outputs)](
        compressed_kv,
        num_outputs, cu_kv_comp, start_pos,
        kv_cache, block_offsets,
        layer_idx, tokens_per_block, head_dim,
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(3),
        block_offsets.stride(1), block_offsets.stride(3),
        compressed_kv.stride(0), compressed_kv.stride(1),
        BLOCK_SIZE=block_size,
    )
