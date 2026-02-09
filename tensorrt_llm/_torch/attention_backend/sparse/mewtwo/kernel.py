import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    ],
    key=["head_dim", "compress_ratio"],
)
@triton.jit
def paged_kv_compress_kernel(
    kv_score_ptr,
    ape_ptr,
    kv_lens_ptr,
    start_pos_ptr,
    cu_seq_lens_ptr,
    cu_kv_comp_ptr,
    paged_kv_ptr,
    paged_score_ptr,
    block_table_kv_ptr,
    block_table_score_ptr,
    output_ptr,
    compressed_mask_ptr,
    compress_ratio,
    head_dim,
    page_size,
    max_blocks,
    stride_in_t,
    stride_in_h,
    stride_ape_r,
    stride_ape_h,
    stride_cache_blk,
    stride_cache_p,
    stride_cache_h,
    stride_bt_b,
    stride_bt_s,
    stride_out_c,
    stride_out_h,
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

    state_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    state_mask = state_offsets < state_dim
    head_mask = state_offsets < head_dim

    start_pos = tl.load(start_pos_ptr + batch_idx)
    kv_len = tl.load(kv_lens_ptr + batch_idx)
    input_offset = tl.load(cu_seq_lens_ptr + batch_idx)
    output_offset = tl.load(cu_kv_comp_ptr + batch_idx)

    # Phase 1: Write tokens to paged cache (sequential write based on token_idx)
    for t in range(NEXT_N):
        token_idx = start_pos + t
        if token_idx < kv_len:
            ape_idx = token_idx % compress_ratio
            # Sequential write: use token_idx directly, block table handles mapping
            write_pos = token_idx
            logical_block = write_pos // page_size
            block_offset = write_pos % page_size

            # Use separate block tables for kv and score
            phys_block_kv = tl.load(
                block_table_kv_ptr + batch_idx * stride_bt_b + logical_block * stride_bt_s
            )
            phys_block_score = tl.load(
                block_table_score_ptr + batch_idx * stride_bt_b + logical_block * stride_bt_s
            )
            cache_base_kv = phys_block_kv * stride_cache_blk + block_offset * stride_cache_p
            cache_base_score = phys_block_score * stride_cache_blk + block_offset * stride_cache_p
            in_ptr = kv_score_ptr + (input_offset + t) * stride_in_t
            new_kv = tl.load(in_ptr + state_offsets * stride_in_h, mask=state_mask, other=0.0).to(
                tl.float32
            )
            new_score = tl.load(
                in_ptr + (state_dim + state_offsets) * stride_in_h, mask=state_mask, other=0.0
            ).to(tl.float32)
            ape_val = tl.load(
                ape_ptr + ape_idx * stride_ape_r + state_offsets * stride_ape_h,
                mask=state_mask,
                other=0.0,
            ).to(tl.float32)

            tl.store(
                paged_kv_ptr + cache_base_kv + state_offsets * stride_cache_h,
                new_kv,
                mask=state_mask,
            )
            tl.store(
                paged_score_ptr + cache_base_score + state_offsets * stride_cache_h,
                new_score + ape_val,
                mask=state_mask,
            )

    # Phase 2: Count compressions and store mask
    last_token_idx = start_pos + NEXT_N - 1
    num_compressions = (last_token_idx + 1) // compress_ratio - start_pos // compress_ratio
    if block_idx == 0:
        tl.store(compressed_mask_ptr + batch_idx, num_compressions > 0)

    # Phase 3: Perform reductions and write to packed output
    for c in range(NEXT_N):
        if c < num_compressions:
            compress_idx = start_pos // compress_ratio + c
            # Current chunk: tokens [compress_idx*ratio : (compress_idx+1)*ratio]
            curr_chunk_start = compress_idx * compress_ratio

            running_max = tl.full([BLOCK_SIZE], float("-inf"), dtype=tl.float32)
            running_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            running_wsum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

            if IS_OVERLAP:
                # Previous chunk: tokens [(compress_idx-1)*ratio : compress_idx*ratio]
                # Read with first half features (:head_dim)
                prev_chunk_start = curr_chunk_start - compress_ratio
                for r in range(compress_ratio):
                    pos = prev_chunk_start + r
                    log_blk = pos // page_size
                    off = pos % page_size
                    pblk_kv = tl.load(
                        block_table_kv_ptr + batch_idx * stride_bt_b + log_blk * stride_bt_s
                    )
                    pblk_score = tl.load(
                        block_table_score_ptr + batch_idx * stride_bt_b + log_blk * stride_bt_s
                    )
                    base_kv = pblk_kv * stride_cache_blk + off * stride_cache_p
                    base_score = pblk_score * stride_cache_blk + off * stride_cache_p

                    # Read first half features (state_offsets is already 0..BLOCK_SIZE-1)
                    k = tl.load(
                        paged_kv_ptr + base_kv + state_offsets * stride_cache_h,
                        mask=head_mask,
                        other=0.0,
                    ).to(tl.float32)
                    s = tl.load(
                        paged_score_ptr + base_score + state_offsets * stride_cache_h,
                        mask=head_mask,
                        other=float("-inf"),
                    ).to(tl.float32)

                    new_max = tl.maximum(running_max, s)
                    scale = tl.exp(running_max - new_max)
                    term = tl.exp(s - new_max)
                    running_sum = running_sum * scale + term
                    running_wsum = running_wsum * scale + k * term
                    running_max = new_max

                # Current chunk: read with second half features (head_dim:2*head_dim)
                for r in range(compress_ratio):
                    pos = curr_chunk_start + r
                    log_blk = pos // page_size
                    off = pos % page_size
                    pblk_kv = tl.load(
                        block_table_kv_ptr + batch_idx * stride_bt_b + log_blk * stride_bt_s
                    )
                    pblk_score = tl.load(
                        block_table_score_ptr + batch_idx * stride_bt_b + log_blk * stride_bt_s
                    )
                    base_kv = pblk_kv * stride_cache_blk + off * stride_cache_p
                    base_score = pblk_score * stride_cache_blk + off * stride_cache_p

                    # Read second half features (offset by head_dim)
                    k = tl.load(
                        paged_kv_ptr + base_kv + (head_dim + state_offsets) * stride_cache_h,
                        mask=head_mask,
                        other=0.0,
                    ).to(tl.float32)
                    s = tl.load(
                        paged_score_ptr + base_score + (head_dim + state_offsets) * stride_cache_h,
                        mask=head_mask,
                        other=float("-inf"),
                    ).to(tl.float32)

                    new_max = tl.maximum(running_max, s)
                    scale = tl.exp(running_max - new_max)
                    term = tl.exp(s - new_max)
                    running_sum = running_sum * scale + term
                    running_wsum = running_wsum * scale + k * term
                    running_max = new_max
            else:
                # Non-overlap: just read from current chunk
                for r in range(compress_ratio):
                    pos = curr_chunk_start + r
                    log_blk = pos // page_size
                    off = pos % page_size
                    pblk_kv = tl.load(
                        block_table_kv_ptr + batch_idx * stride_bt_b + log_blk * stride_bt_s
                    )
                    pblk_score = tl.load(
                        block_table_score_ptr + batch_idx * stride_bt_b + log_blk * stride_bt_s
                    )
                    base_kv = pblk_kv * stride_cache_blk + off * stride_cache_p
                    base_score = pblk_score * stride_cache_blk + off * stride_cache_p

                    k = tl.load(
                        paged_kv_ptr + base_kv + state_offsets * stride_cache_h,
                        mask=head_mask,
                        other=0.0,
                    ).to(tl.float32)
                    s = tl.load(
                        paged_score_ptr + base_score + state_offsets * stride_cache_h,
                        mask=head_mask,
                        other=float("-inf"),
                    ).to(tl.float32)

                    new_max = tl.maximum(running_max, s)
                    scale = tl.exp(running_max - new_max)
                    term = tl.exp(s - new_max)
                    running_sum = running_sum * scale + term
                    running_wsum = running_wsum * scale + k * term
                    running_max = new_max

            # Write to packed output: [total_outputs, head_dim]
            out_ptr = output_ptr + (output_offset + c) * stride_out_c + state_offsets * stride_out_h
            result = running_wsum / running_sum
            tl.store(out_ptr, result.to(output_ptr.dtype.element_ty), mask=head_mask)


@triton.jit
def prefill_reduction_kernel(
    kv_score_desc_ptr,
    ape_desc_ptr,
    kv_lens_ptr,
    start_pos_ptr,
    cu_seq_lens_ptr,
    cu_kv_comp_ptr,
    output_ptr,
    compressed_mask_ptr,
    paged_kv_ptr,
    paged_score_ptr,
    block_table_kv_ptr,
    block_table_score_ptr,
    head_dim,
    page_size,
    max_blocks,
    state_dim,
    stride_out_c,
    stride_out_h,
    stride_cache_blk,
    stride_cache_p,
    stride_cache_h,
    stride_bt_b,
    stride_bt_s,
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

    Grid: (batch_size, max_outputs_per_batch, num_head_chunks)
    Each block computes one compressed output for one head_dim chunk.
    Parallel processing of head_dim chunks eliminates sequential kernel launches.
    """
    batch_idx = tl.program_id(0)
    local_output_idx = tl.program_id(1)
    head_chunk_idx = tl.program_id(2)  # NEW: parallel head chunk processing

    # Compute head_offset from chunk index
    head_offset = head_chunk_idx * BLOCK_SIZE

    start_pos = tl.load(start_pos_ptr + batch_idx)
    kv_len = tl.load(kv_lens_ptr + batch_idx)
    input_offset = tl.load(cu_seq_lens_ptr + batch_idx)
    output_offset = tl.load(cu_kv_comp_ptr + batch_idx)

    seqlen = kv_len - start_pos
    num_outputs = tl.maximum(seqlen // COMPRESS_RATIO, 1)

    if local_output_idx >= num_outputs:
        return

    # Early exit if this chunk is beyond head_dim
    if head_offset >= head_dim:
        return

    # head_offsets are relative to head_offset (chunk offset)
    head_offsets = tl.arange(0, BLOCK_SIZE)
    # Mask for valid positions within this chunk and within head_dim
    head_mask = (head_offset + head_offsets) < head_dim
    tl.multiple_of(head_offsets, 8)

    coff = 2 if IS_OVERLAP else 1
    actual_num_outputs = seqlen // COMPRESS_RATIO
    should_compress = local_output_idx < actual_num_outputs

    # Write compressed_mask (only from first output block and first head chunk per batch)
    if local_output_idx == 0 and head_offset == 0:
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
        # Write at absolute token positions so that the decode kernel
        # (which also uses absolute positions via write_pos = token_idx)
        # can read the previous chunk at the correct locations.
        if IS_OVERLAP and cutoff >= COMPRESS_RATIO:
            base_row = cutoff - COMPRESS_RATIO
            row_mask = r_offsets < COMPRESS_RATIO

            # Write at absolute positions [start_pos + cutoff - ratio : start_pos + cutoff)
            write_pos = start_pos + cutoff - COMPRESS_RATIO + r_offsets
            log_blk = write_pos // page_size
            blk_off = write_pos % page_size
            pblk_kv = tl.load(bt_base_kv + log_blk * stride_bt_s, mask=row_mask, other=0)
            pblk_score = tl.load(bt_base_score + log_blk * stride_bt_s, mask=row_mask, other=0)

            for col_idx in tl.static_range(2):
                # col_off includes head_offset for chunked processing
                col_off = col_idx * head_dim + head_offset
                kv = (
                    kv_score_desc_ptr.load([input_offset + base_row, col_off])
                    .reshape(COMPRESS_RATIO, BLOCK_SIZE)
                    .to(tl.float32)
                )
                sc = (
                    kv_score_desc_ptr.load([input_offset + base_row, state_dim + col_off])
                    .reshape(COMPRESS_RATIO, BLOCK_SIZE)
                    .to(tl.float32)
                )
                ape = (
                    ape_desc_ptr.load([0, col_off])
                    .reshape(COMPRESS_RATIO, BLOCK_SIZE)
                    .to(tl.float32)
                )
                cache_col = col_off + head_offsets
                ptrs_kv = (
                    paged_kv_ptr
                    + pblk_kv[:, None] * stride_cache_blk
                    + blk_off[:, None] * stride_cache_p
                    + cache_col[None, :] * stride_cache_h
                )
                ptrs_sc = (
                    paged_score_ptr
                    + pblk_score[:, None] * stride_cache_blk
                    + blk_off[:, None] * stride_cache_p
                    + cache_col[None, :] * stride_cache_h
                )
                tl.store(ptrs_kv, kv, mask=row_mask[:, None] & head_mask[None, :])
                tl.store(ptrs_sc, sc + ape, mask=row_mask[:, None] & head_mask[None, :])

        # Remainder tokens
        # Write at absolute positions [start_pos + cutoff : start_pos + cutoff + remainder)
        if remainder > 0:
            base_row = cutoff
            row_mask = r_offsets < remainder

            write_pos = start_pos + cutoff + r_offsets
            log_blk = write_pos // page_size
            blk_off = write_pos % page_size
            pblk_kv = tl.load(bt_base_kv + log_blk * stride_bt_s, mask=row_mask, other=0)
            pblk_score = tl.load(bt_base_score + log_blk * stride_bt_s, mask=row_mask, other=0)

            for col_idx in tl.static_range(2):
                if col_idx < coff:
                    # col_off includes head_offset for chunked processing
                    col_off = col_idx * head_dim + head_offset
                    kv = (
                        kv_score_desc_ptr.load([input_offset + base_row, col_off])
                        .reshape(COMPRESS_RATIO, BLOCK_SIZE)
                        .to(tl.float32)
                    )
                    sc = (
                        kv_score_desc_ptr.load([input_offset + base_row, state_dim + col_off])
                        .reshape(COMPRESS_RATIO, BLOCK_SIZE)
                        .to(tl.float32)
                    )
                    ape = (
                        ape_desc_ptr.load([0, col_off])
                        .reshape(COMPRESS_RATIO, BLOCK_SIZE)
                        .to(tl.float32)
                    )

                    cache_col = col_off + head_offsets
                    ptrs_kv = (
                        paged_kv_ptr
                        + pblk_kv[:, None] * stride_cache_blk
                        + blk_off[:, None] * stride_cache_p
                        + cache_col[None, :] * stride_cache_h
                    )
                    ptrs_sc = (
                        paged_score_ptr
                        + pblk_score[:, None] * stride_cache_blk
                        + blk_off[:, None] * stride_cache_p
                        + cache_col[None, :] * stride_cache_h
                    )
                    tl.store(ptrs_kv, kv, mask=row_mask[:, None] & head_mask[None, :])
                    tl.store(ptrs_sc, sc + ape, mask=row_mask[:, None] & head_mask[None, :])

    # Reduction
    if not should_compress:
        return

    running_max = tl.full([BLOCK_SIZE], float("-inf"), dtype=tl.float32)
    running_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    running_wsum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    if IS_OVERLAP:
        # Previous segment (col=0 + head_offset)
        if local_output_idx > 0:
            input_start = (local_output_idx - 1) * COMPRESS_RATIO
            k = (
                kv_score_desc_ptr.load([input_offset + input_start, head_offset])
                .reshape(COMPRESS_RATIO, BLOCK_SIZE)
                .to(tl.float32)
            )
            s = (
                kv_score_desc_ptr.load([input_offset + input_start, state_dim + head_offset])
                .reshape(COMPRESS_RATIO, BLOCK_SIZE)
                .to(tl.float32)
            )
            s = s + ape_desc_ptr.load([0, head_offset]).reshape(COMPRESS_RATIO, BLOCK_SIZE).to(
                tl.float32
            )

            s_max = tl.max(s, axis=0)
            new_max = tl.maximum(running_max, s_max)
            scale = tl.exp(running_max - new_max)
            exp_s = tl.exp(s - new_max[None, :])
            running_sum = running_sum * scale + tl.sum(exp_s, axis=0)
            running_wsum = running_wsum * scale + tl.sum(k * exp_s, axis=0)
            running_max = new_max

        # Current segment (col=head_dim + head_offset)
        cur_start = local_output_idx * COMPRESS_RATIO
        k = (
            kv_score_desc_ptr.load([input_offset + cur_start, head_dim + head_offset])
            .reshape(COMPRESS_RATIO, BLOCK_SIZE)
            .to(tl.float32)
        )
        s = (
            kv_score_desc_ptr.load([input_offset + cur_start, state_dim + head_dim + head_offset])
            .reshape(COMPRESS_RATIO, BLOCK_SIZE)
            .to(tl.float32)
        )
        s = s + ape_desc_ptr.load([0, head_dim + head_offset]).reshape(
            COMPRESS_RATIO, BLOCK_SIZE
        ).to(tl.float32)

        s_max = tl.max(s, axis=0)
        new_max = tl.maximum(running_max, s_max)
        scale = tl.exp(running_max - new_max)
        exp_s = tl.exp(s - new_max[None, :])
        running_sum = running_sum * scale + tl.sum(exp_s, axis=0)
        running_wsum = running_wsum * scale + tl.sum(k * exp_s, axis=0)
        running_max = new_max
    else:
        # Non-overlap: single segment (col=head_offset)
        input_start = local_output_idx * COMPRESS_RATIO
        k = (
            kv_score_desc_ptr.load([input_offset + input_start, head_offset])
            .reshape(COMPRESS_RATIO, BLOCK_SIZE)
            .to(tl.float32)
        )
        s = (
            kv_score_desc_ptr.load([input_offset + input_start, state_dim + head_offset])
            .reshape(COMPRESS_RATIO, BLOCK_SIZE)
            .to(tl.float32)
        )
        s = s + ape_desc_ptr.load([0, head_offset]).reshape(COMPRESS_RATIO, BLOCK_SIZE).to(
            tl.float32
        )

        s_max = tl.max(s, axis=0)
        new_max = tl.maximum(running_max, s_max)
        scale = tl.exp(running_max - new_max)
        exp_s = tl.exp(s - new_max[None, :])
        running_sum = running_sum * scale + tl.sum(exp_s, axis=0)
        running_wsum = running_wsum * scale + tl.sum(k * exp_s, axis=0)
        running_max = new_max

    # Output to correct head_dim position (head_offset + head_offsets)
    out_ptr = (
        output_ptr
        + (output_offset + local_output_idx) * stride_out_c
        + (head_offset + head_offsets) * stride_out_h
    )
    result = running_wsum / running_sum
    tl.store(out_ptr, result.to(output_ptr.dtype.element_ty), mask=head_mask)


def kv_compress_triton(
    kv_score: torch.Tensor,
    ape: torch.Tensor,
    kv_lens: torch.Tensor,
    start_pos: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    cu_new_comp_kv: torch.Tensor,
    kv_comp: torch.Tensor,
    compressed_mask: torch.Tensor,
    paged_kv: torch.Tensor,
    paged_score: torch.Tensor,
    block_table_kv: torch.Tensor,
    block_table_score: torch.Tensor = None,
    compress_ratio: int = None,
    head_dim: int = None,
    overlap: bool = False,
    page_size: int = 32,
    next_n: int = 1,
):
    """Decode kernel: write tokens to cache, perform conditional compression.

    Args:
        kv_score: [m, 2*state_dim] fused input (m = bsz * next_n)
        ape: [compress_ratio, state_dim] positional embedding
        kv_lens: [bsz] total KV length per batch (past + current tokens)
        start_pos: [bsz] starting position (past KV length). Can be None to auto-compute from kv_lens.
        cu_seq_lens: [bsz+1] cumulative input offsets into kv_score.
        cu_new_comp_kv: [bsz+1] cumulative output offsets.
        kv_comp: [total_outputs, head_dim] pre-allocated output buffer
        compressed_mask: [bsz] pre-allocated bool mask buffer
        paged_kv/paged_score: [num_blocks, page_size, state_dim]
        block_table_kv: [bsz, max_blocks] block table for kv cache
        block_table_score: [bsz, max_blocks] block table for score cache (if None, uses block_table_kv)
        next_n: Tokens per request (1 for decode, >1 for MTP)
    """
    if block_table_score is None:
        block_table_score = block_table_kv

    batch_size = kv_lens.shape[0]

    # Auto-compute start_pos from kv_lens if not provided
    if start_pos is None:
        start_pos = kv_lens - next_n

    coff = 2 if overlap else 1
    state_dim = coff * head_dim

    def grid(meta):
        return (batch_size, triton.cdiv(state_dim, meta["BLOCK_SIZE"]))

    paged_kv_compress_kernel[grid](
        kv_score,
        ape,
        kv_lens,
        start_pos,
        cu_seq_lens,
        cu_new_comp_kv,
        paged_kv,
        paged_score,
        block_table_kv,
        block_table_score,
        kv_comp,
        compressed_mask,
        compress_ratio,
        head_dim,
        page_size,
        block_table_kv.shape[1],
        kv_score.stride(0),
        kv_score.stride(1),
        ape.stride(0),
        ape.stride(1),
        paged_kv.stride(0),
        paged_kv.stride(1),
        paged_kv.stride(2),
        block_table_kv.stride(0),
        block_table_kv.stride(1),
        kv_comp.stride(0),
        kv_comp.stride(1),
        IS_OVERLAP=overlap,
        NEXT_N=next_n,
    )


def kv_compress_prefill_triton(
    kv_score: torch.Tensor,
    ape: torch.Tensor,
    kv_lens: torch.Tensor,
    start_pos: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    cu_new_comp_kv: torch.Tensor,
    kv_comp: torch.Tensor,
    compressed_mask: torch.Tensor,
    paged_kv: torch.Tensor,
    paged_score: torch.Tensor,
    block_table_kv: torch.Tensor,
    block_table_score: torch.Tensor = None,
    compress_ratio: int = None,
    head_dim: int = None,
    overlap: bool = False,
    page_size: int = 32,
):
    """Prefill kernel: bulk compression with TMA-optimized loads.

    Args:
        kv_score: [m, 2*state_dim] fused packed input (m = total tokens across batch)
        ape: [compress_ratio, state_dim] positional embedding
        kv_lens: [bsz] total KV length per batch (past + current tokens)
        start_pos: [bsz] starting position (past KV length). Can be None for pure prefill (start_pos=0).
        cu_seq_lens: [bsz+1] cumulative input offsets into kv_score.
        cu_new_comp_kv: [bsz+1] cumulative output offsets.
        kv_comp: [total_outputs, head_dim] pre-allocated output buffer
        compressed_mask: [bsz] pre-allocated bool mask buffer
        paged_kv/paged_score: [num_blocks, page_size, state_dim]
        block_table_kv: [bsz, max_blocks] block table for kv cache
        block_table_score: [bsz, max_blocks] block table for score cache (if None, uses block_table_kv)
    """
    if block_table_score is None:
        block_table_score = block_table_kv

    batch_size = kv_lens.shape[0]

    # Auto-compute start_pos if not provided (assume pure prefill, start_pos=0)
    if start_pos is None:
        start_pos = torch.zeros(batch_size, device=kv_lens.device, dtype=torch.int32)

    coff = 2 if overlap else 1
    state_dim = coff * head_dim

    seq_lens = kv_lens - start_pos
    num_outputs_per_batch = torch.clamp(seq_lens // compress_ratio, min=1)
    max_outputs = num_outputs_per_batch.max().item()

    # Compute safe BLOCK_SIZE to avoid shared memory overflow
    # The kernel loads multiple [compress_ratio, BLOCK_SIZE] tensors
    # Shared memory limit is ~232KB, use conservative 200KB
    SHARED_MEM_LIMIT = 230000  # bytes
    NUM_LIVE_TENSORS = 4  # k, s, ape, exp_s in worst case
    max_block_size = SHARED_MEM_LIMIT // (compress_ratio * 4 * NUM_LIVE_TENSORS)
    # Round down to power of 2, minimum 64
    block_size = max(64, 1 << (max_block_size.bit_length() - 1)) if max_block_size >= 64 else 64
    block_size = min(block_size, head_dim)

    kv_score_desc = TensorDescriptor.from_tensor(kv_score, [compress_ratio, block_size])
    ape_desc = TensorDescriptor.from_tensor(ape, [compress_ratio, block_size])

    # Process head_dim in chunks of block_size - now parallel via 3D grid
    num_head_chunks = (head_dim + block_size - 1) // block_size

    # Compute optimal num_warps based on block_size
    # More warps for larger block sizes to improve parallelism
    num_warps = 4 if block_size <= 128 else 8

    # Single kernel launch with 3D grid for parallel head chunk processing
    # Grid: (batch_size, max_outputs, num_head_chunks)
    # This eliminates sequential kernel launches and enables concurrent execution
    # num_stages=2 enables software pipelining for TMA loads
    prefill_reduction_kernel[(batch_size, max_outputs, num_head_chunks)](
        kv_score_desc,
        ape_desc,
        kv_lens,
        start_pos,
        cu_seq_lens,
        cu_new_comp_kv,
        kv_comp,
        compressed_mask,
        paged_kv,
        paged_score,
        block_table_kv,
        block_table_score,
        head_dim,
        page_size,
        block_table_kv.shape[1],
        state_dim,
        kv_comp.stride(0),
        kv_comp.stride(1),
        paged_kv.stride(0),
        paged_kv.stride(1),
        paged_kv.stride(2),
        block_table_kv.stride(0),
        block_table_kv.stride(1),
        IS_OVERLAP=overlap,
        COMPRESS_RATIO=compress_ratio,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=2,  # Enable software pipelining for overlapped TMA loads
    )


# ============================================================================
# Unified Compressed KV Cache Scatter Kernel (supports default and blockwise FP8 modes)
# ============================================================================


@triton.jit
def compressed_kv_scatter_kernel(
    # Input: compressed KV [total_outputs, head_dim] (any dtype: bf16/fp16/fp8/etc)
    compressed_kv_ptr,
    # Input: Scale bytes [total_tokens, scale_size] (only used when IS_BLOCKWISE_FP8=True)
    kv_scale_ptr,
    # Metadata
    num_outputs_ptr,  # [bsz] number of outputs per batch
    cu_kv_comp_ptr,  # [bsz+1] cumulative output offsets
    start_pos_ptr,  # [bsz] position offset (past compressed KV length)
    # KV cache: [num_blocks, kv_factor, tokens_per_block * head_dim] or flat bytes per block
    kv_cache_ptr,
    # Block offsets: [num_seqs, max_blocks_per_seq]
    block_offsets_ptr,
    # Dimensions
    tokens_per_block,
    head_dim,
    scale_size,  # Only used when IS_BLOCKWISE_FP8=True
    # Strides for kv_cache
    stride_cache_blk,
    stride_cache_token,
    stride_cache_elem,  # Element stride (for default: stride(2), for blockwise FP8: byte stride)
    # Strides for block_offsets
    stride_bo_batch,
    stride_bo_blk,
    # Strides for input
    stride_in_token,
    stride_in_elem,
    # Stride for scale input (only used when IS_BLOCKWISE_FP8=True)
    stride_scale_token,
    # Constexpr parameters
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_S: tl.constexpr,
    IS_BLOCKWISE_FP8: tl.constexpr,
    IS_PERTENSOR_FP8: tl.constexpr,
):
    """Unified scatter kernel for compressed KV to paged cache.

    When IS_BLOCKWISE_FP8=True (blockwise FP8 with separate scales):
        - Input: [total_tokens, head_dim] fp8 (FP8 data)
        - Scale: [total_tokens, scale_size] uint8 (f32 bytes)
        - Cache: [num_blocks, tokens_per_block * head_dim + tokens_per_block * scale_size] fp8
        - Non-interleaved layout per block: [k0, k1, ..., kN, scale0, scale1, ..., scaleN]
        - Scatters both FP8 data and scales

    When IS_PERTENSOR_FP8=True (per-tensor FP8):
        - Input: [total_outputs, head_dim] uint8 (FP8 bytes)
        - Cache: [num_blocks, kv_factor, tokens_per_block * head_dim] fp8e4nv
        - Bitcasts uint8 -> fp8e4nv before storing

    When both are False (default mode):
        - Input: [total_outputs, head_dim] any dtype (bf16/fp16/fp8/etc)
        - Cache: [num_blocks, kv_factor, tokens_per_block * head_dim]
        - Scatters KV data directly (dtype-agnostic)

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
        block_offsets_ptr + batch_idx * stride_bo_batch + logical_block * stride_bo_blk
    )

    # Global token index in input
    global_token_idx = output_offset + local_output_idx

    if IS_BLOCKWISE_FP8:
        # FP8 mode: non-interleaved layout within each block:
        #   [k0, k1, ..., kN, scale0, scale1, ..., scaleN]
        # FP8 data region: block_base + token_offset * head_dim
        # Scale data region: block_base + tokens_per_block * head_dim + token_offset * scale_size
        block_base = phys_block * stride_cache_blk
        fp8_cache_offset = block_base + token_offset * head_dim * stride_cache_elem
        scale_cache_offset = (
            block_base
            + (tokens_per_block * head_dim + token_offset * scale_size) * stride_cache_elem
        )

        # ===== Scatter FP8 data =====
        # Process head_dim elements in chunks of BLOCK_SIZE_H
        # Input is fp8, cache is fp8 - direct load/store
        for h_start in range(0, head_dim, BLOCK_SIZE_H):
            h_offsets = h_start + tl.arange(0, BLOCK_SIZE_H)
            h_mask = h_offsets < head_dim

            # Load FP8 data directly (no other= to avoid int32->fp8 cast error)
            fp8_data = tl.load(
                compressed_kv_ptr + global_token_idx * stride_in_token + h_offsets * stride_in_elem,
                mask=h_mask,
            )

            # Store to cache (FP8 data in first section of block)
            tl.store(
                kv_cache_ptr + fp8_cache_offset + h_offsets * stride_cache_elem,
                fp8_data,
                mask=h_mask,
            )

        # ===== Scatter scale data =====
        # Scale data in second section of block (after all FP8 data)
        # Scale input is uint8 (f32 bytes), cache may be fp8 - bitcast to match cache dtype
        for s_start in range(0, scale_size, BLOCK_SIZE_S):
            s_offsets = s_start + tl.arange(0, BLOCK_SIZE_S)
            s_mask = s_offsets < scale_size

            # Load scale bytes (uint8)
            scale_data = tl.load(
                kv_scale_ptr + global_token_idx * stride_scale_token + s_offsets,
                mask=s_mask,
            )

            # Bitcast uint8 to fp8e4nv to match cache dtype (same bytes, different type)
            scale_data_fp8 = scale_data.to(tl.float8e4nv, bitcast=True)

            # Store to cache
            tl.store(
                kv_cache_ptr + scale_cache_offset + s_offsets * stride_cache_elem,
                scale_data_fp8,
                mask=s_mask,
            )
    elif IS_PERTENSOR_FP8:
        # Per-tensor FP8 mode: input is uint8 (FP8 bytes), cache is fp8e4nv
        # Need to bitcast uint8 -> fp8e4nv before storing
        cache_base = phys_block * stride_cache_blk + token_offset * head_dim * stride_cache_elem

        for h_start in range(0, head_dim, BLOCK_SIZE_H):
            h_offsets = h_start + tl.arange(0, BLOCK_SIZE_H)
            h_mask = h_offsets < head_dim

            # Load uint8 data (FP8 bytes)
            data = tl.load(
                compressed_kv_ptr + global_token_idx * stride_in_token + h_offsets * stride_in_elem,
                mask=h_mask,
            )
            # Bitcast uint8 -> fp8e4nv for storing to fp8 cache
            data_fp8 = data.to(tl.float8e4nv, bitcast=True)
            tl.store(
                kv_cache_ptr + cache_base + h_offsets * stride_cache_elem, data_fp8, mask=h_mask
            )
    else:
        # Default mode: cache layout is [num_blocks, kv_factor, tokens * head_dim]
        # Input and cache have the same dtype (bf16/fp16/etc) - direct copy
        cache_base = phys_block * stride_cache_blk + token_offset * head_dim * stride_cache_elem

        for h_start in range(0, head_dim, BLOCK_SIZE_H):
            h_offsets = h_start + tl.arange(0, BLOCK_SIZE_H)
            h_mask = h_offsets < head_dim

            data = tl.load(
                compressed_kv_ptr + global_token_idx * stride_in_token + h_offsets * stride_in_elem,
                mask=h_mask,
            )
            tl.store(kv_cache_ptr + cache_base + h_offsets * stride_cache_elem, data, mask=h_mask)


def compressed_kv_scatter(
    compressed_kv: torch.Tensor,  # [total_outputs, head_dim] packed, any dtype
    num_comp_tokens: torch.Tensor,  # [bsz] number of compressed tokens per batch
    cu_new_comp_kv: torch.Tensor,  # [bsz+1] cumulative output offsets
    start_pos: torch.Tensor,  # [bsz] compressed cache position
    kv_cache: torch.Tensor,  # [num_blocks, kv_factor, tokens_per_block * head_dim]
    block_offsets: torch.Tensor,  # [num_seqs, max_blocks_per_seq]
    tokens_per_block: int,
    head_dim: int,
    kv_cache_dtype: str = "default",
    kv_scale: torch.Tensor = None,  # [total_tokens, scale_size] uint8, for fp8_blockwise
):
    """Scatter compressed KV to paged cache.

    Supports multiple KV cache formats:
    - "default": Any dtype (bf16, fp16, etc.) - the kernel handles it automatically
    - "fp8_blockwise": FP8 data with blockwise scales (requires kv_scale)
    - "fp8_pertensor": FP8 data with per-tensor scale (scale stored separately)

    Args:
        compressed_kv: [total_outputs, head_dim] packed format
            - For "default": any dtype (bf16, fp16, etc.)
            - For "fp8_blockwise": fp8 (FP8 data)
            - For "fp8_pertensor": uint8 (FP8 bytes viewed as uint8)
        num_comp_tokens: [bsz] number of valid outputs per batch
        cu_new_comp_kv: [bsz+1] cumulative output offsets
        start_pos: [bsz] starting position in compressed cache
        kv_cache: [num_blocks, kv_factor, tokens_per_block * head_dim]
            - For "default": same dtype as input
            - For "fp8_blockwise": fp8 (non-interleaved: [k0..kN, scale0..scaleN] per block)
            - For "fp8_pertensor": fp8e4nv
        block_offsets: [num_seqs, max_blocks_per_seq]
        tokens_per_block: Tokens per cache block
        head_dim: Hidden dimension (number of elements/bytes)
        kv_cache_dtype: "default", "fp8_blockwise", or "fp8_pertensor"
        kv_scale: Scale bytes for fp8_blockwise mode [total_tokens, scale_size]
    """
    if compressed_kv.numel() == 0:
        return

    batch_size = num_comp_tokens.shape[0]
    max_outputs = num_comp_tokens.max().item()

    if max_outputs == 0:
        return

    block_size_h = min(triton.next_power_of_2(head_dim), 512)

    if kv_cache_dtype == "fp8_blockwise":
        assert kv_scale is not None, "kv_scale required for fp8_blockwise mode"
        scale_size = kv_scale.shape[1]
        block_size_s = min(triton.next_power_of_2(scale_size), 64)

        compressed_kv_scatter_kernel[(batch_size, max_outputs)](
            compressed_kv,
            kv_scale,
            num_comp_tokens,
            cu_new_comp_kv,
            start_pos,
            kv_cache,
            block_offsets,
            tokens_per_block,
            head_dim,
            scale_size,
            kv_cache.stride(0),
            kv_cache.stride(1) if kv_cache.dim() > 1 else 1,
            kv_cache.stride(-1),
            block_offsets.stride(0),
            block_offsets.stride(1),
            compressed_kv.stride(0),
            compressed_kv.stride(1) if compressed_kv.dim() > 1 else 1,
            kv_scale.stride(0),
            BLOCK_SIZE_H=block_size_h,
            BLOCK_SIZE_S=block_size_s,
            IS_BLOCKWISE_FP8=True,
            IS_PERTENSOR_FP8=False,
        )
    elif kv_cache_dtype == "fp8_pertensor":
        # Per-tensor FP8: input is uint8 (FP8 bytes), cache is fp8e4nv
        # Dummy scale tensor (not used)
        dummy_scale = compressed_kv

        compressed_kv_scatter_kernel[(batch_size, max_outputs)](
            compressed_kv,
            dummy_scale,
            num_comp_tokens,
            cu_new_comp_kv,
            start_pos,
            kv_cache,
            block_offsets,
            tokens_per_block,
            head_dim,
            0,  # scale_size not used
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            block_offsets.stride(0),
            block_offsets.stride(1),
            compressed_kv.stride(0),
            compressed_kv.stride(1),
            1,  # stride_scale_token not used
            BLOCK_SIZE_H=block_size_h,
            BLOCK_SIZE_S=16,  # Not used
            IS_BLOCKWISE_FP8=False,
            IS_PERTENSOR_FP8=True,
        )
    else:
        # Default mode: dtype-agnostic scatter
        # Dummy scale tensor (not used in non-FP8 mode)
        dummy_scale = compressed_kv

        compressed_kv_scatter_kernel[(batch_size, max_outputs)](
            compressed_kv,
            dummy_scale,  # Not used when IS_BLOCKWISE_FP8=False
            num_comp_tokens,
            cu_new_comp_kv,
            start_pos,
            kv_cache,
            block_offsets,
            tokens_per_block,
            head_dim,
            0,  # scale_size not used
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            block_offsets.stride(0),
            block_offsets.stride(1),
            compressed_kv.stride(0),
            compressed_kv.stride(1),
            1,  # stride_scale_token not used
            BLOCK_SIZE_H=block_size_h,
            BLOCK_SIZE_S=16,  # Not used
            IS_BLOCKWISE_FP8=False,
            IS_PERTENSOR_FP8=False,
        )
