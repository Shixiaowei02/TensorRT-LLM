import math

import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]


def _next_power_of_2(n):
    """Round up to the next power of 2."""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1

# ============================================================================
# cuTile Kernels (primary implementations)
#
# These cuTile-based kernels are the primary implementations. The deprecated
# Triton kernels have been moved to kernel_triton.py and are re-exported above
# for backward compatibility. They use NVIDIA's CUDA Tile DSL with native
# N-D gather/scatter indexing, avoiding host-side flatten/reshape for
# torch.compile compatibility.
# ============================================================================


@ct.kernel
def paged_kv_compress_cutile_kernel(
    # N-D data tensors (indexed via N-D gather/scatter)
    kv_score,  # [m, 2*state_dim]
    ape,  # [compress_ratio, state_dim]
    paged_kv,  # [num_blocks, page_size, state_dim]
    paged_score,  # [num_blocks, page_size, state_dim]
    block_table_kv,  # [bsz, max_blocks]
    block_table_score,  # [bsz, max_blocks]
    output,  # [total_outputs, head_dim]
    # Metadata (1D, scalar-indexed)
    kv_lens,  # [bsz]
    start_pos_tensor,  # [bsz]
    cu_seq_lens,  # [bsz+1]
    cu_kv_comp,  # [bsz+1]
    compressed_mask,  # [bsz]
    # Runtime scalars
    page_size: int,
    state_dim: int,
    # Compile-time constants
    head_dim: ConstInt,
    COMPRESS_RATIO: ConstInt,  # 4 or 128
    IS_OVERLAP: ConstBool,
    NEXT_N: ConstInt,
    BLOCK_SIZE: ConstInt,
):
    """Decode kernel: update paged state + conditional compression.

    cuTile equivalent of paged_kv_compress_kernel (Triton).
    Grid: (batch_size, cdiv(state_dim, BLOCK_SIZE))
    """
    batch_idx = ct.bid(0)
    block_idx = ct.bid(1)

    state_offsets = block_idx * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=ct.int32)

    # Load per-batch metadata (scalar gathers)
    sp = ct.gather(start_pos_tensor, batch_idx, padding_value=0)
    kv_len = ct.gather(kv_lens, batch_idx, padding_value=0)
    in_off = ct.gather(cu_seq_lens, batch_idx, padding_value=0)
    out_off = ct.gather(cu_kv_comp, batch_idx, padding_value=0)

    # ================================================================
    # Phase 1: Write NEXT_N tokens to paged cache
    # ================================================================
    for t in range(NEXT_N):
        token_idx = sp + t
        if token_idx < kv_len:
            ape_idx = token_idx % COMPRESS_RATIO

            # Block table lookups (2D index)
            log_blk = token_idx // page_size
            blk_off = token_idx % page_size
            phys_kv = ct.gather(
                block_table_kv, (batch_idx, log_blk), padding_value=0,
            )
            phys_sc = ct.gather(
                block_table_score, (batch_idx, log_blk), padding_value=0,
            )

            # Load kv and score from kv_score (2D)
            kv_data = ct.gather(
                kv_score, (in_off + t, state_offsets), padding_value=0,
            )
            kv_data = ct.astype(kv_data, ct.float32)
            sc_data = ct.gather(
                kv_score, (in_off + t, state_dim + state_offsets), padding_value=0,
            )
            sc_data = ct.astype(sc_data, ct.float32)

            # Load APE (2D)
            ape_data = ct.gather(
                ape, (ape_idx, state_offsets), padding_value=0,
            )
            ape_data = ct.astype(ape_data, ct.float32)

            # Write to paged caches (3D scatter)
            ct.scatter(
                paged_kv, (phys_kv, blk_off, state_offsets),
                ct.astype(kv_data, paged_kv.dtype),
            )
            ct.scatter(
                paged_score, (phys_sc, blk_off, state_offsets),
                ct.astype(sc_data + ape_data, paged_score.dtype),
            )

    # ================================================================
    # Phase 2: Count compressions and store mask
    # ================================================================
    last_token_idx = sp + NEXT_N - 1
    num_compressions = (last_token_idx + 1) // COMPRESS_RATIO - sp // COMPRESS_RATIO
    if block_idx == 0:
        ct.scatter(compressed_mask, batch_idx, num_compressions > 0)

    # ================================================================
    # Phase 3: Reduction (per-token loop + online softmax weighted avg)
    # ================================================================
    # Precompute head masks: in overlap mode state_dim = 2*head_dim,
    # but Phase 3 output is only head_dim wide. Mask out positions >= head_dim.
    head_mask_score = ct.where(
        state_offsets < head_dim,
        ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32),
        ct.full((BLOCK_SIZE,), -math.inf, dtype=ct.float32),
    )
    head_mask_kv = ct.where(
        state_offsets < head_dim,
        ct.full((BLOCK_SIZE,), 1.0, dtype=ct.float32),
        ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32),
    )
    for c in range(NEXT_N):
        if c < num_compressions:
            compress_idx = sp // COMPRESS_RATIO + c
            curr_chunk_start = compress_idx * COMPRESS_RATIO

            running_max = ct.full((BLOCK_SIZE,), -math.inf, dtype=ct.float32)
            running_sum = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)
            running_wsum = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)

            if IS_OVERLAP:
                # --- Previous chunk: first head_dim features ---
                prev_start = curr_chunk_start - COMPRESS_RATIO
                for r in range(COMPRESS_RATIO):
                    pos = prev_start + r
                    log_blk = pos // page_size
                    blk_off = pos % page_size
                    phys_kv = ct.gather(
                        block_table_kv, (batch_idx, log_blk), padding_value=0,
                    )
                    phys_sc = ct.gather(
                        block_table_score, (batch_idx, log_blk), padding_value=0,
                    )
                    k = ct.astype(
                        ct.gather(paged_kv, (phys_kv, blk_off, state_offsets), padding_value=0),
                        ct.float32,
                    )
                    s = ct.astype(
                        ct.gather(paged_score, (phys_sc, blk_off, state_offsets), padding_value=0),
                        ct.float32,
                    )
                    k = k * head_mask_kv
                    s = s + head_mask_score

                    new_max = ct.maximum(running_max, s)
                    scale = ct.exp(running_max - new_max)
                    term = ct.exp(s - new_max)
                    running_sum = running_sum * scale + term
                    running_wsum = running_wsum * scale + k * term
                    running_max = new_max

                # --- Current chunk: second head_dim features (offset by head_dim) ---
                for r in range(COMPRESS_RATIO):
                    pos = curr_chunk_start + r
                    log_blk = pos // page_size
                    blk_off = pos % page_size
                    phys_kv = ct.gather(
                        block_table_kv, (batch_idx, log_blk), padding_value=0,
                    )
                    phys_sc = ct.gather(
                        block_table_score, (batch_idx, log_blk), padding_value=0,
                    )
                    k = ct.astype(
                        ct.gather(paged_kv, (phys_kv, blk_off, head_dim + state_offsets), padding_value=0),
                        ct.float32,
                    )
                    s = ct.astype(
                        ct.gather(paged_score, (phys_sc, blk_off, head_dim + state_offsets), padding_value=0),
                        ct.float32,
                    )
                    k = k * head_mask_kv
                    s = s + head_mask_score

                    new_max = ct.maximum(running_max, s)
                    scale = ct.exp(running_max - new_max)
                    term = ct.exp(s - new_max)
                    running_sum = running_sum * scale + term
                    running_wsum = running_wsum * scale + k * term
                    running_max = new_max

            else:
                # --- Non-overlap: single chunk ---
                for r in range(COMPRESS_RATIO):
                    pos = curr_chunk_start + r
                    log_blk = pos // page_size
                    blk_off = pos % page_size
                    phys_kv = ct.gather(
                        block_table_kv, (batch_idx, log_blk), padding_value=0,
                    )
                    phys_sc = ct.gather(
                        block_table_score, (batch_idx, log_blk), padding_value=0,
                    )
                    k = ct.astype(
                        ct.gather(paged_kv, (phys_kv, blk_off, state_offsets), padding_value=0),
                        ct.float32,
                    )
                    s = ct.astype(
                        ct.gather(paged_score, (phys_sc, blk_off, state_offsets), padding_value=0),
                        ct.float32,
                    )
                    k = k * head_mask_kv
                    s = s + head_mask_score

                    new_max = ct.maximum(running_max, s)
                    scale = ct.exp(running_max - new_max)
                    term = ct.exp(s - new_max)
                    running_sum = running_sum * scale + term
                    running_wsum = running_wsum * scale + k * term
                    running_max = new_max

            # Write compressed output (2D scatter, check_bounds handles OOB)
            result = running_wsum / running_sum
            ct.scatter(
                output, (out_off + c, state_offsets),
                ct.astype(result, output.dtype),
                check_bounds=True,
            )


def kv_compress_cutile(
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
        kv_lens: [bsz] total KV length per batch
        start_pos: [bsz] starting position. None to auto-compute from kv_lens.
        cu_seq_lens: [bsz+1] cumulative input offsets
        cu_new_comp_kv: [bsz+1] cumulative output offsets
        kv_comp: [total_outputs, head_dim] pre-allocated output buffer
        compressed_mask: [bsz] pre-allocated bool mask buffer
        paged_kv/paged_score: [num_blocks, page_size, state_dim]
        block_table_kv: [bsz, max_blocks]
        block_table_score: [bsz, max_blocks] (if None, uses block_table_kv)
        compress_ratio: Compression factor (4 or 128)
        head_dim: Hidden dimension per head
        overlap: Whether to use overlap mode
        page_size: Tokens per cache block
        next_n: Tokens per request (1 for decode, >1 for MTP)
    """
    if block_table_score is None:
        block_table_score = block_table_kv

    batch_size = kv_lens.shape[0]

    if start_pos is None:
        start_pos = kv_lens - next_n

    coff = 2 if overlap else 1
    state_dim = coff * head_dim

    BLOCK_SIZE = min(_next_power_of_2(state_dim), 512)
    grid = (batch_size, (state_dim + BLOCK_SIZE - 1) // BLOCK_SIZE)

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        paged_kv_compress_cutile_kernel,
        (
            kv_score,
            ape,
            paged_kv,
            paged_score,
            block_table_kv,
            block_table_score,
            kv_comp,
            kv_lens,
            start_pos,
            cu_seq_lens,
            cu_new_comp_kv,
            compressed_mask,
            page_size,
            state_dim,
            head_dim,
            compress_ratio,  # COMPRESS_RATIO (compile-time: 4 or 128)
            overlap,  # IS_OVERLAP
            next_n,  # NEXT_N
            BLOCK_SIZE,
        ),
    )


# ============================================================================
# cuTile Prefill Kernel
# ============================================================================


@ct.kernel
def prefill_reduction_cutile_kernel(
    # N-D data tensors (indexed via N-D gather/scatter)
    kv_score,  # [m, 2*state_dim]
    ape,  # [compress_ratio, state_dim]
    paged_kv,  # [num_blocks, page_size, state_dim]
    paged_score,  # [num_blocks, page_size, state_dim]
    block_table_kv,  # [bsz, max_blocks]
    block_table_score,  # [bsz, max_blocks]
    output,  # [total_outputs, head_dim]
    # Metadata (1D, scalar-indexed)
    kv_lens,  # [bsz]
    start_pos_tensor,  # [bsz]
    cu_seq_lens,  # [bsz+1]
    cu_kv_comp,  # [bsz+1]
    compressed_mask,  # [bsz]
    # Runtime scalars
    page_size: int,
    state_dim: int,
    # Compile-time constants
    head_dim: ConstInt,
    COMPRESS_RATIO: ConstInt,  # 4 or 128
    IS_OVERLAP: ConstBool,
    BLOCK_SIZE: ConstInt,
):
    """Prefill kernel: bulk compression with per-token gather/scatter.

    cuTile equivalent of prefill_reduction_kernel (Triton).
    Grid: (batch_size, max_outputs_per_batch, num_head_chunks)
    """
    batch_idx = ct.bid(0)
    local_output_idx = ct.bid(1)
    head_chunk_idx = ct.bid(2)

    head_offset = head_chunk_idx * BLOCK_SIZE
    head_offsets = ct.arange(BLOCK_SIZE, dtype=ct.int32)

    # Load per-batch metadata
    sp = ct.gather(start_pos_tensor, batch_idx, padding_value=0)
    kv_len = ct.gather(kv_lens, batch_idx, padding_value=0)
    input_offset = ct.gather(cu_seq_lens, batch_idx, padding_value=0)
    output_offset = ct.gather(cu_kv_comp, batch_idx, padding_value=0)

    seqlen = kv_len - sp
    num_outputs = ct.maximum(seqlen // COMPRESS_RATIO, 1)

    # Early exit if out-of-range
    if local_output_idx >= num_outputs:
        return
    if head_offset >= head_dim:
        return

    coff = 2 if IS_OVERLAP else 1
    actual_num_outputs = seqlen // COMPRESS_RATIO
    should_compress = local_output_idx < actual_num_outputs

    # Write compressed_mask (first output block and first head chunk only)
    if local_output_idx == 0 and head_offset == 0:
        ct.scatter(compressed_mask, batch_idx, actual_num_outputs > 0)

    # ================================================================
    # Phase 1: State Update (last output block only)
    # ================================================================
    if local_output_idx == num_outputs - 1:
        remainder = seqlen % COMPRESS_RATIO
        cutoff = seqlen - remainder

        # 1a. Last full chunk (overlap only, when cutoff >= COMPRESS_RATIO)
        if IS_OVERLAP and cutoff >= COMPRESS_RATIO:
            for r in range(COMPRESS_RATIO):
                write_pos = sp + cutoff - COMPRESS_RATIO + r
                log_blk = write_pos // page_size
                blk_off = write_pos % page_size
                phys_kv = ct.gather(
                    block_table_kv, (batch_idx, log_blk), padding_value=0,
                )
                phys_sc = ct.gather(
                    block_table_score, (batch_idx, log_blk), padding_value=0,
                )

                base_row = cutoff - COMPRESS_RATIO + r
                for col_idx in range(2):
                    col_off = col_idx * head_dim + head_offset
                    # Load kv, score, ape (2D gather)
                    kv_data = ct.astype(
                        ct.gather(kv_score, (input_offset + base_row, col_off + head_offsets), padding_value=0),
                        ct.float32,
                    )
                    sc_data = ct.astype(
                        ct.gather(kv_score, (input_offset + base_row, state_dim + col_off + head_offsets), padding_value=0),
                        ct.float32,
                    )
                    ape_data = ct.astype(
                        ct.gather(ape, (r, col_off + head_offsets), padding_value=0),
                        ct.float32,
                    )

                    # Write to paged caches (3D scatter)
                    ct.scatter(
                        paged_kv, (phys_kv, blk_off, col_off + head_offsets),
                        ct.astype(kv_data, paged_kv.dtype),
                    )
                    ct.scatter(
                        paged_score, (phys_sc, blk_off, col_off + head_offsets),
                        ct.astype(sc_data + ape_data, paged_score.dtype),
                    )

        # 1b. Remainder tokens
        if remainder > 0:
            for r in range(COMPRESS_RATIO):
                if r < remainder:
                    write_pos = sp + cutoff + r
                    log_blk = write_pos // page_size
                    blk_off = write_pos % page_size
                    phys_kv = ct.gather(
                        block_table_kv, (batch_idx, log_blk), padding_value=0,
                    )
                    phys_sc = ct.gather(
                        block_table_score, (batch_idx, log_blk), padding_value=0,
                    )

                    base_row = cutoff + r
                    for col_idx in range(2):
                        if col_idx < coff:
                            col_off = col_idx * head_dim + head_offset
                            kv_data = ct.astype(
                                ct.gather(kv_score, (input_offset + base_row, col_off + head_offsets), padding_value=0),
                                ct.float32,
                            )
                            sc_data = ct.astype(
                                ct.gather(kv_score, (input_offset + base_row, state_dim + col_off + head_offsets), padding_value=0),
                                ct.float32,
                            )
                            ape_data = ct.astype(
                                ct.gather(ape, (r, col_off + head_offsets), padding_value=0),
                                ct.float32,
                            )

                            ct.scatter(
                                paged_kv, (phys_kv, blk_off, col_off + head_offsets),
                                ct.astype(kv_data, paged_kv.dtype),
                            )
                            ct.scatter(
                                paged_score, (phys_sc, blk_off, col_off + head_offsets),
                                ct.astype(sc_data + ape_data, paged_score.dtype),
                            )

    # ================================================================
    # Phase 2: Reduction (per-token loop + online softmax)
    # ================================================================
    if not should_compress:
        return

    # Precompute head_mask for boundary chunks
    head_mask_score = ct.where(
        head_offset + head_offsets < head_dim,
        ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32),
        ct.full((BLOCK_SIZE,), -math.inf, dtype=ct.float32),
    )
    head_mask_kv = ct.where(
        head_offset + head_offsets < head_dim,
        ct.full((BLOCK_SIZE,), 1.0, dtype=ct.float32),
        ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32),
    )

    running_max = ct.full((BLOCK_SIZE,), -math.inf, dtype=ct.float32)
    running_sum = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)
    running_wsum = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)

    if IS_OVERLAP:
        # Previous segment (col = head_offset, first head_dim features)
        if local_output_idx > 0:
            input_start = (local_output_idx - 1) * COMPRESS_RATIO
            for r in range(COMPRESS_RATIO):
                row = input_offset + input_start + r
                k = ct.astype(
                    ct.gather(kv_score, (row, head_offset + head_offsets), padding_value=0),
                    ct.float32,
                )
                s = ct.astype(
                    ct.gather(kv_score, (row, state_dim + head_offset + head_offsets), padding_value=0),
                    ct.float32,
                )
                a = ct.astype(
                    ct.gather(ape, (r, head_offset + head_offsets), padding_value=0),
                    ct.float32,
                )
                s = s + a
                k = k * head_mask_kv
                s = s + head_mask_score

                new_max = ct.maximum(running_max, s)
                scale = ct.exp(running_max - new_max)
                term = ct.exp(s - new_max)
                running_sum = running_sum * scale + term
                running_wsum = running_wsum * scale + k * term
                running_max = new_max

        # Current segment (col = head_dim + head_offset, second head_dim features)
        cur_start = local_output_idx * COMPRESS_RATIO
        for r in range(COMPRESS_RATIO):
            row = input_offset + cur_start + r
            k = ct.astype(
                ct.gather(kv_score, (row, head_dim + head_offset + head_offsets), padding_value=0),
                ct.float32,
            )
            s = ct.astype(
                ct.gather(kv_score, (row, state_dim + head_dim + head_offset + head_offsets), padding_value=0),
                ct.float32,
            )
            a = ct.astype(
                ct.gather(ape, (r, head_dim + head_offset + head_offsets), padding_value=0),
                ct.float32,
            )
            s = s + a
            k = k * head_mask_kv
            s = s + head_mask_score

            new_max = ct.maximum(running_max, s)
            scale = ct.exp(running_max - new_max)
            term = ct.exp(s - new_max)
            running_sum = running_sum * scale + term
            running_wsum = running_wsum * scale + k * term
            running_max = new_max

    else:
        # Non-overlap: single segment (col = head_offset)
        input_start = local_output_idx * COMPRESS_RATIO
        for r in range(COMPRESS_RATIO):
            row = input_offset + input_start + r
            k = ct.astype(
                ct.gather(kv_score, (row, head_offset + head_offsets), padding_value=0),
                ct.float32,
            )
            s = ct.astype(
                ct.gather(kv_score, (row, state_dim + head_offset + head_offsets), padding_value=0),
                ct.float32,
            )
            a = ct.astype(
                ct.gather(ape, (r, head_offset + head_offsets), padding_value=0),
                ct.float32,
            )
            s = s + a
            k = k * head_mask_kv
            s = s + head_mask_score

            new_max = ct.maximum(running_max, s)
            scale = ct.exp(running_max - new_max)
            term = ct.exp(s - new_max)
            running_sum = running_sum * scale + term
            running_wsum = running_wsum * scale + k * term
            running_max = new_max

    # Output scatter (2D, check_bounds handles OOB for head_offset + head_offsets >= head_dim)
    result = running_wsum / running_sum
    ct.scatter(
        output, (output_offset + local_output_idx, head_offset + head_offsets),
        ct.astype(result, output.dtype),
        check_bounds=True,
    )


def kv_compress_prefill_cutile(
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
    """Prefill kernel: bulk compression with per-token gather/scatter."""
    if block_table_score is None:
        block_table_score = block_table_kv

    batch_size = kv_lens.shape[0]

    if start_pos is None:
        start_pos = torch.zeros(batch_size, device=kv_lens.device, dtype=torch.int32)

    coff = 2 if overlap else 1
    state_dim = coff * head_dim

    # Compute grid dimensions
    seq_lens = kv_lens - start_pos
    num_outputs_per_batch = torch.clamp(seq_lens // compress_ratio, min=1)
    max_outputs = num_outputs_per_batch.max().item()

    # Match Triton's BLOCK_SIZE heuristic for grid parallelism
    _SMEM_LIMIT = 230000
    _NUM_LIVE = 4  # k, s, ape, exp in worst case
    _max_bs = _SMEM_LIMIT // (compress_ratio * 4 * _NUM_LIVE)
    BLOCK_SIZE = max(64, 1 << (_max_bs.bit_length() - 1)) if _max_bs >= 64 else 64
    BLOCK_SIZE = min(BLOCK_SIZE, _next_power_of_2(head_dim))
    num_head_chunks = (head_dim + BLOCK_SIZE - 1) // BLOCK_SIZE

    grid = (batch_size, max_outputs, num_head_chunks)

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        prefill_reduction_cutile_kernel,
        (
            kv_score,
            ape,
            paged_kv,
            paged_score,
            block_table_kv,
            block_table_score,
            kv_comp,
            kv_lens,
            start_pos,
            cu_seq_lens,
            cu_new_comp_kv,
            compressed_mask,
            page_size,
            state_dim,
            head_dim,
            compress_ratio,  # COMPRESS_RATIO
            overlap,  # IS_OVERLAP
            BLOCK_SIZE,
        ),
    )


# ============================================================================
# cuTile Scatter: Compressed KV → Paged Cache
# ============================================================================


@ct.kernel
def compressed_kv_scatter_cutile_kernel(
    # 2D data tensors (indexed via 2D gather/scatter)
    compressed_kv,  # [total_outputs, head_dim]
    kv_scale,  # [num_tokens, num_scale_blocks] fp32 (blockwise FP8 only)
    kv_cache,  # [num_blocks, block_elems] (uint8 for FP8, native dtype for default)
    kv_cache_scale,  # [num_blocks, block_elems_fp32] fp32 view (blockwise FP8 only)
    # 1D metadata (scalar-indexed)
    num_outputs,  # [bsz]
    cu_kv_comp,  # [bsz+1]
    start_pos_tensor,  # [bsz]
    # 2D block table (indexed via 2D gather)
    block_offsets,  # [bsz, max_blocks]
    # Runtime scalar
    tokens_per_block: int,
    # Compile-time constants
    head_dim: ConstInt,
    num_scale_blocks: ConstInt,
    BLOCK_SIZE_H: ConstInt,
    BLOCK_SIZE_S: ConstInt,
    IS_BLOCKWISE_FP8: ConstBool,
):
    """Scatter compressed KV tokens to paged cache.

    cuTile equivalent of compressed_kv_scatter_kernel (Triton).
    Grid: (batch_size, max_outputs_per_batch)

    Uses 2D gather/scatter to index tensors in their original shapes,
    avoiding host-side flatten/reshape ops for torch.compile compatibility.
    """
    batch_idx = ct.bid(0)
    local_output_idx = ct.bid(1)

    # Load per-batch metadata (scalar gathers)
    n_outputs = ct.gather(num_outputs, batch_idx, padding_value=0)
    if local_output_idx >= n_outputs:
        return

    start_pos = ct.gather(start_pos_tensor, batch_idx, padding_value=0)
    cu_offset = ct.gather(cu_kv_comp, batch_idx, padding_value=0)

    # Compute cache position
    cache_pos = start_pos + local_output_idx
    logical_block = cache_pos // tokens_per_block
    token_offset = cache_pos % tokens_per_block

    # Physical block lookup (2D index into block_offsets)
    phys_block = ct.gather(
        block_offsets,
        (batch_idx, logical_block),
        padding_value=0,
    )

    # Global token index in input
    global_token_idx = cu_offset + local_output_idx

    if IS_BLOCKWISE_FP8:
        # FP8 blockwise: non-interleaved layout per block row:
        #   [fp8_data: tpb * head_dim cols | scale_data: tpb * num_scale_blocks cols (fp32)]
        # FP8 col in kv_cache (uint8): token_offset * head_dim + h
        # Scale col in kv_cache_scale (fp32): (tpb * head_dim) // 4 + token_offset * nsb + s
        fp8_col_base = token_offset * head_dim
        scale_col_base = (tokens_per_block * head_dim) // 4 + token_offset * num_scale_blocks

        # Scatter FP8 data (uint8 byte-level)
        for h_start in range(0, head_dim, BLOCK_SIZE_H):
            h_offsets = h_start + ct.arange(BLOCK_SIZE_H, dtype=ct.int32)
            data = ct.gather(compressed_kv, (global_token_idx, h_offsets), padding_value=0)
            ct.scatter(kv_cache, (phys_block, fp8_col_base + h_offsets), data)

        # Scatter scale data (fp32)
        for s_start in range(0, num_scale_blocks, BLOCK_SIZE_S):
            s_offsets = s_start + ct.arange(BLOCK_SIZE_S, dtype=ct.int32)
            scale_data = ct.gather(kv_scale, (global_token_idx, s_offsets), padding_value=0)
            ct.scatter(kv_cache_scale, (phys_block, scale_col_base + s_offsets), scale_data)
    else:
        # Default mode or per-tensor FP8: direct element copy
        col_base = token_offset * head_dim

        for h_start in range(0, head_dim, BLOCK_SIZE_H):
            h_offsets = h_start + ct.arange(BLOCK_SIZE_H, dtype=ct.int32)
            data = ct.gather(compressed_kv, (global_token_idx, h_offsets), padding_value=0)
            ct.scatter(kv_cache, (phys_block, col_base + h_offsets), ct.astype(data, kv_cache.dtype))


def compressed_kv_scatter_cutile(
    compressed_kv: torch.Tensor,  # [total_outputs, head_dim] packed, any dtype
    num_comp_tokens: torch.Tensor,  # [bsz] number of compressed tokens per batch
    cu_new_comp_kv: torch.Tensor,  # [bsz+1] cumulative output offsets
    start_pos: torch.Tensor,  # [bsz] compressed cache position
    kv_cache: torch.Tensor,  # [num_blocks, ...] paged cache
    block_offsets: torch.Tensor,  # [num_seqs, max_blocks_per_seq]
    tokens_per_block: int,
    head_dim: int,
    kv_cache_dtype: str = "default",
    kv_scale: torch.Tensor = None,  # [total_tokens, num_scale_blocks] fp32, for fp8_blockwise
):
    """Scatter compressed KV to paged cache.

    Supports multiple KV cache formats:
    - "default": Any dtype (bf16, fp16, etc.) - direct element copy
    - "fp8_blockwise": FP8 data with blockwise scales (requires kv_scale)
    - "fp8_pertensor": FP8 data with per-tensor scale (uint8 byte copy)

    Uses 2D gather/scatter in the kernel to avoid host-side flatten/reshape
    ops, making the wrapper compatible with torch.compile/graph mode.
    """
    if compressed_kv.numel() == 0:
        return

    batch_size = num_comp_tokens.shape[0]
    max_outputs = num_comp_tokens.max().item()

    if max_outputs == 0:
        return

    block_size_h = min(_next_power_of_2(head_dim), 512)
    grid = (batch_size, max_outputs)

    # Ensure cache is 2D [num_blocks, block_elems] (collapse kv_factor if 3D)
    cache_2d = kv_cache if kv_cache.dim() == 2 else kv_cache.reshape(kv_cache.shape[0], -1)

    if kv_cache_dtype == "fp8_blockwise":
        assert kv_scale is not None, "kv_scale required for fp8_blockwise mode"
        num_scale_blocks = kv_scale.shape[1]
        block_size_s = min(_next_power_of_2(num_scale_blocks), 64)

        kv_uint8 = compressed_kv.view(torch.uint8)  # [total_outputs, head_dim] uint8
        cache_uint8 = cache_2d.view(torch.uint8)  # [num_blocks, block_elems] uint8
        cache_fp32 = cache_uint8.view(torch.float32)  # [num_blocks, block_elems//4] fp32

        ct.launch(
            torch.cuda.current_stream(),
            grid,
            compressed_kv_scatter_cutile_kernel,
            (
                kv_uint8,
                kv_scale,
                cache_uint8,
                cache_fp32,
                num_comp_tokens,
                cu_new_comp_kv,
                start_pos,
                block_offsets,
                tokens_per_block,
                head_dim,
                num_scale_blocks,
                block_size_h,
                block_size_s,
                True,  # IS_BLOCKWISE_FP8
            ),
        )
    elif kv_cache_dtype == "fp8_pertensor":
        cache_uint8 = cache_2d.view(torch.uint8)
        # Dummy fp32 tensor (not accessed when IS_BLOCKWISE_FP8=False)
        dummy_fp32 = cache_uint8.view(torch.float32)

        ct.launch(
            torch.cuda.current_stream(),
            grid,
            compressed_kv_scatter_cutile_kernel,
            (
                compressed_kv,
                dummy_fp32,
                cache_uint8,
                dummy_fp32,
                num_comp_tokens,
                cu_new_comp_kv,
                start_pos,
                block_offsets,
                tokens_per_block,
                head_dim,
                0,  # num_scale_blocks (not used)
                block_size_h,
                16,  # BLOCK_SIZE_S (not used)
                False,  # IS_BLOCKWISE_FP8
            ),
        )
    else:
        # Convert input to match cache dtype (handles bf16→fp8 when cache is fp8;
        # CuTile doesn't support bf16→fp8 conversion inside kernels)
        input_kv = compressed_kv.to(cache_2d.dtype) if compressed_kv.dtype != cache_2d.dtype else compressed_kv

        # Dummy fp32 2D tensor (not accessed when IS_BLOCKWISE_FP8=False)
        dummy_fp32 = torch.empty(1, 1, dtype=torch.float32, device=compressed_kv.device)

        ct.launch(
            torch.cuda.current_stream(),
            grid,
            compressed_kv_scatter_cutile_kernel,
            (
                input_kv,
                dummy_fp32,
                cache_2d,
                dummy_fp32,
                num_comp_tokens,
                cu_new_comp_kv,
                start_pos,
                block_offsets,
                tokens_per_block,
                head_dim,
                0,  # num_scale_blocks (not used)
                block_size_h,
                16,  # BLOCK_SIZE_S (not used)
                False,  # IS_BLOCKWISE_FP8
            ),
        )
