# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test suite for KV compressor kernels (cuTile).

Tests cover: prefill/decode corner cases, state updates, varlen, MTP support.
Run: pytest -s tests/unittest/_torch/attention/sparse/test_compressor_kernel.py
"""

from typing import Tuple

import pytest
import torch
import triton

from tensorrt_llm._torch.attention_backend.sparse.mewtwo.kernel import (
    compressed_kv_scatter_cutile,
    kv_compress_cutile,
    kv_compress_prefill_cutile,
)


def prepare_compress_output(
    cu_new_comp_kv: torch.Tensor,
    batch_size: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pre-allocate output tensors for compression kernels.

    Args:
        cu_new_comp_kv: [bsz+1] cumulative output offsets
        batch_size: Number of batches
        head_dim: Dimension of KV head
        device: Target device
        dtype: Output dtype (default bfloat16)

    Returns:
        kv_comp: [total_outputs, head_dim] output buffer
        compressed_mask: [batch_size] bool mask buffer
    """
    total_outputs = cu_new_comp_kv[-1].item()
    kv_comp = torch.empty(total_outputs, head_dim, device=device, dtype=dtype)
    compressed_mask = torch.empty(batch_size, device=device, dtype=torch.bool)
    return kv_comp, compressed_mask


# ============================================================================
# PyTorch References (mirror model.py logic)
# ============================================================================


def run_pytorch_reference(
    new_kv, new_score, ape, kv_state, score_state, token_idx, compress_ratio, head_dim, overlap
):
    """Decode reference: single token update + conditional compression."""
    bsz = kv_state.shape[0]
    should_compress = (token_idx + 1) % compress_ratio == 0
    ape_ratio = token_idx % compress_ratio

    score_val = new_score + ape[ape_ratio]
    kv_val = new_kv
    output = None

    if overlap:
        update_idx = compress_ratio + ape_ratio
        kv_state[:bsz, update_idx] = kv_val.squeeze(1)
        score_state[:bsz, update_idx] = score_val.squeeze(1)
        if should_compress:
            d = head_dim
            kv_cat = torch.cat(
                [kv_state[:bsz, :compress_ratio, :d], kv_state[:bsz, compress_ratio:, d:]], dim=1
            )
            score_cat = torch.cat(
                [score_state[:bsz, :compress_ratio, :d], score_state[:bsz, compress_ratio:, d:]],
                dim=1,
            )
            output = (kv_cat * score_cat.softmax(dim=1)).sum(dim=1, keepdim=True)
            kv_state[:bsz, :compress_ratio] = kv_state[:bsz, compress_ratio:].clone()
            score_state[:bsz, :compress_ratio] = score_state[:bsz, compress_ratio:].clone()
    else:
        kv_state[:bsz, ape_ratio] = kv_val.squeeze(1)
        score_state[:bsz, ape_ratio] = score_val.squeeze(1)
        if should_compress:
            output = (kv_state[:bsz] * score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)

    return output


def run_pytorch_prefill_reference(
    kv, score, ape, kv_state, score_state, compress_ratio, head_dim, overlap
):
    """Prefill reference: bulk compression + state update."""
    bsz, seqlen, _ = kv.size()
    ratio = compress_ratio
    remainder = seqlen % ratio
    cutoff = seqlen - remainder
    offset = ratio if overlap else 0

    # State update
    if overlap and cutoff >= ratio:
        kv_state[:bsz, :ratio] = kv[:, cutoff - ratio : cutoff]
        score_state[:bsz, :ratio] = score[:, cutoff - ratio : cutoff] + ape

    if remainder > 0:
        kv, kv_state[:bsz, offset : offset + remainder] = kv.split([cutoff, remainder], dim=1)
        score_state[:bsz, offset : offset + remainder] = score[:, cutoff:] + ape[:remainder]
        score = score[:, :cutoff]

    if cutoff == 0:
        return None

    kv = kv.unflatten(1, (-1, ratio))
    score = score.unflatten(1, (-1, ratio)) + ape

    if overlap:
        b, s, r, _ = kv.size()
        d = head_dim

        kv_transformed = torch.zeros(b, s, 2 * ratio, d, device=kv.device, dtype=kv.dtype)
        score_transformed = torch.full(
            (b, s, 2 * ratio, d), float("-inf"), device=score.device, dtype=score.dtype
        )

        kv_transformed[:, :, ratio:] = kv[:, :, :, d:]
        score_transformed[:, :, ratio:] = score[:, :, :, d:]
        kv_transformed[:, 1:, :ratio] = kv[:, :-1, :, :d]
        score_transformed[:, 1:, :ratio] = score[:, :-1, :, :d]

        kv, score = kv_transformed, score_transformed

    output = (kv * score.softmax(dim=2)).sum(dim=2)
    return output if seqlen >= ratio else None


def run_pytorch_prefill_reference_varlen(
    kv_score, ape, kv_lens, start_pos, compress_ratio, head_dim, overlap, kv_state, score_state
):
    """Varlen prefill reference: process each batch independently."""
    bsz = kv_lens.shape[0]
    coff = 2 if overlap else 1
    state_dim = coff * head_dim
    ratio = compress_ratio
    offset = ratio if overlap else 0

    # Compute sequence lengths and cumulative offsets
    seq_lens = kv_lens - start_pos
    cu_seq_lens = torch.zeros(bsz + 1, device=kv_score.device, dtype=torch.int32)
    cu_seq_lens[1:] = torch.cumsum(seq_lens, dim=0)

    outputs = []

    for b in range(bsz):
        seqlen = seq_lens[b].item()
        input_start = cu_seq_lens[b].item()

        kv_score_b = kv_score[input_start : input_start + seqlen]
        kv = kv_score_b[:, :state_dim].unsqueeze(0)
        score = kv_score_b[:, state_dim:].unsqueeze(0)

        remainder = seqlen % ratio
        cutoff = seqlen - remainder

        # State update
        if overlap and cutoff >= ratio:
            kv_state[b : b + 1, :ratio] = kv[:, cutoff - ratio : cutoff]
            score_state[b : b + 1, :ratio] = score[:, cutoff - ratio : cutoff] + ape

        if remainder > 0:
            kv_state[b : b + 1, offset : offset + remainder] = kv[:, cutoff:]
            score_state[b : b + 1, offset : offset + remainder] = (
                score[:, cutoff:] + ape[:remainder]
            )

        if cutoff == 0:
            continue

        kv_comp = kv[:, :cutoff].unflatten(1, (-1, ratio))
        score_comp = score[:, :cutoff].unflatten(1, (-1, ratio)) + ape

        if overlap:
            _, s, r, _ = kv_comp.size()
            d = head_dim

            kv_transformed = torch.zeros(1, s, 2 * ratio, d, device=kv.device, dtype=kv.dtype)
            score_transformed = torch.full(
                (1, s, 2 * ratio, d), float("-inf"), device=score.device, dtype=score.dtype
            )
            kv_transformed[:, :, ratio:] = kv_comp[:, :, :, d:]
            score_transformed[:, :, ratio:] = score_comp[:, :, :, d:]
            kv_transformed[:, 1:, :ratio] = kv_comp[:, :-1, :, :d]
            score_transformed[:, 1:, :ratio] = score_comp[:, :-1, :, :d]
            kv_comp, score_comp = kv_transformed, score_transformed

        output = (kv_comp * score_comp.softmax(dim=2)).sum(dim=2)
        outputs.append(output.squeeze(0))

    if outputs:
        return torch.cat(outputs, dim=0)  # [total_outputs, head_dim]
    return torch.empty(0, head_dim, device=kv_score.device, dtype=torch.float32)


# ============================================================================
# Test Utilities
# ============================================================================


def fuse_kv_score(kv, score):
    """Fuse kv and score into [*, 2*dim]."""
    return torch.cat([kv, score], dim=-1)


def prepare_decode_metadata(
    batch_size: int,
    compress_ratio: int,
    head_dim: int,
    device: torch.device,
    next_n: int = 1,
):
    """Prepare decode metadata for tests."""
    max_compressions = (next_n + compress_ratio - 1) // compress_ratio
    cu_seq_lens = torch.arange(
        0, (batch_size + 1) * next_n, next_n, device=device, dtype=torch.int32
    )
    cu_outputs = torch.arange(
        0, (batch_size + 1) * max_compressions, max_compressions, device=device, dtype=torch.int32
    )
    return cu_seq_lens, cu_outputs


def prepare_prefill_metadata(
    kv_lens: torch.Tensor,
    start_pos: torch.Tensor,
    compress_ratio: int,
    head_dim: int,
    device: torch.device = None,
):
    """Prepare prefill metadata for tests."""
    batch_size = kv_lens.shape[0]
    if device is None:
        device = kv_lens.device

    if start_pos is None:
        start_pos = torch.zeros(batch_size, device=device, dtype=torch.int32)

    seq_lens = kv_lens - start_pos
    num_outputs_per_batch = torch.clamp(seq_lens // compress_ratio, min=1)

    cu_seq_lens = torch.zeros(batch_size + 1, device=device, dtype=torch.int32)
    cu_seq_lens[1:] = torch.cumsum(seq_lens, dim=0)

    cu_outputs = torch.zeros(batch_size + 1, device=device, dtype=torch.int32)
    cu_outputs[1:] = torch.cumsum(num_outputs_per_batch, dim=0)

    return cu_seq_lens, cu_outputs


def create_paged_cache(batch_size, seqlen, compress_ratio, head_dim, overlap, page_size=4):
    """Create paged cache tensors."""
    coff = 2 if overlap else 1
    state_dim = coff * head_dim
    total_positions = seqlen + (compress_ratio if overlap else 0)
    max_blocks = (total_positions + page_size - 1) // page_size
    num_blocks = batch_size * max_blocks

    paged_kv = torch.zeros(num_blocks, page_size, state_dim, device="cuda")
    paged_score = torch.zeros(num_blocks, page_size, state_dim, device="cuda")
    block_table = torch.arange(num_blocks, device="cuda", dtype=torch.int32).view(
        batch_size, max_blocks
    )

    return paged_kv, paged_score, block_table, page_size, max_blocks


def pack_prefill_inputs(kv_list, score_list):
    """Pack variable-length inputs into [m, 2*state_dim] format."""
    seq_lens = torch.tensor([kv.shape[0] for kv in kv_list], device="cuda", dtype=torch.int32)
    kv_score = fuse_kv_score(torch.cat(kv_list, dim=0), torch.cat(score_list, dim=0))
    return kv_score, seq_lens


def create_compressed_kv_cache(batch_size, max_compressed_len, head_dim, tokens_per_block=4):
    """Create paged cache for compressed KV storage."""
    max_blocks_per_seq = (max_compressed_len + tokens_per_block - 1) // tokens_per_block
    num_blocks = batch_size * max_blocks_per_seq
    kv_factor = 1
    kv_cache = torch.zeros(num_blocks, kv_factor, tokens_per_block * head_dim, device="cuda")
    block_offsets = torch.zeros(batch_size, max_blocks_per_seq, device="cuda", dtype=torch.int32)
    for b in range(batch_size):
        block_ids = torch.arange(
            b * max_blocks_per_seq, (b + 1) * max_blocks_per_seq, device="cuda", dtype=torch.int32
        )
        block_offsets[b, :] = block_ids
    return kv_cache, block_offsets, tokens_per_block, max_blocks_per_seq


# ============================================================================
# Correctness Tests
# ============================================================================

PREFILL_CONFIGS = [
    # Overlap mode (ratio=4)
    pytest.param(1, 3, 4, 8, True, id="overlap_seqlen_lt_ratio"),
    pytest.param(1, 4, 4, 8, True, id="overlap_seqlen_eq_ratio"),
    pytest.param(1, 5, 4, 8, True, id="overlap_1chunk_1remainder"),
    pytest.param(1, 8, 4, 8, True, id="overlap_2chunks"),
    pytest.param(1, 9, 4, 8, True, id="overlap_2chunks_1remainder"),
    pytest.param(1, 20, 4, 16, True, id="overlap_5chunks"),
    pytest.param(2, 20, 4, 16, True, id="overlap_multi_batch"),
    pytest.param(4, 32, 4, 128, True, id="overlap_large_head_dim"),
    pytest.param(1, 100, 4, 64, True, id="overlap_25chunks"),
    # Basic mode (ratio=128)
    pytest.param(1, 64, 128, 8, False, id="basic_seqlen_lt_ratio"),
    pytest.param(1, 128, 128, 8, False, id="basic_seqlen_eq_ratio"),
    pytest.param(1, 129, 128, 8, False, id="basic_1chunk_1remainder"),
    pytest.param(1, 256, 128, 8, False, id="basic_2chunks"),
    pytest.param(1, 260, 128, 8, False, id="basic_2chunks_4remainder"),
    pytest.param(2, 512, 128, 64, False, id="basic_multi_batch_large"),
]


@pytest.mark.parametrize("batch_size,seqlen,compress_ratio,head_dim,overlap", PREFILL_CONFIGS)
def test_prefill_corner_cases(batch_size, seqlen, compress_ratio, head_dim, overlap):
    """Test prefill kernel corner cases."""
    coff = 2 if overlap else 1
    state_len, state_dim = coff * compress_ratio, coff * head_dim

    kv = torch.randn(batch_size, seqlen, state_dim, device="cuda")
    score = torch.randn(batch_size, seqlen, state_dim, device="cuda")
    ape = torch.randn(compress_ratio, state_dim, device="cuda")
    kv_state_py = torch.zeros(batch_size, state_len, state_dim, device="cuda")
    score_state_py = torch.full((batch_size, state_len, state_dim), float("-inf"), device="cuda")
    paged_kv, paged_score, block_table, page_size, _ = create_paged_cache(
        batch_size, seqlen, compress_ratio, head_dim, overlap
    )

    out_py = run_pytorch_prefill_reference(
        kv.clone(),
        score.clone(),
        ape,
        kv_state_py,
        score_state_py,
        compress_ratio,
        head_dim,
        overlap,
    )

    kv_score = fuse_kv_score(kv.view(-1, state_dim), score.view(-1, state_dim))
    kv_lens = torch.full((batch_size,), seqlen, device="cuda", dtype=torch.int32)
    start_pos = torch.zeros(batch_size, device="cuda", dtype=torch.int32)
    cu_seq_lens, cu_outputs = prepare_prefill_metadata(
        kv_lens, start_pos, compress_ratio, head_dim, kv_score.device
    )
    kv_comp, compressed_mask = prepare_compress_output(
        cu_outputs, batch_size, head_dim, kv_score.device, torch.bfloat16
    )
    seq_lens = kv_lens - start_pos
    num_outputs_per_batch = torch.clamp(seq_lens // compress_ratio, min=1)
    max_outputs = num_outputs_per_batch.max().item()

    kv_compress_prefill_cutile(
        kv_score,
        ape,
        kv_lens,
        start_pos,
        cu_seq_lens,
        cu_outputs,
        kv_comp,
        compressed_mask,
        paged_kv,
        paged_score,
        block_table,
        max_outputs,
        block_table,
        compress_ratio,
        head_dim,
        overlap,
        page_size,
    )

    num_chunks = seqlen // compress_ratio
    if out_py is None or num_chunks == 0:
        pass  # No compression expected
    elif kv_comp.numel() == 0:
        pytest.fail("cuTile returned empty output but PyTorch returned valid output")
    else:
        out_reshaped = kv_comp.view(batch_size, num_chunks, head_dim)
        assert torch.allclose(out_py.to(kv_comp.dtype), out_reshaped, rtol=2e-3, atol=1e-3), (
            f"Output mismatch: max diff = {(out_py.to(kv_comp.dtype) - out_reshaped).abs().max():.6f}"
        )


DECODE_CONFIGS = [
    pytest.param(1, 4, 8, True, 16, id="overlap_4compressions"),
    pytest.param(2, 4, 8, True, 8, id="overlap_multi_batch"),
    pytest.param(1, 128, 8, False, 256, id="basic_2compressions"),
    pytest.param(1, 4, 128, True, 12, id="overlap_large_head_dim"),
    pytest.param(4, 4, 64, True, 20, id="overlap_multi_batch_large"),
    pytest.param(1, 4, 8, True, 4, id="overlap_1compression"),
    pytest.param(2, 128, 64, False, 384, id="basic_multi_batch_3comp"),
]


@pytest.mark.parametrize("batch_size,compress_ratio,head_dim,overlap,num_steps", DECODE_CONFIGS)
def test_decode_corner_cases(batch_size, compress_ratio, head_dim, overlap, num_steps):
    """Test decode kernel corner cases."""
    coff = 2 if overlap else 1
    state_len, state_dim = coff * compress_ratio, coff * head_dim
    page_size = 8
    max_blocks = (num_steps + compress_ratio + page_size - 1) // page_size

    ape = torch.randn(compress_ratio, state_dim, device="cuda")

    num_blocks = batch_size * max_blocks
    paged_kv = torch.zeros(num_blocks, page_size, state_dim, device="cuda")
    paged_score = torch.zeros(num_blocks, page_size, state_dim, device="cuda")
    block_table = torch.arange(num_blocks, device="cuda", dtype=torch.int32).view(
        batch_size, max_blocks
    )

    kv_state_py = torch.zeros(batch_size, state_len, state_dim, device="cuda")
    score_state_py = torch.full((batch_size, state_len, state_dim), float("-inf"), device="cuda")

    # Pre-compute decode metadata
    cu_seq_lens, cu_outputs = prepare_decode_metadata(
        batch_size, compress_ratio, head_dim, torch.device("cuda"), next_n=1
    )
    kv_comp, compressed_mask = prepare_compress_output(
        cu_outputs, batch_size, head_dim, torch.device("cuda"), torch.bfloat16
    )

    # Pre-fill for overlap mode (all batches)
    if overlap:
        init_kv = torch.randn(compress_ratio, state_dim, device="cuda")
        init_score = torch.randn(compress_ratio, state_dim, device="cuda")
        kv_state_py[:, :compress_ratio] = init_kv
        score_state_py[:, :compress_ratio] = init_score
        for b in range(batch_size):
            for r in range(compress_ratio):
                log_block, offset = r // page_size, r % page_size
                phys_block = block_table[b, log_block].item()
                paged_kv[phys_block, offset] = init_kv[r]
                paged_score[phys_block, offset] = init_score[r]

    for step in range(num_steps):
        new_kv = torch.randn(batch_size, state_dim, device="cuda")
        new_score = torch.randn(batch_size, state_dim, device="cuda")

        # For overlap mode, account for initial compress_ratio tokens in cache
        # token_idx is the absolute position: compress_ratio + step for overlap, step for basic
        token_idx = (compress_ratio + step) if overlap else step
        total_tokens = token_idx + 1

        # PyTorch reference
        out_py = run_pytorch_reference(
            new_kv.unsqueeze(1),
            new_score.unsqueeze(1),
            ape,
            kv_state_py,
            score_state_py,
            token_idx,
            compress_ratio,
            head_dim,
            overlap,
        )

        # cuTile kernel
        kv_score = fuse_kv_score(new_kv, new_score)  # [bsz, 2*state_dim]
        kv_lens = torch.full((batch_size,), total_tokens, device="cuda", dtype=torch.int32)
        start_pos = torch.full((batch_size,), token_idx, device="cuda", dtype=torch.int32)

        kv_compress_cutile(
            kv_score,
            ape,
            kv_lens,
            start_pos,
            cu_seq_lens,
            cu_outputs,
            kv_comp,
            compressed_mask,
            paged_kv,
            paged_score,
            block_table,
            block_table,
            compress_ratio,
            head_dim,
            overlap,
            page_size,
            next_n=1,
        )

        should_compress = (step + 1) % compress_ratio == 0
        if should_compress:
            assert compressed_mask.all(), f"Step {step}: expected compression but mask is False"
            if out_py is not None:
                for b in range(batch_size):
                    out_idx = cu_outputs[b].item()
                    diff = out_py[b, 0, :head_dim].to(kv_comp.dtype) - kv_comp[out_idx, :]
                    assert torch.allclose(
                        out_py[b, 0, :head_dim].to(kv_comp.dtype),
                        kv_comp[out_idx, :],
                        rtol=1e-2,
                        atol=1e-3,
                    ), f"Step {step}, Batch {b}: mismatch diff={(diff).abs().max():.6f}"
        else:
            assert not compressed_mask.any(), f"Step {step}: unexpected compression"


STATE_UPDATE_CONFIGS = [
    pytest.param(1, 9, 4, 8, True, id="overlap_2chunks_1remainder"),
    # pytest.param(1, 130, 128, 8, False, id="basic_1chunk_2remainder"),
]


@pytest.mark.parametrize("batch_size,seqlen,compress_ratio,head_dim,overlap", STATE_UPDATE_CONFIGS)
def test_prefill_state_update(batch_size, seqlen, compress_ratio, head_dim, overlap):
    """Verify prefill state updates match reference."""
    coff = 2 if overlap else 1
    state_len, state_dim = coff * compress_ratio, coff * head_dim

    kv = torch.arange(batch_size * seqlen * state_dim, device="cuda").reshape(
        batch_size, seqlen, state_dim
    )
    score = torch.arange(batch_size * seqlen * state_dim, device="cuda").reshape(
        batch_size, seqlen, state_dim
    )
    ape = torch.randn(compress_ratio, state_dim, device="cuda")

    kv_state_py = torch.zeros(batch_size, state_len, state_dim, device="cuda")
    score_state_py = torch.full((batch_size, state_len, state_dim), float("-inf"), device="cuda")

    paged_kv, paged_score, block_table, page_size, max_blocks = create_paged_cache(
        batch_size, seqlen, compress_ratio, head_dim, overlap
    )

    # Run PyTorch reference
    _ = run_pytorch_prefill_reference(
        kv.clone(),
        score.clone(),
        ape,
        kv_state_py,
        score_state_py,
        compress_ratio,
        head_dim,
        overlap,
    )

    # Run cuTile kernel
    kv_packed = kv.view(-1, state_dim)
    score_packed = score.view(-1, state_dim)
    kv_score = fuse_kv_score(kv_packed, score_packed)
    kv_lens = torch.full((batch_size,), seqlen, device="cuda", dtype=torch.int32)
    start_pos = torch.zeros(batch_size, device="cuda", dtype=torch.int32)

    # Pre-compute metadata
    cu_seq_lens, cu_outputs = prepare_prefill_metadata(
        kv_lens, start_pos, compress_ratio, head_dim, kv_score.device
    )
    kv_comp, compressed_mask = prepare_compress_output(
        cu_outputs, batch_size, head_dim, kv_score.device, torch.bfloat16
    )
    seq_lens = kv_lens - start_pos
    num_outputs_per_batch = torch.clamp(seq_lens // compress_ratio, min=1)
    max_outputs = num_outputs_per_batch.max().item()

    kv_compress_prefill_cutile(
        kv_score,
        ape,
        kv_lens,
        start_pos,
        cu_seq_lens,
        cu_outputs,
        kv_comp,
        compressed_mask,
        paged_kv,
        paged_score,
        block_table,
        max_outputs,
        block_table,
        compress_ratio,
        head_dim,
        overlap,
        page_size,
    )

    # Extract state from paged cache
    # Prefill writes state at absolute token positions:
    #   overlap:     [cutoff-ratio : cutoff] (last full chunk) and [cutoff : cutoff+remainder]
    #   non-overlap: [cutoff : cutoff+remainder]
    remainder = seqlen % compress_ratio
    cutoff = seqlen - remainder
    offset = compress_ratio if overlap else 0

    kv_state_kernel = torch.zeros_like(kv_state_py)
    score_state_kernel = torch.full_like(score_state_py, float("-inf"))

    # Recover last full chunk for overlap mode
    # Kernel writes at absolute positions [cutoff-ratio : cutoff]
    # PyTorch reference stores at virtual positions [0 : ratio]
    if overlap and cutoff >= compress_ratio:
        for r in range(compress_ratio):
            abs_pos = cutoff - compress_ratio + r
            log_block = abs_pos // page_size
            block_offset = abs_pos % page_size
            phys_block = block_table[0, log_block].item()
            kv_state_kernel[0, r] = paged_kv[phys_block, block_offset]
            score_state_kernel[0, r] = paged_score[phys_block, block_offset]

    # Recover remainder
    # Kernel writes at absolute positions [cutoff : cutoff+remainder]
    # PyTorch reference stores at virtual positions [offset : offset+remainder]
    if remainder > 0:
        for r in range(remainder):
            abs_pos = cutoff + r
            log_block = abs_pos // page_size
            block_offset = abs_pos % page_size
            phys_block = block_table[0, log_block].item()
            kv_state_kernel[0, offset + r] = paged_kv[phys_block, block_offset]
            score_state_kernel[0, offset + r] = paged_score[phys_block, block_offset]

    assert torch.allclose(kv_state_py, kv_state_kernel, atol=1e-5), (
        f"KV state mismatch: {(kv_state_py - kv_state_kernel).abs().max():.6f}"
    )
    assert torch.allclose(score_state_py, score_state_kernel, atol=1e-5), (
        f"Score state mismatch: {(score_state_py - score_state_kernel).abs().max():.6f}"
    )


PREFILL_VARLEN_CONFIGS = [
    pytest.param([8, 12, 4, 16], 4, 8, True, id="varlen_overlap_mixed"),
    pytest.param([128, 256, 64, 192], 128, 8, False, id="varlen_basic_mixed"),
    pytest.param([20, 20, 20], 4, 16, True, id="varlen_uniform"),
    pytest.param([3, 5, 8, 4], 4, 8, True, id="varlen_some_lt_ratio"),
]


@pytest.mark.parametrize("seq_lens_list,compress_ratio,head_dim,overlap", PREFILL_VARLEN_CONFIGS)
def test_prefill_varlen(seq_lens_list, compress_ratio, head_dim, overlap):
    """Test prefill with variable-length sequences."""
    batch_size = len(seq_lens_list)
    coff = 2 if overlap else 1
    state_len, state_dim = coff * compress_ratio, coff * head_dim

    # Create variable-length packed input
    total_tokens = sum(seq_lens_list)
    kv_packed = torch.randn(total_tokens, state_dim, device="cuda")
    score_packed = torch.randn(total_tokens, state_dim, device="cuda")
    kv_score = fuse_kv_score(kv_packed, score_packed)
    ape = torch.randn(compress_ratio, state_dim, device="cuda")

    # Per-batch metadata
    kv_lens = torch.tensor(seq_lens_list, device="cuda", dtype=torch.int32)
    start_pos = torch.zeros(batch_size, device="cuda", dtype=torch.int32)

    # PyTorch reference state
    kv_state_py = torch.zeros(batch_size, state_len, state_dim, device="cuda")
    score_state_py = torch.full((batch_size, state_len, state_dim), float("-inf"), device="cuda")

    # Paged cache setup
    page_size = 4
    max_seqlen = max(seq_lens_list)
    total_positions = max_seqlen + (compress_ratio if overlap else 0)
    max_blocks = (total_positions + page_size - 1) // page_size
    num_blocks = batch_size * max_blocks
    paged_kv = torch.zeros(num_blocks, page_size, state_dim, device="cuda")
    paged_score = torch.zeros(num_blocks, page_size, state_dim, device="cuda")
    block_table = torch.arange(num_blocks, device="cuda", dtype=torch.int32).view(
        batch_size, max_blocks
    )

    # PyTorch reference (varlen)
    out_py = run_pytorch_prefill_reference_varlen(
        kv_score.clone(),
        ape,
        kv_lens,
        start_pos,
        compress_ratio,
        head_dim,
        overlap,
        kv_state_py,
        score_state_py,
    )

    # cuTile kernel
    cu_seq_lens, cu_outputs = prepare_prefill_metadata(
        kv_lens, start_pos, compress_ratio, head_dim, kv_score.device
    )
    kv_comp, compressed_mask = prepare_compress_output(
        cu_outputs, batch_size, head_dim, kv_score.device, torch.bfloat16
    )
    seq_lens = kv_lens - start_pos
    num_outputs_per_batch = torch.clamp(seq_lens // compress_ratio, min=1)
    max_outputs = num_outputs_per_batch.max().item()

    kv_compress_prefill_cutile(
        kv_score,
        ape,
        kv_lens,
        start_pos,
        cu_seq_lens,
        cu_outputs,
        kv_comp,
        compressed_mask,
        paged_kv,
        paged_score,
        block_table,
        max_outputs,
        block_table,
        compress_ratio,
        head_dim,
        overlap,
        page_size,
    )

    # Check output - extract only valid outputs from packed output
    # cu_outputs uses min=1 for CUDA graph, but kernel only writes where seqlen >= ratio
    actual_outputs_per_batch = [s // compress_ratio for s in seq_lens_list]
    total_actual_outputs = sum(actual_outputs_per_batch)

    if total_actual_outputs == 0:
        # No compression expected
        assert out_py.numel() == 0, "Expected empty output when no compression"
    elif out_py.numel() == 0:
        pytest.fail("PyTorch returned empty output but expected valid output")
    else:
        # Extract valid outputs from kernel's packed output
        # cu_outputs[b] gives offset for batch b, but includes min=1 padding
        valid_outputs = []
        offset = 0
        for b, actual_count in enumerate(actual_outputs_per_batch):
            # cu_outputs uses clamped count, so we need to compute actual offset
            clamped_count = max(seq_lens_list[b] // compress_ratio, 1)
            if actual_count > 0:
                valid_outputs.append(kv_comp[offset : offset + actual_count])
            offset += clamped_count

        if valid_outputs:
            out_kernel_valid = torch.cat(valid_outputs, dim=0)
            assert torch.allclose(
                out_py.to(out_kernel_valid.dtype), out_kernel_valid, rtol=1e-4, atol=1e-5
            ), (
                f"Output mismatch: max diff = {(out_py.to(out_kernel_valid.dtype) - out_kernel_valid).abs().max():.6f}"
            )
        else:
            assert out_py.numel() == 0, "Expected empty output"


PREFILL_DECODE_CONFIGS = [
    pytest.param(1, 20, 4, 16, True, 12, id="overlap_prefill20_decode12"),
    pytest.param(1, 256, 128, 16, False, 128, id="basic_prefill256_decode128"),
    pytest.param(2, 32, 4, 64, True, 20, id="overlap_multi_batch"),
    pytest.param(1, 5, 4, 8, True, 8, id="overlap_prefill_with_remainder"),
    pytest.param(1, 512, 128, 32, False, 256, id="basic_large"),
]


@pytest.mark.parametrize(
    "batch_size,prefill_len,compress_ratio,head_dim,overlap,decode_steps", PREFILL_DECODE_CONFIGS
)
def test_prefill_then_decode(
    batch_size, prefill_len, compress_ratio, head_dim, overlap, decode_steps
):
    """Test prefill followed by decode (simulates inference)."""
    coff = 2 if overlap else 1
    state_len, state_dim = coff * compress_ratio, coff * head_dim
    page_size = 8
    total_len = prefill_len + decode_steps
    max_blocks = (total_len + (compress_ratio if overlap else 0) + page_size - 1) // page_size

    ape = torch.randn(compress_ratio, state_dim, device="cuda")

    # PyTorch reference state
    kv_state_py = torch.zeros(batch_size, state_len, state_dim, device="cuda")
    score_state_py = torch.full((batch_size, state_len, state_dim), float("-inf"), device="cuda")

    # Paged cache
    num_blocks = batch_size * max_blocks
    paged_kv = torch.zeros(num_blocks, page_size, state_dim, device="cuda")
    paged_score = torch.zeros(num_blocks, page_size, state_dim, device="cuda")
    block_table = torch.arange(num_blocks, device="cuda", dtype=torch.int32).view(
        batch_size, max_blocks
    )

    # 1. Prefill
    kv_prefill = torch.randn(batch_size, prefill_len, state_dim, device="cuda")
    score_prefill = torch.randn(batch_size, prefill_len, state_dim, device="cuda")

    # PyTorch prefill
    out_py_prefill = run_pytorch_prefill_reference(
        kv_prefill.clone(),
        score_prefill.clone(),
        ape,
        kv_state_py,
        score_state_py,
        compress_ratio,
        head_dim,
        overlap,
    )

    # cuTile prefill
    kv_packed = kv_prefill.view(-1, state_dim)
    score_packed = score_prefill.view(-1, state_dim)
    kv_score_prefill = fuse_kv_score(kv_packed, score_packed)
    kv_lens_prefill = torch.full((batch_size,), prefill_len, device="cuda", dtype=torch.int32)
    start_pos_prefill = torch.zeros(batch_size, device="cuda", dtype=torch.int32)

    # Pre-compute prefill metadata
    cu_seq_lens, cu_outputs = prepare_prefill_metadata(
        kv_lens_prefill, start_pos_prefill, compress_ratio, head_dim, kv_score_prefill.device
    )
    kv_comp, compressed_mask = prepare_compress_output(
        cu_outputs, batch_size, head_dim, kv_score_prefill.device, torch.bfloat16
    )
    seq_lens = kv_lens_prefill - start_pos_prefill
    num_outputs_per_batch = torch.clamp(seq_lens // compress_ratio, min=1)
    max_outputs = num_outputs_per_batch.max().item()

    kv_compress_prefill_cutile(
        kv_score_prefill,
        ape,
        kv_lens_prefill,
        start_pos_prefill,
        cu_seq_lens,
        cu_outputs,
        kv_comp,
        compressed_mask,
        paged_kv,
        paged_score,
        block_table,
        max_outputs,
        block_table,
        compress_ratio,
        head_dim,
        overlap,
        page_size,
    )

    if out_py_prefill is not None and kv_comp.numel() > 0:
        num_chunks = prefill_len // compress_ratio
        out_reshaped = kv_comp.view(batch_size, num_chunks, head_dim)
        assert torch.allclose(
            out_py_prefill.to(kv_comp.dtype), out_reshaped, rtol=1e-4, atol=1e-5
        ), (
            f"Prefill output mismatch: {(out_py_prefill.to(kv_comp.dtype) - out_reshaped).abs().max():.6f}"
        )

    # 2. Decode (continue from where prefill left off)
    decode_start = (prefill_len // compress_ratio) * compress_ratio
    remainder = prefill_len % compress_ratio

    # Pre-compute decode metadata
    cu_seq_lens_decode, cu_outputs_decode = prepare_decode_metadata(
        batch_size, compress_ratio, head_dim, torch.device("cuda"), next_n=1
    )
    kv_comp_decode, compressed_mask_decode = prepare_compress_output(
        cu_outputs_decode, batch_size, head_dim, torch.device("cuda"), torch.bfloat16
    )

    for i in range(decode_steps):
        step = decode_start + remainder + i  # Continue from where prefill left off

        new_kv = torch.randn(batch_size, state_dim, device="cuda")
        new_score = torch.randn(batch_size, state_dim, device="cuda")

        # Both prefill and decode use absolute token positions in the state cache.
        # The prefill kernel writes at [cutoff-ratio:cutoff] and [cutoff:cutoff+remainder],
        # so the decode kernel continues from position step = prefill_len + i.
        token_idx = step
        total_tokens = token_idx + 1

        # PyTorch reference
        out_py = run_pytorch_reference(
            new_kv.unsqueeze(1),
            new_score.unsqueeze(1),
            ape,
            kv_state_py,
            score_state_py,
            token_idx,
            compress_ratio,
            head_dim,
            overlap,
        )

        # cuTile decode
        kv_score = fuse_kv_score(new_kv, new_score)
        kv_lens = torch.full((batch_size,), total_tokens, device="cuda", dtype=torch.int32)
        start_pos = torch.full((batch_size,), token_idx, device="cuda", dtype=torch.int32)

        kv_compress_cutile(
            kv_score,
            ape,
            kv_lens,
            start_pos,
            cu_seq_lens_decode,
            cu_outputs_decode,
            kv_comp_decode,
            compressed_mask_decode,
            paged_kv,
            paged_score,
            block_table,
            block_table,
            compress_ratio,
            head_dim,
            overlap,
            page_size,
            next_n=1,
        )

        should_compress = (token_idx + 1) % compress_ratio == 0
        if should_compress and out_py is not None:
            assert compressed_mask_decode.all(), (
                f"Step {i} (token_idx={token_idx}): expected compression"
            )
            for b in range(batch_size):
                out_idx = cu_outputs_decode[b].item()
                diff = out_py[b, 0, :head_dim].to(kv_comp_decode.dtype) - kv_comp_decode[out_idx, :]
                assert torch.allclose(
                    out_py[b, 0, :head_dim].to(kv_comp_decode.dtype),
                    kv_comp_decode[out_idx, :],
                    rtol=1e-4,
                    atol=1e-5,
                ), (
                    f"Decode step {i} (token_idx={token_idx}), Batch {b}: mismatch diff={(diff).abs().max():.6f}"
                )


# MTP (Multi-Token Prediction) Tests
MTP_CONFIGS = [
    pytest.param(1, 4, 8, True, 4, id="overlap_next4"),
    pytest.param(2, 4, 8, True, 7, id="overlap_multi_batch_next7"),
    pytest.param(1, 4, 8, False, 5, id="basic_next5"),
    pytest.param(1, 4, 64, True, 3, id="overlap_next3_large_head"),
    pytest.param(4, 4, 32, True, 8, id="overlap_multi_batch_next8"),
    pytest.param(1, 4, 16, True, 1, id="overlap_next1_single"),
    pytest.param(2, 128, 32, False, 4, id="basic_multi_batch_next4"),
]


@pytest.mark.parametrize("batch_size,compress_ratio,head_dim,overlap,next_n", MTP_CONFIGS)
def test_decode_mtp(batch_size, compress_ratio, head_dim, overlap, next_n):
    """Test decode with multiple tokens per request (MTP)."""
    coff = 2 if overlap else 1
    state_len, state_dim = coff * compress_ratio, coff * head_dim
    page_size = 8
    num_steps = compress_ratio * 4  # Enough to trigger multiple compressions
    max_blocks = (num_steps + compress_ratio + page_size - 1) // page_size

    ape = torch.randn(compress_ratio, state_dim, device="cuda")

    num_blocks = batch_size * max_blocks
    paged_kv = torch.zeros(num_blocks, page_size, state_dim, device="cuda")
    paged_score = torch.zeros(num_blocks, page_size, state_dim, device="cuda")
    block_table = torch.arange(num_blocks, device="cuda", dtype=torch.int32).view(
        batch_size, max_blocks
    )

    kv_state_py = torch.zeros(batch_size, state_len, state_dim, device="cuda")
    score_state_py = torch.full((batch_size, state_len, state_dim), float("-inf"), device="cuda")

    # Pre-fill for overlap mode
    if overlap:
        init_kv = torch.randn(compress_ratio, state_dim, device="cuda")
        init_score = torch.randn(compress_ratio, state_dim, device="cuda")
        kv_state_py[:, :compress_ratio] = init_kv
        score_state_py[:, :compress_ratio] = init_score
        for b in range(batch_size):
            for r in range(compress_ratio):
                log_block, offset = r // page_size, r % page_size
                phys_block = block_table[b, log_block].item()
                paged_kv[phys_block, offset] = init_kv[r]
                paged_score[phys_block, offset] = init_score[r]

    # Process multiple tokens at once
    # For overlap mode, account for initial compress_ratio tokens in cache
    base_token_idx = compress_ratio if overlap else 0

    step = 0
    while step < num_steps:
        actual_n = min(next_n, num_steps - step)

        # Generate next_n tokens
        new_kv = torch.randn(batch_size * actual_n, state_dim, device="cuda")
        new_score = torch.randn(batch_size * actual_n, state_dim, device="cuda")

        # PyTorch reference: process one token at a time
        py_outputs = []
        for t in range(actual_n):
            token_idx = base_token_idx + step + t
            kv_t = new_kv[t::actual_n].unsqueeze(1)  # Get token t for all batches
            score_t = new_score[t::actual_n].unsqueeze(1)
            out_py = run_pytorch_reference(
                kv_t,
                score_t,
                ape,
                kv_state_py,
                score_state_py,
                token_idx,
                compress_ratio,
                head_dim,
                overlap,
            )
            if out_py is not None:
                py_outputs.append((token_idx, out_py))

        # cuTile: process all tokens at once
        kv_score = fuse_kv_score(new_kv, new_score)
        abs_start = base_token_idx + step
        kv_lens = torch.full((batch_size,), abs_start + actual_n, device="cuda", dtype=torch.int32)
        start_pos = torch.full((batch_size,), abs_start, device="cuda", dtype=torch.int32)

        # Compute decode metadata for actual_n (may differ from next_n at end of loop)
        cu_seq_lens, cu_outputs = prepare_decode_metadata(
            batch_size, compress_ratio, head_dim, torch.device("cuda"), next_n=actual_n
        )
        kv_comp, compressed_mask = prepare_compress_output(
            cu_outputs, batch_size, head_dim, torch.device("cuda"), torch.bfloat16
        )

        kv_compress_cutile(
            kv_score,
            ape,
            kv_lens,
            start_pos,
            cu_seq_lens,
            cu_outputs,
            kv_comp,
            compressed_mask,
            paged_kv,
            paged_score,
            block_table,
            block_table,
            compress_ratio,
            head_dim,
            overlap,
            page_size,
            next_n=actual_n,
        )

        # Verify outputs match (packed [total_outputs, head_dim] format)
        if len(py_outputs) > 0:
            assert compressed_mask.all(), f"Step {step}: expected compression"
            for i, (token_idx, out_py) in enumerate(py_outputs):
                for b in range(batch_size):
                    out_idx = cu_outputs[b].item() + i
                    diff = out_py[b, 0, :head_dim].to(kv_comp.dtype) - kv_comp[out_idx, :]
                    assert torch.allclose(
                        out_py[b, 0, :head_dim].to(kv_comp.dtype),
                        kv_comp[out_idx, :],
                        rtol=1e-4,
                        atol=5e-5,  # Loose a bit for type conversion
                    ), f"Token {token_idx}, Batch {b}: mismatch diff={(diff).abs().max():.6f}"

        step += actual_n


# Scatter kernel test configs
SCATTER_CONFIGS = [
    # Basic cases
    pytest.param(1, 8, [4], [0], 4, id="single_batch_basic"),
    pytest.param(2, 16, [2, 3], [0, 1], 4, id="multi_batch_basic"),
    pytest.param(4, 32, [1, 2, 1, 3], [2, 0, 1, 4], 4, id="varied_outputs"),
    # Corner cases: single output
    pytest.param(1, 8, [1], [0], 4, id="single_output"),
    pytest.param(1, 16, [1], [5], 4, id="single_output_nonzero_start"),
    # Corner cases: cross block boundary
    pytest.param(1, 8, [6], [2], 4, id="cross_block_boundary"),
    pytest.param(2, 16, [5, 7], [3, 1], 4, id="multi_batch_cross_block"),
    # Corner cases: large head_dim
    pytest.param(1, 128, [3], [0], 4, id="large_head_dim"),
    pytest.param(2, 256, [2, 2], [0, 4], 4, id="very_large_head_dim"),
    # Corner cases: different tokens_per_block
    pytest.param(1, 16, [8], [0], 8, id="large_block_size"),
    pytest.param(2, 16, [3, 4], [0, 2], 2, id="small_block_size"),
    # Corner cases: high start positions (stress paging)
    pytest.param(1, 16, [2], [100], 4, id="high_start_pos"),
    pytest.param(2, 8, [1, 1], [50, 75], 4, id="high_start_pos_multi"),
    # Corner cases: uneven batches
    pytest.param(3, 16, [1, 5, 2], [0, 0, 10], 4, id="uneven_outputs"),
    pytest.param(4, 8, [10, 1, 8, 3], [0, 5, 2, 12], 4, id="highly_uneven"),
]


@pytest.mark.parametrize(
    "batch_size,head_dim,num_outputs_list,start_positions,tokens_per_block", SCATTER_CONFIGS
)
def test_compressed_kv_scatter(
    batch_size, head_dim, num_outputs_list, start_positions, tokens_per_block
):
    """Test compressed KV scatter kernel."""
    num_outputs = torch.tensor(num_outputs_list, device="cuda", dtype=torch.int32)
    cu_new_comp_kv = torch.zeros(batch_size + 1, device="cuda", dtype=torch.int32)
    cu_new_comp_kv[1:] = torch.cumsum(num_outputs, dim=0)
    total_outputs = cu_new_comp_kv[-1].item()
    compressed_kv = torch.randn(total_outputs, head_dim, device="cuda")
    start_pos = torch.tensor(start_positions, device="cuda", dtype=torch.int32)

    max_compressed_len = max(start_positions) + max(num_outputs_list) + 4
    kv_cache, block_offsets, _, _ = create_compressed_kv_cache(
        batch_size, max_compressed_len, head_dim, tokens_per_block
    )
    max_outputs = num_outputs.max().item()

    compressed_kv_scatter_cutile(
        compressed_kv,
        num_outputs,
        cu_new_comp_kv,
        start_pos,
        kv_cache,
        block_offsets,
        tokens_per_block,
        head_dim,
        max_outputs,
    )

    for b in range(batch_size):
        for i in range(num_outputs_list[b]):
            cache_pos = start_positions[b] + i
            logical_block = cache_pos // tokens_per_block
            token_offset = cache_pos % tokens_per_block
            phys_block = block_offsets[b, logical_block].item()
            expected = compressed_kv[cu_new_comp_kv[b].item() + i]
            actual = kv_cache[
                phys_block, 0, token_offset * head_dim : (token_offset + 1) * head_dim
            ]
            assert torch.allclose(expected, actual.to(expected.dtype), rtol=1e-5, atol=1e-6), (
                f"Mismatch at batch={b}, output={i}, cache_pos={cache_pos}"
            )


# ============================================================================
# FP8 Blockwise Scatter Tests
# ============================================================================


@pytest.mark.parametrize("head_dim", [128, 256, 512])
@pytest.mark.parametrize("batch_size,tokens_per_req", [(1, 32), (3, 32)])
def test_fp8_scatter_kernel(head_dim, batch_size, tokens_per_req):
    """
    Compare cuTile FP8 scatter kernel vs CUDA kernel (torch.ops.trtllm.indexer_k_cache_scatter_op).

    The CUDA kernel only supports head_dim=128, so for larger head_dim we compare
    against the Python reference. For head_dim=128, we use the CUDA kernel as golden.
    """
    from tensorrt_llm.quantization.utils import fp8_utils

    torch.manual_seed(123)

    block_size = 64
    num_tokens = batch_size * tokens_per_req
    max_seq_len = 512

    # Compute scale size (1x128 blockwise quantization)
    num_scale_blocks = (head_dim + 127) // 128
    scale_size = num_scale_blocks * 4
    per_token_size = head_dim + scale_size

    # Allocate cache with enough blocks
    max_blocks = (max_seq_len + block_size - 1) // block_size
    num_blocks = batch_size * max_blocks

    # Cache for cuTile kernel - use fp8 dtype to match compressor usage
    # Non-interleaved layout per block: [k0, k1, ..., kN, scale0, scale1, ..., scaleN]
    # Total size per block = block_size * head_dim + block_size * scale_size = block_size * per_token_size
    total_block_elems = block_size * per_token_size
    kv_cache_cutile = torch.zeros(
        num_blocks, total_block_elems, device="cuda", dtype=torch.float8_e4m3fn
    )
    # Cache for golden (CUDA kernel or Python reference) - uint8 for byte comparison
    kv_cache_golden = torch.zeros(num_blocks, total_block_elems, device="cuda", dtype=torch.uint8)

    # Block offsets: [batch_size, max_blocks]
    block_offsets = torch.zeros(batch_size, max_blocks, device="cuda", dtype=torch.int32)
    for b in range(batch_size):
        for blk in range(max_blocks):
            block_offsets[b, blk] = b * max_blocks + blk

    # Generate test data
    k_original = torch.randn(num_tokens, head_dim, device="cuda", dtype=torch.bfloat16)
    k_fp8, k_scale = fp8_utils.fp8_quantize_1x128_sf_transpose(k_original)

    # Prepare data for kernel
    # FP8 data: pass as fp8 (same bytes, kernel handles it)
    k_fp8_contiguous = k_fp8.contiguous().view(num_tokens, head_dim)
    # Scale data: uint8 bytes (needed for golden reference only)
    k_scale_bytes = k_scale.contiguous().flatten().view(torch.uint8).view(num_tokens, scale_size)

    # Metadata
    num_comp_tokens = torch.full((batch_size,), tokens_per_req, device="cuda", dtype=torch.int32)
    cu_new_comp_kv = torch.zeros(batch_size + 1, device="cuda", dtype=torch.int32)
    cu_new_comp_kv[1:] = num_comp_tokens.cumsum(0)
    start_pos = torch.zeros(batch_size, device="cuda", dtype=torch.int32)
    max_outputs = num_comp_tokens.max().item()

    # ========== cuTile Kernel ==========
    compressed_kv_scatter_cutile(
        k_fp8_contiguous,
        num_comp_tokens,
        cu_new_comp_kv,
        start_pos,
        kv_cache_cutile,
        block_offsets,
        block_size,
        head_dim,
        max_outputs=max_outputs,
        kv_cache_dtype="fp8_blockwise",
        kv_scale=k_scale,
    )
    torch.cuda.synchronize()

    # ========== Golden: CUDA kernel for head_dim=128, Python for others ==========
    # Non-interleaved layout: [k0, k1, ..., kN, scale0, scale1, ..., scaleN] per block
    # FP8 offset for token i: block_base + i * head_dim
    # Scale offset for token i: block_base + block_size * head_dim + i * scale_size
    if head_dim == 128:
        # Use CUDA kernel as golden (it only supports head_dim=128)
        # Need to compute flat slot mappings for the CUDA kernel
        slot_mapping_fp8 = torch.zeros(num_tokens, device="cuda", dtype=torch.int64)
        slot_mapping_scale = torch.zeros(num_tokens, device="cuda", dtype=torch.int64)

        global_token_idx = 0
        for b in range(batch_size):
            for local_idx in range(tokens_per_req):
                cache_pos = int(start_pos[b].item()) + local_idx
                logical_block = cache_pos // block_size
                token_offset = cache_pos % block_size
                phys_block = int(block_offsets[b, logical_block].item())

                # Non-interleaved flat byte index for FP8 data
                fp8_offset = phys_block * total_block_elems + token_offset * head_dim
                # Non-interleaved flat byte index for scale data
                scale_offset = (
                    phys_block * total_block_elems
                    + block_size * head_dim
                    + token_offset * scale_size
                )

                slot_mapping_fp8[global_token_idx] = fp8_offset
                slot_mapping_scale[global_token_idx] = scale_offset
                global_token_idx += 1

        # Prepare byte-level data for CUDA kernel
        k_fp8_bytes = k_fp8.contiguous().flatten().view(torch.uint8).view(num_tokens, head_dim)

        # Reshape cache for CUDA kernel: [num_blocks, block_size, 1, per_token_size]
        kv_cache_golden_reshaped = kv_cache_golden.view(num_blocks, block_size, per_token_size)

        torch.ops.trtllm.indexer_k_cache_scatter_op(
            k_fp8_bytes,
            k_scale_bytes,
            kv_cache_golden_reshaped.unsqueeze(2),
            slot_mapping_fp8,
            slot_mapping_scale,
        )
        torch.cuda.synchronize()

        # Flatten back for comparison
        kv_cache_golden = kv_cache_golden_reshaped.view(num_blocks, total_block_elems)
    else:
        # Use Python reference for larger head_dim (CUDA kernel doesn't support)
        # Non-interleaved layout: FP8 region then scale region within each block
        # Prepare byte-level data for Python reference
        k_fp8_bytes = k_fp8.contiguous().flatten().view(torch.uint8).view(num_tokens, head_dim)

        global_token_idx = 0
        for b in range(batch_size):
            for local_idx in range(tokens_per_req):
                cache_pos = int(start_pos[b].item()) + local_idx
                logical_block = cache_pos // block_size
                token_offset = cache_pos % block_size
                phys_block = int(block_offsets[b, logical_block].item())

                # Write FP8 data (first section of block)
                fp8_start = token_offset * head_dim
                kv_cache_golden[phys_block, fp8_start : fp8_start + head_dim] = k_fp8_bytes[
                    global_token_idx
                ]
                # Write scale data (second section of block)
                scale_start = block_size * head_dim + token_offset * scale_size
                kv_cache_golden[phys_block, scale_start : scale_start + scale_size] = k_scale_bytes[
                    global_token_idx
                ]

                global_token_idx += 1

    # ========== Validation ==========
    # Compare as bytes (view both as uint8)
    cutile_bytes = kv_cache_cutile.view(torch.uint8)
    golden_bytes = kv_cache_golden.view(torch.uint8)

    if torch.equal(cutile_bytes, golden_bytes):
        print(f"PASS: head_dim={head_dim}, batch={batch_size}, tokens={num_tokens}")
    else:
        # Find differences
        diff_mask = cutile_bytes != golden_bytes
        num_diffs = diff_mask.sum().item()
        total_bytes = cutile_bytes.numel()

        # Show first few differences
        diff_indices = torch.nonzero(diff_mask.view(-1))[:5]
        for idx in diff_indices:
            flat_idx = idx.item()
            print(
                f"  Byte {flat_idx}: cuTile={cutile_bytes.view(-1)[flat_idx].item()}, "
                f"Golden={golden_bytes.view(-1)[flat_idx].item()}"
            )

        raise AssertionError(
            f"cuTile kernel differs from golden: {num_diffs}/{total_bytes} bytes ({100 * num_diffs / total_bytes:.4f}%)"
        )


@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("batch_size,tokens_per_req", [(1, 16), (2, 32)])
def test_fp8_pertensor_scatter_cutile(head_dim, batch_size, tokens_per_req):
    """Test CuTile per-tensor FP8 scatter against Python reference."""
    torch.manual_seed(42)

    tokens_per_block = 32
    num_tokens = batch_size * tokens_per_req
    max_seq_len = 256

    max_blocks = (max_seq_len + tokens_per_block - 1) // tokens_per_block
    num_blocks = batch_size * max_blocks
    kv_factor = 1

    # fp8e4nv cache
    kv_cache = torch.zeros(
        num_blocks, kv_factor, tokens_per_block * head_dim, device="cuda", dtype=torch.float8_e4m3fn
    )

    block_offsets = torch.zeros(batch_size, max_blocks, device="cuda", dtype=torch.int32)
    for b in range(batch_size):
        for blk in range(max_blocks):
            block_offsets[b, blk] = b * max_blocks + blk

    # Generate FP8 data (cast bf16 -> fp8, then view as uint8)
    kv_comp = torch.randn(num_tokens, head_dim, device="cuda", dtype=torch.bfloat16)
    kv_fp8 = kv_comp.to(torch.float8_e4m3fn)
    kv_uint8 = kv_fp8.view(torch.uint8)

    num_comp_tokens = torch.full((batch_size,), tokens_per_req, device="cuda", dtype=torch.int32)
    cu_new_comp_kv = torch.zeros(batch_size + 1, device="cuda", dtype=torch.int32)
    cu_new_comp_kv[1:] = num_comp_tokens.cumsum(0)
    start_pos = torch.zeros(batch_size, device="cuda", dtype=torch.int32)
    max_outputs = num_comp_tokens.max().item()

    # Run CuTile kernel
    compressed_kv_scatter_cutile(
        kv_uint8,
        num_comp_tokens,
        cu_new_comp_kv,
        start_pos,
        kv_cache,
        block_offsets,
        tokens_per_block,
        head_dim,
        max_outputs=max_outputs,
        kv_cache_dtype="fp8_pertensor",
    )
    torch.cuda.synchronize()

    # Verify against expected values
    for b in range(batch_size):
        for i in range(tokens_per_req):
            cache_pos = i
            logical_block = cache_pos // tokens_per_block
            token_offset = cache_pos % tokens_per_block
            phys_block = block_offsets[b, logical_block].item()
            token_idx = cu_new_comp_kv[b].item() + i

            expected_bytes = kv_uint8[token_idx]
            actual_bytes = kv_cache[
                phys_block, 0, token_offset * head_dim : (token_offset + 1) * head_dim
            ].view(torch.uint8)

            assert torch.equal(expected_bytes, actual_bytes), (
                f"Per-tensor FP8 mismatch at batch={b}, token={i}"
            )


# ============================================================================
# Benchmarks: cuTile Kernels vs PyTorch Reference
# ============================================================================


def benchmark_scatter_all_backends():
    """Benchmark cuTile vs PyTorch scatter kernels using triton.testing.do_bench."""

    print("\n" + "=" * 70)
    print("Scatter Kernel Benchmark: cuTile vs PyTorch")
    print("=" * 70)

    configs = [
        # (batch_size, head_dim, total_outputs, tokens_per_block, name)
        (1, 512, 32, 32, "b1_h512_t32"),
        (8, 512, 64, 32, "b8_h512_t64"),
        (32, 512, 128, 32, "b32_h512_t128"),
        (1, 512, 256, 32, "b1_h512_t256"),
        (8, 512, 512, 32, "b8_h512_t512"),
        (1, 128, 32, 8, "b1_h128_t32"),
        (8, 128, 64, 8, "b8_h128_t64"),
        (32, 128, 128, 8, "b32_h128_t128"),
    ]

    results = []
    for batch_size, head_dim, total_outputs, tokens_per_block, name in configs:
        outputs_per_batch = total_outputs // batch_size
        num_outputs = torch.full((batch_size,), outputs_per_batch, device="cuda", dtype=torch.int32)
        cu_kv_comp = torch.zeros(batch_size + 1, device="cuda", dtype=torch.int32)
        cu_kv_comp[1:] = torch.cumsum(num_outputs, dim=0)

        compressed_kv = torch.randn(total_outputs, head_dim, device="cuda")
        start_pos = torch.zeros(batch_size, device="cuda", dtype=torch.int32)

        max_compressed_len = outputs_per_batch + 4
        kv_cache, block_offsets, _, _ = create_compressed_kv_cache(
            batch_size, max_compressed_len, head_dim, tokens_per_block
        )

        kv_cache_cutile = kv_cache.clone()
        max_outputs = num_outputs.max().item()

        def cutile_fn():
            compressed_kv_scatter_cutile(
                compressed_kv,
                num_outputs,
                cu_kv_comp,
                start_pos,
                kv_cache_cutile,
                block_offsets,
                tokens_per_block,
                head_dim,
                max_outputs=max_outputs,
            )

        def pytorch_fn():
            for b in range(batch_size):
                for i in range(outputs_per_batch):
                    cache_pos = i
                    logical_block = cache_pos // tokens_per_block
                    token_offset = cache_pos % tokens_per_block
                    phys_block = block_offsets[b, logical_block].item()
                    kv_cache[
                        phys_block, 0, token_offset * head_dim : (token_offset + 1) * head_dim
                    ] = compressed_kv[cu_kv_comp[b].item() + i]

        cutile_ms = triton.testing.do_bench(cutile_fn, warmup=25, rep=100)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=25, rep=100)

        cutile_us = cutile_ms * 1000
        pytorch_us = pytorch_ms * 1000

        results.append(
            {
                "name": name,
                "cutile_us": cutile_us,
                "pytorch_us": pytorch_us,
                "speedup": pytorch_us / cutile_us if cutile_us > 0 else float("inf"),
            }
        )

    # Print results
    print(f"\n{'Config':<30} {'cuTile (us)':>12} {'PyTorch (us)':>12} {'Speedup':>10}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['name']:<30} {r['cutile_us']:>12.2f} "
            f"{r['pytorch_us']:>12.2f} {r['speedup']:>9.2f}x"
        )
    print("=" * 70)

    return results


def benchmark_compress_kernel():
    """Benchmark cuTile vs PyTorch compress (decode) kernels using triton.testing.do_bench."""

    print("\n" + "=" * 70)
    print("Compress Kernel Benchmark: cuTile vs PyTorch (decode)")
    print("=" * 70)

    configs = [
        # (batch_size, compress_ratio, head_dim, overlap, page_size, name)
        (1, 4, 512, True, 32, "b1_r4_d512_overlap"),
        (8, 4, 512, True, 32, "b8_r4_d512_overlap"),
        (32, 4, 512, True, 32, "b32_r4_d512_overlap"),
        (1, 128, 512, False, 32, "b1_r128_d512"),
        (8, 128, 512, False, 32, "b8_r128_d512"),
        (32, 128, 512, False, 32, "b32_r128_d512"),
        (1, 4, 128, True, 8, "b1_r4_d128_overlap"),
        (8, 4, 128, True, 8, "b8_r4_d128_overlap"),
        (32, 4, 128, True, 8, "b32_r4_d128_overlap"),
    ]

    results = []
    for batch_size, compress_ratio, head_dim, overlap, page_size, name in configs:
        coff = 2 if overlap else 1
        state_dim = coff * head_dim
        max_blocks = (compress_ratio * 2 + page_size - 1) // page_size

        # Prepare inputs
        ape = torch.randn(compress_ratio, state_dim, device="cuda")
        new_kv = torch.randn(batch_size, state_dim, device="cuda")
        new_score = torch.randn(batch_size, state_dim, device="cuda")
        kv_score = fuse_kv_score(new_kv, new_score)

        # PyTorch state
        state_len = coff * compress_ratio
        kv_state_py = torch.zeros(batch_size, state_len, state_dim, device="cuda")
        score_state_py = torch.full(
            (batch_size, state_len, state_dim), float("-inf"), device="cuda"
        )

        # Paged cache
        num_blocks = batch_size * max_blocks
        paged_kv = torch.zeros(num_blocks, page_size, state_dim, device="cuda")
        paged_score = torch.zeros(num_blocks, page_size, state_dim, device="cuda")
        block_table = torch.arange(num_blocks, device="cuda", dtype=torch.int32).view(
            batch_size, max_blocks
        )

        # Use step = compress_ratio - 1 to trigger compression.
        step = compress_ratio - 1
        token_idx = (compress_ratio + step) if overlap else step
        kv_lens = torch.full((batch_size,), token_idx + 1, device="cuda", dtype=torch.int32)
        start_pos = torch.full((batch_size,), token_idx, device="cuda", dtype=torch.int32)
        cu_seq_lens, cu_outputs = prepare_decode_metadata(
            batch_size, compress_ratio, head_dim, torch.device("cuda"), next_n=1
        )
        kv_comp, compressed_mask = prepare_compress_output(
            cu_outputs, batch_size, head_dim, torch.device("cuda"), torch.bfloat16
        )

        paged_kv_cutile = paged_kv.clone()
        paged_score_cutile = paged_score.clone()
        kv_comp_cutile = kv_comp.clone()

        def cutile_fn():
            kv_compress_cutile(
                kv_score,
                ape,
                kv_lens,
                start_pos,
                cu_seq_lens,
                cu_outputs,
                kv_comp_cutile,
                compressed_mask,
                paged_kv_cutile,
                paged_score_cutile,
                block_table,
                block_table,
                compress_ratio,
                head_dim,
                overlap,
                page_size,
                next_n=1,
            )

        def pytorch_fn():
            run_pytorch_reference(
                new_kv.unsqueeze(1),
                new_score.unsqueeze(1),
                ape,
                kv_state_py.clone(),
                score_state_py.clone(),
                step,
                compress_ratio,
                head_dim,
                overlap,
            )

        cutile_ms = triton.testing.do_bench(cutile_fn, warmup=25, rep=100)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=25, rep=100)

        cutile_us = cutile_ms * 1000
        pytorch_us = pytorch_ms * 1000

        results.append(
            {
                "name": name,
                "cutile_us": cutile_us,
                "pytorch_us": pytorch_us,
                "speedup": pytorch_us / cutile_us if cutile_us > 0 else float("inf"),
            }
        )

    # Print results
    print(f"\n{'Config':<30} {'cuTile (us)':>12} {'PyTorch (us)':>12} {'Speedup':>10}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['name']:<30} {r['cutile_us']:>12.2f} "
            f"{r['pytorch_us']:>12.2f} {r['speedup']:>9.2f}x"
        )
    print("=" * 70)

    return results


def benchmark_compress_prefill_kernel():
    """Benchmark cuTile vs PyTorch compress (prefill) kernels using triton.testing.do_bench."""

    print("\n" + "=" * 70)
    print("Compress Prefill Kernel Benchmark: cuTile vs PyTorch")
    print("=" * 70)

    configs = [
        # (batch_size, seqlen, compress_ratio, head_dim, overlap, page_size, name)
        (1, 128, 4, 512, True, 32, "prefill_b1_s128_r4_d512"),
        (4, 256, 4, 512, True, 32, "prefill_b4_s256_r4_d512"),
        (8, 512, 4, 512, True, 32, "prefill_b8_s512_r4_d512"),
        (1, 512, 128, 512, False, 32, "prefill_b1_s512_r128_d512"),
        (4, 512, 128, 512, False, 32, "prefill_b4_s512_r128_d512"),
        (8, 1024, 128, 512, False, 32, "prefill_b8_s1024_r128_d512"),
        (1, 128, 4, 128, True, 8, "prefill_b1_s128_r4_d128"),
        (8, 512, 4, 128, True, 8, "prefill_b8_s512_r4_d128"),
        (32, 512, 4, 128, True, 8, "prefill_b32_s512_r4_d128"),
    ]

    results = []
    for batch_size, seqlen, compress_ratio, head_dim, overlap, page_size, name in configs:
        coff = 2 if overlap else 1
        state_dim = coff * head_dim

        # Prepare inputs
        kv = torch.randn(batch_size, seqlen, state_dim, device="cuda")
        score = torch.randn(batch_size, seqlen, state_dim, device="cuda")
        ape = torch.randn(compress_ratio, state_dim, device="cuda")

        # PyTorch state
        state_len = coff * compress_ratio
        kv_state_py = torch.zeros(batch_size, state_len, state_dim, device="cuda")
        score_state_py = torch.full(
            (batch_size, state_len, state_dim), float("-inf"), device="cuda"
        )

        # Shared inputs
        kv_score = fuse_kv_score(kv.view(-1, state_dim), score.view(-1, state_dim))
        kv_lens = torch.full((batch_size,), seqlen, device="cuda", dtype=torch.int32)
        start_pos = torch.zeros(batch_size, device="cuda", dtype=torch.int32)
        cu_seq_lens, cu_outputs = prepare_prefill_metadata(
            kv_lens, start_pos, compress_ratio, head_dim, kv_score.device
        )
        kv_comp, compressed_mask = prepare_compress_output(
            cu_outputs, batch_size, head_dim, kv_score.device, torch.bfloat16
        )
        paged_kv, paged_score, block_table, _, _ = create_paged_cache(
            batch_size, seqlen, compress_ratio, head_dim, overlap, page_size
        )

        paged_kv_cutile = paged_kv.clone()
        paged_score_cutile = paged_score.clone()
        kv_comp_cutile = kv_comp.clone()
        compressed_mask_cutile = compressed_mask.clone()

        def cutile_fn():
            seq_lens = kv_lens - start_pos
            num_outputs_per_batch = torch.clamp(seq_lens // compress_ratio, min=1)
            max_outputs = num_outputs_per_batch.max().item()
            kv_compress_prefill_cutile(
                kv_score,
                ape,
                kv_lens,
                start_pos,
                cu_seq_lens,
                cu_outputs,
                kv_comp_cutile,
                compressed_mask_cutile,
                paged_kv_cutile,
                paged_score_cutile,
                block_table,
                max_outputs,
                block_table,
                compress_ratio,
                head_dim,
                overlap,
                page_size,
            )

        def pytorch_fn():
            run_pytorch_prefill_reference(
                kv.clone(),
                score.clone(),
                ape,
                kv_state_py.clone(),
                score_state_py.clone(),
                compress_ratio,
                head_dim,
                overlap,
            )

        cutile_ms = triton.testing.do_bench(cutile_fn, warmup=25, rep=100)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=25, rep=100)

        cutile_us = cutile_ms * 1000
        pytorch_us = pytorch_ms * 1000

        results.append(
            {
                "name": name,
                "cutile_us": cutile_us,
                "pytorch_us": pytorch_us,
                "speedup": pytorch_us / cutile_us if cutile_us > 0 else float("inf"),
            }
        )

    # Print results
    print(f"\n{'Config':<30} {'cuTile (us)':>12} {'PyTorch (us)':>12} {'Speedup':>10}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['name']:<30} {r['cutile_us']:>12.2f} "
            f"{r['pytorch_us']:>12.2f} {r['speedup']:>9.2f}x"
        )
    print("=" * 70)

    return results


def run_all_benchmarks():
    """Run all kernel benchmarks."""
    print("\n" + "=" * 80)
    print("Compressor Kernel Benchmarks")
    print("=" * 80)

    # cuTile vs PyTorch
    benchmark_scatter_all_backends()
    benchmark_compress_kernel()
    benchmark_compress_prefill_kernel()


if __name__ == "__main__":
    run_all_benchmarks()
