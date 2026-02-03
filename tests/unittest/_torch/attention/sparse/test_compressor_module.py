#!/usr/bin/env python3
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests comparing Compressor with RefCompressor."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorrt_llm._torch.attention_backend.interface import (
    MLAParams,
    PositionalEmbeddingParams,
    PositionEmbeddingType,
    RotaryScalingType,
)
from tensorrt_llm._torch.attention_backend.sparse.dsa import rotate_activation
from tensorrt_llm._torch.attention_backend.sparse.mewtwo.compressor import Compressor
from tensorrt_llm._torch.attention_backend.sparse.mewtwo.mewtwo import MewtwoAttentionType
from tensorrt_llm._torch.modules.rotary_embedding import RopeParams

# ============================================================================
# Dummy Metadata and KVCacheManager for Testing
# ============================================================================


class DummyKVCacheManager:
    """Dummy KVCacheManager that returns pre-allocated buffers for testing."""

    def __init__(
        self,
        kv_cache: torch.Tensor,
        paged_kv_state: torch.Tensor,
        paged_score_state: torch.Tensor,
    ):
        self._buffers = {
            MewtwoAttentionType.COMPRESS: kv_cache,
            MewtwoAttentionType.COMPRESSOR_STATE: paged_kv_state,
            MewtwoAttentionType.COMPRESSOR_SCORE: paged_score_state,
        }

    def get_buffers(self, layer_idx: int, attention_type: MewtwoAttentionType) -> torch.Tensor:
        """Return pre-allocated buffer for the given attention type."""
        return self._buffers[attention_type]


class DummyAttentionMetadata:
    """Dummy attention metadata for testing Compressor.forward."""

    def __init__(
        self,
        num_contexts: int,
        num_generations: int,
        num_ctx_tokens: int,
        num_tokens: int,
        kv_cache_manager: DummyKVCacheManager,
        block_tables: dict,
        cu_seq_lens: dict,
        cu_kv_comp: dict,
        compressed_position_ids: dict,
        compressed_kv_lens: dict,
        compressed_start_positions: dict,
        slot_mapping_fp8: torch.Tensor = None,
        slot_mapping_scale: torch.Tensor = None,
    ):
        self.num_contexts = num_contexts
        self.num_generations = num_generations
        self.num_ctx_tokens = num_ctx_tokens
        self.num_tokens = num_tokens
        self.kv_cache_manager = kv_cache_manager
        self.block_tables = block_tables
        self.cu_seq_lens = cu_seq_lens
        self.cu_kv_comp = cu_kv_comp
        self.compressed_position_ids = compressed_position_ids
        self.compressed_kv_lens = compressed_kv_lens
        self.compressed_start_positions = compressed_start_positions
        self.slot_mapping_fp8 = slot_mapping_fp8
        self.slot_mapping_scale = slot_mapping_scale


# ============================================================================
# Reference Implementation (DO NOT MODIFY)
# ============================================================================


@dataclass
class ModelArgs:
    """Model arguments for Compressor."""

    max_batch_size: int = 4
    max_seq_len: int = 4096
    dim: int = 4096
    head_dim: int = 512
    rope_head_dim: int = 64
    norm_eps: float = 1e-6
    compress_ratios: Tuple[int, ...] = (1, 1, 4, 128, 4, 128, 4)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


class Linear(nn.Module):
    """Simple linear layer (fp32 weights for reference)."""

    def __init__(self, in_features: int, out_features: int, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype or torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


def apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """Apply rotary positional embeddings."""
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


class RefCompressor(nn.Module):
    """Reference Compressor implementation for testing."""

    def __init__(
        self, args: ModelArgs, compress_ratio: int = 4, head_dim: int = 512, rotate: bool = False
    ):
        super().__init__()
        self.dim = args.dim
        self.head_dim = head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = head_dim - args.rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        coff = 1 + self.overlap

        self.ape = nn.Parameter(
            torch.empty(compress_ratio, coff * self.head_dim, dtype=torch.float32)
        )
        self.wkv = Linear(self.dim, coff * self.head_dim, dtype=torch.float32)
        self.wgate = Linear(self.dim, coff * self.head_dim, dtype=torch.float32)
        self.norm = RMSNorm(self.head_dim, args.norm_eps)
        self.kv_cache = None
        self.register_buffer(
            "kv_state",
            torch.zeros(
                args.max_batch_size,
                coff * compress_ratio,
                coff * self.head_dim,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "score_state",
            torch.full(
                (args.max_batch_size, coff * compress_ratio, coff * self.head_dim),
                float("-inf"),
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def overlap_transform(self, tensor: torch.Tensor, value=0):
        b, s, _, _ = tensor.size()
        ratio, d = self.compress_ratio, self.head_dim
        new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new_tensor

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor):
        assert self.kv_cache is not None
        bsz, seqlen, _ = x.size()
        ratio, overlap, d = self.compress_ratio, self.overlap, self.head_dim
        dtype = x.dtype
        x = x.float()
        kv = self.wkv(x)
        score = self.wgate(x)
        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            freqs_cis = freqs_cis[:cutoff:ratio]
            offset = ratio if overlap else 0
            if overlap and cutoff >= ratio:
                self.kv_state[:bsz, :ratio] = kv[:, cutoff - ratio : cutoff]
                self.score_state[:bsz, :ratio] = score[:, cutoff - ratio : cutoff] + self.ape
            if remainder > 0:
                kv, self.kv_state[:bsz, offset : offset + remainder] = kv.split(
                    [cutoff, remainder], dim=1
                )
                self.score_state[:bsz, offset : offset + remainder] = (
                    score[:, cutoff:] + self.ape[:remainder]
                )
                score = score[:, :cutoff]
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio)) + self.ape
            if overlap:
                kv = self.overlap_transform(kv, 0)
                score = self.overlap_transform(score, float("-inf"))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            should_compress = (start_pos + 1) % self.compress_ratio == 0
            score += self.ape[start_pos % ratio]
            if overlap:
                self.kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv_state = torch.cat(
                        [self.kv_state[:bsz, :ratio, :d], self.kv_state[:bsz, ratio:, d:]], dim=1
                    )
                    score_state = torch.cat(
                        [self.score_state[:bsz, :ratio, :d], self.score_state[:bsz, ratio:, d:]],
                        dim=1,
                    )
                    kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
                    self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:]
                    self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:]
            else:
                self.kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(
                        dim=1, keepdim=True
                    )

        if not should_compress:
            return
        kv = self.norm(kv.to(dtype))
        apply_rotary_emb(kv[..., -self.rope_head_dim :], freqs_cis)
        if self.rotate:
            kv = rotate_activation(kv)
        if start_pos == 0:
            self.kv_cache[:bsz, : seqlen // ratio] = kv
        else:
            self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
        return kv


# ============================================================================
# Test Configuration & Helpers
# ============================================================================

DEVICE = "cuda"
DTYPE = torch.bfloat16
DIM, HEAD_DIM, ROPE_DIM = 4096, 512, 64
MAX_BATCH, MAX_SEQ, PAGE_SIZE = 4, 4096, 32
ORI_SEQ_LEN = 65536
ROPE_THETA, ROPE_FACTOR, BETA_FAST, BETA_SLOW = 40000.0, 4, 32, 1


def precompute_freqs_cis(
    dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow
) -> torch.Tensor:
    """Precompute rotary embeddings."""

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def assert_similar(t1: Optional[torch.Tensor], t2: Optional[torch.Tensor], name: str = "Output"):
    """Assert tensors are similar (cosine sim >= 0.999)."""
    if t1 is None and t2 is None:
        return
    assert t1 is not None and t2 is not None, f"{name}: One is None"
    assert t1.shape == t2.shape, f"{name}: Shape mismatch {t1.shape} vs {t2.shape}"
    t1, t2 = t1.float().flatten(), t2.float().flatten()
    cos_sim = F.cosine_similarity(t1.unsqueeze(0), t2.unsqueeze(0)).item()
    # Also check magnitude to avoid scaled-but-equal-direction false positives
    max_diff = (t1 - t2).abs().max().item()
    scale = max(t1.abs().max().item(), t2.abs().max().item(), 1e-3)
    rel_err = max_diff / scale
    assert cos_sim >= 0.999, f"{name}: cos_sim={cos_sim:.6f}"
    assert rel_err <= 5e-2, f"{name}: rel_err={rel_err:.6f}, max_diff={max_diff:.6f}"


def read_paged_cache_tokens(
    kv_cache: torch.Tensor,
    block_offsets: torch.Tensor,
    batch_idx: int,
    num_tokens: int,
    tokens_per_block: int,
) -> torch.Tensor:
    """Materialize paged compressed cache for a batch into a contiguous view."""
    blocks_needed = (num_tokens + tokens_per_block - 1) // tokens_per_block
    blocks = []
    for blk in range(blocks_needed):
        block_id = int(block_offsets[0, batch_idx, 0, blk].item())
        block = kv_cache[block_id, 0].view(tokens_per_block, HEAD_DIM)
        blocks.append(block)
    return torch.cat(blocks, dim=0)[:num_tokens]


def run_ref_segmented_forward(
    ref: RefCompressor,
    tokens: torch.Tensor,
    freqs_cis: torch.Tensor,
    segments: list[tuple[int, int]],
) -> Optional[torch.Tensor]:
    """Run ref forward per segment, concatenating non-None outputs."""
    outputs = []
    cursor = 0
    for start_pos, seg_len in segments:
        seg_tokens = tokens[:, cursor : cursor + seg_len]
        freq_slice = freqs_cis[start_pos : start_pos + seg_len]
        # For decode segments (start_pos > 0), forward expects seqlen=1; step tokens.
        if start_pos > 0 and seg_len > 1:
            for i in range(seg_len):
                pos = start_pos + i
                out = ref(seg_tokens[:, i : i + 1], pos, freq_slice[i : i + 1])
                if out is not None:
                    outputs.append(out)
        else:
            out = ref(seg_tokens, start_pos, freq_slice)
            if out is not None:
                outputs.append(out)
        cursor += seg_len
    assert cursor <= tokens.size(1), "Segment lengths exceed provided tokens"
    if not outputs:
        return None
    return torch.cat(outputs, dim=1)


class CompressorWrapper:
    """Wrapper around Compressor to manage caches and provide a simpler test interface."""

    def __init__(self, compress_ratio: int = 4, rotate: bool = False, layer_idx: int = 0):
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.layer_idx = layer_idx
        coff = 2 if self.overlap else 1

        # Create MLAParams
        mla_params = MLAParams(
            hidden_size=DIM,
            qk_rope_head_dim=ROPE_DIM,
            qk_nope_head_dim=HEAD_DIM - ROPE_DIM,
        )

        # Create RoPE
        rope_params = RopeParams(
            dim=ROPE_DIM,
            theta=ROPE_THETA,
            max_positions=4096,
            beta_fast=BETA_FAST,
            beta_slow=BETA_SLOW,
            scale=ROPE_FACTOR,
            mscale=1.0,
            mscale_all_dim=1.0,
            original_max_positions=ORI_SEQ_LEN,
            scale_type=RotaryScalingType.yarn,
        )

        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.yarn,
            rope=rope_params,
            is_neox=False,
        )

        # Create the new Compressor
        self.compressor = Compressor(
            mla_params=mla_params,
            layer_idx=layer_idx,
            compress_ratio=compress_ratio,
            norm_eps=1e-6,
            skip_create_weights_in_init=False,
            pos_embd_params=pos_embd_params,
            page_size=PAGE_SIZE,
            dtype=DTYPE,
        ).to(DEVICE)

        # Allocate paged state caches
        total_pos = MAX_SEQ + (compress_ratio if self.overlap else 0)
        max_blocks = (total_pos + PAGE_SIZE - 1) // PAGE_SIZE
        num_blocks = MAX_BATCH * max_blocks

        self.paged_kv = torch.zeros(
            num_blocks, PAGE_SIZE, coff * HEAD_DIM, device=DEVICE, dtype=torch.float32
        )
        self.paged_score = torch.full_like(self.paged_kv, float("-inf"))
        self.block_table_kv = torch.arange(num_blocks, device=DEVICE, dtype=torch.int32).view(
            MAX_BATCH, max_blocks
        )
        self.block_table_score = self.block_table_kv.clone()

        # Allocate compressed KV cache
        # tokens_per_block must match page_size used by Compressor for scatter
        self.tokens_per_block = PAGE_SIZE
        max_compressed = MAX_SEQ // compress_ratio
        max_comp_blocks = (max_compressed + self.tokens_per_block - 1) // self.tokens_per_block
        num_comp_blocks = MAX_BATCH * max_comp_blocks

        self.kv_cache = torch.zeros(
            num_comp_blocks, 1, self.tokens_per_block * HEAD_DIM, device=DEVICE, dtype=DTYPE
        )
        self.block_offsets = torch.zeros(
            1, MAX_BATCH, 2, max_comp_blocks, device=DEVICE, dtype=torch.int32
        )
        for b in range(MAX_BATCH):
            for blk in range(max_comp_blocks):
                self.block_offsets[0, b, :, blk] = b * max_comp_blocks + blk

        # Per-batch state tracking
        self.kv_state = torch.zeros(
            MAX_BATCH, coff * compress_ratio, coff * HEAD_DIM, dtype=torch.float32, device=DEVICE
        )
        self.score_state = torch.full(
            (MAX_BATCH, coff * compress_ratio, coff * HEAD_DIM),
            float("-inf"),
            dtype=torch.float32,
            device=DEVICE,
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int | torch.Tensor,
        freqs_cis: torch.Tensor,
        batch_indices: torch.Tensor = None,
        seq_lens: torch.Tensor = None,
    ):
        """Wrapper forward that matches the reference Compressor interface.
        
        Supports mixed prefill+decode when seq_lens and start_pos tensor are provided.
        """
        ratio = self.compress_ratio

        # Handle variable-length sequences
        if seq_lens is not None:
            # Mixed batch mode with variable-length sequences
            seq_lens = seq_lens.to(torch.int32)
            bsz = seq_lens.size(0)
            if isinstance(start_pos, torch.Tensor):
                start_pos_tensor = start_pos.to(torch.int32)
            else:
                start_pos_tensor = torch.full((bsz,), start_pos, dtype=torch.int32, device=DEVICE)
            
            # Flatten input tokens
            if x.ndim == 3:
                x_flat = x.view(-1, DIM)
            else:
                x_flat = x
            total_tokens = int(seq_lens.sum().item())
            x_flat = x_flat[:total_tokens]
            
            # Determine which sequences are context (prefill) vs generation (decode)
            is_context = start_pos_tensor == 0
            num_contexts = int(is_context.sum().item())
            num_generations = bsz - num_contexts
            
            # Reorder: contexts first, then generations
            ctx_indices = torch.where(is_context)[0]
            gen_indices = torch.where(~is_context)[0]
            reorder_indices = torch.cat([ctx_indices, gen_indices])
            
            seq_lens_reordered = seq_lens[reorder_indices]
            start_pos_reordered = start_pos_tensor[reorder_indices]
            
            # Compute token offsets for reordering
            cu_seq_original = torch.zeros(bsz + 1, dtype=torch.int32, device=DEVICE)
            cu_seq_original[1:] = seq_lens.cumsum(0)
            
            # Reorder tokens: contexts first, then generations
            token_indices = []
            for idx in reorder_indices:
                start = cu_seq_original[idx].item()
                end = cu_seq_original[idx + 1].item()
                token_indices.extend(range(start, end))
            x_flat = x_flat[token_indices]
            
            num_ctx_tokens = int(seq_lens_reordered[:num_contexts].sum().item()) if num_contexts > 0 else 0
            num_gen_tokens = int(seq_lens_reordered[num_contexts:].sum().item()) if num_generations > 0 else 0
            seq_lens = seq_lens_reordered
            start_pos_for_kv = start_pos_reordered
            
            # Store reorder_indices for block table selection later
            batch_indices_for_blocks = reorder_indices
        else:
            # Original single-mode logic
            bsz, seqlen, _ = x.size()
            x_flat = x.view(-1, DIM)
            
            if seqlen == 1:
                # Decode mode
                num_contexts = 0
                num_generations = bsz
                num_ctx_tokens = 0
                num_gen_tokens = bsz
                seq_lens = torch.ones(bsz, dtype=torch.int32, device=DEVICE)
                start_pos_for_kv = torch.full((bsz,), start_pos, dtype=torch.int32, device=DEVICE)
            else:
                # Prefill mode
                num_contexts = bsz
                num_generations = 0
                num_ctx_tokens = bsz * seqlen
                num_gen_tokens = 0
                seq_lens = torch.full((bsz,), seqlen, dtype=torch.int32, device=DEVICE)
                start_pos_for_kv = torch.zeros(bsz, dtype=torch.int32, device=DEVICE)
            
            # No reordering needed in simple mode
            batch_indices_for_blocks = None

        cu_seq_lens = torch.zeros(bsz + 1, dtype=torch.int32, device=DEVICE)
        cu_seq_lens[1:] = seq_lens.cumsum(0)

        # Compute KV lengths (past + current) per sequence
        kv_lens = start_pos_for_kv + seq_lens

        # Compute number of compressed outputs per batch
        # For prefill (start_pos=0): compress every ratio tokens
        # For decode (start_pos>0): compress only if we complete a chunk
        is_prefill = start_pos_for_kv == 0
        num_comp_prefill = seq_lens // ratio
        should_compress_decode = ((start_pos_for_kv + seq_lens) % ratio == 0).to(torch.int32)
        num_comp = torch.where(is_prefill, num_comp_prefill, should_compress_decode)

        cu_kv_comp = torch.zeros(bsz + 1, dtype=torch.int32, device=DEVICE)
        cu_kv_comp[1:] = num_comp.cumsum(0)

        # Compute compressed start positions per sequence
        compressed_start_pos = start_pos_for_kv // ratio

        # Create position IDs for compressed outputs
        total_outputs = cu_kv_comp[-1].item()
        num_position_ids = max(total_outputs, 1)
        position_ids = torch.zeros(num_position_ids, dtype=torch.int64, device=DEVICE)
        offset = 0
        for b in range(bsz):
            n_out = num_comp[b].item()
            if is_prefill[b]:
                # Prefill: positions 0, ratio, 2*ratio, ...
                for i in range(n_out):
                    position_ids[offset + i] = i * ratio
            else:
                # Decode: position is start_pos
                for i in range(n_out):
                    position_ids[offset + i] = start_pos_for_kv[b].item()
            offset += n_out

        # Build dummy KV cache manager
        dummy_kv_cache_manager = DummyKVCacheManager(
            kv_cache=self.kv_cache,
            paged_kv_state=self.paged_kv,
            paged_score_state=self.paged_score,
        )

        # Build block_tables dict keyed by MewtwoAttentionType
        # Use batch_indices if provided, or batch_indices_for_blocks from reordering
        if batch_indices is not None:
            block_table_kv_selected = self.block_table_kv[batch_indices]
            block_table_score_selected = self.block_table_score[batch_indices]
        elif batch_indices_for_blocks is not None:
            block_table_kv_selected = self.block_table_kv[batch_indices_for_blocks]
            block_table_score_selected = self.block_table_score[batch_indices_for_blocks]
        else:
            block_table_kv_selected = self.block_table_kv[:bsz]
            block_table_score_selected = self.block_table_score[:bsz]
        
        block_tables = {
            MewtwoAttentionType.COMPRESS: self.block_offsets,
            MewtwoAttentionType.COMPRESSOR_STATE: block_table_kv_selected,
            MewtwoAttentionType.COMPRESSOR_SCORE: block_table_score_selected,
        }

        # Build dicts keyed by compress_ratio
        cu_seq_lens_dict = {ratio: cu_seq_lens}
        cu_kv_comp_dict = {ratio: cu_kv_comp}
        compressed_position_ids_dict = {ratio: position_ids}
        compressed_kv_lens_dict = {ratio: kv_lens}
        compressed_start_positions_dict = {ratio: compressed_start_pos}

        # Build dummy attention metadata
        metadata = DummyAttentionMetadata(
            num_contexts=num_contexts,
            num_generations=num_generations,
            num_ctx_tokens=num_ctx_tokens,
            num_tokens=num_ctx_tokens + num_gen_tokens,
            kv_cache_manager=dummy_kv_cache_manager,
            block_tables=block_tables,
            cu_seq_lens=cu_seq_lens_dict,
            cu_kv_comp=cu_kv_comp_dict,
            compressed_position_ids=compressed_position_ids_dict,
            compressed_kv_lens=compressed_kv_lens_dict,
            compressed_start_positions=compressed_start_positions_dict,
        )

        # Call the new compressor forward
        kv_comp = self.compressor(
            x=x_flat,
            metadata=metadata,
        )

        # Reshape output to [bsz, num_compressed, head_dim]
        total_outputs = cu_kv_comp[-1].item()
        if total_outputs == 0:
            return None

        # Split packed output back to per-batch
        outputs = []
        for b in range(bsz):
            start = cu_kv_comp[b].item()
            end = cu_kv_comp[b + 1].item()
            if end > start:
                outputs.append(kv_comp[start:end])
            else:
                outputs.append(None)

        # If all batches have the same number of outputs, stack them
        non_none_outputs = [o for o in outputs if o is not None]
        if len(non_none_outputs) == 0:
            return None
        elif all(o is not None and o.size(0) == outputs[0].size(0) for o in outputs):
            return torch.stack(outputs, dim=0)
        else:
            # Concatenate all non-None outputs along dim=0, then unsqueeze for batch dim
            # This handles mixed batches with different compression counts
            return torch.cat(non_none_outputs, dim=0).unsqueeze(0)

    def reset_state(self):
        """Reset paged caches for new sequence."""
        self.paged_kv.zero_()
        self.paged_score.fill_(float("-inf"))


def setup_compressors(compress_ratio: int = 4, rotate: bool = False):
    """Create synced RefCompressor + Compressor with all caches initialized."""
    args = ModelArgs()
    overlap = compress_ratio == 4

    # Reference compressor
    ref = RefCompressor(args, compress_ratio, HEAD_DIM, rotate).to(DEVICE)
    ref.ape.data.normal_(0, 0.02)
    ref.wkv.weight.data.normal_(0, 0.02)
    ref.wgate.weight.data.normal_(0, 0.02)
    ref.kv_cache = torch.zeros(
        MAX_BATCH, MAX_SEQ // compress_ratio, HEAD_DIM, device=DEVICE, dtype=DTYPE
    )

    # Compressor wrapper
    comp = CompressorWrapper(compress_ratio, rotate)

    # Copy weights from ref to compressor
    coff = 2 if overlap else 1
    # The compressor uses combined wkv_gate linear, need to copy weights appropriately
    # wkv_gate output is [state_dim * 2] = [coff * head_dim * 2]
    # First half is kv, second half is gate/score
    comp.compressor.wkv_gate.weight.data[: coff * HEAD_DIM] = ref.wkv.weight.data.clone()
    comp.compressor.wkv_gate.weight.data[coff * HEAD_DIM :] = ref.wgate.weight.data.clone()
    comp.compressor.ape.data.copy_(ref.ape.data)
    comp.compressor.norm.weight.data.copy_(ref.norm.weight.data)

    return ref, comp


@pytest.fixture(autouse=True)
def seed():
    """Seed RNG for reproducibility."""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.parametrize(
    "batch,seqlen,ratio",
    [
        (1, 128, 4),
        (2, 130, 4),
        (1, 2, 4),
        (2, 128, 8),  # basic configs
        (1, 64, 4),
        (2, 256, 4),
        (4, 128, 4),  # batch/seqlen variations
    ],
)
def test_prefill(batch, seqlen, ratio):
    """Test prefill mode."""
    ref, comp = setup_compressors(ratio)
    freqs = precompute_freqs_cis(
        ROPE_DIM, MAX_SEQ, ORI_SEQ_LEN, ROPE_THETA, ROPE_FACTOR, BETA_FAST, BETA_SLOW
    ).to(DEVICE)[:seqlen]
    x = torch.randn(batch, seqlen, DIM, device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        out_ref = ref(x, 0, freqs)
        out_comp = comp.forward(x, 0, freqs)

    assert_similar(out_ref, out_comp)
    if out_ref is not None:
        num_tokens = out_ref.size(1)
        for b in range(batch):
            cached_ref = ref.kv_cache[b : b + 1, :num_tokens]
            cached_comp = read_paged_cache_tokens(
                comp.kv_cache, comp.block_offsets, b, num_tokens, comp.tokens_per_block
            ).unsqueeze(0)
            assert_similar(cached_ref, out_ref[b : b + 1], f"Prefill ref cache[{b}]")
            assert_similar(cached_comp, out_comp[b : b + 1], f"Prefill comp cache[{b}]")
            assert_similar(cached_ref, cached_comp, f"Prefill cache parity[{b}]")


@pytest.mark.parametrize(
    "prefill,steps,batch,ratio",
    [
        (128, 8, 1, 4),
        (128, 8, 2, 4),
        (128, 24, 1, 4),
        (128, 8, 1, 8),
        (128, 4, 1, 128),
    ],
)
def test_decode(prefill, steps, batch, ratio):
    """Test prefill + decode."""
    ref, comp = setup_compressors(ratio)
    freqs = precompute_freqs_cis(
        ROPE_DIM, MAX_SEQ, ORI_SEQ_LEN, ROPE_THETA, ROPE_FACTOR, BETA_FAST, BETA_SLOW
    ).to(DEVICE)

    # Prefill
    x = torch.randn(batch, prefill, DIM, device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        assert_similar(ref(x, 0, freqs[:prefill]), comp.forward(x, 0, freqs[:prefill]), "Prefill")

    # Decode
    for step in range(steps):
        pos = prefill + step
        x = torch.randn(batch, 1, DIM, device=DEVICE, dtype=DTYPE)
        with torch.no_grad():
            out_ref = ref(x, pos, freqs[pos : pos + 1])
            out_comp = comp.forward(x, pos, freqs[pos : pos + 1])
            assert_similar(out_ref, out_comp, f"Decode[{step}]")
            if out_ref is not None:
                num_tokens = pos // ratio + 1
                for b in range(batch):
                    cached_ref = ref.kv_cache[b : b + 1, :num_tokens]
                    cached_comp = read_paged_cache_tokens(
                        comp.kv_cache, comp.block_offsets, b, num_tokens, comp.tokens_per_block
                    ).unsqueeze(0)
                    assert_similar(
                        cached_ref[:, -1:], out_ref[b : b + 1], f"Decode cache ref[{b}] step{step}"
                    )
                    assert_similar(
                        cached_comp[:, -1:],
                        out_comp[b : b + 1],
                        f"Decode cache comp[{b}] step{step}",
                    )
                    assert_similar(cached_ref, cached_comp, f"Decode cache parity[{b}] step{step}")


def test_varlen_batch():
    """Test variable-length prefill batch, compare with reference."""
    seq_lens = [64, 96, 128]
    ratio = 4

    # Test each sequence independently
    for i, slen in enumerate(seq_lens):
        ref, comp = setup_compressors(ratio)
        freqs = precompute_freqs_cis(
            ROPE_DIM, MAX_SEQ, ORI_SEQ_LEN, ROPE_THETA, ROPE_FACTOR, BETA_FAST, BETA_SLOW
        ).to(DEVICE)

        x = torch.randn(1, slen, DIM, device=DEVICE, dtype=DTYPE)

        with torch.no_grad():
            # Reset ref's state for each independent sequence
            ref.kv_state.zero_()
            ref.score_state.fill_(float("-inf"))
            out_ref = ref(x, 0, freqs[:slen])
            out_comp = comp.forward(x, 0, freqs[:slen])

        assert_similar(out_ref, out_comp, f"Varlen seq{i}")


def test_mixed_batch():
    """Test mixed context + generation requests in a single forward call.
    
    Simulates a realistic mixed batch with:
    - 1 context request: 8 tokens, start_pos=0
    - 1 generation request: 1 token, start_pos=127 (triggers compression at 128)
    
    Generation requests have seqlen=1 and require pre-populated state.
    """
    ratio = 4
    ref, comp = setup_compressors(ratio)
    freqs = precompute_freqs_cis(
        ROPE_DIM, MAX_SEQ, ORI_SEQ_LEN, ROPE_THETA, ROPE_FACTOR, BETA_FAST, BETA_SLOW
    ).to(DEVICE)

    # Context request: 8 tokens at start_pos=0 → 2 compressed outputs
    ctx_len = 8
    x_ctx = torch.randn(1, ctx_len, DIM, device=DEVICE, dtype=DTYPE)
    
    # Generation request: 1 token at start_pos=127 → triggers compression at 128
    # (127 + 1) % 4 == 0, so compression is triggered
    gen_start_pos = 127
    x_gen = torch.randn(1, 1, DIM, device=DEVICE, dtype=DTYPE)
    
    # For ref: need to pre-populate state by running prefill of gen_start_pos tokens
    # This simulates the generation request having processed gen_start_pos tokens already
    x_gen_prefill = torch.randn(1, gen_start_pos, DIM, device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        # === Reference: run each request separately ===
        
        # Context request on ref (batch 0)
        ref.kv_state.zero_()
        ref.score_state.fill_(float("-inf"))
        out_ref_ctx = ref(x_ctx, 0, freqs[:ctx_len])
        
        # Generation request on ref (batch 0, but independent - reset state first)
        # Pre-populate state by running prefill of gen_start_pos tokens
        ref.kv_state.zero_()
        ref.score_state.fill_(float("-inf"))
        _ = ref(x_gen_prefill, 0, freqs[:gen_start_pos])  # Sets up state
        # Now run the decode token
        out_ref_gen = ref(x_gen, gen_start_pos, freqs[gen_start_pos:gen_start_pos + 1])
        
        # Collect non-None outputs
        ref_outputs = [o for o in [out_ref_ctx, out_ref_gen] if o is not None]
        out_ref = torch.cat(ref_outputs, dim=1) if ref_outputs else None
        
        # === Compressor: single forward with mixed batch ===
        # Flatten tokens: [ctx_tokens, gen_token]
        x_flat = torch.cat([x_ctx.squeeze(0), x_gen.squeeze(0)], dim=0)
        
        # Sequence lengths and start positions for each request
        seq_lens = torch.tensor([ctx_len, 1], dtype=torch.int32, device=DEVICE)
        start_pos_tensor = torch.tensor([0, gen_start_pos], dtype=torch.int32, device=DEVICE)
        
        # Pre-populate compressor's paged state for generation request (batch idx 1)
        # by running prefill on batch 1 first
        comp.reset_state()
        comp.forward(x_gen_prefill, 0, freqs[:gen_start_pos], batch_indices=torch.tensor([1], device=DEVICE, dtype=torch.int32))
        
        # Now run mixed batch
        out_comp = comp.forward(x_flat, start_pos_tensor, freqs, seq_lens=seq_lens)

    if out_ref is None:
        assert out_comp is None, "Mixed batch: expected no compression output"
    else:
        assert_similar(out_ref, out_comp, "Mixed batch context+generation single forward")


# ============================================================================
# FP8 Blockwise Quantization Tests
# ============================================================================


def has_deep_gemm():
    """Check if DeepGEMM is available."""
    try:
        from tensorrt_llm import deep_gemm
        return deep_gemm is not None
    except Exception:
        return False


def skip_pre_hopper():
    """Skip test on pre-Hopper GPUs."""
    from tensorrt_llm._utils import get_sm_version
    sm_version = get_sm_version()
    return pytest.mark.skipif(
        sm_version < 90,
        reason=f"FP8 requires Hopper or later (SM >= 90), got SM {sm_version}"
    )


def compute_slot_mappings_fp8(
    num_tokens: int,
    head_dim: int,
    block_size: int,
    block_offsets: torch.Tensor,
    start_pos: torch.Tensor,
    num_comp_tokens: torch.Tensor,
    cu_kv_comp: torch.Tensor,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute slot mappings for FP8 blockwise cache layout.
    
    Note: These slot mappings are for the CUDA kernel interface (indexer_k_cache_scatter_op).
    The Triton kernel uses block_offsets directly and doesn't need these mappings.
    
    The cache layout is: [num_blocks, block_size, head_dim + scale_size]
    where scale_size = 4 bytes (float32) per token.
    
    Args:
        num_tokens: Total number of compressed tokens
        head_dim: Head dimension
        block_size: Tokens per block
        block_offsets: Block offset table [1, batch, 2, max_blocks]
        start_pos: Start positions per batch
        num_comp_tokens: Number of compressed tokens per batch
        cu_kv_comp: Cumulative compressed token counts
        device: Device to create tensors on
        
    Returns:
        slot_mapping_fp8: Flat byte indices for FP8 data
        slot_mapping_scale: Flat byte indices for scale data
    """
    scale_size = 4  # float32 = 4 bytes
    per_token_size = head_dim + scale_size
    block_stride = block_size * per_token_size
    
    slot_mapping_fp8 = torch.zeros(num_tokens, dtype=torch.int64, device=device)
    slot_mapping_scale = torch.zeros(num_tokens, dtype=torch.int64, device=device)
    
    bsz = num_comp_tokens.size(0)
    for b in range(bsz):
        num_comp = int(num_comp_tokens[b].item())
        s_pos = int(start_pos[b].item())
        cu_start = int(cu_kv_comp[b].item())
        
        for i in range(num_comp):
            token_pos = s_pos + i
            block_idx = token_pos // block_size
            pos_in_block = token_pos % block_size
            block_id = int(block_offsets[0, b, 0, block_idx].item())
            
            # FP8 data starts at beginning of block
            fp8_offset = block_id * block_stride + pos_in_block * head_dim
            # Scale data starts after FP8 data in block
            scale_base_offset = block_size * head_dim
            scale_offset = block_id * block_stride + scale_base_offset + pos_in_block * scale_size
            
            slot_mapping_fp8[cu_start + i] = fp8_offset
            slot_mapping_scale[cu_start + i] = scale_offset
    
    return slot_mapping_fp8, slot_mapping_scale


@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
@skip_pre_hopper()
@pytest.mark.parametrize("batch,seqlen", [(1, 128), (2, 64)])
def test_fp8_blockwise_compressor(batch, seqlen):
    """Test Compressor with blockwise FP8 quantization."""
    ratio = 4
    
    # Create MLAParams
    mla_params = MLAParams(
        hidden_size=DIM,
        qk_rope_head_dim=ROPE_DIM,
        qk_nope_head_dim=HEAD_DIM - ROPE_DIM,
    )
    
    # Create RoPE params
    rope_params = RopeParams(
        dim=ROPE_DIM,
        theta=ROPE_THETA,
        max_positions=4096,
        beta_fast=BETA_FAST,
        beta_slow=BETA_SLOW,
        scale=ROPE_FACTOR,
        mscale=1.0,
        mscale_all_dim=1.0,
        original_max_positions=ORI_SEQ_LEN,
        scale_type=RotaryScalingType.yarn,
    )
    
    pos_embd_params = PositionalEmbeddingParams(
        type=PositionEmbeddingType.yarn,
        rope=rope_params,
        is_neox=False,
    )
    
    # Create FP8 blockwise compressor
    compressor = Compressor(
        mla_params=mla_params,
        layer_idx=0,
        compress_ratio=ratio,
        norm_eps=1e-6,
        skip_create_weights_in_init=False,
        pos_embd_params=pos_embd_params,
        page_size=PAGE_SIZE,
        dtype=DTYPE,
        kv_cache_dtype="fp8_blockwise",
    ).to(DEVICE)
    
    # Initialize weights
    compressor.wkv_gate.weight.data.normal_(0, 0.02)
    compressor.ape.data.normal_(0, 0.02)
    
    # Allocate paged state caches
    overlap = ratio == 4
    coff = 2 if overlap else 1
    total_pos = MAX_SEQ + (ratio if overlap else 0)
    max_blocks = (total_pos + PAGE_SIZE - 1) // PAGE_SIZE
    num_blocks = MAX_BATCH * max_blocks
    
    paged_kv = torch.zeros(
        num_blocks, PAGE_SIZE, coff * HEAD_DIM, device=DEVICE, dtype=torch.float32
    )
    paged_score = torch.full_like(paged_kv, float("-inf"))
    block_table_kv = torch.arange(num_blocks, device=DEVICE, dtype=torch.int32).view(
        MAX_BATCH, max_blocks
    )
    block_table_score = block_table_kv.clone()
    
    # Allocate compressed KV cache for FP8
    # FP8 cache layout: [num_blocks, block_size, 1, head_dim + scale_size]
    tokens_per_block = PAGE_SIZE
    scale_size = 4  # float32
    per_token_size = HEAD_DIM + scale_size
    max_compressed = MAX_SEQ // ratio
    max_comp_blocks = (max_compressed + tokens_per_block - 1) // tokens_per_block
    num_comp_blocks = MAX_BATCH * max_comp_blocks
    
    # Use uint8 cache for FP8 storage (3D: [num_blocks, block_size, per_token_size])
    kv_cache_fp8 = torch.zeros(
        num_comp_blocks, tokens_per_block, per_token_size, device=DEVICE, dtype=torch.uint8
    )
    block_offsets = torch.zeros(
        1, MAX_BATCH, 2, max_comp_blocks, device=DEVICE, dtype=torch.int32
    )
    for b in range(MAX_BATCH):
        for blk in range(max_comp_blocks):
            block_offsets[0, b, :, blk] = b * max_comp_blocks + blk
    
    # Create dummy KV cache manager
    dummy_kv_cache_manager = DummyKVCacheManager(
        kv_cache=kv_cache_fp8,
        paged_kv_state=paged_kv,
        paged_score_state=paged_score,
    )
    
    # Compute expected outputs
    num_compressed = seqlen // ratio
    cu_kv_comp = torch.zeros(batch + 1, dtype=torch.int32, device=DEVICE)
    cu_kv_comp[1:] = torch.arange(1, batch + 1) * num_compressed
    total_comp_tokens = int(cu_kv_comp[-1].item())
    
    seq_lens = torch.full((batch,), seqlen, dtype=torch.int32, device=DEVICE)
    kv_lens = seq_lens.clone()
    cu_seq_lens = torch.zeros(batch + 1, dtype=torch.int32, device=DEVICE)
    cu_seq_lens[1:] = seq_lens.cumsum(0)
    
    start_pos = torch.zeros(batch, dtype=torch.int32, device=DEVICE)
    num_comp_tokens = cu_kv_comp[1:] - cu_kv_comp[:-1]
    
    # Compute position IDs for compressed tokens
    position_ids = torch.zeros(max(total_comp_tokens, 1), dtype=torch.int64, device=DEVICE)
    offset = 0
    for b in range(batch):
        for i in range(num_compressed):
            position_ids[offset + i] = i * ratio
        offset += num_compressed
    
    # Compute FP8 slot mappings
    slot_mapping_fp8, slot_mapping_scale = compute_slot_mappings_fp8(
        num_tokens=total_comp_tokens,
        head_dim=HEAD_DIM,
        block_size=tokens_per_block,
        block_offsets=block_offsets,
        start_pos=start_pos,
        num_comp_tokens=num_comp_tokens,
        cu_kv_comp=cu_kv_comp,
        device=DEVICE,
    )
    
    # Build block_tables dict
    block_tables = {
        MewtwoAttentionType.COMPRESS: block_offsets,
        MewtwoAttentionType.COMPRESSOR_STATE: block_table_kv[:batch],
        MewtwoAttentionType.COMPRESSOR_SCORE: block_table_score[:batch],
    }
    
    # Build metadata
    metadata = DummyAttentionMetadata(
        num_contexts=batch,
        num_generations=0,
        num_ctx_tokens=batch * seqlen,
        num_tokens=batch * seqlen,
        kv_cache_manager=dummy_kv_cache_manager,
        block_tables=block_tables,
        cu_seq_lens={ratio: cu_seq_lens},
        cu_kv_comp={ratio: cu_kv_comp},
        compressed_position_ids={ratio: position_ids},
        compressed_kv_lens={ratio: kv_lens},
        compressed_start_positions={ratio: start_pos},
        slot_mapping_fp8=slot_mapping_fp8,
        slot_mapping_scale=slot_mapping_scale,
    )
    
    # Create input
    x = torch.randn(batch * seqlen, DIM, device=DEVICE, dtype=DTYPE)
    
    with torch.no_grad():
        result = compressor(x, metadata)
    
    # FP8 blockwise mode returns (kv_fp8, kv_scale) tuple
    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    kv_fp8, kv_scale = result
    
    # Verify output shapes
    assert kv_fp8.shape == (total_comp_tokens, HEAD_DIM), \
        f"FP8 shape mismatch: {kv_fp8.shape} vs expected ({total_comp_tokens}, {HEAD_DIM})"
    num_scale_blocks = (HEAD_DIM + 127) // 128
    assert kv_scale.shape == (total_comp_tokens, num_scale_blocks), \
        f"Scale shape mismatch: {kv_scale.shape} vs expected ({total_comp_tokens}, {num_scale_blocks})"
    
    # Verify FP8 data was written to cache
    # Extract FP8 bytes from cache for first token (3D indexing: [block, token_in_block, data])
    if total_comp_tokens > 0:
        block_id = int(block_offsets[0, 0, 0, 0].item())
        fp8_bytes = kv_cache_fp8[block_id, 0, :HEAD_DIM]
        assert fp8_bytes.any(), "FP8 data should be non-zero in cache"
        
        # Extract scale from cache
        scale_bytes = kv_cache_fp8[block_id, 0, HEAD_DIM:HEAD_DIM + scale_size]
        scale_value = scale_bytes.view(torch.float32).item()
        assert scale_value != 0, "Scale should be non-zero"
    
    print(f"FP8 blockwise compressor test passed: batch={batch}, seqlen={seqlen}")


@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
@skip_pre_hopper()
def test_fp8_blockwise_roundtrip():
    """Verify FP8 quantization values survive write cycle."""
    torch.manual_seed(42)
    
    ratio = 4
    batch, seqlen = 1, 128
    num_compressed = seqlen // ratio
    
    # Create compressor
    mla_params = MLAParams(
        hidden_size=DIM,
        qk_rope_head_dim=ROPE_DIM,
        qk_nope_head_dim=HEAD_DIM - ROPE_DIM,
    )
    
    rope_params = RopeParams(
        dim=ROPE_DIM,
        theta=ROPE_THETA,
        max_positions=4096,
        beta_fast=BETA_FAST,
        beta_slow=BETA_SLOW,
        scale=ROPE_FACTOR,
        mscale=1.0,
        mscale_all_dim=1.0,
        original_max_positions=ORI_SEQ_LEN,
        scale_type=RotaryScalingType.yarn,
    )
    
    pos_embd_params = PositionalEmbeddingParams(
        type=PositionEmbeddingType.yarn,
        rope=rope_params,
        is_neox=False,
    )
    
    compressor = Compressor(
        mla_params=mla_params,
        layer_idx=0,
        compress_ratio=ratio,
        norm_eps=1e-6,
        skip_create_weights_in_init=False,
        pos_embd_params=pos_embd_params,
        page_size=PAGE_SIZE,
        dtype=DTYPE,
        kv_cache_dtype="fp8_blockwise",
    ).to(DEVICE)
    
    compressor.wkv_gate.weight.data.normal_(0, 0.02)
    compressor.ape.data.normal_(0, 0.02)
    
    # Setup caches
    overlap = ratio == 4
    coff = 2 if overlap else 1
    total_pos = MAX_SEQ + (ratio if overlap else 0)
    max_blocks = (total_pos + PAGE_SIZE - 1) // PAGE_SIZE
    num_blocks = MAX_BATCH * max_blocks
    
    paged_kv = torch.zeros(num_blocks, PAGE_SIZE, coff * HEAD_DIM, device=DEVICE, dtype=torch.float32)
    paged_score = torch.full_like(paged_kv, float("-inf"))
    block_table_kv = torch.arange(num_blocks, device=DEVICE, dtype=torch.int32).view(MAX_BATCH, max_blocks)
    block_table_score = block_table_kv.clone()
    
    tokens_per_block = PAGE_SIZE
    scale_size = 4
    per_token_size = HEAD_DIM + scale_size
    max_compressed = MAX_SEQ // ratio
    max_comp_blocks = (max_compressed + tokens_per_block - 1) // tokens_per_block
    num_comp_blocks = MAX_BATCH * max_comp_blocks
    
    # 3D cache: [num_blocks, block_size, per_token_size]
    kv_cache_fp8 = torch.zeros(
        num_comp_blocks, tokens_per_block, per_token_size, device=DEVICE, dtype=torch.uint8
    )
    block_offsets = torch.zeros(1, MAX_BATCH, 2, max_comp_blocks, device=DEVICE, dtype=torch.int32)
    for b in range(MAX_BATCH):
        for blk in range(max_comp_blocks):
            block_offsets[0, b, :, blk] = b * max_comp_blocks + blk
    
    dummy_kv_cache_manager = DummyKVCacheManager(
        kv_cache=kv_cache_fp8,
        paged_kv_state=paged_kv,
        paged_score_state=paged_score,
    )
    
    cu_kv_comp = torch.tensor([0, num_compressed], dtype=torch.int32, device=DEVICE)
    seq_lens = torch.tensor([seqlen], dtype=torch.int32, device=DEVICE)
    cu_seq_lens = torch.tensor([0, seqlen], dtype=torch.int32, device=DEVICE)
    start_pos = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    num_comp_tokens = torch.tensor([num_compressed], dtype=torch.int32, device=DEVICE)
    position_ids = torch.arange(0, seqlen, ratio, dtype=torch.int64, device=DEVICE)
    
    slot_mapping_fp8, slot_mapping_scale = compute_slot_mappings_fp8(
        num_tokens=num_compressed,
        head_dim=HEAD_DIM,
        block_size=tokens_per_block,
        block_offsets=block_offsets,
        start_pos=start_pos,
        num_comp_tokens=num_comp_tokens,
        cu_kv_comp=cu_kv_comp,
        device=DEVICE,
    )
    
    block_tables = {
        MewtwoAttentionType.COMPRESS: block_offsets,
        MewtwoAttentionType.COMPRESSOR_STATE: block_table_kv[:1],
        MewtwoAttentionType.COMPRESSOR_SCORE: block_table_score[:1],
    }
    
    metadata = DummyAttentionMetadata(
        num_contexts=1,
        num_generations=0,
        num_ctx_tokens=seqlen,
        num_tokens=seqlen,
        kv_cache_manager=dummy_kv_cache_manager,
        block_tables=block_tables,
        cu_seq_lens={ratio: cu_seq_lens},
        cu_kv_comp={ratio: cu_kv_comp},
        compressed_position_ids={ratio: position_ids},
        compressed_kv_lens={ratio: seq_lens},
        compressed_start_positions={ratio: start_pos},
        slot_mapping_fp8=slot_mapping_fp8,
        slot_mapping_scale=slot_mapping_scale,
    )
    
    x = torch.randn(seqlen, DIM, device=DEVICE, dtype=DTYPE)
    
    with torch.no_grad():
        result = compressor(x, metadata)
    
    # FP8 blockwise mode returns (kv_fp8, kv_scale) tuple directly
    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    kv_fp8, kv_scale = result
    
    # Verify scales were written correctly
    # kv_scale shape is [num_tokens, num_scale_blocks] where num_scale_blocks = (head_dim + 127) // 128
    for i in range(min(num_compressed, 5)):  # Check first 5 tokens
        block_idx = i // tokens_per_block
        pos_in_block = i % tokens_per_block
        block_id = int(block_offsets[0, 0, 0, block_idx].item())
        
        # Extract stored scale bytes (3D indexing)
        scale_bytes = kv_cache_fp8[block_id, pos_in_block, HEAD_DIM:HEAD_DIM + scale_size]
        stored_scale = scale_bytes.view(torch.float32).item()
        
        # Compare with returned scale (first scale block for simplicity)
        expected_scale = kv_scale[i, 0].item()
        assert abs(expected_scale - stored_scale) < 1e-5, \
            f"Token {i}: scale mismatch (expected={expected_scale:.6f}, stored={stored_scale:.6f})"
    
    print("FP8 blockwise roundtrip test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
