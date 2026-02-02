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
        block_tables = {
            MewtwoAttentionType.COMPRESS: self.block_offsets,
            MewtwoAttentionType.COMPRESSOR_STATE: self.block_table_kv[:bsz],
            MewtwoAttentionType.COMPRESSOR_SCORE: self.block_table_score[:bsz],
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
        if all(o is not None and o.size(0) == outputs[0].size(0) for o in outputs):
            return torch.stack(outputs, dim=0)
        else:
            # Return first batch output for compatibility with existing tests
            return outputs[0].unsqueeze(0) if outputs[0] is not None else None

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
    """Test prefill + decode in a single forward call, compare with reference."""
    ref, comp = setup_compressors()
    freqs = precompute_freqs_cis(
        ROPE_DIM, MAX_SEQ, ORI_SEQ_LEN, ROPE_THETA, ROPE_FACTOR, BETA_FAST, BETA_SLOW
    ).to(DEVICE)
    ratio = 4

    # Prefill sequence followed by decode tokens
    # Treat as 2 sequences: one prefill (8 tokens), one decode (1 token at position 8)
    prefill_len = 8
    decode_len = 1
    decode_start_pos = prefill_len  # Decode starts after prefill
    
    x_prefill = torch.randn(1, prefill_len, DIM, device=DEVICE, dtype=DTYPE)
    x_decode = torch.randn(1, decode_len, DIM, device=DEVICE, dtype=DTYPE)
    
    # Flatten tokens for mixed batch: [prefill_tokens, decode_tokens]
    x_flat = torch.cat([x_prefill.squeeze(0), x_decode.squeeze(0)], dim=0)  # [9, DIM]
    
    # Sequence lengths and start positions for mixed batch
    seq_lens = torch.tensor([prefill_len, decode_len], dtype=torch.int32, device=DEVICE)
    start_pos = torch.tensor([0, decode_start_pos], dtype=torch.int32, device=DEVICE)

    with torch.no_grad():
        # Reference: run prefill, then decode
        ref.kv_state.zero_()
        ref.score_state.fill_(float("-inf"))

        out_ref_prefill = ref(x_prefill, 0, freqs[:prefill_len])
        out_ref_decode = ref(x_decode, decode_start_pos, freqs[decode_start_pos:decode_start_pos + decode_len])
        
        # Concatenate ref outputs (filter None)
        ref_outputs = [o for o in [out_ref_prefill, out_ref_decode] if o is not None]
        out_ref = torch.cat(ref_outputs, dim=1) if ref_outputs else None
        
        # Compressor: single forward with mixed batch
        comp.reset_state()
        out_comp = comp.forward(x_flat, start_pos, freqs, seq_lens=seq_lens)

    if out_ref is None:
        assert out_comp is None, "Mixed batch: expected no compression output"
    else:
        assert_similar(out_ref, out_comp, "Mixed batch prefill+decode single forward")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
