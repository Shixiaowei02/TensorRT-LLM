#!/usr/bin/env python3
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests comparing Compressor with RefCompressor."""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
from tensorrt_llm._torch.attention_backend.sparse.mewtwo import MewtwoCacheManager
from tensorrt_llm._torch.attention_backend.sparse.mewtwo.compressor import Compressor
from tensorrt_llm._torch.attention_backend.sparse.mewtwo.mewtwo import MewtwoAttentionType
from tensorrt_llm._torch.modules.rotary_embedding import RopeParams
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import DataType, SamplingConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MewtwoSparseAttentionConfig
from tensorrt_llm.mapping import Mapping

# ============================================================================
# Dummy Metadata for Testing
# ============================================================================


class DummyAttentionMetadata:
    """Dummy attention metadata for testing Compressor.forward."""

    def __init__(
        self,
        num_contexts: int,
        num_generations: int,
        num_ctx_tokens: int,
        num_tokens: int,
        kv_cache_manager: MewtwoCacheManager,
        block_tables: dict,
        cu_seq_lens: dict,
        cu_new_comp_kv: dict,
        compressed_position_ids: dict,
        compressed_kv_lens: dict,
        past_kv_lens: dict,
        num_compressed_tokens: dict,
        slot_mapping_fp8: torch.Tensor = None,
        slot_mapping_scale: torch.Tensor = None,
    ):
        self.num_contexts = num_contexts
        self.num_generations = num_generations
        self.num_ctx_tokens = num_ctx_tokens
        self.num_tokens = num_tokens
        self.kv_cache_manager = kv_cache_manager
        self.block_tables = block_tables
        self.cu_seq_lens_cuda = cu_seq_lens
        self.cu_new_comp_kv_cuda = cu_new_comp_kv
        self.compressed_position_ids_cuda = compressed_position_ids
        self.compressed_kv_lens_cuda = compressed_kv_lens
        self.past_kv_lens_cuda = past_kv_lens
        self.slot_mapping_fp8 = slot_mapping_fp8
        self.slot_mapping_scale = slot_mapping_scale
        self.num_compressed_tokens = num_compressed_tokens


# ============================================================================
# Reference Implementation (DO NOT MODIFY)
# ============================================================================


@dataclass
class ModelArgs:
    """Model arguments for Compressor."""

    max_batch_size: int = 16
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
INDEX_HEAD_DIM = 128  # Fixed head_dim for indexer (INDEXER_COMPRESS)
MAX_BATCH, MAX_SEQ, PAGE_SIZE = 16, 4096, 128
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


def dequantize_blockwise_fp8(
    kv_fp8: torch.Tensor, kv_scale: torch.Tensor, block_size: int = 128
) -> torch.Tensor:
    """Dequantize blockwise FP8 data.

    Args:
        kv_fp8: [num_tokens, head_dim] FP8 tensor
        kv_scale: [num_tokens, num_scale_blocks] scale factors (one per 128 elements)

    Returns:
        Dequantized float tensor
    """
    num_tokens, head_dim = kv_fp8.shape
    num_blocks = (head_dim + block_size - 1) // block_size

    kv_float = kv_fp8.float()
    kv_dequant = torch.zeros_like(kv_float)

    for b in range(num_blocks):
        start = b * block_size
        end = min(start + block_size, head_dim)
        kv_dequant[:, start:end] = kv_float[:, start:end] * kv_scale[:, b : b + 1]

    return kv_dequant


def assert_fp8_similar(
    fp8_result: tuple,
    ref_result: torch.Tensor,
    kv_cache_dtype: str,
    name: str = "FP8 Output",
):
    """Assert FP8 result matches reference after dequantization.

    Uses FP8-appropriate tolerances: cos_sim >= 0.99, rel_err <= 10%
    """
    kv_fp8, scale = fp8_result

    if kv_cache_dtype == "fp8_blockwise":
        kv_dequant = dequantize_blockwise_fp8(kv_fp8, scale)
    else:  # fp8_pertensor
        kv_dequant = kv_fp8.float() * scale.item()

    ref_float = ref_result.float().flatten()
    dequant_flat = kv_dequant.flatten()

    # Cosine similarity (>= 0.99 for FP8)
    cos_sim = F.cosine_similarity(dequant_flat.unsqueeze(0), ref_float.unsqueeze(0)).item()
    assert cos_sim >= 0.99, f"{name}: cos_sim={cos_sim:.4f} < 0.99"

    # Relative error (<= 10% for FP8)
    max_diff = (dequant_flat - ref_float).abs().max().item()
    scale_val = max(ref_float.abs().max().item(), 1e-3)
    rel_err = max_diff / scale_val
    assert rel_err <= 0.1, f"{name}: rel_err={rel_err:.4f} > 0.1"


def read_paged_cache_tokens(
    kv_cache: torch.Tensor,
    block_offsets: torch.Tensor,
    batch_idx: int,
    num_tokens: int,
    tokens_per_block: int,
) -> torch.Tensor:
    """Materialize paged compressed cache for a batch into a contiguous view.

    Args:
        kv_cache: Cache buffer with shape [num_blocks, tokens_per_block, head_dim]
        block_offsets: Block offset table with shape [num_seqs, max_blocks]
        batch_idx: Index of the batch to read
        num_tokens: Number of tokens to read
        tokens_per_block: Tokens per cache block

    Returns:
        Tensor containing the read values with shape [num_tokens, head_dim]
    """
    blocks_needed = (num_tokens + tokens_per_block - 1) // tokens_per_block
    # Extract block indices for this batch
    block_indices = block_offsets[batch_idx, :blocks_needed].tolist()
    # Read all blocks at once and reshape
    _, _, dim_per_token = kv_cache.shape
    return kv_cache[block_indices].reshape(-1, dim_per_token)[:num_tokens]


def build_fp8_golden_cache(
    kv_fp8: torch.Tensor,
    kv_scale: torch.Tensor,
    cache_shape: tuple,
    block_offsets: torch.Tensor,
    batch: int,
    num_compressed: int,
    tokens_per_block: int,
    kv_cache_dtype: str,
    head_dim: int = HEAD_DIM,
) -> torch.Tensor:
    """Build Python golden reference cache from compressor's FP8 output.

    This validates the cache scatter operation by manually placing the compressor's
    FP8 output into the expected cache positions. Quantization correctness is
    validated separately via assert_fp8_similar.

    Args:
        kv_fp8: Compressor's FP8 output [total_tokens, head_dim]
        kv_scale: Compressor's scale output
        cache_shape: Shape of the cache tensor
        block_offsets: Block offset table [num_seqs, max_blocks]
        batch: Batch size
        num_compressed: Compressed tokens per batch
        tokens_per_block: Tokens per cache block
        kv_cache_dtype: "fp8_blockwise" or "fp8_pertensor"
        head_dim: Head dimension (default HEAD_DIM=512, use INDEX_HEAD_DIM=128 for indexer)

    Returns:
        golden_cache: Expected cache tensor for comparison
    """
    total_comp_tokens = batch * num_compressed
    golden_cache = torch.zeros(cache_shape, device=kv_fp8.device, dtype=torch.uint8)

    # Convert FP8 to bytes
    kv_fp8_bytes = kv_fp8.contiguous().view(torch.uint8).view(total_comp_tokens, head_dim)

    if kv_cache_dtype == "fp8_blockwise":
        # Blockwise: non-interleaved layout per block:
        # [k0, k1, ..., kN, scale0, scale1, ..., scaleN]
        num_scale_blocks = (head_dim + 127) // 128
        scale_size = num_scale_blocks * 4
        kv_scale_bytes = (
            kv_scale.flatten().contiguous().view(torch.uint8).view(total_comp_tokens, scale_size)
        )

        # Flatten to 2D [num_blocks, block_stride] for flat byte indexing
        num_blocks = cache_shape[0]
        golden_flat = golden_cache.view(num_blocks, -1)

        # Scatter into cache using non-interleaved offsets
        global_token = 0
        for b in range(batch):
            for i in range(num_compressed):
                block_idx = i // tokens_per_block
                pos_in_block = i % tokens_per_block
                block_id = int(block_offsets[b, block_idx].item())

                # FP8 data in first section of block
                fp8_start = pos_in_block * head_dim
                golden_flat[block_id, fp8_start : fp8_start + head_dim] = kv_fp8_bytes[global_token]
                # Scale data in second section of block
                scale_start = tokens_per_block * head_dim + pos_in_block * scale_size
                golden_flat[block_id, scale_start : scale_start + scale_size] = kv_scale_bytes[
                    global_token
                ]
                global_token += 1

        # Reshape back to original shape
        golden_cache = golden_flat.view(cache_shape)
    else:
        # Per-tensor: cache layout is [num_blocks, tokens_per_block, head_dim]
        # Same as blockwise but without scale bytes
        global_token = 0
        for b in range(batch):
            for i in range(num_compressed):
                block_idx = i // tokens_per_block
                pos_in_block = i % tokens_per_block
                block_id = int(block_offsets[b, block_idx].item())

                golden_cache[block_id, pos_in_block, :head_dim] = kv_fp8_bytes[global_token]
                global_token += 1

    return golden_cache


def assert_fp8_cache_match(
    kernel_cache: torch.Tensor,
    golden_cache: torch.Tensor,
    kv_cache_dtype: str,
    name: str = "FP8 Cache",
    head_dim: int = HEAD_DIM,
):
    """Assert kernel cache matches Python golden reference."""
    # Convert both to uint8 bytes for comparison (kernel_cache may be Float8_e4m3fn)
    kernel_bytes = kernel_cache.view(torch.uint8)
    golden_bytes = golden_cache.view(torch.uint8)

    if torch.equal(kernel_bytes, golden_bytes):
        return  # Perfect match

    diff_mask = kernel_bytes != golden_bytes
    num_diffs = diff_mask.sum().item()
    total_bytes = kernel_bytes.numel()

    # Build detailed error message
    msg = (
        f"{name}: {num_diffs}/{total_bytes} byte differences ({100 * num_diffs / total_bytes:.4f}%)"
    )

    if kv_cache_dtype == "fp8_blockwise":
        # Reshape to original layout for detailed analysis
        kernel_reshaped = kernel_bytes.view(kernel_cache.shape)
        golden_reshaped = golden_bytes.view(golden_cache.shape)
        fp8_diffs = (
            (kernel_reshaped[:, :, :head_dim] != golden_reshaped[:, :, :head_dim]).sum().item()
        )
        scale_diffs = (
            (kernel_reshaped[:, :, head_dim:] != golden_reshaped[:, :, head_dim:]).sum().item()
        )

        msg += f" [FP8: {fp8_diffs}, Scale: {scale_diffs}]"

    # Show first few differences
    diff_indices = torch.nonzero(diff_mask.view(-1))[:5]
    for idx in diff_indices:
        flat_idx = idx.item()
        msg += f"\n  Byte {flat_idx}: kernel={kernel_cache.view(-1)[flat_idx].item()}, "
        msg += f"golden={golden_cache.view(-1)[flat_idx].item()}"

    raise AssertionError(msg)


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

    # Class-level constants for MewtwoCacheManager
    WINDOW_SIZE = 128
    VOCAB_SIZE = 129280

    def __init__(
        self,
        compress_ratio: int = 4,
        rotate: bool = False,
        layer_idx: int = 0,
        kv_cache_dtype: str = "default",
        is_indexer: bool = False,
    ):
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.layer_idx = layer_idx
        self.kv_cache_dtype = kv_cache_dtype
        self.is_indexer = is_indexer
        coff = 2 if self.overlap else 1

        # Create MLAParams
        # For indexer mode, use INDEX_HEAD_DIM instead of HEAD_DIM
        target_head_dim = INDEX_HEAD_DIM if is_indexer else HEAD_DIM
        mla_params = MLAParams(
            hidden_size=DIM,
            qk_rope_head_dim=ROPE_DIM,
            qk_nope_head_dim=target_head_dim - ROPE_DIM,
        )

        # Create RoPE - use no scaling to match precompute_freqs_cis behavior
        # (precompute_freqs_cis only applies yarn scaling when seqlen > ORI_SEQ_LEN)
        rope_params = RopeParams(
            dim=ROPE_DIM,
            theta=ROPE_THETA,
            max_positions=4096,
            beta_fast=BETA_FAST,
            beta_slow=BETA_SLOW,
            scale=1.0,  # No scaling
            mscale=1.0,
            mscale_all_dim=1.0,
            original_max_positions=ORI_SEQ_LEN,
            scale_type=RotaryScalingType.none,  # Match precompute_freqs_cis (no yarn for pos < ORI_SEQ_LEN)
        )

        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.rope_gpt_neox,  # Basic RoPE without yarn
            rope=rope_params,
            is_neox=False,
        )

        # Create the Compressor
        self.compressor = Compressor(
            mla_params=mla_params,
            layer_idx=layer_idx,
            compress_ratio=compress_ratio,
            norm_eps=1e-6,
            skip_create_weights_in_init=False,
            pos_embd_params=pos_embd_params,
            dtype=DTYPE,
            kv_cache_dtype=kv_cache_dtype,
            is_indexer=is_indexer,
        ).to(DEVICE)

        # Create MewtwoCacheManager
        self.cache_manager = self._create_mewtwo_cache_manager(compress_ratio)
        # COMPRESS cache has tokens_per_block from cache manager's compressed_block_sizes
        self.tokens_per_block = self.cache_manager.compressed_block_sizes[self.layer_idx]

        # Track active requests for the cache manager
        self.active_requests: Dict[int, LlmRequest] = {}
        self.next_request_id = 0

        # Store reference to kv_cache for test compatibility (read from cache manager)
        self._update_kv_cache_reference()

        # Create block_offsets for test compatibility (will be updated per forward)
        max_compressed = MAX_SEQ // compress_ratio
        max_comp_blocks = (max_compressed + self.tokens_per_block - 1) // self.tokens_per_block
        self.block_offsets = torch.zeros(
            MAX_BATCH, max_comp_blocks, device=DEVICE, dtype=torch.int32
        )

    def cleanup(self):
        """Free all active requests and shut down the cache manager.

        Must be called before the wrapper is discarded, otherwise the cache
        manager's destructor will fail with ResourceBusyError and GPU memory
        will leak across tests.
        """
        if not hasattr(self, "cache_manager"):
            return
        for req in self.active_requests.values():
            self.cache_manager.free_resources(req)
        self.active_requests.clear()
        self.cache_manager.shutdown()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    def _create_mewtwo_cache_manager(self, compress_ratio: int) -> MewtwoCacheManager:
        """Create a MewtwoCacheManager for testing."""
        # Single layer with the given compress ratio
        compress_ratios = [compress_ratio]

        # Create sparse attention config
        sparse_attn_config = MewtwoSparseAttentionConfig(
            index_head_dim=INDEX_HEAD_DIM,
            window_size=self.WINDOW_SIZE,
            compress_ratios=compress_ratios,
        )

        # Create KV cache config
        max_num_tokens = MAX_SEQ * MAX_BATCH
        kv_cache_config = KvCacheConfig(
            enable_block_reuse=False,
            max_tokens=max_num_tokens,
            event_buffer_max_size=0,
        )

        # Create mapping (single GPU, no parallelism)
        mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)

        if self.kv_cache_dtype in ["fp8_pertensor", "fp8_blockwise"]:
            cache_dtype = DataType.FP8
        else:
            cache_dtype = DataType.BF16

        # Create cache manager
        cache_manager = MewtwoCacheManager(
            kv_cache_config=kv_cache_config,
            kv_cache_type=CacheTypeCpp.SELFKONLY,
            num_layers=len(compress_ratios),
            num_kv_heads=1,
            head_dim=HEAD_DIM,
            tokens_per_block=PAGE_SIZE,
            max_seq_len=MAX_SEQ,
            max_batch_size=MAX_BATCH,
            mapping=mapping,
            dtype=cache_dtype,
            compressor_dtype=DataType.FLOAT,  # State caches always use FP32
            vocab_size=self.VOCAB_SIZE,
            max_num_tokens=max_num_tokens,
            sparse_attn_config=sparse_attn_config,
        )

        return cache_manager

    def _update_kv_cache_reference(self):
        """Update the kv_cache reference from cache manager for test compatibility."""
        compress_type = (
            MewtwoAttentionType.INDEXER_COMPRESS
            if self.is_indexer
            else MewtwoAttentionType.COMPRESS
        )
        self.kv_cache = self.cache_manager.get_buffers(self.layer_idx, compress_type)

    def _create_request(self, request_id: int, prompt_len: int) -> LlmRequest:
        """Helper to create a test LlmRequest (following test_mewtwo_cache_manager pattern).

        Args:
            request_id: Unique request identifier
            prompt_len: Prompt length (number of tokens)

        Returns:
            LlmRequest instance
        """
        input_tokens = list(range(prompt_len))
        request = LlmRequest(
            request_id=request_id,
            max_new_tokens=1024,
            input_tokens=input_tokens,
            sampling_config=SamplingConfig(),
            is_streaming=False,
        )
        return request

    def _prepare_requests_for_batch(
        self,
        bsz: int,
        seq_lens: torch.Tensor,
        start_pos: torch.Tensor,
        is_prefill: torch.Tensor,
        batch_indices: torch.Tensor = None,
    ) -> Tuple[List[LlmRequest], ScheduledRequests]:
        """Prepare requests for a batch using prepare_resources pattern from test_mewtwo_cache_manager.

        Args:
            batch_indices: Optional tensor mapping batch positions to external batch indices.
                          If provided, requests are stored/retrieved using these indices.
                          If None, sequential indices (0, 1, 2...) are used.

        Returns:
            Tuple of (requests list, scheduled_batch) for later update_resources call
        """
        requests = []
        context_requests = []
        generation_requests = []

        # Use batch_indices if provided, otherwise use sequential indices
        if batch_indices is not None:
            ext_indices = [int(batch_indices[b].item()) for b in range(bsz)]
        else:
            ext_indices = list(range(bsz))

        # Separate prefill and generation indices
        prefill_indices = []
        gen_indices = []
        for b in range(bsz):
            if is_prefill[b]:
                prefill_indices.append(b)
            else:
                gen_indices.append(b)

        # Handle prefill requests - create LlmRequest directly (following test_mewtwo_cache_manager pattern)
        for b in prefill_indices:
            seq_len = int(seq_lens[b].item())
            req = self._create_request(self.next_request_id, seq_len)
            ext_idx = ext_indices[b]
            self.active_requests[ext_idx] = req
            context_requests.append(req)
            self.next_request_id += 1

        # Handle generation requests - reuse existing requests or create new ones
        for b in gen_indices:
            ext_idx = ext_indices[b]
            req = self.active_requests.get(ext_idx)
            if req is None:
                # Need to create a new request for generation (prefill was done in previous call)
                pos = int(start_pos[b].item())
                req = self._create_request(self.next_request_id, pos)
                req.state = LlmRequestState.GENERATION_IN_PROGRESS
                req.context_current_position = pos
                req.add_new_token(pos, 0)  # Simulate having processed tokens
                self.active_requests[ext_idx] = req
                self.next_request_id += 1
            else:
                # Existing request from previous prefill - mark as generation
                req.state = LlmRequestState.GENERATION_IN_PROGRESS
            generation_requests.append(req)

        # Build final request list in batch order
        for b in range(bsz):
            ext_idx = ext_indices[b]
            requests.append(self.active_requests[ext_idx])

        # Build scheduled batch and call prepare_resources (following test_mewtwo_cache_manager pattern)
        scheduled_batch = ScheduledRequests()
        scheduled_batch.context_requests = context_requests
        scheduled_batch.generation_requests = generation_requests
        self.cache_manager.prepare_resources(scheduled_batch)

        return requests, scheduled_batch

    def _get_block_table_for_request(
        self,
        req: LlmRequest,
        attn_type: MewtwoAttentionType,
    ) -> torch.Tensor:
        """Get block table for a request and attention type."""
        page_indices = self.cache_manager.get_cache_indices(
            request_id=req.py_request_id,
            layer_idx=self.layer_idx,
            attn_type=attn_type,
        )
        return torch.tensor(page_indices, dtype=torch.int32, device=DEVICE)

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

            num_ctx_tokens = (
                int(seq_lens_reordered[:num_contexts].sum().item()) if num_contexts > 0 else 0
            )
            num_gen_tokens = (
                int(seq_lens_reordered[num_contexts:].sum().item()) if num_generations > 0 else 0
            )
            seq_lens = seq_lens_reordered
            past_kv_lens = start_pos_reordered

            # Use reorder_indices for request mapping (maps reordered position to original batch index)
            batch_indices_for_requests = reorder_indices
            start_pos_for_kv = past_kv_lens

            # Get number of compressed tokens
            num_ctx_compressed_tokens = (seq_lens[:num_contexts] // ratio).sum().item()
            num_gen_compressed_tokens = num_generations
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
                past_kv_lens = torch.full((bsz,), start_pos, dtype=torch.int32, device=DEVICE)
                num_ctx_compressed_tokens = 0
                num_gen_compressed_tokens = bsz
            else:
                # Prefill mode
                num_contexts = bsz
                num_generations = 0
                num_ctx_tokens = bsz * seqlen
                num_gen_tokens = 0
                seq_lens = torch.full((bsz,), seqlen, dtype=torch.int32, device=DEVICE)
                past_kv_lens = torch.zeros(bsz, dtype=torch.int32, device=DEVICE)
                num_ctx_compressed_tokens = (seq_lens // ratio).sum().item()
                num_gen_compressed_tokens = 0

            # Use batch_indices for request mapping if provided
            batch_indices_for_requests = batch_indices
            start_pos_for_kv = past_kv_lens

        # Prepare requests for the cache manager
        is_prefill = start_pos_for_kv == 0
        requests, scheduled_batch = self._prepare_requests_for_batch(
            bsz, seq_lens, start_pos_for_kv, is_prefill, batch_indices_for_requests
        )

        cu_seq_lens = torch.zeros(bsz + 1, dtype=torch.int32, device=DEVICE)
        cu_seq_lens[1:] = seq_lens.cumsum(0)
        num_compressed_tokens = num_ctx_compressed_tokens + num_gen_compressed_tokens

        # Compute KV lengths (past + current) per sequence
        kv_lens = past_kv_lens + seq_lens

        # Compute number of compressed outputs per batch
        # For prefill (start_pos=0): compress every ratio tokens
        # For decode (start_pos>0): compress only if we complete a chunk
        num_comp_prefill = seq_lens // ratio
        should_compress_decode = ((past_kv_lens + seq_lens) % ratio == 0).to(torch.int32)
        num_comp = torch.where(is_prefill, num_comp_prefill, should_compress_decode)

        cu_new_comp_kv = torch.zeros(bsz + 1, dtype=torch.int32, device=DEVICE)
        cu_new_comp_kv[1:] = num_comp.cumsum(0)

        # Create position IDs for compressed outputs
        position_ids = torch.zeros(num_compressed_tokens, dtype=torch.int64, device=DEVICE)
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
                    position_ids[offset + i] = past_kv_lens[b].item()
            offset += n_out

        # Determine attention types based on is_indexer
        if self.is_indexer:
            compress_type = MewtwoAttentionType.INDEXER_COMPRESS
            state_type = MewtwoAttentionType.INDEXER_COMPRESSOR_STATE
            score_type = MewtwoAttentionType.INDEXER_COMPRESSOR_SCORE
        else:
            compress_type = MewtwoAttentionType.COMPRESS
            state_type = MewtwoAttentionType.COMPRESSOR_STATE
            score_type = MewtwoAttentionType.COMPRESSOR_SCORE

        # Build block_tables dict keyed by MewtwoAttentionType using cache manager
        block_table_compress_list = []
        block_table_kv_state_list = []
        block_table_score_state_list = []

        for b, req in enumerate(requests):
            block_table_compress_list.append(self._get_block_table_for_request(req, compress_type))
            block_table_kv_state_list.append(self._get_block_table_for_request(req, state_type))
            block_table_score_state_list.append(self._get_block_table_for_request(req, score_type))

        # Pad and stack block tables to handle variable-length block indices
        max_blocks_compress = max(bt.size(0) for bt in block_table_compress_list)
        block_table_compress = torch.zeros(
            bsz, max_blocks_compress, dtype=torch.int32, device=DEVICE
        )
        for b in range(bsz):
            bt = block_table_compress_list[b]
            block_table_compress[b, : bt.size(0)] = bt

        max_blocks_state = max(bt.size(0) for bt in block_table_kv_state_list)
        block_table_kv_state = torch.zeros(bsz, max_blocks_state, dtype=torch.int32, device=DEVICE)
        block_table_score_state = torch.zeros(
            bsz, max_blocks_state, dtype=torch.int32, device=DEVICE
        )
        for b in range(bsz):
            bt_kv = block_table_kv_state_list[b]
            bt_score = block_table_score_state_list[b]
            block_table_kv_state[b, : bt_kv.size(0)] = bt_kv
            block_table_score_state[b, : bt_score.size(0)] = bt_score

        # Update block_offsets for test compatibility
        self.block_offsets = block_table_compress

        block_tables = {
            compress_type: block_table_compress,
            state_type: block_table_kv_state,
            score_type: block_table_score_state,
        }

        # Both prefill and decode kernels use absolute token positions for the
        # state cache, so pass the absolute kv_lens directly.
        # Build dicts keyed by compress_ratio
        cu_new_comp_kv_dict = {ratio: cu_new_comp_kv}
        compressed_position_ids_dict = {ratio: position_ids}
        compressed_kv_lens_dict = {ratio: kv_lens}
        past_kv_lens_dict = {ratio: past_kv_lens // ratio}
        num_compressed_tokens_dict = {ratio: num_compressed_tokens}

        # Build attention metadata using MewtwoCacheManager
        metadata = DummyAttentionMetadata(
            num_contexts=num_contexts,
            num_generations=num_generations,
            num_ctx_tokens=num_ctx_tokens,
            num_tokens=num_ctx_tokens + num_gen_tokens,
            kv_cache_manager=self.cache_manager,
            block_tables=block_tables,
            cu_seq_lens=cu_seq_lens,
            cu_new_comp_kv=cu_new_comp_kv_dict,
            compressed_position_ids=compressed_position_ids_dict,
            compressed_kv_lens=compressed_kv_lens_dict,
            past_kv_lens=past_kv_lens_dict,
            num_compressed_tokens=num_compressed_tokens_dict,
        )

        # Update kv_cache reference for test compatibility
        self._update_kv_cache_reference()

        # Call the compressor forward
        result = self.compressor(x=x_flat, metadata=metadata)

        # Update request state and call update_resources after processing
        for b, req in enumerate(requests):
            if is_prefill[b]:
                req.context_current_position = int(seq_lens[b].item())
            # Call add_new_token for BOTH prefill and generation requests
            # Token count should be total sequence length after processing (start_pos + seq_len)
            token_count = int((start_pos_for_kv[b] + seq_lens[b]).item())
            req.add_new_token(token_count, 0)
        self.cache_manager.update_resources(scheduled_batch)

        # For FP8 modes, return tuple directly (kv_fp8, kv_scale)
        if isinstance(result, tuple):
            return result

        # For default mode, reshape output to [bsz, num_compressed, head_dim]
        kv_comp = result
        total_outputs = cu_new_comp_kv[-1].item()
        if total_outputs == 0:
            return None

        # Split packed output back to per-batch
        outputs = []
        for b in range(bsz):
            start = cu_new_comp_kv[b].item()
            end = cu_new_comp_kv[b + 1].item()
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
        """Reset cache manager state for new sequences."""
        # Shutdown existing cache manager to release resources properly
        if hasattr(self, "cache_manager") and self.cache_manager is not None:
            self.cache_manager.shutdown()

        # Clear active requests and reset request ID counter
        self.active_requests.clear()
        self.next_request_id = 0

        # Recreate the cache manager to reset all caches
        self.cache_manager = self._create_mewtwo_cache_manager(self.compress_ratio)
        self._update_kv_cache_reference()


def setup_compressors(
    compress_ratio: int = 4,
    rotate: bool = False,
    kv_cache_dtype: str = "default",
):
    """Create synced RefCompressor + Compressor with all caches initialized.

    Args:
        compress_ratio: Compression ratio (default 4)
        rotate: Whether to apply Hadamard rotation
        kv_cache_dtype: Cache dtype - "default", "fp8_blockwise", or "fp8_pertensor"

    Returns:
        ref: RefCompressor (bf16 reference)
        comp: CompressorWrapper (with specified kv_cache_dtype)
    """
    args = ModelArgs()
    overlap = compress_ratio == 4

    # For fp8_blockwise with ratio=4, use INDEXER_COMPRESS type which has scale space allocated
    # INDEXER_COMPRESS is only registered for sparse layers and uses INDEX_HEAD_DIM=128
    is_indexer = kv_cache_dtype == "fp8_blockwise"

    # Use appropriate head_dim based on is_indexer
    target_head_dim = INDEX_HEAD_DIM if is_indexer else HEAD_DIM

    # Reference compressor (bf16)
    ref = RefCompressor(args, compress_ratio, target_head_dim, rotate).to(DEVICE)
    ref.ape.data.normal_(0, 0.02)
    ref.wkv.weight.data.normal_(0, 0.02)
    ref.wgate.weight.data.normal_(0, 0.02)
    ref.kv_cache = torch.zeros(
        MAX_BATCH, MAX_SEQ // compress_ratio, target_head_dim, device=DEVICE, dtype=DTYPE
    )

    # Compressor wrapper with specified kv_cache_dtype
    comp = CompressorWrapper(
        compress_ratio, rotate, kv_cache_dtype=kv_cache_dtype, is_indexer=is_indexer
    )

    # Copy weights from ref to compressor
    coff = 2 if overlap else 1
    comp.compressor.wkv_gate.weight.data[: coff * target_head_dim] = ref.wkv.weight.data.clone()
    comp.compressor.wkv_gate.weight.data[coff * target_head_dim :] = ref.wgate.weight.data.clone()
    comp.compressor.ape.data.copy_(ref.ape.data)
    comp.compressor.norm.weight.data.copy_(ref.norm.weight.data)

    return ref, comp


@pytest.fixture(autouse=True)
def seed():
    """Seed RNG for reproducibility and ensure GPU cleanup between tests."""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    yield
    # Force garbage collection so CompressorWrapper.__del__ → cleanup() runs
    # before the next test allocates new cache managers.
    import gc

    gc.collect()
    torch.cuda.empty_cache()


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
        (1, 256, 128),
        (2, 512, 128),
        (16, 234, 128),  # ratio=128 coverage
    ],
)
def test_prefill(batch, seqlen, ratio):
    """Test prefill mode."""
    # rotate=True required: Compressor unconditionally applies Hadamard rotation
    ref, comp = setup_compressors(ratio, rotate=True)
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
    # rotate=True required: Compressor unconditionally applies Hadamard rotation
    ref, comp = setup_compressors(ratio, rotate=True)
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
        # rotate=True required: Compressor unconditionally applies Hadamard rotation
        ref, comp = setup_compressors(ratio, rotate=True)
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
    # rotate=True required: Compressor unconditionally applies Hadamard rotation
    ref, comp = setup_compressors(ratio, rotate=True)
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
        out_ref_gen = ref(x_gen, gen_start_pos, freqs[gen_start_pos : gen_start_pos + 1])

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
        comp.forward(
            x_gen_prefill,
            0,
            freqs[:gen_start_pos],
            batch_indices=torch.tensor([1], device=DEVICE, dtype=torch.int32),
        )

        # Now run mixed batch
        out_comp = comp.forward(x_flat, start_pos_tensor, freqs, seq_lens=seq_lens)

    if out_ref is None:
        assert out_comp is None, "Mixed batch: expected no compression output"
    else:
        assert_similar(out_ref, out_comp, "Mixed batch context+generation single forward")


# ============================================================================
# FP8 Blockwise Quantization Tests
# ============================================================================


@pytest.mark.parametrize(
    "batch,seqlen,ratio",
    [
        (1, 128, 4),
        (2, 64, 4),
        (4, 256, 4),
        (16, 256, 4),
        (16, 512, 4),
    ],
)
def test_fp8_blockwise_compressor(batch, seqlen, ratio):
    """Test FP8 blockwise Compressor against RefCompressor (bf16 reference).

    FP8 blockwise uses INDEXER_COMPRESS cache type which has INDEX_HEAD_DIM=128.
    This test validates FP8 quantization and cache scatter for the indexer path.
    """
    num_compressed = seqlen // ratio

    ref, comp = setup_compressors(ratio, rotate=True, kv_cache_dtype="fp8_blockwise")
    freqs = precompute_freqs_cis(
        ROPE_DIM, MAX_SEQ, ORI_SEQ_LEN, ROPE_THETA, ROPE_FACTOR, BETA_FAST, BETA_SLOW
    ).to(DEVICE)[:seqlen]
    x = torch.randn(batch, seqlen, DIM, device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        out_ref = ref(x, 0, freqs)
        out_comp = comp.forward(x, 0, freqs)

    assert isinstance(out_comp, tuple), f"Expected tuple, got {type(out_comp)}"
    kv_fp8, kv_scale = out_comp

    # Verify shapes - INDEXER_COMPRESS uses INDEX_HEAD_DIM=128
    total_comp_tokens = batch * num_compressed
    num_scale_blocks = (INDEX_HEAD_DIM + 127) // 128
    assert kv_fp8.shape == (total_comp_tokens, INDEX_HEAD_DIM), (
        f"Expected shape {(total_comp_tokens, INDEX_HEAD_DIM)}, got {kv_fp8.shape}"
    )
    assert kv_scale.shape == (total_comp_tokens, num_scale_blocks), (
        f"Expected scale shape {(total_comp_tokens, num_scale_blocks)}, got {kv_scale.shape}"
    )

    # Verify scales are positive and valid
    assert (kv_scale > 0).all(), "All scales should be positive"

    # Compare dequantized FP8 with RefCompressor output (both use INDEX_HEAD_DIM=128)
    assert_fp8_similar(
        out_comp, out_ref.view(-1, INDEX_HEAD_DIM), "fp8_blockwise", "FP8 vs RefCompressor"
    )

    # Verify cache scatter
    golden_cache = build_fp8_golden_cache(
        kv_fp8,
        kv_scale,
        comp.kv_cache.shape,
        comp.block_offsets,
        batch,
        num_compressed,
        comp.tokens_per_block,
        "fp8_blockwise",
        head_dim=INDEX_HEAD_DIM,
    )
    assert_fp8_cache_match(
        comp.kv_cache,
        golden_cache,
        "fp8_blockwise",
        "Blockwise cache layout",
        head_dim=INDEX_HEAD_DIM,
    )


@pytest.mark.parametrize(
    "batch,seqlen,ratio",
    [
        (1, 128, 4),
        (2, 256, 4),
        (1, 256, 128),
        (2, 512, 128),  # ratio=128 coverage
    ],
)
def test_fp8_pertensor_compressor(batch, seqlen, ratio):
    """Test per-tensor FP8 Compressor against RefCompressor (bf16 reference)."""
    num_compressed = seqlen // ratio

    ref, comp = setup_compressors(ratio, rotate=True, kv_cache_dtype="fp8_pertensor")
    freqs = precompute_freqs_cis(
        ROPE_DIM, MAX_SEQ, ORI_SEQ_LEN, ROPE_THETA, ROPE_FACTOR, BETA_FAST, BETA_SLOW
    ).to(DEVICE)[:seqlen]
    x = torch.randn(batch, seqlen, DIM, device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        out_ref = ref(x, 0, freqs)
        out_comp = comp.forward(x, 0, freqs)

    assert isinstance(out_comp, tuple), f"Expected tuple, got {type(out_comp)}"
    kv_fp8, kv_scale = out_comp

    # Verify shapes
    total_comp_tokens = batch * num_compressed
    assert kv_fp8.shape == (total_comp_tokens, HEAD_DIM)
    assert kv_scale.numel() == 1, f"Per-tensor scale should be scalar, got {kv_scale.shape}"
    assert kv_scale.item() > 0, "Scale should be positive"

    # Compare dequantized FP8 with reference
    assert_fp8_similar(
        out_comp, out_ref.view(-1, HEAD_DIM), "fp8_pertensor", "FP8 vs RefCompressor"
    )

    # Verify scale is fixed at 1.0 (compressor uses static scale following trtllm.py convention)
    assert kv_scale.item() == 1.0, (
        f"Per-tensor scale should be fixed at 1.0, got {kv_scale.item():.6f}"
    )

    # Verify cache scatter (cache is FP8 dtype, same as compressor output)
    golden_cache = build_fp8_golden_cache(
        kv_fp8,
        kv_scale,
        comp.kv_cache.shape,
        comp.block_offsets,
        batch,
        num_compressed,
        comp.tokens_per_block,
        "fp8_pertensor",
    )
    assert_fp8_cache_match(comp.kv_cache, golden_cache, "fp8_pertensor", "Per-tensor cache layout")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
