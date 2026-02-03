from typing import Literal, Optional

import torch
import torch.nn as nn

from tensorrt_llm._torch.attention_backend.interface import MLAParams, PositionalEmbeddingParams
from tensorrt_llm._torch.attention_backend.sparse.dsa import rotate_activation
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.modules.rotary_embedding import RotaryEmbedding
from tensorrt_llm._torch.utils import maybe_compiled_cat
from tensorrt_llm.quantization.utils import fp8_utils

from .kernel import (
    compressed_kv_scatter,
    kv_compress_prefill_triton,
    kv_compress_triton,
)
from .mewtwo import MewtwoAttentionType, MewtwoTrtllmAttentionMetadata

# KV cache dtype options
KVCacheDtype = Literal["default", "fp8_blockwise"]


class Compressor(nn.Module):
    """KV compressor using Triton kernels.

    This module uses paged memory management and Triton kernels for efficient KV compression.

    Args:
        mla_params: MLA parameters
        compress_ratio: Compression ratio (e.g., 4 means compress 4 tokens into 1)
        norm_eps: RMSNorm epsilon
        skip_create_weights_in_init: Whether to skip creating weights in init
        page_size: Size of each page in paged cache (default 32)
        pos_embd_params: Positional embedding parameters
        dtype: Data type
        kv_cache_dtype: KV cache data type - "default" uses dtype, "fp8_blockwise" uses
            blockwise FP8 quantization (1x128 block size) like Indexer compressor
    """

    def __init__(
        self,
        mla_params: MLAParams,
        layer_idx: int,
        compress_ratio: int,
        norm_eps: float,
        skip_create_weights_in_init: bool,
        pos_embd_params: PositionalEmbeddingParams,
        page_size: int = 32,
        dtype: Optional[torch.dtype] = torch.bfloat16,
        kv_cache_dtype: KVCacheDtype = "default",
    ):
        super().__init__()
        self.dim = mla_params.hidden_size
        self.head_dim = mla_params.qk_rope_head_dim + mla_params.qk_nope_head_dim
        self.rope_head_dim = mla_params.qk_rope_head_dim
        self.nope_head_dim = mla_params.qk_nope_head_dim
        self.overlap = compress_ratio == 4
        self.state_dim = 2 * self.head_dim if self.overlap else self.head_dim
        self.compress_ratio = compress_ratio
        self.page_size = page_size
        self.eps = norm_eps
        self.layer_idx = layer_idx
        self.kv_cache_dtype = kv_cache_dtype
        self.dtype = dtype

        # Linear layers for KV and gate projections
        self.wkv_gate = Linear(
            self.dim,
            self.state_dim * 2,
            bias=False,
            dtype=torch.float32,
            quant_config=None,
            skip_create_weights_in_init=skip_create_weights_in_init,
            use_custom_cublas_mm=True,
        )
        self.norm = RMSNorm(hidden_size=self.head_dim, eps=self.eps, dtype=dtype)

        # RoPE
        self.rotary_emb = RotaryEmbedding(
            pos_embd_params.rope,
            head_dim=self.rope_head_dim,
            is_neox=pos_embd_params.is_neox,
        )

        # Other parameters
        self.ape = nn.Parameter(torch.empty(compress_ratio, self.state_dim, dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        metadata: MewtwoTrtllmAttentionMetadata,
    ) -> Optional[torch.Tensor]:
        """Forward pass for paged compression.

        This method matches the interface of Compressor.forward() from model.py,
        but uses paged memory and Triton kernels internally.

        Args:
            x: Input tensor [m, dim]
            metadata: attention metadata

        Returns:
            Compressed KV tensor [bsz, num_compressed, head_dim] or None if no compression
        """

        # Get inputs from metadata
        num_contexts = metadata.num_contexts
        num_generations = metadata.num_generations
        num_ctx_tokens = metadata.num_ctx_tokens
        num_gen_tokens = metadata.num_tokens - metadata.num_ctx_tokens
        kv_cache = metadata.kv_cache_manager.get_buffers(
            self.layer_idx, MewtwoAttentionType.COMPRESS
        )
        paged_kv_state = metadata.kv_cache_manager.get_buffers(
            self.layer_idx, MewtwoAttentionType.COMPRESSOR_STATE
        )
        paged_score_state = metadata.kv_cache_manager.get_buffers(
            self.layer_idx, MewtwoAttentionType.COMPRESSOR_SCORE
        )
        block_table = metadata.block_tables[MewtwoAttentionType.COMPRESS]
        block_table_kv_state = metadata.block_tables[MewtwoAttentionType.COMPRESSOR_STATE]
        block_table_score_state = metadata.block_tables[MewtwoAttentionType.COMPRESSOR_SCORE]
        cu_kv_comp = metadata.cu_kv_comp[self.compress_ratio]
        kv_lens = metadata.compressed_kv_lens[self.compress_ratio]

        bsz = num_contexts + num_generations
        ratio, overlap = self.compress_ratio, self.overlap
        dtype = x.dtype

        # Compute number of compressed tokens per batch from cumulative offsets
        num_comp_tokens = cu_kv_comp[1:] - cu_kv_comp[:-1]

        # Project input to KV and score
        x_float = x.float()
        kv_score = self.wkv_gate(x_float)

        # Allocate output buffer
        # TODO: fix the CUDA graph compatibility issue
        kv_comp = torch.empty(
            max(cu_kv_comp[-1].item(), 1), self.head_dim, device=x.device, dtype=dtype
        )
        compressed_mask = torch.empty(bsz, device=x.device, dtype=torch.bool)

        # Update kv/score state and compress kv
        if num_contexts > 0:
            # Prefill mode: use TMA-optimized kernel
            cu_kv_comp_ctx = cu_kv_comp[: num_contexts + 1]
            kv_compress_prefill_triton(
                kv_score=kv_score[:num_ctx_tokens],
                ape=self.ape,
                kv_lens=kv_lens[:num_contexts],
                start_pos=None,
                cu_seq_lens=metadata.cu_seq_lens[self.compress_ratio],
                cu_kv_comp=cu_kv_comp_ctx,
                kv_comp=kv_comp,
                compressed_mask=compressed_mask[:num_contexts],
                paged_kv=paged_kv_state,
                paged_score=paged_score_state,
                block_table_kv=block_table_kv_state[:num_contexts],
                block_table_score=block_table_score_state[:num_contexts],
                compress_ratio=ratio,
                head_dim=self.head_dim,
                overlap=overlap,
                page_size=self.page_size,
            )
        if num_generations > 0:
            # Decode mode: ALWAYS call kernel to update paged cache
            # The kernel will write tokens to cache and only produce compressed
            # output when compression is triggered
            cu_kv_comp_gen = cu_kv_comp[num_contexts:]
            kv_compress_triton(
                kv_score=kv_score[num_ctx_tokens:],
                ape=self.ape,
                kv_lens=kv_lens[num_contexts : num_contexts + num_generations],
                start_pos=None,
                cu_seq_lens=metadata.cu_seq_lens[self.compress_ratio],
                cu_kv_comp=cu_kv_comp_gen,
                kv_comp=kv_comp,
                compressed_mask=compressed_mask[num_contexts:],
                paged_kv=paged_kv_state,
                paged_score=paged_score_state,
                block_table_kv=block_table_kv_state[num_contexts : num_contexts + num_generations],
                block_table_score=block_table_score_state[
                    num_contexts : num_contexts + num_generations
                ],
                compress_ratio=ratio,
                head_dim=self.head_dim,
                overlap=overlap,
                page_size=self.page_size,
                next_n=num_gen_tokens // num_generations,
            )

        # RMSNorm
        kv_comp = self.norm(kv_comp)

        # RoPE
        kv_comp_nope, kv_comp_pe = kv_comp.split([self.nope_head_dim, self.rope_head_dim], dim=-1)
        kv_comp_pe = self.rotary_emb(
            metadata.compressed_position_ids[self.compress_ratio], [kv_comp_pe]
        )
        kv_comp = maybe_compiled_cat([kv_comp_nope, kv_comp_pe[0]], dim=-1)

        # Hadamard rotation
        kv_comp = rotate_activation(kv_comp)

        # Scatter compressed KV to KV cache
        if self.kv_cache_dtype == "fp8_blockwise":
            # Blockwise FP8 quantization and scatter (like Indexer compressor)
            return self._update_kv_cache_fp8_blockwise(
                kv_comp=kv_comp,
                num_comp_tokens=num_comp_tokens[:bsz],
                cu_kv_comp=cu_kv_comp,
                start_pos=metadata.compressed_start_positions[self.compress_ratio][:bsz],
                kv_cache=kv_cache,
                block_offsets=block_table,
                metadata=metadata,
            )
        else:
            # Default: scatter with original dtype
            compressed_kv_scatter(
                compressed_kv=kv_comp,
                num_comp_tokens=num_comp_tokens[:bsz],
                cu_kv_comp=cu_kv_comp,
                start_pos=metadata.compressed_start_positions[self.compress_ratio][:bsz],
                kv_cache=kv_cache,
                block_offsets=block_table,
                tokens_per_block=self.page_size,
                head_dim=self.head_dim,
            )
            return kv_comp

    def _update_kv_cache_fp8_blockwise(
        self,
        kv_comp: torch.Tensor,
        num_comp_tokens: torch.Tensor,
        cu_kv_comp: torch.Tensor,
        start_pos: torch.Tensor,
        kv_cache: torch.Tensor,
        block_offsets: torch.Tensor,
        metadata: MewtwoTrtllmAttentionMetadata,
    ) -> None:
        """
        Update KV cache with blockwise FP8 quantization.
        
        Uses Triton kernel for scattering FP8 data and scales, supporting
        arbitrary head_dim (unlike the CUDA kernel which is limited to hdim=128).
        
        Args:
            kv_comp: Compressed KV tensor, shape [total_comp_tokens, head_dim]
            num_comp_tokens: Number of compressed tokens per batch
            cu_kv_comp: Cumulative compressed token counts
            start_pos: Start positions for each batch in compressed cache
            kv_cache: KV cache buffer [num_blocks, block_size, per_token_size]
            block_offsets: Block offsets for each batch
            metadata: Attention metadata (unused, kept for API compatibility)
        """
        num_tokens = kv_comp.shape[0]
        if num_tokens == 0:
            return

        head_dim = self.head_dim

        # Quantize to blockwise FP8
        kv_fp8, kv_scale = fp8_utils.fp8_quantize_1x128_sf_transpose(
            kv_comp, use_ue8m0=False
        )

        # Convert to bytes (flatten before viewing to ensure stride(-1) == 1)
        kv_fp8_bytes = kv_fp8.contiguous().flatten().view(torch.uint8).view(num_tokens, head_dim)
        scale_size = kv_scale.shape[1] * 4  # float32 = 4 bytes
        kv_scale_bytes = kv_scale.contiguous().flatten().view(torch.uint8).view(num_tokens, scale_size)

        # Use Triton kernel which supports arbitrary head_dim
        compressed_kv_scatter(
            kv_fp8_bytes,
            num_comp_tokens,
            cu_kv_comp,
            start_pos,
            kv_cache,
            block_offsets,
            self.page_size,
            head_dim,
            kv_cache_dtype="fp8_blockwise",
            kv_scale=kv_scale_bytes,
        )
        return kv_fp8, kv_scale 
