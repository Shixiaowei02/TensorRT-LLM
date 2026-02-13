from typing import TYPE_CHECKING, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn

from tensorrt_llm._torch.attention_backend.interface import MLAParams, PositionalEmbeddingParams
from tensorrt_llm._torch.attention_backend.sparse.dsa import rotate_activation
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.modules.rotary_embedding import RotaryEmbedding
from tensorrt_llm._torch.utils import maybe_compiled_cat
from tensorrt_llm.quantization.utils import fp8_utils

from .kernel import compressed_kv_scatter_cutile, kv_compress_cutile, kv_compress_prefill_cutile

if TYPE_CHECKING:
    from .mewtwo import MewtwoTrtllmAttentionMetadata

# KV cache dtype options:
#   "default"      - bf16/fp16, no quantization
#   "fp8_pertensor" - FP8 with single per-tensor scale (stored separately)
#   "fp8_blockwise" - FP8 with per-128-element blockwise scales (interleaved in cache)
KVCacheDtype = Literal["default", "fp8_pertensor", "fp8_blockwise"]


class Compressor(nn.Module):
    """KV compressor using Triton kernels with paged memory management.

    Args:
        mla_params: MLA parameters containing hidden_size and head dimensions
        layer_idx: Layer index for cache management
        compress_ratio: Compression ratio (e.g., 4 compresses 4 tokens into 1)
        norm_eps: RMSNorm epsilon
        skip_create_weights_in_init: Whether to skip weight initialization
        pos_embd_params: Positional embedding parameters for RoPE
        dtype: Data type for computation
        kv_cache_dtype: Cache quantization mode ("default", "fp8_pertensor", "fp8_blockwise")
    """

    def __init__(
        self,
        mla_params: MLAParams,
        layer_idx: int,
        compress_ratio: int,
        norm_eps: float,
        skip_create_weights_in_init: bool,
        pos_embd_params: PositionalEmbeddingParams,
        dtype: Optional[torch.dtype] = torch.bfloat16,
        kv_cache_dtype: KVCacheDtype = "default",
        is_indexer: bool = False,
    ):
        super().__init__()
        # Dimensions
        self.dim = mla_params.hidden_size
        self.head_dim = mla_params.qk_rope_head_dim + mla_params.qk_nope_head_dim
        self.rope_head_dim = mla_params.qk_rope_head_dim
        self.nope_head_dim = mla_params.qk_nope_head_dim

        # Compression config
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.state_dim = 2 * self.head_dim if self.overlap else self.head_dim

        # Cache config
        self.layer_idx = layer_idx
        self.kv_cache_dtype = kv_cache_dtype
        self.is_indexer = is_indexer

        # Modules
        self.wkv_gate = Linear(
            self.dim,
            self.state_dim * 2,
            bias=False,
            dtype=torch.float32,
            quant_config=None,
            skip_create_weights_in_init=skip_create_weights_in_init,
            use_custom_cublas_mm=True,
        )
        self.norm = RMSNorm(hidden_size=self.head_dim, eps=norm_eps, dtype=dtype)
        self.rotary_emb = RotaryEmbedding(
            pos_embd_params.rope,
            head_dim=self.rope_head_dim,
            is_neox=pos_embd_params.is_neox,
        )

        # Learnable absolute positional encoding for compression
        self.ape = nn.Parameter(torch.empty(compress_ratio, self.state_dim, dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        metadata: "MewtwoTrtllmAttentionMetadata",
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], None]:
        """Forward pass for paged KV compression.

        Args:
            x: Input tensor [num_tokens, dim]
            metadata: Attention metadata with cache info

        Returns:
            - "default" mode: kv_comp tensor [total_compressed, head_dim]
            - "fp8_blockwise" mode: (kv_fp8, kv_scale) tuple
            - "fp8_pertensor" mode: (kv_fp8, scale) tuple
        """
        # Import at runtime to avoid circular dependency
        from .mewtwo import MewtwoAttentionType

        # Extract metadata
        num_contexts = metadata.num_contexts
        num_generations = metadata.num_generations
        num_ctx_tokens = metadata.num_ctx_tokens
        bsz = num_contexts + num_generations

        # Determine attention types based on whether this is an indexer compressor
        if self.is_indexer:
            compress_type = MewtwoAttentionType.INDEXER_COMPRESS
            state_type = MewtwoAttentionType.INDEXER_COMPRESSOR_STATE
            score_type = MewtwoAttentionType.INDEXER_COMPRESSOR_SCORE
        else:
            compress_type = MewtwoAttentionType.COMPRESS
            state_type = MewtwoAttentionType.COMPRESSOR_STATE
            score_type = MewtwoAttentionType.COMPRESSOR_SCORE

        # Get cache buffers
        kv_cache = metadata.kv_cache_manager.get_buffers(self.layer_idx, compress_type)
        paged_kv_state = metadata.kv_cache_manager.get_buffers(self.layer_idx, state_type)
        paged_score_state = metadata.kv_cache_manager.get_buffers(self.layer_idx, score_type)

        # Get block tables
        block_table = metadata.block_tables[(self.compress_ratio, compress_type)]
        block_table_kv_state = metadata.block_tables[(self.compress_ratio, state_type)]
        block_table_score_state = metadata.block_tables[(self.compress_ratio, score_type)]

        # Get tokens_per_block from cache manager
        # state_tokens_per_block: for state/score caches (used in compress kernels)
        # compress_tokens_per_block: for compressed KV cache (used in scatter)
        state_tokens_per_block = metadata.kv_cache_manager.tokens_per_block
        compress_tokens_per_block = metadata.kv_cache_manager.compressed_block_sizes[self.layer_idx]

        # Get compression metadata
        cu_new_comp_kv = metadata.cu_new_comp_kv_cuda[self.compress_ratio]
        kv_lens = metadata.kv_lens_cuda_runtime
        total_num_comp_tokens = metadata.num_total_compressed_tokens[self.compress_ratio]
        num_comp_tokens = metadata.new_comp_kv_lens_cuda[self.compress_ratio][:bsz]
        max_num_comp_tokens = metadata.max_num_compressed_tokens[self.compress_ratio]
        max_ctx_comp_kv_lens, _, max_comp_kv_lens = max_num_comp_tokens

        # Project input to KV and score
        kv_score = self.wkv_gate(x.float())

        # Allocate output buffer
        kv_comp = torch.empty(total_num_comp_tokens, self.head_dim, device=x.device, dtype=x.dtype)
        compressed_mask = torch.empty(bsz, device=x.device, dtype=torch.bool)

        # Run compression kernels
        if num_contexts > 0:
            kv_compress_prefill_cutile(
                kv_score=kv_score[:num_ctx_tokens],
                ape=self.ape,
                kv_lens=kv_lens[:num_contexts],
                start_pos=None,
                cu_seq_lens=metadata.cu_seq_lens_cuda,
                cu_new_comp_kv=cu_new_comp_kv[: num_contexts + 1],
                kv_comp=kv_comp,
                compressed_mask=compressed_mask[:num_contexts],
                paged_kv=paged_kv_state,
                paged_score=paged_score_state,
                block_table_kv=block_table_kv_state[:num_contexts],
                block_table_score=block_table_score_state[:num_contexts],
                compress_ratio=self.compress_ratio,
                head_dim=self.head_dim,
                overlap=self.overlap,
                page_size=state_tokens_per_block,
                max_outputs=max_ctx_comp_kv_lens,
            )

        if num_generations > 0:
            num_gen_tokens = metadata.num_tokens - num_ctx_tokens
            kv_compress_cutile(
                kv_score=kv_score[num_ctx_tokens:],
                ape=self.ape,
                kv_lens=kv_lens[num_contexts:],
                start_pos=None,
                cu_seq_lens=metadata.cu_seq_lens_cuda,
                cu_new_comp_kv=cu_new_comp_kv[num_contexts:],
                kv_comp=kv_comp,
                compressed_mask=compressed_mask[num_contexts:],
                paged_kv=paged_kv_state,
                paged_score=paged_score_state,
                block_table_kv=block_table_kv_state[num_contexts:],
                block_table_score=block_table_score_state[num_contexts:],
                compress_ratio=self.compress_ratio,
                head_dim=self.head_dim,
                overlap=self.overlap,
                page_size=state_tokens_per_block,
                next_n=num_gen_tokens // num_generations,
            )

        # If there are no compressed tokens, there should be no generation requests.
        # Directly return the compressed tokens.
        if total_num_comp_tokens == 0:
            if self.kv_cache_dtype == "fp8_blockwise":
                return None, None
            else:
                return kv_comp

        # Post-processing: RMSNorm -> RoPE -> Hadamard rotation
        kv_comp = self.norm(kv_comp)
        kv_comp_nope, kv_comp_pe = kv_comp.split([self.nope_head_dim, self.rope_head_dim], dim=-1)

        kv_comp_pe = self.rotary_emb(
            metadata.compressed_position_ids_cuda[self.compress_ratio][: kv_comp_pe.shape[0]],
            [kv_comp_pe],
        )
        kv_comp = maybe_compiled_cat([kv_comp_nope, kv_comp_pe[0]], dim=-1)
        kv_comp = rotate_activation(kv_comp)

        # Scatter to cache with appropriate quantization
        start_pos = metadata.past_kv_lens_cuda[self.compress_ratio][:bsz]

        if self.kv_cache_dtype == "fp8_blockwise":
            return self._scatter_fp8_blockwise(
                kv_comp,
                num_comp_tokens,
                cu_new_comp_kv,
                start_pos,
                kv_cache,
                block_table,
                compress_tokens_per_block,
                max_comp_kv_lens,
            )
        elif self.kv_cache_dtype == "fp8_pertensor":
            return self._scatter_fp8_pertensor(
                kv_comp,
                num_comp_tokens,
                cu_new_comp_kv,
                start_pos,
                kv_cache,
                block_table,
                compress_tokens_per_block,
                max_comp_kv_lens,
            )
        else:
            compressed_kv_scatter_cutile(
                kv_comp,
                num_comp_tokens,
                cu_new_comp_kv,
                start_pos,
                kv_cache,
                block_table,
                compress_tokens_per_block,
                self.head_dim,
                max_comp_kv_lens,
            )
            return kv_comp

    def _scatter_fp8_blockwise(
        self,
        kv_comp: torch.Tensor,
        num_comp_tokens: torch.Tensor,
        cu_new_comp_kv: torch.Tensor,
        start_pos: torch.Tensor,
        kv_cache: torch.Tensor,
        block_offsets: torch.Tensor,
        tokens_per_block: int,
        max_outputs: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Quantize to blockwise FP8 and scatter to cache.

        Returns:
            (kv_fp8, kv_scale) tuple, or None if empty
        """
        num_tokens = kv_comp.shape[0]
        if num_tokens == 0:
            return None

        # Quantize with per-128-element scales
        kv_fp8, kv_scale = fp8_utils.fp8_quantize_1x128_sf_transpose(kv_comp, use_ue8m0=False)

        # kv_fp8: [num_tokens, head_dim] in float8_e4m3fn - pass directly
        # kv_scale: [num_tokens, num_scale_blocks] in float32 - pass directly

        compressed_kv_scatter_cutile(
            kv_fp8.contiguous().view(num_tokens, self.head_dim),
            num_comp_tokens,
            cu_new_comp_kv,
            start_pos,
            kv_cache,
            block_offsets,
            tokens_per_block,
            self.head_dim,
            max_outputs=max_outputs,
            kv_cache_dtype="fp8_blockwise",
            kv_scale=kv_scale,
        )
        return kv_fp8, kv_scale

    def _scatter_fp8_pertensor(
        self,
        kv_comp: torch.Tensor,
        num_comp_tokens: torch.Tensor,
        cu_new_comp_kv: torch.Tensor,
        start_pos: torch.Tensor,
        kv_cache: torch.Tensor,
        block_offsets: torch.Tensor,
        tokens_per_block: int,
        max_outputs: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Quantize to per-tensor FP8 and scatter to cache.

        Uses static scale = 1.0
        kv_scale_quant_orig: dequantization scale = 1.0
        kv_scale_orig_quant: quantization scale = 1.0

        Returns:
            (kv_fp8, kv_scale_quant_orig) tuple, or None if empty
        """
        num_tokens = kv_comp.shape[0]
        if num_tokens == 0:
            return None

        # Static scale = 1.0 (no scaling, following trtllm.py convention)
        # kv_scale_quant_orig = dequant scale = 1.0
        # kv_scale_orig_quant = quant scale = 1.0
        kv_scale_quant_orig = torch.ones(1, dtype=torch.float32, device=kv_comp.device)

        # Quantize with scale = 1.0 (direct cast to FP8)
        kv_fp8 = kv_comp.to(torch.float8_e4m3fn)

        compressed_kv_scatter_cutile(
            kv_fp8.view(torch.uint8),
            num_comp_tokens,
            cu_new_comp_kv,
            start_pos,
            kv_cache,
            block_offsets,
            tokens_per_block,
            self.head_dim,
            max_outputs=max_outputs,
            kv_cache_dtype="fp8_pertensor",
        )
        return kv_fp8, kv_scale_quant_orig
