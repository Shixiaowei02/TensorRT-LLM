from enum import IntEnum
from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
import torch.nn as nn

from tensorrt_llm._torch.attention_backend.interface import MLAParams, PositionalEmbeddingParams
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.modules.rotary_embedding import RotaryEmbedding

from .kernel import kv_compress_cutile, kv_compress_prefill_cutile

if TYPE_CHECKING:
    from .mewtwo import MewtwoTrtllmAttentionMetadata


class KVCacheDtype(IntEnum):
    """KV cache quantization mode (values match C++ cache_mode parameter)."""

    DEFAULT = 0  # bf16/fp16, no quantization
    FP8_PERTENSOR = 1  # FP8 with single per-tensor scale
    FP8_BLOCKWISE = 2  # FP8 with per-128-element blockwise scales


_KV_CACHE_DTYPE_MAP = {
    "default": KVCacheDtype.DEFAULT,
    "fp8_pertensor": KVCacheDtype.FP8_PERTENSOR,
    "fp8_blockwise": KVCacheDtype.FP8_BLOCKWISE,
}


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
        kv_cache_dtype: Cache quantization mode (KVCacheDtype enum or string)
        rotate_activation: Whether to apply Hadamard transform in postprocessing (False to skip)
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
        kv_cache_dtype: Union[str, KVCacheDtype] = KVCacheDtype.DEFAULT,
        is_indexer: bool = False,
        rotate_activation: bool = True,
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
        if isinstance(kv_cache_dtype, str):
            kv_cache_dtype = _KV_CACHE_DTYPE_MAP[kv_cache_dtype]
        self.kv_cache_dtype: KVCacheDtype = kv_cache_dtype
        self.is_indexer = is_indexer
        self.rotate_activation = rotate_activation

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
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass for paged KV compression.

        Args:
            x: Input tensor [num_tokens, dim]
            metadata: Attention metadata with cache info

        Returns:
            (kv_data, scale) tuple:
            - default / fp8_pertensor: (kv_comp, None)
            - fp8_blockwise indexer:   (fp8_output, scale_output)
            - no compressed tokens:    (None, None)
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
        max_ctx_comp_kv_lens = metadata.max_ctx_compressed_tokens[self.compress_ratio]

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
                start_pos=metadata.cached_token_lens_cuda[:num_contexts],
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
                next_n=metadata.num_gen_tokens_per_seq,
            )

        # Scatter to cache with appropriate quantization (all modes fused)
        start_pos = metadata.past_kv_lens_cuda[self.compress_ratio][:bsz]
        total_tokens = kv_comp.shape[0]

        # Allocate FP8 output buffers for blockwise indexer; other modes ignore them
        fp8_output = None
        scale_output = None
        if self.kv_cache_dtype == KVCacheDtype.FP8_BLOCKWISE and self.is_indexer:
            num_scale_blocks = self.head_dim // 128
            fp8_output = torch.empty(
                total_tokens, self.head_dim, dtype=torch.uint8, device=kv_comp.device
            )
            scale_output = torch.empty(
                total_tokens, num_scale_blocks, dtype=torch.float32, device=kv_comp.device
            )

        position_ids = metadata.compressed_position_ids_cuda[self.compress_ratio][:total_tokens]

        # Fused postprocess + scatter: RMSNorm + RoPE + Hadamard + paged cache write
        torch.ops.trtllm.compressor_postprocess_scatter(
            kv_comp,
            None,
            self.norm.weight,
            self.norm.variance_epsilon,
            self.rotary_emb.rotary_cos_sin,
            position_ids,
            self.nope_head_dim,
            self.rope_head_dim,
            kv_cache,
            num_comp_tokens,
            cu_new_comp_kv,
            start_pos,
            block_table,
            compress_tokens_per_block,
            int(self.kv_cache_dtype),
            self.rotate_activation,
            fp8_output,
            scale_output,
        )

        if fp8_output is not None:
            return fp8_output.view(torch.float8_e4m3fn), scale_output
        return kv_comp, None
