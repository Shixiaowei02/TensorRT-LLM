import math
from enum import Enum
from typing import TYPE_CHECKING, Dict, Optional, Set, Tuple

import torch

from tensorrt_llm._torch.attention_backend.interface import MLAParams, PositionalEmbeddingParams
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention
from tensorrt_llm._torch.modules.multi_stream_utils import maybe_execute_in_parallel
from tensorrt_llm.models.modeling_utils import QuantConfig

from ..dsa import DSAtrtllmAttentionMetadata, Indexer, _to_float
from ..kernel import mewtwo_local_to_global_indices
from .compressor import Compressor

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import SparseAttentionConfig


def build_window_local_indices(
    token_positions: torch.Tensor,
    window_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Build SWA local indices for all tokens.
    """
    positions = token_positions.unsqueeze(1)  # [num_tokens, 1]
    offsets = torch.arange(window_size, dtype=torch.int32, device=device)

    # matrix[i, j] = max(0, pos[i] - window_size + 1) + j
    swa_start = (positions - window_size + 1).clamp(min=0)
    swa_indices = swa_start + offsets

    swa_indices = torch.where(swa_indices > positions, -1, swa_indices)
    return swa_indices.to(torch.int32)


def build_compressed_local_indices(
    token_positions: torch.Tensor,
    compress_ratio: int,
    max_compressed_indices: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Build compressed local indices for compress_ratio=128.
    For each token, indices are arange(0, (pos+1) // compress_ratio).
    """
    num_tokens = token_positions.shape[0]
    # Number of valid compressed indices per token
    num_valid = (token_positions + 1) // compress_ratio  # [num_tokens]

    # Create output filled with -1
    indices = torch.full((num_tokens, max_compressed_indices), -1, dtype=torch.int32, device=device)

    # Generate sequential indices: 0, 1, 2, ..., max_compressed_indices-1
    col_indices = torch.arange(max_compressed_indices, dtype=torch.int32, device=device)

    # Mask: valid where col_idx < num_valid[row]
    valid_mask = col_indices.unsqueeze(0) < num_valid.unsqueeze(1)

    # Fill valid positions with sequential indices
    indices = torch.where(valid_mask, col_indices.unsqueeze(0).expand(num_tokens, -1), indices)

    return indices


class MewtwoTrtllmAttentionMetadata(DSAtrtllmAttentionMetadata):
    # The set of compress ratios for the layers
    compress_ratio_set: Set[int]
    # The number of total compressed tokens for each compress ratio
    num_compressed_tokens: Dict[int, int] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compress_ratio_set = set(self.compress_ratio)

    def __post_init__(self):
        super().__post_init__()
        capture_graph = self.is_cuda_graph

        # Create buffers for the compressor
        # cu_seq_lens_cuda is the cumulative sequence lengths for the requests
        self.cu_seq_lens_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences + 1,),
            dtype=torch.int,
            cache_name="cu_seq_lens_cuda",
            capture_graph=capture_graph,
        )
        self.cu_seq_lens = torch.empty_like(self.cu_seq_lens_cuda, device="cpu", pin_memory=True)
        self.cu_seq_lens[0] = 0

        # cu_new_comp_kv_cuda is the cumulative number of new compressed tokens for the requests
        self.cu_new_comp_kv_cuda = {
            compress_ratio: self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_sequences + 1,),
                dtype=torch.int,
                cache_name=f"cu_new_comp_kv_cuda_{compress_ratio}",
                capture_graph=capture_graph,
            )
            for compress_ratio in self.compress_ratio_set
        }
        self.cu_new_comp_kv = {
            compress_ratio: torch.empty_like(
                self.cu_new_comp_kv_cuda[compress_ratio], device="cpu", pin_memory=True
            )
            for compress_ratio in self.compress_ratio_set
        }
        for compress_ratio in self.compress_ratio_set:
            self.cu_new_comp_kv[compress_ratio][0] = 0

        # compressed_kv_lens_cuda is the number of compressed tokens for the requests
        self.compressed_kv_lens_cuda = {
            compress_ratio: self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_sequences,),
                dtype=torch.int,
                cache_name=f"compressed_kv_lens_cuda_{compress_ratio}",
                capture_graph=capture_graph,
            )
            for compress_ratio in self.compress_ratio_set
        }
        self.compressed_kv_lens = {
            compress_ratio: torch.empty_like(
                self.compressed_kv_lens_cuda[compress_ratio], device="cpu", pin_memory=True
            )
            for compress_ratio in self.compress_ratio_set
        }

        # past_kv_lens_cuda is the number of past compressed tokens for the requests
        self.past_kv_lens_cuda = {
            compress_ratio: self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_sequences,),
                dtype=torch.int,
                cache_name=f"past_kv_lens_cuda_{compress_ratio}",
                capture_graph=capture_graph,
            )
            for compress_ratio in self.compress_ratio_set
        }
        self.past_kv_lens = {
            compress_ratio: torch.empty_like(
                self.past_kv_lens_cuda[compress_ratio], device="cpu", pin_memory=True
            )
            for compress_ratio in self.compress_ratio_set
        }

        # compressed_position_ids_cuda is the compressed position ids for the requests
        self.compressed_position_ids_cuda = {
            compress_ratio: self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_tokens,),
                dtype=torch.int,
                cache_name=f"compressed_position_ids_cuda_{compress_ratio}",
                capture_graph=capture_graph,
            )
            for compress_ratio in self.compress_ratio_set
        }
        self.compressed_position_ids = {
            compress_ratio: torch.empty_like(
                self.compressed_position_ids_cuda[compress_ratio], device="cpu", pin_memory=True
            )
            for compress_ratio in self.compress_ratio_set
        }

        # empty topk indices buffer with all -1s in the tensor
        self.empty_topk_indices_buffer = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_tokens, self.sparse_mla_topk),
            cache_name="empty_topk_indices_buffer",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.empty_topk_indices_buffer.fill_(-1)

        # SWA local indices
        self.swa_local_indices_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_tokens, self.sparse_attention_config.window_size),
            cache_name="swa_local_indices_cuda",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )

        # Compute max_compressed_indices for CUDA graph compatibility
        self.max_compressed_indices = {
            1: 0,  # No compressed indices
            4: self.sparse_mla_topk,  # index_topk from indexer
            128: math.ceil(self.max_seq_len / 128),  # All compressed tokens
        }

        # Compressed local indices for compress_ratio=128
        # Note: ratio=4 uses dynamic topk_indices from indexer, so we only pre-allocate for ratio=128
        self.compressed_local_indices_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_tokens, self.max_compressed_indices[128]),
            cache_name="compressed_local_indices_cuda",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )

        # Block tables for cache buffers
        self.cache_buffer_block_offsets = {
            compress_ratio: self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_sequences, self.kv_cache_manager.max_blocks_per_seq),
                cache_name=f"cache_buffer_block_offsets_{compress_ratio}",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
            for compress_ratio in self.compress_ratio_set
        }
        self.host_cache_buffer_block_offsets = {
            compress_ratio: torch.empty_like(
                self.cache_buffer_block_offsets[compress_ratio], device="cpu", pin_memory=True
            )
            for compress_ratio in self.compress_ratio_set
        }

        # sparse_mla_topk_lens: actual token count per token for each compress_ratio (SWA + compressed)
        # Shape: [max_num_tokens] per compress_ratio
        self.sparse_mla_topk_lens = {
            compress_ratio: self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_tokens,),
                cache_name=f"sparse_mla_topk_lens_{compress_ratio}",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
            for compress_ratio in self.compress_ratio_set
        }

    def prepare_for_cache_block_offsets(self):
        """
        Prepare block offsets for cache buffers.
        """
        # Build cache buffer block offsets for all compress_ratios
        for compress_ratio in self.compress_ratio_set:
            for i in range(self.num_seqs):
                request_id = self.request_ids[i]
                if compress_ratio == 1:
                    layer_idx = 0
                    attn_type = MewtwoAttentionType.SWA
                else:
                    attn_type = MewtwoAttentionType.INDEXER_COMPRESS
                    if compress_ratio == 4:
                        layer_idx = 2
                    else:
                        layer_idx = 3
                cache_buffer_blocks = self.kv_cache_manager.get_cache_indices(
                    request_id, layer_idx=layer_idx, attn_type=attn_type
                )
                self.host_cache_buffer_block_offsets[compress_ratio][
                    i, : len(cache_buffer_blocks)
                ].copy_(torch.tensor(cache_buffer_blocks, dtype=torch.int32, device="cpu"))
            self.cache_buffer_block_offsets[compress_ratio][: self.num_seqs].copy_(
                self.host_cache_buffer_block_offsets[compress_ratio][: self.num_seqs],
                non_blocking=True,
            )

    def prepare_for_mewtwo_indices(self):
        """Prepare SWA local indices, compressed indices for ratio=128, and sparse_mla_topk_lens."""
        window_size = self.sparse_attention_config.window_size
        num_requests = self.num_seqs
        device = self.swa_local_indices_cuda.device

        # Build token positions tensor
        cached_token_lens = torch.tensor(
            self.kv_cache_params.num_cached_tokens_per_seq[:num_requests],
            dtype=torch.int32,
            device=device,
        )

        # Vectorized: create positions for all tokens
        token_positions = torch.zeros(self.num_tokens, dtype=torch.int32, device=device)
        token_idx = 0
        for req_idx in range(num_requests):
            seq_len = self.seq_lens[req_idx]
            base_pos = cached_token_lens[req_idx]
            token_positions[token_idx : token_idx + seq_len] = base_pos + torch.arange(
                seq_len, dtype=torch.int32, device=device
            )
            token_idx += seq_len

        # Build SWA local indices using helper function
        swa_indices = build_window_local_indices(
            token_positions[: self.num_tokens], window_size, device
        )
        self.swa_local_indices_cuda[: self.num_tokens].copy_(swa_indices)

        # Build compressed local indices for compress_ratio=128
        # Note: ratio=4 uses dynamic topk_indices from indexer, not pre-allocated
        compressed_indices = build_compressed_local_indices(
            token_positions[: self.num_tokens],
            compress_ratio=128,
            max_compressed_indices=self.max_compressed_indices[128],
            device=device,
        )
        self.compressed_local_indices_cuda[: self.num_tokens].copy_(compressed_indices)

        # Build sparse_mla_topk_lens: count of actual attached tokens per token
        self.prepare_sparse_mla_topk_lens(token_positions[: self.num_tokens])

    def prepare_sparse_mla_topk_lens(self, token_positions: torch.Tensor):
        """
        Prepare sparse_mla_topk_lens: actual number of attached tokens per token.

        For each compress_ratio:
        - compress_ratio=1: min(kv_len, window_size)
        - compress_ratio=4: min(kv_len, window_size) + min(kv_len // 4, sparse_mla_topk)
        - compress_ratio=128: min(kv_len, window_size) + kv_len // 128
        """
        window_size = self.sparse_attention_config.window_size

        # kv_len for each token = pos + 1
        kv_lens = token_positions + 1  # [num_tokens]

        swa_count = torch.minimum(kv_lens, torch.full_like(kv_lens, window_size))

        for compress_ratio in self.compress_ratio_set:
            if compress_ratio == 1:
                # SWA only
                total_count = swa_count
            elif compress_ratio == 4:
                # SWA + indexer topk
                compressed_count = torch.minimum(
                    kv_lens // compress_ratio, torch.full_like(kv_lens, self.sparse_mla_topk)
                )
                total_count = swa_count + compressed_count
            elif compress_ratio == 128:
                # SWA + all compressed tokens
                compressed_count = kv_lens // compress_ratio
                total_count = swa_count + compressed_count
            else:
                raise ValueError(f"Unsupported compress_ratio: {compress_ratio}")

            self.sparse_mla_topk_lens[compress_ratio][: self.num_tokens].copy_(
                total_count.to(torch.int32)
            )

    def prepare(self):
        super().super().prepare()

        cached_token_lens = torch.tensor(
            self.kv_cache_params.num_cached_tokens_per_seq,
            dtype=torch.int,
            device="cpu",
        )
        kv_lens = cached_token_lens + self.seq_lens_kv
        num_requests = self.num_contexts + self.num_generations
        num_gen_tokens = self.num_tokens - self.num_ctx_tokens

        # For indices conversion
        self.prepare_for_indices_conversion()

        # For indexer k cache
        self.prepare_for_indexer_k_cache()

        # For block offsets
        self.prepare_for_cache_block_offsets()

        # For mewtwo indices
        self.prepare_for_mewtwo_indices()

        # Prepare metadata for indexer
        MewtwoIndexer.prepare(metadata=self)

        # Prepare buffers for the compressor
        # prepare cu_seq_lens_cuda and cu_seq_lens
        self.cu_seq_lens[1 : num_requests + 1] = self.seq_lens.cumsum(0)
        self.cu_seq_lens_cuda[: num_requests + 1].copy_(
            self.cu_seq_lens[: num_requests + 1], non_blocking=True
        )

        # Prepare num_compressed_tokens, cu_new_comp_kv_cuda/cu_new_comp_kv and,
        # compressed_kv_lens_cuda/compressed_kv_lens
        num_gen_tokens_per_seq = num_gen_tokens // self.num_generations
        for compress_ratio in self.compress_ratio_set:
            num_comp_kv_lens = kv_lens[:num_requests] // compress_ratio
            past_comp_kv_lens = cached_token_lens // compress_ratio
            new_comp_kv_lens = num_comp_kv_lens - past_comp_kv_lens
            self.cu_new_comp_kv[compress_ratio][1 : num_requests + 1] = new_comp_kv_lens.cumsum(0)
            self.cu_new_comp_kv_cuda[compress_ratio][: num_requests + 1].copy_(
                self.cu_new_comp_kv[compress_ratio][: num_requests + 1], non_blocking=True
            )
            self.compressed_kv_lens[compress_ratio][:num_requests] = num_comp_kv_lens
            self.compressed_kv_lens_cuda[compress_ratio][:num_requests].copy_(
                self.compressed_kv_lens[compress_ratio][:num_requests], non_blocking=True
            )
            num_ctx_compressed_tokens = num_comp_kv_lens[: self.num_contexts].sum().item()
            # To support CUDA graph, generation requests should use a constant number of compressed tokens.
            num_gen_compressed_tokens = self.num_generations * math.ceil(
                num_gen_tokens_per_seq / compress_ratio
            )
            self.num_compressed_tokens[compress_ratio] = (
                num_ctx_compressed_tokens + num_gen_compressed_tokens
            )

        # Prepare past_kv_lens_cuda/past_kv_lens
        for compress_ratio in self.compress_ratio_set:
            self.past_kv_lens[compress_ratio][:num_requests] = (
                cached_token_lens[:num_requests] // compress_ratio
            )
            self.past_kv_lens_cuda[compress_ratio][:num_requests].copy_(
                self.past_kv_lens[compress_ratio][:num_requests], non_blocking=True
            )

        # Prepare compressed_position_ids_cuda/compressed_position_ids
        for compress_ratio in self.compress_ratio_set:
            position_ids = []
            for i in range(self.num_contexts):
                past_kv_lens = self.past_kv_lens[compress_ratio][i].item()
                kv_lens = self.compressed_kv_lens[compress_ratio][i].item()
                position_ids.extend(list(range(past_kv_lens, kv_lens)))
            for i in range(self.num_generations):
                # Use a constant number of new compressed KV tokens for each generation request
                # to support CUDA graph.
                past_kv_lens = self.past_kv_lens[compress_ratio][self.num_contexts + i].item()
                new_kv_lens = math.ceil(num_gen_tokens_per_seq / compress_ratio)
                position_ids.extend(list(range(past_kv_lens, past_kv_lens + new_kv_lens)))
            compressed_num_tokens = len(position_ids)
            self.compressed_position_ids[compress_ratio][:compressed_num_tokens] = (
                torch.tensor(position_ids, dtype=torch.int, device="cuda") * compress_ratio
            )
            self.compressed_position_ids_cuda[compress_ratio][:compressed_num_tokens].copy_(
                self.compressed_position_ids[compress_ratio][:compressed_num_tokens],
                non_blocking=True,
            )


class MewtwoAttentionType(Enum):
    SWA = 0
    COMPRESS = 1
    COMPRESSOR_STATE = 2
    COMPRESSOR_SCORE = 3
    INDEXER_COMPRESS = 4
    INDEXER_COMPRESSOR_STATE = 5
    INDEXER_COMPRESSOR_SCORE = 6


class MewtwoIndexer(Indexer):
    def __init__(
        self,
        quant_config: Optional[QuantConfig],
        pos_embd_params: Optional[PositionalEmbeddingParams],
        mla_params: Optional[MLAParams],
        skip_create_weights_in_init: bool,
        sparse_attention_config: "SparseAttentionConfig",
        dtype: Optional[torch.dtype],
        compress_ratio: int = 1,
        layer_idx: int = 0,
        aux_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__(
            quant_config,
            pos_embd_params,
            mla_params,
            skip_create_weights_in_init,
            sparse_attention_config,
            dtype,
            compress_ratio,
            layer_idx,
            aux_stream,
        )
        rms_norm_eps = 1e-6
        self.compressor = Compressor(
            mla_params,
            layer_idx,
            compress_ratio,
            rms_norm_eps,
            skip_create_weights_in_init,
            pos_embd_params,
            dtype=dtype,
            kv_cache_dtype="fp8_blockwise",
        )

    def _qk_projection_and_rope(self, qr: torch.Tensor, position_ids: torch.Tensor):
        """Project Q and apply RoPE"""
        q = self.wq_b(qr)
        q = q.view(-1, self.n_heads, self.head_dim)
        q_nope, q_pe = q.split([self.head_dim - self.rope_dim, self.rope_dim], dim=-1)
        q_pe = self.rotary_emb(position_ids, [q_pe])[0]
        return q_pe, q_nope

    def forward(
        self,
        qr: torch.Tensor,
        hidden_states: torch.Tensor,
        metadata: MewtwoTrtllmAttentionMetadata,
        position_ids: torch.Tensor,
        indexer_k: torch.Tensor,
    ):
        # compress k
        k_fp8, k_scale = self.compressor(indexer_k, metadata)

        # multi-stream q proj/rope and weights proj
        q, weights = maybe_execute_in_parallel(
            lambda: self._qk_projection_and_rope(qr, position_ids),
            lambda: self.weights_proj(_to_float(hidden_states)),
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )

        # quantize q
        q_pe, q_nope = q
        q_fp8, q_scale = self._prep_q_or_k(q_pe, q_nope)
        q_fp8 = q_fp8.view(-1, self.n_heads, self.head_dim)
        q_scale = q_scale.view(-1, self.n_heads, 1)

        # weights scale
        weights = self._weight_scale(weights, q_scale)

        # If there are no compressed tokens, return an topk indices buffer with all -1s in the tensor.
        if k_fp8 is None:
            topk_indices = metadata.empty_topk_indices_buffer[: hidden_states.shape[0]]
        else:
            topk_indices = self.sparse_attn_indexer(
                metadata, hidden_states, q_fp8, k_fp8, k_scale, weights
            )
        return topk_indices


class MewtwoTrtllmAttention(TrtllmAttention):
    Metadata = MewtwoTrtllmAttentionMetadata

    def __init__(
        self,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: Optional[int] = None,
        quant_config: Optional[QuantConfig] = None,
        q_scaling: Optional[float] = None,
        pos_embd_params: Optional[PositionalEmbeddingParams] = None,
        mla_params: Optional[MLAParams] = None,
        skip_create_weights_in_init: bool = False,
        attention_chunk_size: Optional[int] = None,
        sparse_attention_config: Optional["SparseAttentionConfig"] = None,
        dtype: Optional[torch.dtype] = None,
        aux_stream: Optional[torch.cuda.Stream] = None,
        **kwargs,
    ):
        assert sparse_attention_config is not None, (
            "sparse_attention_config is required for MewtwoTrtllmAttention and cannot be None"
        )
        TrtllmAttention.__init__(
            self,
            layer_idx,
            num_heads,
            head_dim,
            sparse_attention_config=sparse_attention_config,
            num_kv_heads=num_kv_heads,
            quant_config=quant_config,
            q_scaling=q_scaling,
            pos_embd_params=pos_embd_params,
            mla_params=mla_params,
            skip_create_weights_in_init=skip_create_weights_in_init,
            attention_chunk_size=attention_chunk_size,
            **kwargs,
        )

        self.compress_ratio = sparse_attention_config.compress_ratios[layer_idx]

        if self.compress_ratio == 4:
            self.indexer = MewtwoIndexer(
                quant_config,
                pos_embd_params,
                mla_params,
                skip_create_weights_in_init,
                sparse_attention_config,
                dtype,
                layer_idx,
                aux_stream,
            )

        if self.compress_ratio > 1:
            rms_norm_eps = 1e-6
            self.compressor = Compressor(
                mla_params,
                layer_idx,
                self.compress_ratio,
                rms_norm_eps,
                skip_create_weights_in_init,
                pos_embd_params,
                dtype=dtype,
            )

    def sparse_attn_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: MewtwoTrtllmAttentionMetadata,
        hidden_states: Optional[torch.Tensor] = None,
        qr: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        topk_indices: Optional[
            torch.Tensor
        ] = None,  # compressed indices from indexer (for compress_ratio=4)
        is_generation: bool = True,
        **kwargs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Convert local indices (SWA + compressed) to global pool indices."""
        layer_idx = self.layer_idx
        kv_cache_manager = metadata.kv_cache_manager

        # Get buffer pointers directly from kv_cache_manager
        swa_pool_ptr = kv_cache_manager.layer_attn_to_pool_ptr[
            MewtwoAttentionType.SWA.value, layer_idx
        ]
        swa_buffer_ptr = kv_cache_manager.layer_attn_to_buffer_ptr[
            MewtwoAttentionType.SWA.value, layer_idx
        ]

        # Token stride
        token_stride = kv_cache_manager.get_token_bytes(layer_idx, MewtwoAttentionType.SWA)

        # Select indices/tables based on phase
        if is_generation:
            start_idx = metadata.num_ctx_tokens
            end_idx = metadata.num_tokens
            req_start = metadata.num_contexts
            req_end = metadata.num_seqs
            req_offset = metadata.num_contexts
        else:
            start_idx = 0
            end_idx = metadata.num_ctx_tokens
            req_start = 0
            req_end = metadata.num_contexts
            req_offset = 0

        req_id = (metadata.req_idx_per_token[start_idx:end_idx] - req_offset).to(torch.int32)
        swa_local_indices = metadata.swa_local_indices_cuda[start_idx:end_idx]
        block_table_swa = metadata.cache_buffer_block_offsets[1][req_start:req_end]

        # Handle compressed based on compress_ratio
        if self.compress_ratio == 1:
            # SWA only
            global_indices = mewtwo_local_to_global_indices(
                req_id=req_id,
                block_table_swa=block_table_swa,
                swa_local_indices=swa_local_indices,
                swa_pool_ptr=swa_pool_ptr,
                swa_buffer_ptr=swa_buffer_ptr,
                tokens_per_block=kv_cache_manager.tokens_per_block,
                token_stride=token_stride,
                compress_ratio=1,
            )
        else:
            # SWA + compressed indices
            compressed_buffer_ptr = kv_cache_manager.layer_attn_to_buffer_ptr[
                MewtwoAttentionType.COMPRESS.value, layer_idx
            ]
            block_table_compressed = metadata.cache_buffer_block_offsets[self.compress_ratio][
                req_start:req_end
            ]
            if self.compress_ratio == 4:
                assert topk_indices is not None, "topk_indices is required when compress_ratio=4"
                compressed_local_indices = topk_indices
            else:
                compressed_local_indices = metadata.compressed_local_indices_cuda[start_idx:end_idx]
            global_indices = mewtwo_local_to_global_indices(
                req_id=req_id,
                block_table_swa=block_table_swa,
                swa_local_indices=swa_local_indices,
                swa_pool_ptr=swa_pool_ptr,
                swa_buffer_ptr=swa_buffer_ptr,
                tokens_per_block=kv_cache_manager.tokens_per_block,
                token_stride=token_stride,
                block_table_compressed=block_table_compressed,
                compressed_local_indices=compressed_local_indices,
                compressed_buffer_ptr=compressed_buffer_ptr,
                compress_ratio=self.compress_ratio,
                num_compressed_indices=metadata.max_compressed_indices[self.compress_ratio],
            )

        return global_indices, None
