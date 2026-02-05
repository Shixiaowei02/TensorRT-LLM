import math
from enum import Enum
from typing import TYPE_CHECKING, Dict, Optional, Set, Tuple

import torch

from tensorrt_llm._torch.attention_backend.interface import MLAParams, PositionalEmbeddingParams
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention
from tensorrt_llm._torch.modules.multi_stream_utils import maybe_execute_in_parallel
from tensorrt_llm.models.modeling_utils import QuantConfig

from ..dsa import DSAtrtllmAttentionMetadata, Indexer, _to_float
from .compressor import Compressor

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import SparseAttentionConfig


class MewtwoTrtllmAttentionMetadata(DSAtrtllmAttentionMetadata):
    # The set of compress ratios for the layers
    compress_ratio_set: Set[int]
    # The number of total compressed tokens for each compress ratio
    num_compressed_tokens: Dict[int, int]

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
                kv_lens = self.compressed_kv_lens[i].item()
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

        compress_ratio = sparse_attention_config.compress_ratios[layer_idx]

        if compress_ratio == 4:
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

        if compress_ratio > 1:
            rms_norm_eps = 1e-6
            self.compressor = Compressor(
                mla_params,
                layer_idx,
                compress_ratio,
                rms_norm_eps,
                skip_create_weights_in_init,
                pos_embd_params,
                dtype=dtype,
            )

    def sparse_attn_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: DSAtrtllmAttentionMetadata,
        hidden_states: Optional[torch.Tensor] = None,
        qr: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
        is_generation: bool = True,
        **kwargs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Transform the local topk indices to global topk indices in paged kv cache
        topk_indices_global, _ = transform_local_topk_and_prepare_pool_view(
            topk_indices,
            metadata,
            self.get_local_layer_idx(metadata),
            is_generation,
            compress_ratio=self.compress_ratio,
        )
        return topk_indices_global, None
