import math
from enum import Enum
from typing import TYPE_CHECKING, Dict, Optional, Set, Tuple

import torch

from tensorrt_llm._torch.attention_backend.interface import MLAParams, PositionalEmbeddingParams
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention, TrtllmAttentionMetadata
from tensorrt_llm._torch.modules.multi_stream_utils import maybe_execute_in_parallel
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.models.modeling_utils import QuantConfig

from ..dsa import DSAtrtllmAttentionMetadata, Indexer, _to_float
from ..kernel import mewtwo_local_to_global_indices
from .compressor import Compressor

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import SparseAttentionConfig

MEWTWO_SPARSE_RATIO = 4
MEWTWO_OVERLAP_COMPRESSOR_RATIO = 4


class MewtwoAttentionType(Enum):
    SWA = 0
    COMPRESS = 1
    COMPRESSOR_STATE = 2
    COMPRESSOR_SCORE = 3
    INDEXER_COMPRESS = 4
    INDEXER_COMPRESSOR_STATE = 5
    INDEXER_COMPRESSOR_SCORE = 6


def is_overlap_compressor(compress_ratio: int) -> bool:
    """
    Check if the compressor of the given layer is working in the overlap mode.
    """
    return compress_ratio == MEWTWO_OVERLAP_COMPRESSOR_RATIO


def is_sparse_layer(compress_ratio: int) -> bool:
    """
    Check if the given layer is a sparse layer.
    """
    return compress_ratio == MEWTWO_SPARSE_RATIO


def is_compress_layer(compress_ratio: int) -> bool:
    """
    Check if the given layer is a compress layer.
    """
    return compress_ratio > 1


def compress_ratio_has_attention(compress_ratio: int, attn_type: MewtwoAttentionType) -> bool:
    """
    Check if the given compress ratio has the given attention type.
    """
    is_sparse = is_sparse_layer(compress_ratio)
    is_compress = is_compress_layer(compress_ratio)

    if attn_type == MewtwoAttentionType.SWA:
        return True
    elif attn_type == MewtwoAttentionType.COMPRESS:
        return is_compress
    elif attn_type == MewtwoAttentionType.COMPRESSOR_STATE:
        return is_compress
    elif attn_type == MewtwoAttentionType.COMPRESSOR_SCORE:
        return is_compress
    elif attn_type == MewtwoAttentionType.INDEXER_COMPRESS:
        return is_sparse
    elif attn_type == MewtwoAttentionType.INDEXER_COMPRESSOR_STATE:
        return is_sparse
    elif attn_type == MewtwoAttentionType.INDEXER_COMPRESSOR_SCORE:
        return is_sparse


def get_attn_dim(
    head_dim: int, index_head_dim: int, compress_ratio: int, attn_type: MewtwoAttentionType
) -> int:
    """
    Get the dimension of the attention type for a specific layer.
    """
    state_factor = 2 if is_overlap_compressor(compress_ratio) else 1
    if attn_type == MewtwoAttentionType.SWA:
        return head_dim
    elif attn_type == MewtwoAttentionType.COMPRESS:
        return head_dim
    elif attn_type == MewtwoAttentionType.COMPRESSOR_STATE:
        return state_factor * head_dim
    elif attn_type == MewtwoAttentionType.COMPRESSOR_SCORE:
        return state_factor * head_dim
    elif attn_type == MewtwoAttentionType.INDEXER_COMPRESS:
        return index_head_dim
    elif attn_type == MewtwoAttentionType.INDEXER_COMPRESSOR_STATE:
        return state_factor * index_head_dim
    elif attn_type == MewtwoAttentionType.INDEXER_COMPRESSOR_SCORE:
        return state_factor * index_head_dim


def get_token_bytes(
    head_dim: int,
    index_head_dim: int,
    compress_ratio: int,
    attn_type: MewtwoAttentionType,
    has_fp8_kv_cache: bool,
) -> int:
    """
    Get the token bytes for a specific layer and attention type.

    Args:
        head_dim: The head dimension
        index_head_dim: The index head dimension
        compress_ratio: The compress ratio
        attn_type: The attention type
        has_fp8_kv_cache: Whether the KV cache uses FP8 quantization

    Returns:
        The number of bytes per token, including scaling factor
    """
    if not compress_ratio_has_attention(compress_ratio, attn_type):
        raise ValueError(
            f"Layer with compress ratio {compress_ratio} does not have attention type {attn_type}"
        )

    attn_dim = get_attn_dim(head_dim, index_head_dim, compress_ratio, attn_type)

    # Default dtype is bfloat16 (2 bytes), or fp8 (1 byte) when FP8 kv cache is enabled
    dtype_bytes = 1 if has_fp8_kv_cache else 2
    # (indexer) compressor state and score always use float32
    if attn_type in [
        MewtwoAttentionType.COMPRESSOR_STATE,
        MewtwoAttentionType.COMPRESSOR_SCORE,
        MewtwoAttentionType.INDEXER_COMPRESSOR_STATE,
        MewtwoAttentionType.INDEXER_COMPRESSOR_SCORE,
    ]:
        dtype_bytes = 4  # (indexer) compressor state and score use float32
    # indexer compress always uses fp8
    elif attn_type == MewtwoAttentionType.INDEXER_COMPRESS:
        dtype_bytes = 1  # indexer compress use fp8

    scale_size = 0
    # indexer compress has scaling factor
    if attn_type == MewtwoAttentionType.INDEXER_COMPRESS:
        quant_block_size = 128
        scale_size = index_head_dim // quant_block_size * 4  # indexer scale is float32

    return attn_dim * dtype_bytes + scale_size


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
    # The set of (compress ratio, attention type) for the layers
    attention_type_set: Set[Tuple[int, MewtwoAttentionType]]
    # The number of total compressed tokens for each compress ratio
    num_total_compressed_tokens: Dict[int, int] = {}
    # The max number of compressed tokens for each compress ratio
    max_num_compressed_tokens: Dict[int, Tuple[int, int, int]] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __post_init__(self):
        super().__post_init__()
        capture_graph = self.is_cuda_graph
        self.compress_ratio_set = set(self.compress_ratios)

        attention_types = []
        for compress_ratio in self.compress_ratio_set:
            if compress_ratio == 1:
                attention_types.append((self.compress_ratios[0], MewtwoAttentionType.SWA))
            elif compress_ratio == 4:
                attention_types.append((self.compress_ratios[0], MewtwoAttentionType.SWA))
                attention_types.append((compress_ratio, MewtwoAttentionType.COMPRESS))
                attention_types.append((compress_ratio, MewtwoAttentionType.COMPRESSOR_STATE))
                attention_types.append((compress_ratio, MewtwoAttentionType.COMPRESSOR_SCORE))
                attention_types.append((compress_ratio, MewtwoAttentionType.INDEXER_COMPRESS))
                attention_types.append(
                    (compress_ratio, MewtwoAttentionType.INDEXER_COMPRESSOR_STATE)
                )
                attention_types.append(
                    (compress_ratio, MewtwoAttentionType.INDEXER_COMPRESSOR_SCORE)
                )
            else:
                attention_types.append((self.compress_ratios[0], MewtwoAttentionType.SWA))
                attention_types.append((compress_ratio, MewtwoAttentionType.COMPRESS))
                attention_types.append((compress_ratio, MewtwoAttentionType.COMPRESSOR_STATE))
                attention_types.append((compress_ratio, MewtwoAttentionType.COMPRESSOR_SCORE))
        self.attention_type_set = set(attention_types)

        # Create buffers for the compressor
        # cu_seq_lens_cuda is the cumulative sequence lengths for the requests
        self.cu_seq_lens_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences + 1,),
            dtype=torch.int,
            cache_name="cu_seq_lens_cuda",
            capture_graph=capture_graph,
        )
        self.cu_seq_lens = torch.empty_like(
            self.cu_seq_lens_cuda, device="cpu", pin_memory=prefer_pinned()
        )
        self.cu_seq_lens[0] = 0

        # new_comp_kv_lens_cuda is the number of new compressed tokens for the requests
        self.new_comp_kv_lens_cuda = {
            compress_ratio: self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_sequences,),
                dtype=torch.int,
                cache_name=f"new_comp_kv_lens_cuda_{compress_ratio}",
                capture_graph=capture_graph,
            )
            for compress_ratio in self.compress_ratio_set
        }
        self.new_comp_kv_lens = {
            compress_ratio: torch.empty_like(
                self.new_comp_kv_lens_cuda[compress_ratio], device="cpu", pin_memory=prefer_pinned()
            )
            for compress_ratio in self.compress_ratio_set
        }

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
                self.cu_new_comp_kv_cuda[compress_ratio], device="cpu", pin_memory=prefer_pinned()
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
                self.compressed_kv_lens_cuda[compress_ratio],
                device="cpu",
                pin_memory=prefer_pinned(),
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
                self.past_kv_lens_cuda[compress_ratio], device="cpu", pin_memory=prefer_pinned()
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
                self.compressed_position_ids_cuda[compress_ratio],
                device="cpu",
                pin_memory=prefer_pinned(),
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

        # Compute max_compressed_indices for CUDA graph compatibility.
        # For ratio=4, the indexer selects index_topk compressed tokens.
        # For ratio=128, use max_seq_len / 128 rounded up to next power of 2
        raw_128 = math.ceil(self.max_seq_len / 128)
        po2_128 = 1 << (raw_128 - 1).bit_length() if raw_128 > 0 else 1
        self.max_compressed_indices = {
            1: 0,  # No compressed indices
            4: self.sparse_mla_topk,  # index_topk from indexer
            128: po2_128,  # All compressed tokens, rounded to power of 2
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
                self.cache_buffer_block_offsets[compress_ratio],
                device="cpu",
                pin_memory=prefer_pinned(),
            )
            for compress_ratio in self.compress_ratio_set
        }

        self.block_tables = {
            attention_type: self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_sequences, self.kv_cache_manager.max_blocks_per_seq),
                cache_name=f"block_tables_{attention_type}",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
            for attention_type in self.attention_type_set
        }
        self.host_block_tables = {
            attention_type: torch.empty_like(
                self.block_tables[attention_type], device="cpu", pin_memory=prefer_pinned()
            )
            for attention_type in self.attention_type_set
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

        # cached_token_lens_cuda: number of tokens already cached per request
        self.cached_token_lens_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences,),
            cache_name="cached_token_lens_cuda",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )

    def prepare_for_block_tables(self):
        """
        Prepare block tables for cache buffers.
        """
        for compress_ratio, attention_type in self.attention_type_set:
            host_block_table = self.kv_cache_manager.get_batch_attn_offset(
                request_ids=self.request_ids,
                beam_width=1,
                num_contexts=self.num_contexts,
                num_seqs=self.num_seqs,
                attn_type=attention_type,
                compress_ratio=compress_ratio,
            )
            key = (compress_ratio, attention_type)
            self.host_block_tables[key][: self.num_seqs] = host_block_table[: self.num_seqs]
            self.block_tables[key][: self.num_seqs].copy_(
                self.host_block_tables[key][: self.num_seqs], non_blocking=True
            )

        # Build cache buffer block offsets for all compress_ratios
        for compress_ratio in self.compress_ratio_set:
            if compress_ratio == 1:
                attn_type = MewtwoAttentionType.SWA
            else:
                attn_type = MewtwoAttentionType.COMPRESS

            self.host_cache_buffer_block_offsets[compress_ratio][: self.num_seqs].copy_(
                self.kv_cache_manager.get_batch_attn_offset(
                    self.request_ids,
                    beam_width=self.beam_width,
                    num_contexts=self.num_contexts,
                    num_seqs=self.num_seqs,
                    attn_type=attn_type,
                    compress_ratio=compress_ratio,
                )
            )
            self.cache_buffer_block_offsets[compress_ratio][: self.num_seqs].copy_(
                self.host_cache_buffer_block_offsets[compress_ratio][: self.num_seqs],
                non_blocking=True,
            )

    def prepare_for_mewtwo_indices(self, token_positions=None):
        """Prepare SWA/compressed local indices and sparse_mla_topk_lens."""
        window_size = self.sparse_attention_config.window_size
        device = self.swa_local_indices_cuda.device

        if token_positions is None:
            num_requests = self.num_seqs
            cached_token_lens = torch.tensor(
                self.kv_cache_params.num_cached_tokens_per_seq[:num_requests],
                dtype=torch.int32,
                device=device,
            )
            token_positions = torch.zeros(self.num_tokens, dtype=torch.int32, device=device)
            token_idx = 0
            for req_idx in range(num_requests):
                seq_len = self.seq_lens[req_idx]
                base_pos = cached_token_lens[req_idx]
                token_positions[token_idx : token_idx + seq_len] = base_pos + torch.arange(
                    seq_len, dtype=torch.int32, device=device
                )
                token_idx += seq_len
            token_positions = token_positions[: self.num_tokens]

        num_tokens = token_positions.shape[0]

        swa_indices = build_window_local_indices(token_positions, window_size, device)
        self.swa_local_indices_cuda[:num_tokens].copy_(swa_indices)

        # Only build ratio=128 here; ratio=4 uses dynamic topk_indices from indexer
        compressed_indices = build_compressed_local_indices(
            token_positions,
            compress_ratio=128,
            max_compressed_indices=self.max_compressed_indices[128],
            device=device,
        )
        self.compressed_local_indices_cuda[:num_tokens].copy_(compressed_indices)

        self.prepare_sparse_mla_topk_lens(token_positions)

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
        TrtllmAttentionMetadata.prepare(self)

        cached_token_lens = torch.tensor(
            self.kv_cache_params.num_cached_tokens_per_seq,
            dtype=torch.int,
            device="cpu",
        )
        kv_lens = cached_token_lens + self.seq_lens_kv
        num_requests = self.num_contexts + self.num_generations
        num_gen_tokens = self.num_tokens - self.num_ctx_tokens

        self.cached_token_lens_cuda[:num_requests].copy_(
            cached_token_lens[:num_requests].to(torch.int32), non_blocking=True
        )

        # Cache buffer data pointers
        # If MTP is enabled, enlarge the compress ratios by max_draft_tokens - 1
        extend_compress_ratios = self.compress_ratios + [self.compress_ratios[-1]] * (
            self.max_draft_tokens - 1
        )
        self.swa_buffer_ptrs = {
            layer_idx: self.kv_cache_manager.get_buffers(
                layer_idx, MewtwoAttentionType.SWA
            ).data_ptr()
            for layer_idx in self.kv_cache_manager.pp_layers
        }
        self.compressed_buffer_ptrs = {
            layer_idx: self.kv_cache_manager.get_buffers(
                layer_idx, MewtwoAttentionType.COMPRESS
            ).data_ptr()
            for layer_idx in self.kv_cache_manager.pp_layers
            if is_compress_layer(extend_compress_ratios[layer_idx])
        }

        # Per-ratio base pointer for sparse MLA = min(swa_pool_ptr, compressed_pool_ptr).
        swa_pool_ptr = self.kv_cache_manager.swa_pool_ptr
        self.sparse_mla_base_ptrs = {
            1: swa_pool_ptr,
        }
        for ratio, compress_pool_ptr in self.kv_cache_manager.compress_pool_ptrs.items():
            self.sparse_mla_base_ptrs[ratio] = min(swa_pool_ptr, compress_pool_ptr)

        # For indices conversion
        self.prepare_for_indices_conversion()

        has_sparse_layers = MEWTWO_SPARSE_RATIO in self.compress_ratio_set

        # For indexer k cache (only needed when sparse layers exist)
        if has_sparse_layers:
            self.prepare_for_indexer_k_cache()

        # For block offsets
        self.prepare_for_block_tables()

        # For mewtwo indices
        self.prepare_for_mewtwo_indices()

        # Prepare metadata for indexer (only needed when sparse layers exist)
        if has_sparse_layers:
            MewtwoIndexer.prepare(metadata=self)

        # Prepare buffers for the compressor
        # prepare cu_seq_lens_cuda and cu_seq_lens
        self.cu_seq_lens[1 : num_requests + 1] = self.seq_lens.cumsum(0)
        self.cu_seq_lens_cuda[: num_requests + 1].copy_(
            self.cu_seq_lens[: num_requests + 1], non_blocking=True
        )

        # Prepare num_total_compressed_tokens, cu_new_comp_kv_cuda/cu_new_comp_kv and,
        # compressed_kv_lens_cuda/compressed_kv_lens
        num_gen_tokens_per_seq = (
            num_gen_tokens // self.num_generations if self.num_generations > 0 else 0
        )
        self.num_gen_tokens_per_seq = num_gen_tokens_per_seq
        for compress_ratio in self.compress_ratio_set:
            num_comp_kv_lens = kv_lens[:num_requests] // compress_ratio
            past_comp_kv_lens = cached_token_lens // compress_ratio
            new_comp_kv_lens = num_comp_kv_lens - past_comp_kv_lens
            self.new_comp_kv_lens[compress_ratio][:num_requests] = new_comp_kv_lens
            self.new_comp_kv_lens_cuda[compress_ratio][:num_requests].copy_(
                self.new_comp_kv_lens[compress_ratio][:num_requests], non_blocking=True
            )
            self.cu_new_comp_kv[compress_ratio][1 : num_requests + 1] = new_comp_kv_lens.cumsum(0)
            self.cu_new_comp_kv_cuda[compress_ratio][: num_requests + 1].copy_(
                self.cu_new_comp_kv[compress_ratio][: num_requests + 1], non_blocking=True
            )
            self.compressed_kv_lens[compress_ratio][:num_requests] = num_comp_kv_lens
            self.compressed_kv_lens_cuda[compress_ratio][:num_requests].copy_(
                self.compressed_kv_lens[compress_ratio][:num_requests], non_blocking=True
            )
            num_ctx_compressed_tokens = new_comp_kv_lens[: self.num_contexts].sum().item()
            # To support CUDA graph, generation requests should use a constant number of compressed tokens.
            num_gen_compressed_tokens = self.num_generations * math.ceil(
                num_gen_tokens_per_seq / compress_ratio
            )
            self.num_total_compressed_tokens[compress_ratio] = (
                num_ctx_compressed_tokens + num_gen_compressed_tokens
            )
            max_ctx_comp_kv_lens, max_gen_comp_kv_lens = 0, 0
            if self.num_contexts > 0:
                max_ctx_comp_kv_lens = new_comp_kv_lens[: self.num_contexts].max().item()
            if self.num_generations > 0:
                max_gen_comp_kv_lens = (
                    new_comp_kv_lens[self.num_contexts : self.num_seqs].max().item()
                )
            max_comp_kv_lens = max(max_ctx_comp_kv_lens, max_gen_comp_kv_lens)
            self.max_num_compressed_tokens[compress_ratio] = (
                max_ctx_comp_kv_lens,
                max_gen_comp_kv_lens,
                max_comp_kv_lens,
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

    def on_update_kv_lens(self):
        """Recompute kv-lens-dependent mewtwo metadata on device."""
        super().on_update_kv_lens()

        batch_size = self.num_seqs
        num_contexts = self.num_contexts
        num_generations = self.num_generations
        num_tokens = self.num_tokens
        device = self.kv_lens_cuda.device
        kv_lens = self.kv_lens_cuda[:batch_size]
        seq_lens = self._seq_lens_cuda[:batch_size]
        cached_tokens = kv_lens - seq_lens

        # Recompute cu_seq_lens and req_idx_per_token
        self.cu_seq_lens_cuda[:1].zero_()
        torch.cumsum(seq_lens.to(torch.int), dim=0, out=self.cu_seq_lens_cuda[1 : batch_size + 1])
        token_idx = torch.arange(num_tokens, dtype=torch.int32, device=device)
        self.req_idx_per_token[:num_tokens].copy_(
            torch.searchsorted(
                self.cu_seq_lens_cuda[1 : batch_size + 1].to(torch.int32), token_idx, right=True
            )
        )

        # Per-token positions: cached_tokens[req] + intra-sequence offset
        req_idx = self.req_idx_per_token[:num_tokens]
        base_pos = cached_tokens[req_idx].to(torch.int32)
        offsets = token_idx - self.cu_seq_lens_cuda[req_idx].to(torch.int32)
        token_positions = base_pos + offsets

        num_gen_tokens = num_tokens - self.num_ctx_tokens
        self.num_gen_tokens_per_seq = (
            num_gen_tokens // num_generations if num_generations > 0 else 0
        )
        num_gen_tokens_per_seq = self.num_gen_tokens_per_seq

        # Per-ratio: update compressed/past/new KV lens and position IDs.
        # num_total_compressed_tokens and max_num_compressed_tokens are NOT
        # updated here; prepare() sets them as stable upper bounds.
        for compress_ratio in self.compress_ratio_set:
            compressed_kv = (kv_lens // compress_ratio).to(torch.int)
            self.compressed_kv_lens_cuda[compress_ratio][:batch_size].copy_(compressed_kv)

            past_kv = (cached_tokens // compress_ratio).to(torch.int)
            self.past_kv_lens_cuda[compress_ratio][:batch_size].copy_(past_kv)

            new_comp = compressed_kv - past_kv
            self.new_comp_kv_lens_cuda[compress_ratio][:batch_size].copy_(new_comp)

            self.cu_new_comp_kv_cuda[compress_ratio][:1].zero_()
            torch.cumsum(
                new_comp, dim=0, out=self.cu_new_comp_kv_cuda[compress_ratio][1 : batch_size + 1]
            )

            # Compressed position IDs (layout computed locally)
            new_gen_comp = (
                math.ceil(num_gen_tokens_per_seq / compress_ratio)
                if num_gen_tokens_per_seq > 0
                else 0
            )
            gen_comp = num_generations * new_gen_comp
            ctx_comp = new_comp[:num_contexts].sum().item() if num_contexts > 0 else 0

            if ctx_comp > 0:
                ctx_idx = torch.arange(ctx_comp, dtype=torch.int32, device=device)
                ctx_cu = self.cu_new_comp_kv_cuda[compress_ratio][: num_contexts + 1].to(
                    torch.int32
                )
                ctx_req = torch.searchsorted(ctx_cu[1:], ctx_idx, right=True)
                ctx_offset = ctx_idx - ctx_cu[ctx_req]
                self.compressed_position_ids_cuda[compress_ratio][:ctx_comp].copy_(
                    ((past_kv[:num_contexts][ctx_req] + ctx_offset) * compress_ratio).to(torch.int)
                )
            if gen_comp > 0 and num_generations > 0:
                gen_past = past_kv[num_contexts:batch_size]
                gen_offsets = torch.arange(new_gen_comp, dtype=torch.int32, device=device)
                gen_pos = gen_past.unsqueeze(1) + gen_offsets.unsqueeze(0)
                self.compressed_position_ids_cuda[compress_ratio][
                    ctx_comp : ctx_comp + gen_comp
                ].copy_((gen_pos.reshape(-1) * compress_ratio).to(torch.int))

        self.prepare_for_mewtwo_indices(token_positions)


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
        index_head_dim = sparse_attention_config.index_head_dim
        indexer_mla_params = MLAParams(
            hidden_size=mla_params.hidden_size,
            qk_rope_head_dim=mla_params.qk_rope_head_dim,
            qk_nope_head_dim=index_head_dim - mla_params.qk_rope_head_dim,
        )
        self.compressor = Compressor(
            indexer_mla_params,
            layer_idx,
            compress_ratio,
            rms_norm_eps,
            skip_create_weights_in_init,
            pos_embd_params,
            dtype=dtype,
            kv_cache_dtype="fp8_blockwise",
            is_indexer=True,
        )

    def _qk_projection_and_rope(self, qr: torch.Tensor, position_ids: torch.Tensor):
        """Project Q and apply RoPE"""
        q = self.wq_b(qr)
        q = q.view(-1, self.n_heads, self.head_dim)
        num_tokens = q.shape[0]
        q_nope, q_pe = q.split([self.head_dim - self.rope_dim, self.rope_dim], dim=-1)
        q_pe = self.rotary_emb(position_ids, [q_pe.reshape(num_tokens, -1)])[0]
        q_pe = q_pe.view(num_tokens, self.n_heads, self.rope_dim)
        return q_pe, q_nope

    def forward(
        self,
        qr: torch.Tensor,
        hidden_states: torch.Tensor,
        metadata: MewtwoTrtllmAttentionMetadata,
        position_ids: torch.Tensor,
    ):
        # compress k
        k_fp8, k_scale = self.compressor(hidden_states, metadata)

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
                self.compress_ratio,
                layer_idx,
                aux_stream,
            )

        if self.compress_ratio > 1:
            rms_norm_eps = 1e-6
            has_fp8_kv_cache = False
            if quant_config is not None:
                has_fp8_kv_cache = quant_config.layer_quant_mode.has_fp8_kv_cache()
            kv_cache_dtype = "fp8_pertensor" if has_fp8_kv_cache else "default"
            self.compressor = Compressor(
                mla_params,
                layer_idx,
                self.compress_ratio,
                rms_norm_eps,
                skip_create_weights_in_init,
                pos_embd_params,
                kv_cache_dtype=kv_cache_dtype,
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

        sparse_mla_base_ptr = metadata.sparse_mla_base_ptrs[self.compress_ratio]

        # Get cached buffer pointers
        swa_buffer_ptr = metadata.swa_buffer_ptrs[layer_idx]

        # Token stride
        index_head_dim = self.sparse_attention_config.index_head_dim
        has_fp8_kv_cache = False
        if self.quant_config is not None:
            has_fp8_kv_cache = self.quant_config.layer_quant_mode.has_fp8_kv_cache()
        token_stride = get_token_bytes(
            self.head_dim,
            index_head_dim,
            self.compress_ratio,
            MewtwoAttentionType.SWA,
            has_fp8_kv_cache,
        )

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
                sparse_mla_base_ptr=sparse_mla_base_ptr,
                swa_buffer_ptr=swa_buffer_ptr,
                tokens_per_block=kv_cache_manager.tokens_per_block,
                token_stride=token_stride,
                compress_ratio=1,
            )
        else:
            # SWA + compressed indices
            compressed_buffer_ptr = metadata.compressed_buffer_ptrs[layer_idx]
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
                sparse_mla_base_ptr=sparse_mla_base_ptr,
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

    def sparse_kv_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: MewtwoTrtllmAttentionMetadata,
        **kwargs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return None, None
