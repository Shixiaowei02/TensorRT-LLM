from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Union

import torch

import tensorrt_llm
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManagerV2
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.mapping import Mapping

from ..dsa import DSAtrtllmAttentionMetadata

ModelConfig = tensorrt_llm.bindings.ModelConfig

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig, SparseAttentionConfig


class MewtwoTrtllmAttentionMetadata(DSAtrtllmAttentionMetadata):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_layer_types = len(set(self.compress_ratio))
        self.layer_type_mapping = {
            compress_ratio: i for i, compress_ratio in enumerate(set(self.compress_ratio))
        }

    def __post_init__(self):
        super().__post_init__()
        capture_graph = self.is_cuda_graph

        # Create buffers for different compress ratios
        self.all_kv_lens_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (
                self.num_layer_types,
                self.max_num_sequences,
            ),
            cache_name="all_kv_lens_cuda",
            dtype=torch.int,
            capture_graph=capture_graph,
        )
        self.all_kv_lens = torch.empty_like(self.all_kv_lens_cuda, device="cpu", pin_memory=True)

        # Create buffers for the compressor

    def prepare(self):
        super().prepare()

        # Prepare buffers for the compressor


class MewtwoAttentionType(Enum):
    SWA = 0
    COMPRESS = 1
    COMPRESSOR_STATE = 2
    COMPRESSOR_SCORE = 3
    INDEXER_COMPRESS = 4
    INDEXER_COMPRESSOR_STATE = 5
    INDEXER_COMPRESSOR_SCORE = 6


class MewtwoCacheManager(KVCacheManagerV2):
    def __init__(
        self,
        kv_cache_config: KvCacheConfig,
        kv_cache_type: CacheTypeCpp,
        *,
        num_layers: int,
        num_kv_heads: Union[int, List[Optional[int]]],
        head_dim: int,
        tokens_per_block: int,
        # Note that max_seq_len is not necessarily equal to kv_cache_config.num_tokens.
        # It's derived from the model's BuildConfig for consistency with the C++ backend.
        max_seq_len: int,
        max_batch_size: int,
        mapping: Mapping,
        dtype: DataType = DataType.HALF,
        spec_config: Optional["DecodingBaseConfig"] = None,
        layer_mask: Optional[List[bool]] = None,
        max_num_tokens: int = 8192,
        model_config: Optional[ModelConfig] = None,
        max_beam_width: int = 1,
        sparse_attn_config: "SparseAttentionConfig",
        **kwargs,
    ) -> None:
        super().__init__(
            kv_cache_config,
            kv_cache_type,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            mapping=mapping,
            dtype=dtype,
            spec_config=spec_config,
            layer_mask=layer_mask,
            max_num_tokens=max_num_tokens,
            model_config=model_config,
            max_beam_width=max_beam_width,
            **kwargs,
        )
