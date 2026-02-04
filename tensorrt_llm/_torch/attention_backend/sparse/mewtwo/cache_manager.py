import math
from typing import Dict, List, Tuple

import torch

from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManagerV2, Role
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor, get_size_in_bytes
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MewtwoSparseAttentionConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BufferConfig,
    DiskCacheTierConfig,
    GpuCacheTierConfig,
    HostCacheTierConfig,
    LayerId,
)
from tensorrt_llm.runtime.kv_cache_manager_v2 import KVCacheManager as KVCacheManagerPy
from tensorrt_llm.runtime.kv_cache_manager_v2 import KVCacheManagerConfig as KVCacheManagerConfigPy

from .mewtwo import MewtwoAttentionType

MEWTWO_SPARSE_RATIO = 4
MEWTWO_OVERLAP_COMPRESSOR_RATIO = 4


class MewtwoCacheManager(KVCacheManagerV2):
    # Mapping from [attention type, layer] to the pool id,
    # shape: [num_attention_types, num_layers]
    layer_attn_to_pool_id: torch.Tensor
    # Mapping from [attention type, layer] to the buffer pointer including layer offset,
    # shape: [num_attention_types, num_layers]
    layer_attn_to_buffer_ptr: torch.Tensor
    # Mapping from [attention type, layer] to the pool pointer (base address),
    # shape: [num_attention_types, num_layers]
    layer_attn_to_pool_ptr: torch.Tensor
    # kv_cache_pool_pointers contains pool pointers for each pool, shape: [num_pools, 2]
    # The second column is always 0.
    kv_cache_pool_pointers: torch.Tensor
    # kv_cache_pool_mapping contains pool id and layer offset for each layer's swa attention,
    # shape: [num_layers, 2]
    kv_cache_pool_mapping: torch.Tensor
    # The block size of the (indexer) compressed cache.
    # For other attention types, block size is tokens_per_block.
    compressed_block_sizes: List[int]

    def __init__(
        self,
        kv_cache_config: KvCacheConfig,
        kv_cache_type: CacheTypeCpp,
        *,
        num_layers: int,
        num_kv_heads: int = 1,
        max_batch_size: int,
        max_beam_width: int = 1,
        tokens_per_block: int,
        vocab_size: int,
        dtype: DataType = DataType.BF16,
        compressor_dtype: DataType = DataType.FLOAT,
        sparse_attn_config: MewtwoSparseAttentionConfig,
        **kwargs,
    ) -> None:
        # Mewtwo specific attributes initialization
        assert kv_cache_type == CacheTypeCpp.SELFKONLY, "Mewtwo only supports SELFKONLY"
        assert num_kv_heads == 1, "Mewtwo only supports MQA, num_kv_heads must be 1"
        assert len(sparse_attn_config.compress_ratios) == num_layers, (
            "The length of compress ratios must be equal to the number of layers"
        )
        # use tokens_per_block == 128 to ensure token is contiguous in the compressed cache
        # TODO(jiaganc): remove this after cache manager supports per layer tokens_per_block
        assert tokens_per_block == 128, "Mewtwo requires tokens_per_block == 128"
        assert dtype in [DataType.BF16, DataType.FP8], (
            f"Unsupported dtype: {dtype}, only support BF16 and FP8"
        )
        assert compressor_dtype in [DataType.FLOAT, DataType.FP8], (
            f"Unsupported compressor dtype: {compressor_dtype}, only support FP32 and FP8"
        )

        self._compress_ratios = sparse_attn_config.compress_ratios
        self._window_size = sparse_attn_config.window_size
        self._indexer_head_dim = sparse_attn_config.index_head_dim
        self._compressor_dtype = compressor_dtype
        self.compressed_block_sizes = [
            tokens_per_block // self._compress_ratios[i] for i in range(num_layers)
        ]

        # indexer kv cache use blockwise FP8 quantization
        self._indexer_dtype = DataType.FP8
        self._indexer_scale_dtype = DataType.FLOAT
        self._indexer_quant_block_size = 128
        assert self._indexer_head_dim % self._indexer_quant_block_size == 0, (
            f"indexer_head_dim {self._indexer_head_dim} must be divisible by {self._indexer_quant_block_size}"
        )
        self._indexer_scale_size = get_size_in_bytes(
            self._indexer_head_dim // self._indexer_quant_block_size, self._indexer_scale_dtype
        )

        # General initialization
        super().__init__(
            kv_cache_config,
            kv_cache_type,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
            tokens_per_block=tokens_per_block,
            vocab_size=vocab_size,
            dtype=dtype,
            **kwargs,
        )
        self.is_vswa = True  # Mewtwo must has VSWA
        # delete the manager created in super().__init__()
        del self.impl

        # Create the KVCacheManagerPy
        max_tokens = int(
            math.ceil(kv_cache_config.max_tokens / kv_cache_config.max_util_for_resume) * 1.2
        )
        quota = GpuCacheTierConfig(quota=int(max_tokens * self._get_cache_bytes_per_token()))
        logger.info(f"Allocated {quota.quota / (1 << 30)} GiB in paged KV cache.")
        self._create_cache_manager(
            tokens_per_block=tokens_per_block,
            vocab_size=vocab_size,
            max_util_for_resume=kv_cache_config.max_util_for_resume,
            quota=quota,
        )

        # Used by the KVCacheManagerV2
        self.num_pools = len(self.impl.layer_grouping)
        self.layer_to_pool_mapping_dict: dict[int, int] = {
            layer_id: self.impl.get_layer_group_id(layer_id)
            for layer_id in range(self._num_manager_layers)
        }
        # Mapping from (pool_id, req_idx, kv) to block indices
        # layer/attn in the same pool will share the same block indices
        self.host_kv_cache_block_offsets = torch.empty(
            (
                self.num_pools,
                (max_batch_size + 1) * max_beam_width,
                # Mewtwo doesn't split key and value, but use 2 for compatibility
                2,
                self.max_blocks_per_seq,
            ),
            dtype=torch.int32,
            pin_memory=True,
            device="cpu",
        )

        # Helper functions to construct the tensors
        def _get_pool_id(layer_idx: int, attn_type: MewtwoAttentionType) -> int:
            if not self._layer_has_attention(layer_idx, attn_type):
                return -1
            return self.impl.get_layer_group_id(self._layer_attn_to_layer_id[layer_idx, attn_type])

        def _get_pool_buffer_ptr(layer_idx: int, attn_type: MewtwoAttentionType) -> int:
            if not self._layer_has_attention(layer_idx, attn_type):
                return 0
            return self.impl.get_mem_pool_base_address(
                self._layer_attn_to_layer_id[layer_idx, attn_type], Role.KEY
            )

        def _get_pool_ptr(pool_id: int) -> int:
            if pool_id == -1:
                return 0
            return self.impl.get_mem_pool_base_address(
                self.impl.layer_grouping[pool_id][0], Role.KEY
            )

        # Used by Sparse Attention
        # shape: [len(AttentionType), num_local_layers]
        self.layer_attn_to_pool_id = torch.tensor(
            [
                [_get_pool_id(layer, attn) for layer in self.pp_layers]
                for attn in MewtwoAttentionType
            ],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        # shape: [len(AttentionType), num_local_layers]
        self.layer_attn_to_buffer_ptr = torch.tensor(
            [
                [_get_pool_buffer_ptr(layer, attn) for layer in self.pp_layers]
                for attn in MewtwoAttentionType
            ],
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        )
        # shape: [len(AttentionType), num_local_layers]
        self.layer_attn_to_pool_ptr = torch.tensor(
            [
                [_get_pool_ptr(_get_pool_id(layer, attn)) for layer in self.pp_layers]
                for attn in MewtwoAttentionType
            ],
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        )

        # These tensors only contain swa attention for compatibility,
        # use layer_attn_to_buffer_ptr, layer_attn_to_pool_id, layer_attn_to_pool_ptr for other attentions
        # shape: [num_pools, 2]
        # The second column is always 0, since Mewtwo doesn't have value cache
        self.kv_cache_pool_pointers = torch.tensor(
            [[_get_pool_ptr(pool_id), 0] for pool_id in range(self.num_pools)],
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        )
        # shape: [num_local_layers, 2]
        # The first column is the pool id, the second column is the buffer offset in blocks
        self.kv_cache_pool_mapping = torch.zeros(
            (self.num_local_layers, 2), dtype=torch.int32, device="cpu", pin_memory=True
        )
        swa_layer_offset = (
            self.layer_attn_to_buffer_ptr[MewtwoAttentionType.SWA.value]
            - self.layer_attn_to_pool_ptr[MewtwoAttentionType.SWA.value]
        )
        swa_bytes_per_block = torch.tensor(
            [self._get_attn_bytes_per_block(MewtwoAttentionType.SWA, i) for i in self.pp_layers]
        )
        swa_layer_idx_in_pool = swa_layer_offset // swa_bytes_per_block
        torch.stack(
            [
                self.layer_attn_to_pool_id[MewtwoAttentionType.SWA.value],
                swa_layer_idx_in_pool,
            ],
            dim=1,
            out=self.kv_cache_pool_mapping,
        )

    def get_buffers(self, layer_idx: int, attn_type: MewtwoAttentionType) -> torch.Tensor:
        """
        Get the buffers for a specific layer and attention type.

        Args:
            layer_idx: The layer index
            attn_type: The attention type

        Returns:
            The buffer tensor (shape: [num_blocks, tokens_per_block, attn_dim])
            For blockwise FP8 layers, shape is [num_blocks, tokens_per_block, attn_dim + scale_size]
        """
        layer_id = self._layer_attn_to_layer_id[(layer_idx, attn_type)]
        addr = self.impl.get_mem_pool_base_address(layer_id, Role.KEY)

        block_size = self.tokens_per_block
        if attn_type in [MewtwoAttentionType.COMPRESS, MewtwoAttentionType.INDEXER_COMPRESS]:
            block_size = self.compressed_block_sizes[layer_idx]

        attn_dim = self._get_attn_dim(layer_idx, attn_type)
        scale_size = 0
        if attn_type == MewtwoAttentionType.INDEXER_COMPRESS:
            scale_size = self._indexer_scale_size

        shape = (
            self.impl.get_page_index_upper_bound(layer_id, Role.KEY),
            block_size,
            attn_dim + scale_size,
        )

        dtype = self.dtype
        # (indexer) compressor state and score use compressor_dtype
        if attn_type in [
            MewtwoAttentionType.COMPRESSOR_STATE,
            MewtwoAttentionType.COMPRESSOR_SCORE,
            MewtwoAttentionType.INDEXER_COMPRESSOR_STATE,
            MewtwoAttentionType.INDEXER_COMPRESSOR_SCORE,
        ]:
            dtype = self._compressor_dtype
        elif attn_type == MewtwoAttentionType.INDEXER_COMPRESS:
            dtype = self._indexer_dtype

        return convert_to_torch_tensor(TensorWrapper(addr, dtype, shape))

    def get_cache_indices(
        self,
        request_id: int,
        layer_idx: int,
        attn_type: MewtwoAttentionType,
    ) -> List[int]:
        """
        Get the cache block indices for a batch of requests at a specific layer and attention type.

        Args:
            request_id: The request id
            layer_idx: The layer index
            attn_type: The attention type

        Returns:
            The cache block indices, shape (max_blocks_per_seq,)
        """
        layer_id = self._layer_attn_to_layer_id[(layer_idx, attn_type)]
        pool_id = self.layer_to_pool_mapping_dict[layer_id]
        return self.kv_cache_map[request_id].get_page_indices(pool_id).tolist()

    def _create_cache_manager(
        self,
        tokens_per_block: int,
        vocab_size: int,
        max_util_for_resume: float,
        quota: GpuCacheTierConfig,
    ) -> None:
        """
        Create the cache manager for Mewtwo.
        """

        layers: List[AttentionLayerConfig] = []
        layer_attn_to_layer_id: Dict[Tuple[int, MewtwoAttentionType], LayerId] = {}

        def _add_layer(
            layer_idx: int, attn_type: MewtwoAttentionType, sliding_window_size: int | None
        ):
            nonlocal layers, layer_attn_to_layer_id
            layer_id = LayerId(len(layers))
            # update the mapping from layer index and attention type to layer id
            layer_attn_to_layer_id[layer_idx, attn_type] = layer_id
            # add the layer to the layers list
            layer_config = AttentionLayerConfig(
                layer_id=layer_id,
                buffers=[
                    BufferConfig(
                        role=Role.KEY, size=self._get_attn_bytes_per_block(attn_type, layer_idx)
                    )
                ],
                sliding_window_size=sliding_window_size,
                num_sink_tokens=None,
            )
            layers.append(layer_config)

        # create the layer config for Mewtwo
        for layer in self.pp_layers:
            compress_ratio = self._compress_ratios[layer]
            is_compress_layer = self._is_compress_layer(layer)
            is_sparse_layer = self._is_sparse_layer(layer)
            is_overlap = self._is_overlap_compressor(layer)

            state_factor = 2 if is_overlap else 1

            # sliding window attention pool
            _add_layer(layer, MewtwoAttentionType.SWA, self._window_size)

            if is_compress_layer:
                # compressed attention pool
                _add_layer(layer, MewtwoAttentionType.COMPRESS, None)
                # compressor state, managed as a sliding window attention cache,
                # including compressor kv states and compressor score states
                # TODO(jiaganc): +1 is a workaround for KVCacheManagerPy
                # https://nvidia.slack.com/archives/C0AACHMDM1N/p1769757646367019?thread_ts=1769742738.600089&cid=C0AACHMDM1N
                compressor_window = state_factor * compress_ratio + 1
                _add_layer(layer, MewtwoAttentionType.COMPRESSOR_STATE, compressor_window)
                _add_layer(layer, MewtwoAttentionType.COMPRESSOR_SCORE, compressor_window)

            # sparse attention layer has indexer
            if is_sparse_layer:
                # indexer kv cache pool, dim is indexer_head_dim
                _add_layer(layer, MewtwoAttentionType.INDEXER_COMPRESS, None)
                # indexer has its own compressor, so a separate compressor state
                # similarly, indexer compressor state is managed as a sliding window attention cache
                indexer_compressor_window = state_factor * compress_ratio
                _add_layer(
                    layer, MewtwoAttentionType.INDEXER_COMPRESSOR_STATE, indexer_compressor_window
                )
                _add_layer(
                    layer, MewtwoAttentionType.INDEXER_COMPRESSOR_SCORE, indexer_compressor_window
                )
        # the mapping from layer index and attention type to layer id
        self._layer_attn_to_layer_id = layer_attn_to_layer_id
        # number of layers in the KVCacheManagerPy
        self._num_manager_layers = len(layers)

        config = KVCacheManagerConfigPy(
            tokens_per_block=tokens_per_block,
            vocab_size=vocab_size,
            cache_tiers=[
                quota,
                # Magic Number for now
                HostCacheTierConfig(quota=8000 << 20),
                DiskCacheTierConfig(quota=1 << 30, path="/workspace/"),
            ],
            max_util_for_resume=max_util_for_resume,
            layers=layers,
        )
        # these two attributes are used by the KVCacheManagerV2
        self.kv_cache_manager_py_config = config
        self.impl = KVCacheManagerPy(config)

    def _is_overlap_compressor(self, layer_idx: int) -> bool:
        """
        Check if the compressor of the given layer is working in the overlap mode.
        """
        return self._compress_ratios[layer_idx] == MEWTWO_OVERLAP_COMPRESSOR_RATIO

    def _is_sparse_layer(self, layer_idx: int) -> bool:
        """
        Check if the given layer is a sparse layer.
        """
        return self._compress_ratios[layer_idx] == MEWTWO_SPARSE_RATIO

    def _is_compress_layer(self, layer_idx: int) -> bool:
        """
        Check if the given layer is a compress layer.
        """
        return self._compress_ratios[layer_idx] > 1

    def _get_attn_dim(self, layer_idx: int, attn_type: MewtwoAttentionType) -> int:
        """
        Get the dimension of the attention type for a specific layer.
        """
        is_overlap = self._is_overlap_compressor(layer_idx)
        state_factor = 2 if is_overlap else 1
        if attn_type == MewtwoAttentionType.SWA:
            return self.head_dim
        elif attn_type == MewtwoAttentionType.COMPRESS:
            return self.head_dim
        elif attn_type == MewtwoAttentionType.COMPRESSOR_STATE:
            return state_factor * self.head_dim
        elif attn_type == MewtwoAttentionType.COMPRESSOR_SCORE:
            return state_factor * self.head_dim
        elif attn_type == MewtwoAttentionType.INDEXER_COMPRESS:
            return self._indexer_head_dim
        elif attn_type == MewtwoAttentionType.INDEXER_COMPRESSOR_STATE:
            return state_factor * self._indexer_head_dim
        elif attn_type == MewtwoAttentionType.INDEXER_COMPRESSOR_SCORE:
            return state_factor * self._indexer_head_dim

    def _get_attn_bytes_per_block(
        self,
        attn_type: MewtwoAttentionType,
        layer_idx: int,
    ) -> int:
        """
        Get the cache bytes per token for a specific pool type and layer.

        Args:
            pool: DataRole of the pool
            layer_idx: Global index of the layer

        Returns:
            Size in bytes for this pool
        """
        if not self._layer_has_attention(layer_idx, attn_type):
            raise ValueError(f"Layer {layer_idx} does not have attention type {attn_type}")

        attn_dim = self._get_attn_dim(layer_idx, attn_type)

        dtype = self.dtype
        # (indexer) compressor state and score use compressor_dtype
        if attn_type in [
            MewtwoAttentionType.COMPRESSOR_STATE,
            MewtwoAttentionType.COMPRESSOR_SCORE,
            MewtwoAttentionType.INDEXER_COMPRESSOR_STATE,
            MewtwoAttentionType.INDEXER_COMPRESSOR_SCORE,
        ]:
            dtype = self._compressor_dtype
        # indexer compress use indexer_dtype
        elif attn_type == MewtwoAttentionType.INDEXER_COMPRESS:
            dtype = self._indexer_dtype

        scale_size = 0
        # indexer compress has scaling factor
        if attn_type == MewtwoAttentionType.INDEXER_COMPRESS:
            scale_size = self._indexer_scale_size

        block_size = self.tokens_per_block
        if attn_type in [MewtwoAttentionType.COMPRESS, MewtwoAttentionType.INDEXER_COMPRESS]:
            block_size = self.compressed_block_sizes[layer_idx]

        return (get_size_in_bytes(attn_dim, dtype) + scale_size) * block_size

    def _get_cache_bytes_per_token(self) -> int:
        """
        Get the average cache bytes per token for Mewtwo. This helper function is used to estimate the cache quota.

        Returns:
            Cache bytes per token across all local layers
        """
        return (
            sum(
                self._get_attn_bytes_per_block(attn, layer)
                for layer, attn in zip(self.pp_layers, MewtwoAttentionType)
                if self._layer_has_attention(layer, attn)
            )
            // self.tokens_per_block
        )

    def _layer_has_attention(self, layer_idx: int, attn_type: MewtwoAttentionType) -> bool:
        """
        Check if the given layer has the given attention type.
        """
        is_sparse = self._is_sparse_layer(layer_idx)
        is_compress = self._is_compress_layer(layer_idx)

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
