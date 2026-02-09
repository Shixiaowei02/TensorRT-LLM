from collections import defaultdict
from typing import Dict, List, Tuple

import torch

from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManagerV2, Role
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor, get_size_in_bytes
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MewtwoSparseAttentionConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.math_utils import ceil_div
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BufferConfig,
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
    # This tensor is for compatibility with AttentionOp, it only contains swa attention.
    # kv_cache_pool_pointers contains pool pointers swa pool, shape: [1, 2]
    # It assume the KVCacheManagerPy has only one pool for swa attention.
    # The second column is always 0.
    kv_cache_pool_pointers: torch.Tensor
    # This tensor is for compatibility with AttentionOp, it only contains swa attention.
    # kv_cache_pool_mapping contains pool id and layer offset for each layer's swa attention,
    # shape: [num_local_layers, 2]
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

        self.index_head_dim = sparse_attn_config.index_head_dim
        self._compress_ratios = sparse_attn_config.compress_ratios
        self._window_size = sparse_attn_config.window_size
        self._compressor_dtype = compressor_dtype
        self.compressed_block_sizes = [
            tokens_per_block // self._compress_ratios[i] for i in range(num_layers)
        ]

        # indexer kv cache use blockwise FP8 quantization
        self._indexer_dtype = DataType.FP8
        self._indexer_scale_dtype = DataType.FLOAT
        self.quant_block_size = 128
        assert self.index_head_dim % self.quant_block_size == 0, (
            f"indexer_head_dim {self.index_head_dim} must be divisible by {self.quant_block_size}"
        )
        self._indexer_scale_size = get_size_in_bytes(
            self.index_head_dim // self.quant_block_size, self._indexer_scale_dtype
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
        self._create_cache_manager(
            tokens_per_block=tokens_per_block,
            kv_cache_config=kv_cache_config,
            vocab_size=vocab_size,
        )

        # Cache the first sparse layer index for use in get_batch_indexer_k_cache_indices
        self._first_sparse_layer_idx = next(
            (layer for layer in self.pp_layers if self._is_sparse_layer(layer)),
            None,
        )

        # Used by the KVCacheManagerV2
        self.num_pools = len(self.impl.layer_grouping)
        self.layer_to_pool_mapping_dict = {
            layer_id: self.impl.get_layer_group_id(layer_id)
            for layer_id in range(self._num_manager_layers)
        }
        # Mapping from (pool_id, req_idx, kv) to block indices
        # layer/attn in the same pool will share the same block indices
        self.host_kv_cache_block_offsets = torch.empty(
            (
                self.num_pools,
                (max_batch_size + 1) * max_beam_width,
                1,
                self.max_blocks_per_seq,
            ),
            dtype=torch.int32,
            pin_memory=True,
            device="cpu",
        )

        # Mewtwo expects cache of all layers with the same attention type and compress ratio
        # to be in the same pool and have the same scale.
        self._assert_layer_pool_scale()

        swa_pool_ptr = self.impl.get_mem_pool_base_address(
            self._layer_attn_to_layer_id[0, MewtwoAttentionType.SWA], Role.KEY
        )
        self.swa_pool_ptr = swa_pool_ptr

        swa_bytes_per_block = self._get_attn_bytes_per_block(MewtwoAttentionType.SWA, 0)

        def _get_layer_offset(layer_idx: int) -> int:
            buffer_ptr = self.impl.get_mem_pool_base_address(
                self._layer_attn_to_layer_id[layer_idx, MewtwoAttentionType.SWA], Role.KEY
            )
            return (buffer_ptr - swa_pool_ptr) // swa_bytes_per_block

        # Tensors for compatibility with AttentionOp, only contains swa attention.
        # Assume the SWA of all layers share the same pool.
        # shape: [1, 2]
        self.kv_cache_pool_pointers = torch.tensor(
            [[swa_pool_ptr, 0]],
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        )
        # shape: [num_local_layers, 2]
        self.kv_cache_pool_mapping = torch.tensor(
            [[0, _get_layer_offset(layer_idx)] for layer_idx in range(self.num_local_layers)],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
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
        base_indices = self.kv_cache_map[request_id].get_base_page_indices(pool_id).tolist()
        scale = self.impl.get_page_index_scale(layer_id, Role.KEY)
        return [idx * scale for idx in base_indices if idx != -1]

    def get_token_bytes(self, layer_idx: int, attn_type: MewtwoAttentionType) -> int:
        """
        Get the token bytes for a specific layer and attention type.

        Args:
            layer_idx: The layer index
            attn_type: The attention type

        Returns:
            The token bytes, shape (1,)
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

        return get_size_in_bytes(attn_dim, dtype) + scale_size

    def _create_cache_manager(
        self,
        tokens_per_block: int,
        kv_cache_config: KvCacheConfig,
        vocab_size: int,
    ) -> None:
        """
        Create the cache manager for Mewtwo.
        """
        # Calculate the quota for KV cache
        quota = float("inf")
        if kv_cache_config.max_tokens is not None:
            max_tokens = int(
                ceil_div(kv_cache_config.max_tokens, kv_cache_config.max_util_for_resume) * 1.2
            )
            quota = int(max_tokens * self._get_cache_bytes_per_token())
            if kv_cache_config.free_gpu_memory_fraction is not None:
                logger.warning(
                    f"Both max_tokens and free_gpu_memory_fraction are set to {kv_cache_config.max_tokens}"
                    f"and {kv_cache_config.free_gpu_memory_fraction}, the smaller value will be used."
                )
        if (
            kv_cache_config.max_gpu_total_bytes is not None
            and kv_cache_config.max_gpu_total_bytes > 0
        ):
            if quota > int(kv_cache_config.max_gpu_total_bytes):
                logger.warning(
                    f"max_gpu_total_bytes {kv_cache_config.max_gpu_total_bytes / (1 << 30)}GiB is smaller than "
                    f"the calculated quota {quota / (1 << 30)}GiB, clamping quota to "
                    f"{kv_cache_config.max_gpu_total_bytes / (1 << 30)}GiB"
                )
            quota = min(quota, int(kv_cache_config.max_gpu_total_bytes))

        assert quota != float("inf"), (
            "Quota not set. Check kv_cache_config.max_tokens or kv_cache_config.max_gpu_total_bytes"
        )
        logger.info(f"KV cache manager v2 device quota set to {quota / (1 << 30)}GiB")

        cache_tiers = [GpuCacheTierConfig(quota=quota)]
        if kv_cache_config.host_cache_size is not None and kv_cache_config.host_cache_size > 0:
            cache_tiers.append(HostCacheTierConfig(quota=kv_cache_config.host_cache_size))
            logger.info(
                f"KV cache manager v2 host cache quota set to {kv_cache_config.host_cache_size / (1 << 30)}GiB"
            )

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
                compressor_window = state_factor * compress_ratio
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
            cache_tiers=cache_tiers,
            max_util_for_resume=kv_cache_config.max_util_for_resume,
            layers=layers,
        )
        # these two attributes are used by the KVCacheManagerV2
        self.kv_cache_manager_py_config = config
        self.impl = KVCacheManagerPy(config)

    def _assert_layer_pool_scale(self) -> None:
        attn_ratio_to_pool_id = defaultdict[MewtwoAttentionType, dict[int, int]](lambda: {})
        attn_ratio_to_scale = defaultdict[MewtwoAttentionType, dict[int, int]](lambda: {})

        comb = [
            (attn_type, layer_idx)
            for attn_type in MewtwoAttentionType
            for layer_idx in self.pp_layers
            if self._layer_has_attention(layer_idx, attn_type)
        ]
        for attn_type, layer_idx in comb:
            compress_ratio = self._compress_ratios[layer_idx]
            layer_id = self._layer_attn_to_layer_id[layer_idx, attn_type]
            pool_id = self.layer_to_pool_mapping_dict[layer_id]
            scale = self.impl.get_page_index_scale(layer_id, Role.KEY)

            # check if the pool id is consistent
            if compress_ratio in attn_ratio_to_pool_id[attn_type]:
                other_pool_id = attn_ratio_to_pool_id[attn_type][compress_ratio]
                assert other_pool_id == pool_id, (
                    f"Layer {layer_idx} with compress ratio {compress_ratio}, "
                    f"its attention type {attn_type.name} has pool id {pool_id}, "
                    f"but another layer with the same compress ratio and attention type has pool id {other_pool_id}."
                    "Mewtwo expects they share the same pool."
                )
            else:
                attn_ratio_to_pool_id[attn_type][compress_ratio] = pool_id

            # check if the scale is consistent
            if compress_ratio in attn_ratio_to_scale[attn_type]:
                other_scale = attn_ratio_to_scale[attn_type][compress_ratio]
                assert other_scale == scale, (
                    f"Layer {layer_idx} with compress ratio {compress_ratio}, "
                    f"its attention type {attn_type.name} has scale {scale}, "
                    f"but another layer with the same compress ratio and attention type has scale {other_scale}."
                    "Mewtwo expects they share the same scale."
                )
            else:
                attn_ratio_to_scale[attn_type][compress_ratio] = scale

        # check if all swa attentions are in the same pool and have the same scale
        swa_pool_ids = set(attn_ratio_to_pool_id[MewtwoAttentionType.SWA].values())
        swa_scales = set(attn_ratio_to_scale[MewtwoAttentionType.SWA].values())
        assert len(swa_pool_ids) == 1, "All swa attentions must be in the same pool"
        assert len(swa_scales) == 1, "All swa attentions must have the same scale"

        self._attn_ratio_to_pool_id = attn_ratio_to_pool_id
        self._attn_ratio_to_scale = attn_ratio_to_scale

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
            return self.index_head_dim
        elif attn_type == MewtwoAttentionType.INDEXER_COMPRESSOR_STATE:
            return state_factor * self.index_head_dim
        elif attn_type == MewtwoAttentionType.INDEXER_COMPRESSOR_SCORE:
            return state_factor * self.index_head_dim

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
        token_bytes = self.get_token_bytes(layer_idx, attn_type)

        block_size = self.tokens_per_block
        if attn_type in [MewtwoAttentionType.COMPRESS, MewtwoAttentionType.INDEXER_COMPRESS]:
            block_size = self.compressed_block_sizes[layer_idx]

        return token_bytes * block_size

    def _get_cache_bytes_per_token(self) -> int:
        """
        Get the average cache bytes per token for Mewtwo. This helper function is used to estimate the cache quota.

        Returns:
            Cache bytes per token across all local layers
        """
        return (
            sum(
                self._get_attn_bytes_per_block(attn, layer)
                for layer in self.pp_layers
                for attn in MewtwoAttentionType
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

    def get_indexer_k_cache_buffers(self, layer_idx: int) -> torch.Tensor:
        """
        Get the buffers for the indexer k cache for a specific layer.
        """
        return self.get_buffers(layer_idx, MewtwoAttentionType.INDEXER_COMPRESS).unsqueeze(2)

    def get_batch_indexer_k_cache_indices(self, request_ids: List[int]) -> List[List[int]]:
        """
        Get the indices for the indexer k cache for a batch of requests.
        """
        return self.get_batch_attn_offset(
            request_ids,
            # use beam_width=1 and num_contexts=0 since we don't support beam search
            1,
            0,
            len(request_ids),
            MewtwoAttentionType.INDEXER_COMPRESS,
            MEWTWO_SPARSE_RATIO,
        ).tolist()

    def copy_batch_block_offsets(
        self,
        dst_tensor: torch.Tensor,
        request_ids: List[int],
        beam_width: int,
        num_contexts: int,
        num_seqs: int,
    ) -> None:
        """For compatibility with AttentionOp, copy only the SWA block offsets."""
        offsets = self.get_batch_attn_offset(
            request_ids,
            beam_width,
            num_contexts,
            num_seqs,
            MewtwoAttentionType.SWA,
            # all compress ratios have SWA attention and they are in the same pool
            self._compress_ratios[0],
        )
        dst_tensor[0, :num_seqs, 0] = offsets

    def get_batch_attn_offset(
        self,
        request_ids: List[int],
        beam_width: int,
        num_contexts: int,
        num_seqs: int,
        attn_type: MewtwoAttentionType,
        compress_ratio: int,
    ) -> torch.Tensor:
        """
        Get the block offsets for a specific attention type for a batch of requests.

        Args:
            request_ids: The request ids
            beam_width: The beam width
            num_contexts: The number of context requests
            num_seqs: The number of sequence requests
            attn_type: The attention type
            compress_ratio: The compress ratio. Used for non-SWA attention types.

        Returns:
            The block offsets, shape (num_seqs, max_blocks_per_seq)
        """
        assert beam_width == 1, "beam_width must be 1 for KVCacheManagerV2"
        assert attn_type == MewtwoAttentionType.SWA or compress_ratio is not None, (
            "compress_ratio must be provided for non-SWA attention types"
        )

        copy_idx = self.index_mapper.get_copy_index(request_ids, num_contexts, beam_width)
        assert copy_idx.shape[0] == num_seqs

        pool_id = self._attn_ratio_to_pool_id[attn_type][compress_ratio]
        scale = self._attn_ratio_to_scale[attn_type][compress_ratio]
        offsets = self.host_kv_cache_block_offsets[pool_id, copy_idx, 0] * scale
        offsets[offsets == -scale] = -1
        return offsets
