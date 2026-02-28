from collections import defaultdict
from typing import Dict, List, Tuple

import torch

from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManagerV2, Role
from tensorrt_llm._utils import (
    TensorWrapper,
    convert_to_torch_tensor,
    get_size_in_bytes,
    prefer_pinned,
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MewtwoSparseAttentionConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime import ModelConfig
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BufferConfig,
    GpuCacheTierConfig,
    HostCacheTierConfig,
    LayerId,
)
from tensorrt_llm.runtime.kv_cache_manager_v2 import KVCacheManagerConfig as KVCacheManagerConfigPy

from .mewtwo import (
    MEWTWO_SPARSE_RATIO,
    MewtwoAttentionType,
    compress_ratio_has_attention,
    get_attn_dim,
    get_token_bytes,
    is_compress_layer,
    is_overlap_compressor,
    is_sparse_layer,
)


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
        max_seq_len: int,
        vocab_size: int,
        dtype: DataType = DataType.BF16,
        compressor_dtype: DataType = DataType.FLOAT,
        sparse_attn_config: MewtwoSparseAttentionConfig,
        **kwargs,
    ) -> None:
        # Mewtwo specific attributes initialization
        assert kv_cache_type == CacheTypeCpp.SELFKONLY, "Mewtwo only supports SELFKONLY"
        assert num_kv_heads == 1, "Mewtwo only supports num_kv_heads == 1"
        assert len(sparse_attn_config.compress_ratios) == num_layers, (
            "The length of compress ratios must be equal to the number of layers"
        )
        assert dtype in [DataType.BF16, DataType.FP8], (
            f"Unsupported dtype: {dtype}, only support BF16 and FP8"
        )
        assert compressor_dtype == DataType.FLOAT, (
            f"Unsupported compressor dtype: {compressor_dtype}, only support FP32/TF32"
        )

        assert tokens_per_block in [128, 256], (
            f"MewtwoCacheManager requires tokens_per_block in [128, 256], got {tokens_per_block}. "
            f"Set kv_cache_config.tokens_per_block to 128 or 256."
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
            max_seq_len=max_seq_len,
            vocab_size=vocab_size,
            dtype=dtype,
            **kwargs,
        )
        self.is_vswa = True  # Mewtwo must has VSWA

        # Mewtwo expects cache of all layers with the same attention type and compress ratio
        # to be in the same pool and have the same scale.
        self._assert_layer_pool_scale()

        # For Mewtwo Attention, the base pointer for SWA pool
        self.swa_pool_ptr = self.impl.get_mem_pool_base_address(
            self._layer_attn_to_layer_id[0, MewtwoAttentionType.SWA], Role.KEY
        )

        self.compress_pool_ptrs = {}
        if 4 in self._compress_ratios:  # indexer compressor
            self.compress_pool_ptrs[4] = self.impl.get_mem_pool_base_address(
                self._layer_attn_to_layer_id[
                    self._compress_ratios.index(4), MewtwoAttentionType.COMPRESS
                ],
                Role.KEY,
            )
        if 128 in self._compress_ratios:  # compressor
            self.compress_pool_ptrs[128] = self.impl.get_mem_pool_base_address(
                self._layer_attn_to_layer_id[
                    self._compress_ratios.index(128), MewtwoAttentionType.COMPRESS
                ],
                Role.KEY,
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

        attn_dim = get_attn_dim(
            self.head_dim, self.index_head_dim, self._compress_ratios[layer_idx], attn_type
        )
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

    def _build_pool_mapping_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        swa_bytes_per_block = self._get_attn_bytes_per_block(MewtwoAttentionType.SWA, 0)
        swa_pool_ptr = self.impl.get_mem_pool_base_address(
            self._layer_attn_to_layer_id[0, MewtwoAttentionType.SWA], Role.KEY
        )

        def _get_layer_offset(layer_idx: int) -> int:
            buffer_ptr = self.impl.get_mem_pool_base_address(
                self._layer_attn_to_layer_id[layer_idx, MewtwoAttentionType.SWA], Role.KEY
            )
            return (buffer_ptr - swa_pool_ptr) // swa_bytes_per_block

        # Tensors for compatibility with AttentionOp, only contains swa attention.
        # Assume the SWA of all layers share the same pool.
        # shape: [1, 2]
        kv_cache_pool_pointers = torch.tensor(
            [[swa_pool_ptr, 0]],
            dtype=torch.int64,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        # shape: [num_local_layers, 2]
        kv_cache_pool_mapping = torch.tensor(
            [[0, _get_layer_offset(layer_idx)] for layer_idx in range(self.num_local_layers)],
            dtype=torch.int32,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        return kv_cache_pool_pointers, kv_cache_pool_mapping

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
        return [idx * scale if idx != -1 else -1 for idx in base_indices]

    def _get_cache_quota(self, max_tokens: int) -> int:
        quota = int(max_tokens * self.get_cache_bytes_per_token())
        # Add extra quota to ensure sufficient space for small max_tokens cases.
        quota += len(MewtwoAttentionType) * (2 << 20)
        return quota

    def _build_cache_config(
        self,
        kv_cache_config: KvCacheConfig,
        *,
        tokens_per_block: int,
        vocab_size: int | None,
        cache_tiers: List[GpuCacheTierConfig | HostCacheTierConfig],
    ) -> KVCacheManagerConfigPy:
        """
        Create the cache manager config for Mewtwo.
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
            is_compress = is_compress_layer(compress_ratio)
            is_sparse = is_sparse_layer(compress_ratio)
            is_overlap = is_overlap_compressor(compress_ratio)

            state_factor = 2 if is_overlap else 1

            # sliding window attention pool
            _add_layer(layer, MewtwoAttentionType.SWA, self._window_size)

            if is_compress:
                # compressed attention pool
                _add_layer(layer, MewtwoAttentionType.COMPRESS, None)
                # compressor state, managed as a sliding window attention cache,
                # including compressor kv states and compressor score states
                compressor_window = state_factor * compress_ratio
                _add_layer(layer, MewtwoAttentionType.COMPRESSOR_STATE, compressor_window)
                _add_layer(layer, MewtwoAttentionType.COMPRESSOR_SCORE, compressor_window)

            # sparse attention layer has indexer
            if is_sparse:
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

        return KVCacheManagerConfigPy(
            tokens_per_block=tokens_per_block,
            vocab_size=vocab_size,
            cache_tiers=cache_tiers,
            max_util_for_resume=kv_cache_config.max_util_for_resume,
            layers=layers,
        )

    def _assert_layer_pool_scale(self) -> None:
        attn_ratio_to_pool_id = defaultdict[MewtwoAttentionType, dict[int, int]](lambda: {})
        attn_ratio_to_scale = defaultdict[MewtwoAttentionType, dict[int, int]](lambda: {})

        comb = [
            (attn_type, layer_idx)
            for attn_type in MewtwoAttentionType
            for layer_idx in self.pp_layers
            if compress_ratio_has_attention(self._compress_ratios[layer_idx], attn_type)
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

    def _get_attn_bytes_per_block(
        self,
        attn_type: MewtwoAttentionType,
        layer_idx: int,
    ) -> int:
        """
        Get the cache bytes per token for a specific attention type and layer.
        """
        has_fp8_kv_cache = self.dtype == DataType.FP8
        token_bytes = get_token_bytes(
            self.head_dim,
            self.index_head_dim,
            self._compress_ratios[layer_idx],
            attn_type,
            has_fp8_kv_cache,
        )

        block_size = self.tokens_per_block
        if attn_type in [MewtwoAttentionType.COMPRESS, MewtwoAttentionType.INDEXER_COMPRESS]:
            block_size = self.compressed_block_sizes[layer_idx]

        return token_bytes * block_size

    def get_cache_bytes_per_token(self) -> int:
        """Get the average cache bytes per token for Mewtwo."""
        has_fp8_kv_cache = self.dtype == DataType.FP8
        compress_ratios = [self._compress_ratios[layer] for layer in self.pp_layers]
        return self._estimate_bytes_per_token(
            self.head_dim,
            self.index_head_dim,
            compress_ratios,
            has_fp8_kv_cache,
        )

    def get_layer_bytes_per_token(
        self,
        local_layer_idx: int,
        data_role: Role,
    ) -> int:
        raise NotImplementedError(
            "Mewtwo doesn't support get_layer_bytes_per_token, use _get_attn_bytes_per_block"
        )

    @staticmethod
    def _estimate_bytes_per_token(
        head_dim: int,
        index_head_dim: int,
        compress_ratios: List[int],
        has_fp8_kv_cache,
    ) -> int:
        total_bytes = 0
        for ratio in compress_ratios:
            for attn in MewtwoAttentionType:
                if compress_ratio_has_attention(ratio, attn):
                    attn_bytes = get_token_bytes(
                        head_dim,
                        index_head_dim,
                        ratio,
                        attn,
                        has_fp8_kv_cache,
                    )
                    # compressed attention will be compressed by the compress ratio
                    if attn in [MewtwoAttentionType.COMPRESS, MewtwoAttentionType.INDEXER_COMPRESS]:
                        attn_bytes //= ratio
                    total_bytes += attn_bytes
        return total_bytes

    def get_indexer_k_cache_buffers(self, layer_idx: int) -> torch.Tensor:
        """
        Get the buffers for the indexer k cache for a specific layer.
        """
        buffer = self.get_buffers(layer_idx, MewtwoAttentionType.INDEXER_COMPRESS).unsqueeze(2)
        return buffer.view(torch.uint8)

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
        dst_tensor[0, :num_seqs, :, :] = offsets[:, None, :]

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

    @staticmethod
    def get_cache_size_per_token(model_config: ModelConfig, mapping: Mapping, **kwargs):
        config = model_config.pretrained_config
        head_dim = config.kv_lora_rank + config.qk_rope_head_dim
        index_head_dim = model_config.sparse_attention_config.index_head_dim
        pp_layers = mapping.pp_layers(model_config.get_num_attention_layers())
        compress_ratios = [
            model_config.sparse_attention_config.compress_ratios[layer] for layer in pp_layers
        ]
        quant_config = model_config.quant_config
        if quant_config is not None:
            has_fp8_kv_cache = quant_config.quant_mode.has_fp8_kv_cache()
        else:
            has_fp8_kv_cache = False
        return MewtwoCacheManager._estimate_bytes_per_token(
            head_dim,
            index_head_dim,
            compress_ratios,
            has_fp8_kv_cache,
        )

    def check_invalid_values_in_kv_cache(self, fill_with_zero: bool = False) -> bool:
        some_checks_unavailable = False
        has_invalid_values = torch.tensor(
            [False], dtype=torch.bool, device=torch.cuda.current_device()
        )
        pool_handled = set()

        # Handle each layer from start to end to traverse the whole KV cache.
        for (layer, attn), layer_id in self._layer_attn_to_layer_id.items():
            pool_id = self.layer_to_pool_mapping_dict[layer_id]
            if pool_id in pool_handled:
                continue
            buffer = self.get_buffers(layer, attn)
            # process in chunks of 256 pages to avoid OoM
            for i in range(0, buffer.shape[0], 256):
                buffer_slice = buffer[i : i + 256]
                try:
                    has_invalid_values.logical_or_(torch.isnan(buffer_slice).any())
                    has_invalid_values.logical_or_(torch.isinf(buffer_slice).any())
                except NotImplementedError:
                    some_checks_unavailable = True
            if fill_with_zero:
                buffer.zero_()
            pool_handled.add(pool_id)
        torch.cuda.synchronize()

        if some_checks_unavailable:
            logger.warning(
                "`torch.isnan` or `torch.isinf` is not implemented for current kv cache dtype, "
                "related checks are skipped"
            )
        return bool(has_invalid_values)
