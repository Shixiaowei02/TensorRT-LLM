from typing import Dict, List, Tuple

import pytest
import torch

from tensorrt_llm._torch.attention_backend.sparse.mewtwo import MewtwoCacheManager
from tensorrt_llm._torch.attention_backend.sparse.mewtwo.mewtwo import MewtwoAttentionType
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._utils import binding_to_torch_dtype
from tensorrt_llm.bindings import DataType, SamplingConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MewtwoSparseAttentionConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2._common import BAD_PAGE_INDEX

_RequestCache = Dict[Tuple[int, MewtwoAttentionType], torch.Tensor]


class TestMewtwoCacheManager:
    # mewtwo specific param
    head_dim = 512
    index_head_dim = 128
    window_size = 128
    vocab_size = 129280
    sparse_layer_ratio = 4
    overlap_compress_layer_ratio = 4

    # cache manager specific param
    tokens_per_block = 128
    max_batch_size = 16
    max_seq_len = 1024

    def _is_compress_layer(self, compress_ratio: int) -> bool:
        """Check if a layer uses compression based on its compress ratio.

        Args:
            compress_ratio: The compression ratio for the layer

        Returns:
            True if the layer uses compression (ratio > 1)
        """
        return compress_ratio > 1

    def _is_sparse_layer(self, compress_ratio: int) -> bool:
        """Check if a layer uses sparse attention based on its compress ratio.

        Args:
            compress_ratio: The compression ratio for the layer

        Returns:
            True if the layer uses sparse attention (ratio == 4)
        """
        return compress_ratio == self.sparse_layer_ratio

    def _is_overlap_compressor(self, compress_ratio: int) -> bool:
        """Check if a layer uses overlap compressor based on its compress ratio.

        Args:
            compress_ratio: The compression ratio for the layer

        Returns:
            True if the layer uses overlap compressor (ratio == 4)
        """
        return compress_ratio == self.overlap_compress_layer_ratio

    def _get_window_size(self, compress_ratio: int, attn_type: MewtwoAttentionType) -> int:
        """Get the window size for a layer based on its compress ratio and attention type.

        Args:
            compress_ratio: The compression ratio for the layer
            attn_type: The attention type

        Returns:
            The window size for the layer
        """
        state_factor = 2 if self._is_overlap_compressor(compress_ratio) else 1
        if attn_type == MewtwoAttentionType.SWA:
            return self.window_size
        elif attn_type in [
            MewtwoAttentionType.COMPRESSOR_STATE,
            MewtwoAttentionType.COMPRESSOR_SCORE,
            MewtwoAttentionType.INDEXER_COMPRESSOR_STATE,
            MewtwoAttentionType.INDEXER_COMPRESSOR_SCORE,
        ]:
            return state_factor * compress_ratio
        elif attn_type in [MewtwoAttentionType.COMPRESS, MewtwoAttentionType.INDEXER_COMPRESS]:
            return None

    def _create_mewtwo_cache_manager(
        self,
        tokens_per_block: int,
        compress_ratios: List[int],
        dtype: DataType,
        compressor_dtype: DataType,
    ) -> Tuple[MewtwoCacheManager, MewtwoSparseAttentionConfig]:
        """Helper to create a MewtwoCacheManager for testing."""

        # Create sparse attention config
        sparse_attn_config = MewtwoSparseAttentionConfig(
            index_head_dim=self.index_head_dim,
            window_size=self.window_size,
            compress_ratios=compress_ratios,
        )

        # Create KV cache config
        max_num_tokens = self.max_seq_len * self.max_batch_size
        kv_cache_config = KvCacheConfig(
            enable_block_reuse=False,
            max_tokens=max_num_tokens,
            event_buffer_max_size=0,
        )

        # Create mapping (single GPU, no parallelism)
        mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)

        # Create cache manager
        cache_manager = MewtwoCacheManager(
            kv_cache_config=kv_cache_config,
            kv_cache_type=CacheTypeCpp.SELFKONLY,
            num_layers=len(compress_ratios),
            num_kv_heads=1,
            head_dim=self.head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=self.max_seq_len,
            max_batch_size=self.max_batch_size,
            mapping=mapping,
            dtype=dtype,
            compressor_dtype=compressor_dtype,
            vocab_size=self.vocab_size,
            max_num_tokens=max_num_tokens,
            sparse_attn_config=sparse_attn_config,
        )

        return cache_manager, sparse_attn_config

    def _create_request(self, request_id: int, prompt_len: int) -> LlmRequest:
        """Helper to create a test LlmRequest.

        Args:
            request_id: Unique request identifier
            prompt_len: Prompt length (number of tokens)

        Returns:
            LlmRequest instance
        """
        input_tokens = list(range(prompt_len))
        request = LlmRequest(
            request_id=request_id,
            max_new_tokens=1024,
            input_tokens=input_tokens,
            sampling_config=SamplingConfig(),
            is_streaming=False,
        )

        return request

    def _create_random_cache(
        self,
        seq_len: int,
        head_dim: int,
        sparse_attn_config: MewtwoSparseAttentionConfig,
        dtype: torch.dtype,
        compressor_dtype: torch.dtype,
        device: torch.device = torch.device("cuda"),
    ) -> Dict[Tuple[int, MewtwoAttentionType], torch.Tensor]:
        """Helper to create random cache values for all layers and attention types.

        Args:
            seq_len: Sequence length
            head_dim: Head dimension for regular attention
            sparse_attn_config: Sparse attention configuration

        Returns:
            Dictionary mapping (layer_idx, attn_type) to random tensor values
        """
        cache: Dict[Tuple[int, MewtwoAttentionType], torch.Tensor] = {}

        for layer, ratio in enumerate(sparse_attn_config.compress_ratios):
            is_overlap = self._is_overlap_compressor(ratio)

            cache[layer, MewtwoAttentionType.SWA] = torch.randn(
                (seq_len, head_dim), dtype=dtype, device=device
            )

            if self._is_compress_layer(ratio):
                compressor_dim = 2 * head_dim if is_overlap else head_dim
                cache[layer, MewtwoAttentionType.COMPRESS] = torch.randn(
                    (seq_len // ratio, head_dim), dtype=dtype, device=device
                )
                cache[layer, MewtwoAttentionType.COMPRESSOR_STATE] = torch.randn(
                    (seq_len, compressor_dim), dtype=compressor_dtype, device=device
                )
                cache[layer, MewtwoAttentionType.COMPRESSOR_SCORE] = torch.randn(
                    (seq_len, compressor_dim), dtype=compressor_dtype, device=device
                )

            if self._is_sparse_layer(ratio):
                indexer_dim = sparse_attn_config.index_head_dim
                indexer_compressor_dim = 2 * indexer_dim if is_overlap else indexer_dim
                cache[layer, MewtwoAttentionType.INDEXER_COMPRESS] = torch.randn(
                    (seq_len // ratio, indexer_dim), dtype=dtype, device=device
                )
                cache[layer, MewtwoAttentionType.INDEXER_COMPRESSOR_STATE] = torch.randn(
                    (seq_len, indexer_compressor_dim), dtype=compressor_dtype, device=device
                )
                cache[layer, MewtwoAttentionType.INDEXER_COMPRESSOR_SCORE] = torch.randn(
                    (seq_len, indexer_compressor_dim), dtype=compressor_dtype, device=device
                )

        return cache

    def _prefill_write_paged_cache(
        self,
        buffer: torch.Tensor,
        block_indices: List[int],
        values: torch.Tensor,
    ) -> None:
        """Write context values to a paged cache buffer.

        Args:
            buffer: The cache buffer to write to (shape: [num_blocks, tokens_per_block, dim_per_token])
            block_indices: List of block indices to write to
            values: Values to write (shape: [seq_len, dim_per_token])
        """
        assert buffer.size(2) == values.size(1), f"{buffer.size(2)=} != {values.size(1)=}"
        tokens_per_block = buffer.size(1)
        seq_len, dim_per_token = values.shape

        num_blocks = (seq_len + tokens_per_block - 1) // tokens_per_block
        assert all(idx != BAD_PAGE_INDEX for idx in block_indices[:num_blocks]), (
            f"{block_indices[:num_blocks]=} contains BAD_PAGE_INDEX"
        )

        if seq_len % tokens_per_block != 0:
            # pad the values to the nearest multiple of tokens_per_block
            pad_len = tokens_per_block - (seq_len % tokens_per_block)
            values = torch.cat(
                [
                    values,
                    torch.randn((pad_len, dim_per_token), dtype=values.dtype, device=values.device),
                ],
                dim=0,
            )

        values_blocks = values.reshape(num_blocks, tokens_per_block, dim_per_token)
        buffer[block_indices[:num_blocks]] = values_blocks

    def _decode_write_paged_cache(
        self,
        buffer: torch.Tensor,
        block_indices: List[int],
        token_idx: int,
        value: torch.Tensor,
    ) -> None:
        """Simulate the decode phrase. Write one new token to the cache.

        Args:
            buffer: The cache buffer to write to (shape: [num_blocks, tokens_per_block, dim_per_token])
            block_indices: List of block indices to write to
            token_idx: Index of the new token to write
            value: Value to write (shape: [dim_per_token])
        """
        assert buffer.size(2) == value.size(0), f"{buffer.size(2)=} != {value.size(0)=}"
        num_blocks, tokens_per_block, _ = buffer.shape

        block_idx = token_idx // tokens_per_block
        block_offset = token_idx % tokens_per_block
        assert block_idx < num_blocks, f"{block_idx=} >= {num_blocks=}"
        assert block_indices[block_idx] != BAD_PAGE_INDEX, (
            f"{block_indices[block_idx]=} == BAD_PAGE_INDEX"
        )

        buffer[block_indices[block_idx], block_offset] = value

    def _read_paged_cache(
        self, buffer: torch.Tensor, block_indices: List[int], seq_len: int, window_size: int | None
    ) -> torch.Tensor:
        """Read values from a paged cache buffer.

        Args:
            buffer: The cache buffer to read from (shape: [num_blocks, tokens_per_block, dim_per_token])
            block_indices: List of block/page indices to read from
            seq_len: Sequence length
            window_size: sliding window size to read from the cache

        Returns:
            Tensor containing the read values (shape: [seq_len, dim_per_token] or [window_size, dim_per_token]
            if window_size is given and seq_len > window_size)
        """
        _, tokens_per_block, dim_per_token = buffer.shape

        # check if all blocks within the window are valid
        end_block_idx = (seq_len + tokens_per_block - 1) // tokens_per_block
        if window_size is not None:
            start_block_idx = (seq_len - window_size + tokens_per_block - 1) // tokens_per_block
        else:
            start_block_idx = 0
        assert all(idx != BAD_PAGE_INDEX for idx in block_indices[start_block_idx:end_block_idx]), (
            f"{block_indices[start_block_idx:end_block_idx]=} contains BAD_PAGE_INDEX"
        )

        # read values from the cache
        values = buffer[block_indices].reshape(-1, dim_per_token)[:seq_len]
        if window_size is not None and seq_len > window_size:
            values = values[-window_size:]
        return values

    def _write_request_prefill(
        self,
        req: LlmRequest,
        prompt_len: int,
        cache_manager: MewtwoCacheManager,
        cache_values: _RequestCache,
    ) -> None:
        """Write cache values for a request to the cache manager.

        Args:
            req: The request to write cache for
            prompt_len: Prompt length
            cache_manager: The cache manager instance
            cache_values: Dictionary mapping (layer_idx, attn_type) to tensor values
        """
        compress_ratios = cache_manager._compress_ratios
        for (layer_idx, attn_type), values in cache_values.items():
            page_indices = cache_manager.get_cache_indices(
                request_id=req.py_request_id, layer_idx=layer_idx, attn_type=attn_type
            )

            if attn_type in [MewtwoAttentionType.COMPRESS, MewtwoAttentionType.INDEXER_COMPRESS]:
                seq_len = prompt_len // compress_ratios[layer_idx]
            else:
                seq_len = prompt_len

            self._prefill_write_paged_cache(
                buffer=cache_manager.get_buffers(layer_idx, attn_type),
                block_indices=page_indices,
                values=values[:seq_len],
            )

    def _write_request_decode(
        self,
        req: LlmRequest,
        token_idx: int,
        cache_manager: MewtwoCacheManager,
        cache_values: _RequestCache,
    ) -> None:
        """Simulate the decode phrase. Write one new token to the cache.

        Args:
            req: The request to write cache for
            token_idx: Index of the new token to write
            cache_manager: The cache manager instance
            cache_values: Dictionary mapping (layer_idx, attn_type) to tensor values
        """
        compress_ratios = cache_manager._compress_ratios
        for (layer_idx, attn_type), values in cache_values.items():
            block_indices = cache_manager.get_cache_indices(
                request_id=req.py_request_id, layer_idx=layer_idx, attn_type=attn_type
            )

            # compute the compressed token index
            if attn_type in [MewtwoAttentionType.COMPRESS, MewtwoAttentionType.INDEXER_COMPRESS]:
                if (token_idx + 1) % compress_ratios[layer_idx] != 0:
                    # skip if current token will not trigger compression
                    continue
                token_idx = token_idx // compress_ratios[layer_idx]

            token_value = values[token_idx]
            self._decode_write_paged_cache(
                buffer=cache_manager.get_buffers(layer_idx, attn_type),
                block_indices=block_indices,
                token_idx=token_idx,
                value=token_value,
            )

    def _read_request(
        self,
        req: LlmRequest,
        seq_len: int,
        cache_manager: MewtwoCacheManager,
        compress_ratios: List[int],
    ) -> _RequestCache:
        """Read cache values for a request from the cache manager.

        Args:
            req: The request to read cache for
            seq_len: Sequence length
            cache_manager: The cache manager instance
            compress_ratios: Compression ratios for each layer

        Returns:
            Dictionary mapping (layer_idx, attn_type) to tensor values read from cache
        """
        cache_values: _RequestCache = {}
        for layer, ratio in enumerate(compress_ratios):
            # list of attentions to read for this layer
            attn_types = [MewtwoAttentionType.SWA]
            if self._is_compress_layer(ratio):
                attn_types.extend(
                    [
                        MewtwoAttentionType.COMPRESS,
                        MewtwoAttentionType.COMPRESSOR_STATE,
                        MewtwoAttentionType.COMPRESSOR_SCORE,
                    ]
                )
            if self._is_sparse_layer(ratio):
                attn_types.extend(
                    [
                        MewtwoAttentionType.INDEXER_COMPRESS,
                        MewtwoAttentionType.INDEXER_COMPRESSOR_STATE,
                        MewtwoAttentionType.INDEXER_COMPRESSOR_SCORE,
                    ]
                )

            # read cache values for each attention type
            for attn_type in attn_types:
                # cache_buffer = cache_manager.get_buffers(layer, attn_type)
                page_indices = cache_manager.get_cache_indices(
                    request_id=req.py_request_id, layer_idx=layer, attn_type=attn_type
                )
                if attn_type in [
                    MewtwoAttentionType.COMPRESS,
                    MewtwoAttentionType.INDEXER_COMPRESS,
                ]:
                    attn_len = seq_len // ratio
                else:
                    attn_len = seq_len
                cache_values[layer, attn_type] = self._read_paged_cache(
                    buffer=cache_manager.get_buffers(layer, attn_type),
                    block_indices=page_indices,
                    seq_len=attn_len,
                    window_size=self._get_window_size(ratio, attn_type),
                )

        return cache_values

    def _assert_cache_equal(
        self, seq_len: int, compress_ratios: List[int], expect: _RequestCache, actual: _RequestCache
    ) -> None:
        """Assert that two cache dictionaries contain equal values.

        Args:
            seq_len: Sequence length
            compress_ratios: Compression ratios for each layer
            expected: Expected cache values
            actual: Actual cache values read from cache manager
        """
        # Check that keys match
        assert set(expect.keys()) == set(actual.keys()), (
            f"Cache keys don't match. Expected: {set(expect.keys())}, Actual: {set(actual.keys())}"
        )

        # Check each tensor value
        for layer_idx, attn_type in expect.keys():
            if attn_type in [MewtwoAttentionType.COMPRESS, MewtwoAttentionType.INDEXER_COMPRESS]:
                attn_len = seq_len // compress_ratios[layer_idx]
            else:
                attn_len = seq_len
            expect_values = expect[layer_idx, attn_type][:attn_len]

            window_size = self._get_window_size(compress_ratios[layer_idx], attn_type)
            if window_size is not None:
                expect_values = expect_values[-window_size:]

            torch.testing.assert_close(
                actual[layer_idx, attn_type],
                expect_values,
                rtol=1e-5,
                atol=1e-5,
                msg=f"Mismatch for layer {layer_idx}, attention type {attn_type.value}",
            )

    @pytest.mark.parametrize("compress_ratios", [[1, 4, 128]])
    @pytest.mark.parametrize("dtype,compressor_dtype", [(DataType.BF16, DataType.FLOAT)])
    @pytest.mark.parametrize("request_lens", [[(512, 520), (128, 130), (160, 170)]])
    def test_write_read_cache(
        self,
        compress_ratios: List[int],
        request_lens: List[Tuple[int, int]],
        dtype: DataType,
        compressor_dtype: DataType,
    ):
        # Create cache manager and sparse attention config
        cache_manager, sparse_attn_config = self._create_mewtwo_cache_manager(
            tokens_per_block=self.tokens_per_block,
            compress_ratios=compress_ratios,
            dtype=dtype,
            compressor_dtype=compressor_dtype,
        )

        # Create requests and their cache values
        requests = list[LlmRequest]()
        cache_values = dict[int, _RequestCache]()
        for req_id, (prompt_len, max_seq_len) in enumerate(request_lens):
            req = self._create_request(req_id, prompt_len)
            requests.append(req)

            # Generate random cache values for this request
            cache_values[req_id] = self._create_random_cache(
                seq_len=max_seq_len,
                head_dim=self.head_dim,
                sparse_attn_config=sparse_attn_config,
                dtype=binding_to_torch_dtype(dtype),
                compressor_dtype=binding_to_torch_dtype(compressor_dtype),
            )

        # Simulate the prefill phrase
        seq_lens = [request_lens[r.py_request_id][0] for r in requests]
        scheduled_batch = ScheduledRequests()
        scheduled_batch.context_requests = requests
        cache_manager.prepare_resources(scheduled_batch)

        # Write context to cache
        for req in requests:
            self._write_request_prefill(
                req=req,
                prompt_len=seq_lens[req.py_request_id],
                cache_manager=cache_manager,
                cache_values=cache_values[req.py_request_id],
            )

        # Update requests state and call update_resources
        for req in requests:
            req.context_current_position = seq_lens[req.py_request_id]
            req.add_new_token(seq_lens[req.py_request_id], 0)
        cache_manager.update_resources(scheduled_batch)

        # For disagg example: cache transmission happens here

        # Read context from cache and verify
        for req in requests:
            actual_cache_values = self._read_request(
                req=req,
                seq_len=seq_lens[req.py_request_id],
                cache_manager=cache_manager,
                compress_ratios=compress_ratios,
            )
            self._assert_cache_equal(
                seq_len=seq_lens[req.py_request_id],
                compress_ratios=compress_ratios,
                expect=cache_values[req.py_request_id],
                actual=actual_cache_values,
            )

        # Simulate the decode phrase
        seq_lens = [seq_len + 1 for seq_len in seq_lens]
        scheduled_batch = ScheduledRequests()
        scheduled_batch.generation_requests = requests
        cache_manager.prepare_resources(scheduled_batch)

        # Write new token to cache
        for req in requests:
            self._write_request_decode(
                req=req,
                token_idx=seq_lens[req.py_request_id] - 1,
                cache_manager=cache_manager,
                cache_values=cache_values[req.py_request_id],
            )

        # Read context from cache and verify
        for req in requests:
            actual_cache_values = self._read_request(
                req=req,
                seq_len=seq_lens[req.py_request_id],
                cache_manager=cache_manager,
                compress_ratios=compress_ratios,
            )
            self._assert_cache_equal(
                seq_len=seq_lens[req.py_request_id],
                compress_ratios=compress_ratios,
                expect=cache_values[req.py_request_id],
                actual=actual_cache_values,
            )

        for req in requests:
            req.add_new_token(seq_lens[req.py_request_id], 0)
        cache_manager.update_resources(scheduled_batch)

        # Clean up
        cache_manager.shutdown()

    @pytest.mark.parametrize("compress_ratios", [[1, 4, 128]])
    @pytest.mark.parametrize("dtype,compressor_dtype", [(DataType.BF16, DataType.FLOAT)])
    def test_layer_attn_to_pool_id(
        self, compress_ratios: List[int], dtype: DataType, compressor_dtype: DataType
    ):
        # Create cache manager and sparse attention config
        num_layers = len(compress_ratios)
        cache_manager, _ = self._create_mewtwo_cache_manager(
            tokens_per_block=self.tokens_per_block,
            compress_ratios=compress_ratios,
            dtype=dtype,
            compressor_dtype=compressor_dtype,
        )

        layer_attn_to_pool_id = cache_manager.layer_attn_to_pool_id
        assert layer_attn_to_pool_id.shape == (len(MewtwoAttentionType), num_layers)

        for layer in range(num_layers):
            attn_to_pool_id = layer_attn_to_pool_id[:, layer]
            assert attn_to_pool_id[MewtwoAttentionType.SWA.value] != -1, (
                f"layer {layer} should have SWA attention"
            )

            if self._is_compress_layer(compress_ratios[layer]):
                assert attn_to_pool_id[MewtwoAttentionType.COMPRESS.value] != -1, (
                    f"layer {layer} should have COMPRESS attention"
                )
                assert attn_to_pool_id[MewtwoAttentionType.COMPRESSOR_STATE.value] != -1, (
                    f"layer {layer} should have COMPRESSOR_STATE attention"
                )
                assert attn_to_pool_id[MewtwoAttentionType.COMPRESSOR_SCORE.value] != -1, (
                    f"layer {layer} should have COMPRESSOR_SCORE attention"
                )
            else:
                assert attn_to_pool_id[MewtwoAttentionType.COMPRESS.value] == -1, (
                    f"layer {layer} should not have COMPRESS attention"
                )
                assert attn_to_pool_id[MewtwoAttentionType.COMPRESSOR_STATE.value] == -1, (
                    f"layer {layer} should not have COMPRESSOR_STATE attention"
                )
                assert attn_to_pool_id[MewtwoAttentionType.COMPRESSOR_SCORE.value] == -1, (
                    f"layer {layer} should not have COMPRESSOR_SCORE attention"
                )

            if self._is_sparse_layer(compress_ratios[layer]):
                assert attn_to_pool_id[MewtwoAttentionType.INDEXER_COMPRESS.value] != -1, (
                    f"layer {layer} should have INDEXER_COMPRESS attention"
                )
                assert attn_to_pool_id[MewtwoAttentionType.INDEXER_COMPRESSOR_STATE.value] != -1, (
                    f"layer {layer} should have INDEXER_COMPRESSOR_STATE attention"
                )
                assert attn_to_pool_id[MewtwoAttentionType.INDEXER_COMPRESSOR_SCORE.value] != -1, (
                    f"layer {layer} should have INDEXER_COMPRESSOR_SCORE attention"
                )
            else:
                assert attn_to_pool_id[MewtwoAttentionType.INDEXER_COMPRESS.value] == -1, (
                    f"layer {layer} should not have INDEXER_COMPRESS attention"
                )
                assert attn_to_pool_id[MewtwoAttentionType.INDEXER_COMPRESSOR_STATE.value] == -1, (
                    f"layer {layer} should not have INDEXER_COMPRESSOR_STATE attention"
                )
                assert attn_to_pool_id[MewtwoAttentionType.INDEXER_COMPRESSOR_SCORE.value] == -1, (
                    f"layer {layer} should not have INDEXER_COMPRESSOR_SCORE attention"
                )

    @pytest.mark.parametrize("compress_ratios", [[1, 4, 128]])
    @pytest.mark.parametrize("dtype,compressor_dtype", [(DataType.BF16, DataType.FLOAT)])
    def test_layer_attn_to_buffer_ptr(
        self, compress_ratios: List[int], dtype: DataType, compressor_dtype: DataType
    ):
        # Create cache manager and sparse attention config
        num_layers = len(compress_ratios)
        cache_manager, _ = self._create_mewtwo_cache_manager(
            tokens_per_block=self.tokens_per_block,
            compress_ratios=compress_ratios,
            dtype=dtype,
            compressor_dtype=compressor_dtype,
        )

        layer_attn_to_buffer_ptr = cache_manager.layer_attn_to_buffer_ptr
        assert layer_attn_to_buffer_ptr.shape == (len(MewtwoAttentionType), num_layers)

        for layer in range(num_layers):
            attn_to_buffer_ptr = layer_attn_to_buffer_ptr[:, layer]
            assert attn_to_buffer_ptr[MewtwoAttentionType.SWA.value] != 0, (
                f"layer {layer} should have SWA buffer pointer"
            )

            if self._is_compress_layer(compress_ratios[layer]):
                assert attn_to_buffer_ptr[MewtwoAttentionType.COMPRESS.value] != 0, (
                    f"layer {layer} should have COMPRESS buffer pointer"
                )
                assert attn_to_buffer_ptr[MewtwoAttentionType.COMPRESSOR_STATE.value] != 0, (
                    f"layer {layer} should have COMPRESSOR_STATE buffer pointer"
                )
                assert attn_to_buffer_ptr[MewtwoAttentionType.COMPRESSOR_SCORE.value] != 0, (
                    f"layer {layer} should have COMPRESSOR_SCORE buffer pointer"
                )
            else:
                assert attn_to_buffer_ptr[MewtwoAttentionType.COMPRESS.value] == 0, (
                    f"layer {layer} should not have COMPRESS buffer pointer"
                )
                assert attn_to_buffer_ptr[MewtwoAttentionType.COMPRESSOR_STATE.value] == 0, (
                    f"layer {layer} should not have COMPRESSOR_STATE buffer pointer"
                )
                assert attn_to_buffer_ptr[MewtwoAttentionType.COMPRESSOR_SCORE.value] == 0, (
                    f"layer {layer} should not have COMPRESSOR_SCORE buffer pointer"
                )

            if self._is_sparse_layer(compress_ratios[layer]):
                assert attn_to_buffer_ptr[MewtwoAttentionType.INDEXER_COMPRESS.value] != 0, (
                    f"layer {layer} should have INDEXER_COMPRESS buffer pointer"
                )
                assert (
                    attn_to_buffer_ptr[MewtwoAttentionType.INDEXER_COMPRESSOR_STATE.value] != 0
                ), f"layer {layer} should have INDEXER_COMPRESSOR_STATE buffer pointer"
                assert (
                    attn_to_buffer_ptr[MewtwoAttentionType.INDEXER_COMPRESSOR_SCORE.value] != 0
                ), f"layer {layer} should have INDEXER_COMPRESSOR_SCORE buffer pointer"
            else:
                assert attn_to_buffer_ptr[MewtwoAttentionType.INDEXER_COMPRESS.value] == 0, (
                    f"layer {layer} should not have INDEXER_COMPRESS buffer pointer"
                )
                assert (
                    attn_to_buffer_ptr[MewtwoAttentionType.INDEXER_COMPRESSOR_STATE.value] == 0
                ), f"layer {layer} should not have INDEXER_COMPRESSOR_STATE buffer pointer"
                assert (
                    attn_to_buffer_ptr[MewtwoAttentionType.INDEXER_COMPRESSOR_SCORE.value] == 0
                ), f"layer {layer} should not have INDEXER_COMPRESSOR_SCORE buffer pointer"

    @pytest.mark.parametrize("compress_ratios", [[1, 4, 128]])
    @pytest.mark.parametrize("dtype,compressor_dtype", [(DataType.BF16, DataType.FLOAT)])
    def test_layer_attn_to_pool_ptr(
        self, compress_ratios: List[int], dtype: DataType, compressor_dtype: DataType
    ):
        # Create cache manager and sparse attention config
        num_layers = len(compress_ratios)
        cache_manager, _ = self._create_mewtwo_cache_manager(
            tokens_per_block=self.tokens_per_block,
            compress_ratios=compress_ratios,
            dtype=dtype,
            compressor_dtype=compressor_dtype,
        )

        layer_attn_to_pool_ptr = cache_manager.layer_attn_to_pool_ptr
        assert layer_attn_to_pool_ptr.shape == (len(MewtwoAttentionType), num_layers)

        for layer in range(num_layers):
            attn_to_pool_ptr = layer_attn_to_pool_ptr[:, layer]
            assert attn_to_pool_ptr[MewtwoAttentionType.SWA.value] != 0, (
                f"layer {layer} should have SWA pool pointer"
            )
            if self._is_compress_layer(compress_ratios[layer]):
                assert attn_to_pool_ptr[MewtwoAttentionType.COMPRESS.value] != 0, (
                    f"layer {layer} should have COMPRESS pool pointer"
                )
                assert attn_to_pool_ptr[MewtwoAttentionType.COMPRESSOR_STATE.value] != 0, (
                    f"layer {layer} should have COMPRESSOR_STATE pool pointer"
                )
                assert attn_to_pool_ptr[MewtwoAttentionType.COMPRESSOR_SCORE.value] != 0, (
                    f"layer {layer} should have COMPRESSOR_SCORE pool pointer"
                )
            else:
                assert attn_to_pool_ptr[MewtwoAttentionType.COMPRESS.value] == 0, (
                    f"layer {layer} should not have COMPRESS pool pointer"
                )
                assert attn_to_pool_ptr[MewtwoAttentionType.COMPRESSOR_STATE.value] == 0, (
                    f"layer {layer} should not have COMPRESSOR_STATE pool pointer"
                )
                assert attn_to_pool_ptr[MewtwoAttentionType.COMPRESSOR_SCORE.value] == 0, (
                    f"layer {layer} should not have COMPRESSOR_SCORE pool pointer"
                )

            if self._is_sparse_layer(compress_ratios[layer]):
                assert attn_to_pool_ptr[MewtwoAttentionType.INDEXER_COMPRESS.value] != 0, (
                    f"layer {layer} should have INDEXER_COMPRESS pool pointer"
                )
                assert attn_to_pool_ptr[MewtwoAttentionType.INDEXER_COMPRESSOR_STATE.value] != 0, (
                    f"layer {layer} should have INDEXER_COMPRESSOR_STATE pool pointer"
                )
                assert attn_to_pool_ptr[MewtwoAttentionType.INDEXER_COMPRESSOR_SCORE.value] != 0, (
                    f"layer {layer} should have INDEXER_COMPRESSOR_SCORE pool pointer"
                )
            else:
                assert attn_to_pool_ptr[MewtwoAttentionType.INDEXER_COMPRESS.value] == 0, (
                    f"layer {layer} should not have INDEXER_COMPRESS pool pointer"
                )
                assert attn_to_pool_ptr[MewtwoAttentionType.INDEXER_COMPRESSOR_STATE.value] == 0, (
                    f"layer {layer} should not have INDEXER_COMPRESSOR_STATE pool pointer"
                )
                assert attn_to_pool_ptr[MewtwoAttentionType.INDEXER_COMPRESSOR_SCORE.value] == 0, (
                    f"layer {layer} should not have INDEXER_COMPRESSOR_SCORE pool pointer"
                )

    @pytest.mark.parametrize("compress_ratios", [[1, 4, 128]])
    @pytest.mark.parametrize("dtype,compressor_dtype", [(DataType.BF16, DataType.FLOAT)])
    def test_kv_cache_pool_pointers(
        self, compress_ratios: List[int], dtype: DataType, compressor_dtype: DataType
    ):
        # Create cache manager and sparse attention config
        cache_manager, _ = self._create_mewtwo_cache_manager(
            tokens_per_block=self.tokens_per_block,
            compress_ratios=compress_ratios,
            dtype=dtype,
            compressor_dtype=compressor_dtype,
        )

        kv_cache_pool_pointers = cache_manager.kv_cache_pool_pointers
        assert kv_cache_pool_pointers.shape == (cache_manager.num_pools, 2)

        # all pool pointers should be non-zero
        assert torch.all(kv_cache_pool_pointers[:, 0] != 0), "all pool pointers should be non-zero"
        # Mewtwo doesn't have value cache, so the second column should be 0
        assert torch.all(kv_cache_pool_pointers[:, 1] == 0), "the second column should be 0"

    @pytest.mark.parametrize("compress_ratios", [[1, 4, 128]])
    @pytest.mark.parametrize("dtype,compressor_dtype", [(DataType.BF16, DataType.FLOAT)])
    def test_kv_cache_pool_mapping(
        self, compress_ratios: List[int], dtype: DataType, compressor_dtype: DataType
    ):
        # Create cache manager and sparse attention config
        num_layers = len(compress_ratios)
        cache_manager, _ = self._create_mewtwo_cache_manager(
            tokens_per_block=self.tokens_per_block,
            compress_ratios=compress_ratios,
            dtype=dtype,
            compressor_dtype=compressor_dtype,
        )

        kv_cache_pool_mapping = cache_manager.kv_cache_pool_mapping
        assert kv_cache_pool_mapping.shape == (num_layers, 2)

        assert torch.all(kv_cache_pool_mapping[:, 0] != -1), (
            "all layers should have swa attention pool"
        )
        assert torch.all(kv_cache_pool_mapping[:, 1] >= 0), (
            "buffer pointer offset should be non-negative"
        )
        assert torch.all(kv_cache_pool_mapping[:, 0] == kv_cache_pool_mapping[0, 0]), (
            "all layers should have the same pool_id"
        )
