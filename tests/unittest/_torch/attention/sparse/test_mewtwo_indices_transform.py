"""
Tests for Mewtwo Index Transform Kernel.
"""

from dataclasses import dataclass
from typing import List

import pytest
import torch

from tensorrt_llm._torch.attention_backend.sparse.kernel import mewtwo_local_to_global_indices
from tensorrt_llm._torch.attention_backend.sparse.mewtwo import (
    MewtwoAttentionType,
    MewtwoCacheManager,
)
from tensorrt_llm._torch.attention_backend.sparse.mewtwo.mewtwo import get_token_bytes
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import DataType, SamplingConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MewtwoSparseAttentionConfig
from tensorrt_llm.mapping import Mapping


@dataclass(kw_only=True, frozen=True)
class Scenario:
    """Test scenario configuration."""

    layer_idx: int = 0
    head_dim: int = 512
    index_head_dim: int = 128
    window_size: int = 128
    vocab_size: int = 129280
    tokens_per_block: int = 128
    max_batch_size: int = 16
    max_seq_len: int = 2048
    dtype: DataType = DataType.BF16
    compressor_dtype: DataType = DataType.FLOAT

    compress_ratio: int = 1
    swa_topk: int = 128
    compressed_topk: int = 512
    compressed_attn_type: MewtwoAttentionType = MewtwoAttentionType.COMPRESS


scenarios = [
    Scenario(compress_ratio=1, swa_topk=128, compressed_topk=0, layer_idx=0),
    Scenario(compress_ratio=4, swa_topk=128, compressed_topk=512, layer_idx=2),
    # Set compressed_topk to 2048 to test all compressed tokens
    # It's not a realistic scenario, but it's helpful to test all compressed tokens.
    Scenario(compress_ratio=128, swa_topk=128, compressed_topk=2048, layer_idx=3),
]

batch_configs = [
    [256, 192],
    [128, 233, 876],
    [1158],
]


def _create_cache_manager(scenario: Scenario, num_layers: int = 1):
    """Create a MewtwoCacheManager for testing."""
    base_ratios = [1, 4, 128]
    compress_ratios = [base_ratios[i % len(base_ratios)] for i in range(num_layers)]
    if scenario.layer_idx < num_layers:
        compress_ratios[scenario.layer_idx] = scenario.compress_ratio
    sparse_attn_config = MewtwoSparseAttentionConfig(
        index_head_dim=scenario.index_head_dim,
        window_size=scenario.window_size,
        compress_ratios=compress_ratios,
    )

    max_tokens = scenario.max_seq_len * scenario.max_batch_size
    cache_manager = MewtwoCacheManager(
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=False,
            max_tokens=max_tokens,
            event_buffer_max_size=0,
        ),
        kv_cache_type=CacheTypeCpp.SELFKONLY,
        num_layers=num_layers,
        num_kv_heads=1,
        head_dim=scenario.head_dim,
        tokens_per_block=scenario.tokens_per_block,
        max_seq_len=scenario.max_seq_len,
        max_batch_size=scenario.max_batch_size,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=scenario.dtype,
        compressor_dtype=scenario.compressor_dtype,
        vocab_size=scenario.vocab_size,
        max_num_tokens=max_tokens,
        sparse_attn_config=sparse_attn_config,
    )
    return cache_manager


def _local_to_physical_idx(local_idx: int, block_table: List[int], tokens_per_block: int) -> int:
    """
    Convert local token index to physical buffer index using block table.
    This simulates: buffer_ptr + block_table[block_idx] * tokens_per_block + token_in_block
    """
    block_idx = local_idx // tokens_per_block
    token_in_block = local_idx % tokens_per_block
    page_idx = block_table[block_idx]
    return page_idx * tokens_per_block + token_in_block


def _build_swa_indices(token_pos: int, window_size: int, swa_topk: int, device) -> torch.Tensor:
    """Build SWA local indices for a token at position token_pos."""
    indices = torch.full((swa_topk,), -1, dtype=torch.int32, device=device)
    if token_pos < window_size:
        indices[: token_pos + 1] = torch.arange(token_pos + 1, dtype=torch.int32, device=device)
    else:
        indices[:window_size] = torch.arange(
            token_pos - window_size + 1, token_pos + 1, dtype=torch.int32, device=device
        )
    return indices


def _build_compressed_indices(
    token_pos: int, compress_ratio: int, compressed_topk: int, device
) -> torch.Tensor:
    """Build compressed local indices for a token at position token_pos."""
    indices = torch.full((compressed_topk,), -1, dtype=torch.int32, device=device)
    num_valid = (token_pos + 1) // compress_ratio
    if compress_ratio == 128:
        indices[:num_valid] = torch.arange(num_valid, dtype=torch.int32, device=device)
    else:
        select_count = min(compressed_topk, num_valid)
        indices[:select_count] = torch.randperm(num_valid, device=device)[:select_count].to(
            torch.int32
        )
    return indices


def _run_test(scenario: Scenario, context_lengths: List[int]):
    """Run index transformation test."""
    device = torch.device("cuda")
    layer_idx = scenario.layer_idx
    has_compressed = scenario.compress_ratio > 1
    total_tokens = sum(context_lengths)

    torch.manual_seed(42)

    # Create cache manager and requests
    cache_manager = _create_cache_manager(scenario, num_layers=7)
    requests = [
        LlmRequest(
            request_id=i,
            max_new_tokens=1024,
            input_tokens=list(range(ctx_len)),
            sampling_config=SamplingConfig(),
            is_streaming=False,
        )
        for i, ctx_len in enumerate(context_lengths)
    ]

    scheduled_batch = ScheduledRequests()
    scheduled_batch.context_requests = requests
    cache_manager.prepare_resources(scheduled_batch)

    # Get pointers and offsets
    swa_pool_ptr = cache_manager.swa_pool_ptr
    swa_buffer_ptr = cache_manager.get_buffers(layer_idx, MewtwoAttentionType.SWA).data_ptr()
    # Use min(swa_pool_ptr, compressed_pool_ptr) as base to ensure non-negative indices
    sparse_mla_base_ptr = swa_pool_ptr
    if has_compressed and scenario.compress_ratio in cache_manager.compress_pool_ptrs:
        sparse_mla_base_ptr = min(
            swa_pool_ptr, cache_manager.compress_pool_ptrs[scenario.compress_ratio]
        )

    # Single token stride for all buffers
    has_fp8_kv_cache = scenario.dtype == DataType.FP8
    token_stride = get_token_bytes(
        scenario.head_dim,
        scenario.index_head_dim,
        scenario.compress_ratio,
        MewtwoAttentionType.SWA,
        has_fp8_kv_cache,
    )
    swa_offset = (swa_buffer_ptr - sparse_mla_base_ptr) // token_stride

    # Get compressed buffer type from scenario property
    compressed_attn_type = scenario.compressed_attn_type

    if has_compressed:
        compressed_buffer_ptr = cache_manager.get_buffers(
            layer_idx, compressed_attn_type
        ).data_ptr()
        compressed_offset = (compressed_buffer_ptr - sparse_mla_base_ptr) // token_stride
        tokens_per_block_compressed = scenario.tokens_per_block // scenario.compress_ratio
    else:
        compressed_buffer_ptr, compressed_offset, tokens_per_block_compressed = (
            0,
            0,
            scenario.tokens_per_block,
        )

    # Get buffers and write random values
    swa_buffer = cache_manager.get_buffers(layer_idx, MewtwoAttentionType.SWA)
    swa_buffer.copy_(torch.randn_like(swa_buffer))

    if has_compressed:
        compressed_buffer = cache_manager.get_buffers(layer_idx, compressed_attn_type)
        # Generate random data in float32 first, then convert to target dtype (for FP8 support)
        random_data = torch.randn(compressed_buffer.shape, dtype=torch.float32, device=device)
        compressed_buffer.copy_(random_data.to(compressed_buffer.dtype))

    # Get block tables
    block_tables_swa = [
        cache_manager.get_cache_indices(req.py_request_id, layer_idx, MewtwoAttentionType.SWA)
        for req in requests
    ]
    block_tables_compressed = (
        [
            cache_manager.get_cache_indices(req.py_request_id, layer_idx, compressed_attn_type)
            for req in requests
        ]
        if has_compressed
        else []
    )

    # Flatten buffers for access
    swa_buffer_flat = swa_buffer.view(-1, scenario.head_dim)
    if has_compressed:
        compressed_buffer_flat = compressed_buffer.view(-1, scenario.head_dim)

    # Pad and convert block tables to tensors
    max_blocks_swa = max(len(bt) for bt in block_tables_swa)
    block_table_swa_t = torch.tensor(
        [bt + [-1] * (max_blocks_swa - len(bt)) for bt in block_tables_swa],
        dtype=torch.int32,
        device=device,
    )

    if has_compressed:
        max_blocks_compressed = max(len(bt) for bt in block_tables_compressed)
        block_table_compressed_t = torch.tensor(
            [bt + [-1] * (max_blocks_compressed - len(bt)) for bt in block_tables_compressed],
            dtype=torch.int32,
            device=device,
        )
    else:
        block_table_compressed_t = None

    # Build inputs for all tokens
    req_ids, token_positions = [], []
    for r, ctx_len in enumerate(context_lengths):
        for pos in range(ctx_len):
            req_ids.append(r)
            token_positions.append(pos)

    req_id = torch.tensor(req_ids, dtype=torch.int32, device=device)

    # Build local indices
    swa_local_indices = torch.stack(
        [
            _build_swa_indices(pos, scenario.window_size, scenario.swa_topk, device)
            for pos in token_positions
        ]
    )

    if has_compressed:
        compressed_local_indices = torch.stack(
            [
                _build_compressed_indices(
                    pos, scenario.compress_ratio, scenario.compressed_topk, device
                )
                for pos in token_positions
            ]
        )
    else:
        compressed_local_indices = None

    # Run kernel
    global_indices = mewtwo_local_to_global_indices(
        req_id=req_id,
        block_table_swa=block_table_swa_t,
        swa_local_indices=swa_local_indices,
        sparse_mla_base_ptr=sparse_mla_base_ptr,
        swa_buffer_ptr=swa_buffer_ptr,
        tokens_per_block=scenario.tokens_per_block,
        token_stride=token_stride,
        block_table_compressed=block_table_compressed_t,
        compressed_local_indices=compressed_local_indices,
        compressed_buffer_ptr=compressed_buffer_ptr,
        compress_ratio=scenario.compress_ratio,
        num_compressed_indices=scenario.compressed_topk if has_compressed else 0,
    )

    # Verify values match using realistic access patterns:
    # - Global: pool_ptr + global_idx
    # - Local: buffer_ptr + block_table[block_idx] * tokens_per_block + token_in_block
    # Output is compact: [valid_swa..., valid_compressed..., -1, -1, ...]
    num_samples = min(32, total_tokens)
    sample_indices = torch.randperm(total_tokens)[:num_samples].tolist()

    for t in sample_indices:
        r = req_ids[t]

        # Count valid SWA and compressed indices from input
        valid_swa_indices = [
            i for i in range(scenario.swa_topk) if swa_local_indices[t, i].item() >= 0
        ]
        valid_swa_count = len(valid_swa_indices)

        if has_compressed:
            valid_compressed_indices = [
                i
                for i in range(scenario.compressed_topk)
                if compressed_local_indices[t, i].item() >= 0
            ]
            valid_compressed_count = len(valid_compressed_indices)
        else:
            valid_compressed_indices = []
            valid_compressed_count = 0

        total_valid = valid_swa_count + valid_compressed_count

        # Verify output is compact: first total_valid positions are valid, rest are -1
        out_row = global_indices[t]
        for out_pos in range(out_row.shape[0]):
            global_idx = out_row[out_pos].item()
            if out_pos >= total_valid:
                assert global_idx == -1, f"Token {t} out[{out_pos}]: expected -1, got {global_idx}"
            else:
                assert global_idx != -1, (
                    f"Token {t} out[{out_pos}]: expected non-negative index, got -1"
                )

        # Verify SWA values at compact positions [0, valid_swa_count)
        for out_pos, input_pos in enumerate(valid_swa_indices):
            local_idx = swa_local_indices[t, input_pos].item()
            global_idx = global_indices[t, out_pos].item()

            # Access via global index: pool_ptr + global_idx
            pool_based_idx = global_idx - swa_offset
            actual = swa_buffer_flat[pool_based_idx]

            # Access via local index: buffer_ptr + block_table lookup
            physical_idx = _local_to_physical_idx(
                local_idx, block_tables_swa[r], scenario.tokens_per_block
            )
            expected = swa_buffer_flat[physical_idx]

            torch.testing.assert_close(
                actual, expected, msg=f"Token {t} SWA out[{out_pos}]: value mismatch"
            )

        # Verify compressed values at compact positions [valid_swa_count, total_valid)
        if has_compressed:
            for i, input_pos in enumerate(valid_compressed_indices):
                out_pos = valid_swa_count + i
                local_idx = compressed_local_indices[t, input_pos].item()
                global_idx = global_indices[t, out_pos].item()

                # Access via global index: pool_ptr + global_idx
                pool_based_idx = global_idx - compressed_offset
                actual = compressed_buffer_flat[pool_based_idx]

                # Access via local index: buffer_ptr + block_table lookup
                physical_idx = _local_to_physical_idx(
                    local_idx, block_tables_compressed[r], tokens_per_block_compressed
                )
                expected = compressed_buffer_flat[physical_idx]

                torch.testing.assert_close(
                    actual, expected, msg=f"Token {t} compressed out[{out_pos}]: value mismatch"
                )

    # Cleanup
    for req, ctx_len in zip(requests, context_lengths):
        req.context_current_position = ctx_len
        req.add_new_token(ctx_len, 0)
    cache_manager.update_resources(scheduled_batch)
    cache_manager.shutdown()


@pytest.mark.parametrize("scenario", scenarios, ids=lambda x: f"compress={x.compress_ratio}")
@pytest.mark.parametrize("context_lengths", batch_configs, ids=lambda x: f"batch={len(x)}")
def test_mewtwo_indices_transform(scenario: Scenario, context_lengths: List[int]):
    """Test Mewtwo index transformation kernel by verifying VALUES."""
    _run_test(scenario, context_lengths)


if __name__ == "__main__":
    _run_test(scenarios[1], [256, 192])
    print("PASSED")
