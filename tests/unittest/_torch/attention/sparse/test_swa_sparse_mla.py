"""
Tests for sliding window attention with sparse MLA.
Currently, we only focus on validating KV cache contents with sliding window behavior.
"""

import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import List, Optional, Tuple

import pytest
import torch

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionInputType,
    MLAParams,
    PositionalEmbeddingParams,
    RopeParams,
)
from tensorrt_llm._torch.attention_backend.sparse.dsa import DSACacheManager
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._utils import get_sm_version, str_dtype_to_binding, torch_dtype_to_str
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.functional import PositionEmbeddingType, RopeEmbeddingUtils
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope_to_k_pe(
    k_pe: torch.Tensor, rope_cos_sin: torch.Tensor, sequence_lengths: List[int]
) -> torch.Tensor:
    """
    Apply RoPE to k_pe tensor.

    Args:
        k_pe: k_pe tensor with shape [total_tokens, qk_rope_head_dim]
        rope_cos_sin: RoPE cos/sin cache
        sequence_lengths: List of sequence lengths (tokens are processed sequentially from position 0)

    Returns:
        Rotated k_pe tensor with same shape as input
    """
    k_pe_ref_list = []
    total_tokens = 0
    for seq_len in sequence_lengths:
        k_pe_seq = k_pe[total_tokens : total_tokens + seq_len].unsqueeze(-2)
        cos, sin = rope_cos_sin[:seq_len].chunk(2, dim=-2)
        k_pe_seq = k_pe_seq.unflatten(-1, [-1, 2]).transpose(-2, -1).flatten(start_dim=-2)
        k_pe_seq = ((k_pe_seq * cos) + (rotate_half(k_pe_seq) * sin)).to(dtype=k_pe_seq.dtype)
        k_pe_seq = k_pe_seq.unflatten(-1, [2, -1]).transpose(-2, -1).flatten(start_dim=-2)
        k_pe_ref_list.append(k_pe_seq)
        total_tokens += seq_len
    return torch.cat(k_pe_ref_list).squeeze(-2)


@dataclass(kw_only=True, frozen=True)
class Scenario:
    """Test scenario configuration."""

    dtype: torch.dtype = torch.bfloat16
    kv_cache_dtype: torch.dtype = torch.bfloat16
    num_layers: int = 1
    num_heads: int = 128
    num_kv_heads: int = 1
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 512
    rope_append: bool = True
    hidden_size: int = 7168
    max_position_embeddings: int = 163840
    rope_theta: float = 10000.0
    rope_beta_fast: int = 32
    rope_beta_slow: int = 1
    rope_factor: float = 40.0
    rope_mscale: float = 1.0
    rope_mscale_all_dim: float = 1.0
    rope_original_max_position_embeddings: int = 4096
    rope_type: str = "yarn"
    model_type: str = "deepseek_v3"
    kv_cache_tokens_per_block: int = 64


@dataclass(kw_only=True, frozen=True)
class RopeConfig:
    """RoPE configuration for tests."""

    hidden_size: int = 7168
    num_attention_heads: int = 128
    rope_scaling: dict = field(
        default_factory=lambda: {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 40.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
            "type": "yarn",
        }
    )
    max_position_embeddings: int = 163840
    rope_theta: float = 10000.0
    qk_rope_head_dim: int = 64
    model_type: str = "deepseek_v3"


def extract_kv_cache_from_blocks(
    kv_cache_manager: DSACacheManager,
    seq_idx: int,
    cache_seq_len: int,
    layer_idx: int = 0,
    request_ids: Optional[List[int]] = None,
    total_seq_len: Optional[int] = None,
    window_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Extract KV cache contents from paged blocks for validation.
    Returns: [cache_seq_len, head_dim] tensor
    """
    if request_ids is None:
        request_ids = [seq_idx]
    block_ids_per_seq = kv_cache_manager.get_batch_cache_indices(request_ids)
    block_ids = block_ids_per_seq[0]

    cache_buf = kv_cache_manager.get_buffers(layer_idx, kv_layout="NHD")
    if cache_buf is None:
        raise RuntimeError("Failed to get KV cache buffer")

    cached_data = torch.concat(cache_buf[block_ids, :].unbind(dim=0), dim=1)

    # Calculate the start offset for window extraction
    # When total_seq_len > window_size, the window starts at (total_seq_len - window_size)
    if total_seq_len is not None and window_size is not None and total_seq_len > window_size:
        start_offset = total_seq_len - window_size
        print(f"  Extracting window: offset={start_offset}, len={cache_seq_len}")
        cached_data = cached_data[0, start_offset : start_offset + cache_seq_len, 0, :]
    else:
        cached_data = cached_data[0, :cache_seq_len, 0, :]

    return cached_data


def validate_sliding_window_contents(
    extracted_cache: torch.Tensor,
    reference_k_nope: torch.Tensor,
    reference_k_pe_rotated: torch.Tensor,
    window_size: int,
    kv_lora_rank: int = 512,
    atol: float = 0.1,
    rtol: float = 0.01,
):
    """Validate both k_nope and rotated k_pe parts."""
    total_logical_tokens = reference_k_nope.shape[0]

    print("\nValidating sliding window:")
    print(f"  Total logical tokens: {total_logical_tokens}")
    print(f"  Window size: {window_size}")
    print(f"  Extracted cache shape: {extracted_cache.shape}")

    if total_logical_tokens <= window_size:
        # All tokens fit in window - validate at natural positions
        for i in range(total_logical_tokens):
            # Validate k_nope
            torch.testing.assert_close(
                extracted_cache[i, :kv_lora_rank],
                reference_k_nope[i],
                atol=atol,
                rtol=rtol,
                msg=f"Token {i} k_nope mismatch (seq fits in window)",
            )
            # Validate rotated k_pe
            torch.testing.assert_close(
                extracted_cache[i, kv_lora_rank:],
                reference_k_pe_rotated[i],
                atol=atol,
                rtol=rtol,
                msg=f"Token {i} k_pe mismatch (seq fits in window)",
            )
        print(f"  All {total_logical_tokens} tokens validated (k_nope + k_pe)")
    else:
        # Only last window_size tokens remain
        recent_start = total_logical_tokens - window_size
        print(f"  Most recent tokens: [{recent_start}:{total_logical_tokens}]")

        for i in range(window_size):
            logical_idx = recent_start + i
            physical_pos = i
            # Validate k_nope
            torch.testing.assert_close(
                extracted_cache[physical_pos, :kv_lora_rank],
                reference_k_nope[logical_idx],
                atol=atol,
                rtol=rtol,
                msg=f"Token {logical_idx} k_nope mismatch at pos {physical_pos}",
            )
            # Validate rotated k_pe
            torch.testing.assert_close(
                extracted_cache[physical_pos, kv_lora_rank:],
                reference_k_pe_rotated[logical_idx],
                atol=atol,
                rtol=rtol,
                msg=f"Token {logical_idx} k_pe mismatch at pos {physical_pos}",
            )
        print(f"  All {window_size} tokens validated (k_nope + k_pe)")


def _allocate_kv_cache_for_generation(kv_cache_manager, request_ids, num_tokens: int):
    """Allocate KV cache blocks for generation phase."""
    for request_id in request_ids:
        for _ in range(num_tokens):
            kv_cache_manager.impl.add_token(request_id)


def _build_sparse_topk_indices(seq_len: int, topk: int, device) -> torch.Tensor:
    """Build topk indices for sparse attention."""
    topk_indices = torch.full((seq_len, topk), -1, dtype=torch.int32, device=device)
    for i in range(seq_len):
        valid_len = min(i + 1, topk)
        topk_indices[i, :valid_len] = torch.randperm(i + 1, device=device)[:valid_len].to(
            torch.int32
        )
    return topk_indices


def _prepare_generation_seqlens(
    num_seqs: int,
    generation_seq_len_q: int,
    cached_lens: List[int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare sequence length tensors for mla_rope_generation."""

    # cu_q_seqlens: [0, gen_len, 2*gen_len, ...]
    q_lens = [generation_seq_len_q] * num_seqs
    cu_q_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(q_lens), dim=0).tolist()),
        dtype=torch.int32,
        device=device,
    )

    # cu_kv_seqlens: cumulative sum of (cached_len + gen_len) for each sequence
    kv_lens = [cached_len + generation_seq_len_q for cached_len in cached_lens]
    cu_kv_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(kv_lens), dim=0).tolist()),
        dtype=torch.int32,
        device=device,
    )

    # fmha_scheduler_counter: initialized to 0
    fmha_scheduler_counter = torch.zeros(1, dtype=torch.uint32, device=device)

    return cu_q_seqlens, cu_kv_seqlens, fmha_scheduler_counter


# Test scenarios
scenarios = [
    Scenario(rope_append=True, num_heads=128, kv_lora_rank=512),
    Scenario(rope_append=False, num_heads=64, kv_lora_rank=448),
]

window_sizes = [128]

context_sequence_lengths_list = [[10], [256], [50, 400, 1500], [3510, 20]]

num_generation_steps_list = [0, 3]

SPARSE_TOPK = 2048


def _run_swa_test(
    scenario: Scenario,
    context_sequence_lengths: List[int],
    window_size: int,
    num_generation_steps: int,
    generation_seq_len_q: int = 1,
):
    """Run sliding window attention test for sparse MLA."""
    device = torch.device("cuda")
    dtype = scenario.dtype
    kv_cache_dtype = scenario.kv_cache_dtype
    num_heads = scenario.num_heads
    num_kv_heads = scenario.num_kv_heads
    kv_lora_rank = scenario.kv_lora_rank
    qk_rope_head_dim = scenario.qk_rope_head_dim
    qk_nope_head_dim = scenario.qk_nope_head_dim
    v_head_dim = scenario.v_head_dim
    rope_append = scenario.rope_append
    kv_cache_tokens_per_block = scenario.kv_cache_tokens_per_block
    num_layers = scenario.num_layers

    if rope_append is False:
        num_heads = 64
    head_dim = kv_lora_rank + qk_rope_head_dim

    # Setup RoPE config
    rope_config = RopeConfig(
        hidden_size=scenario.hidden_size,
        num_attention_heads=scenario.num_heads,
        rope_scaling={
            "beta_fast": scenario.rope_beta_fast,
            "beta_slow": scenario.rope_beta_slow,
            "factor": scenario.rope_factor,
            "mscale": scenario.rope_mscale,
            "mscale_all_dim": scenario.rope_mscale_all_dim,
            "original_max_position_embeddings": scenario.rope_original_max_position_embeddings,
            "type": scenario.rope_type,
        },
        max_position_embeddings=scenario.max_position_embeddings,
        rope_theta=scenario.rope_theta,
        qk_rope_head_dim=scenario.qk_rope_head_dim,
        model_type=scenario.model_type,
    )

    # Sparse config
    sparse_config = DeepSeekSparseAttentionConfig(
        index_n_heads=64,
        index_head_dim=128,
        index_topk=SPARSE_TOPK,
        skip_indexer_for_short_seqs=False,
    )

    # Attention backend
    AttentionCls = get_attention_backend("TRTLLM", sparse_config)

    # MLA params
    pos_embd_params = PositionalEmbeddingParams(
        type=PositionEmbeddingType.yarn,
        rope=RopeParams.from_config(rope_config),
        is_neox=False,
    )
    mla_params = MLAParams(
        q_lora_rank=scenario.q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        qk_nope_head_dim=qk_nope_head_dim,
        v_head_dim=v_head_dim,
        rope_append=rope_append,
        predicted_tokens_per_seq=1,
    )

    def yarn_get_mscale(scale=1, mscale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    mscale_all_dim = pos_embd_params.rope.mscale_all_dim
    scaling_factor = pos_embd_params.rope.scale
    mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
    q_scaling = 1.0 / (mscale * mscale)

    quant_config = None
    if kv_cache_dtype == torch.float8_e4m3fn:
        quant_config = QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8.value)

    # Create attention layer
    attention_layer = AttentionCls(
        layer_idx=0,
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        quant_config=quant_config,
        q_scaling=q_scaling,
        pos_embd_params=pos_embd_params,
        mla_params=mla_params,
        sparse_attention_config=sparse_config,
    )

    # Generate rope_cos_sin for reference RoPE calculation
    rope_cos_sin = (
        torch.tensor(
            RopeEmbeddingUtils.create_sinusoidal_positions_yarn(
                rope_config.max_position_embeddings,
                rope_config.qk_rope_head_dim,
                rope_config.rope_theta,
                rope_config.rope_scaling["factor"],
                rope_config.rope_scaling["original_max_position_embeddings"],
                rope_config.rope_scaling["beta_fast"],
                rope_config.rope_scaling["beta_slow"],
                rope_config.rope_scaling["mscale"],
                rope_config.rope_scaling["mscale_all_dim"],
            )[1],
            dtype=torch.float32,
            device=device,
        )
        .reshape(rope_config.max_position_embeddings, -1, 2)
        .transpose(-2, -1)
    )

    # KV cache setup
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    max_context_seq_len = max(context_sequence_lengths)
    max_seq_len = max_context_seq_len + (num_generation_steps + 1) * generation_seq_len_q
    num_seqs = len(context_sequence_lengths)
    max_tokens = (
        (max_seq_len + kv_cache_tokens_per_block - 1)
        // kv_cache_tokens_per_block
        * kv_cache_tokens_per_block
        * num_seqs
    )

    # Create model config for DSACacheManager
    pretrained_config = SimpleNamespace(
        rms_norm_eps=1e-6,
    )
    model_config = ModelConfig(
        mapping=mapping,
        sparse_attention_config=sparse_config,
        pretrained_config=pretrained_config,
    )

    # Use DSACacheManager with sliding window
    kv_cache_manager = DSACacheManager(
        KvCacheConfig(
            max_tokens=max_tokens,
            enable_block_reuse=False,
            max_attention_window=[window_size],
        ),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=num_layers,
        num_kv_heads=1,
        head_dim=head_dim,
        tokens_per_block=kv_cache_tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=num_seqs,
        mapping=mapping,
        dtype=str_dtype_to_binding(torch_dtype_to_str(kv_cache_dtype)),
        sparse_attn_config=sparse_config,
        model_config=model_config,
    )

    # Create inputs
    total_ctx_tokens = sum(context_sequence_lengths)
    total_gen_tokens = num_seqs * num_generation_steps * generation_seq_len_q
    total_tokens = total_ctx_tokens + total_gen_tokens

    torch.manual_seed(42)
    all_k_nope = torch.randn(total_tokens, kv_lora_rank, dtype=dtype, device=device)
    all_k_pe = torch.randn(total_tokens, qk_rope_head_dim, dtype=dtype, device=device)

    # Store reference k_nope and k_pe separately per sequence
    reference_k_nope_per_seq = []
    reference_k_pe_per_seq = []
    ctx_offset = 0
    for seq_idx, ctx_len in enumerate(context_sequence_lengths):
        seq_k_nope = []
        seq_k_pe = []
        # Add context tokens for this sequence
        for i in range(ctx_len):
            seq_k_nope.append(all_k_nope[ctx_offset + i])
            seq_k_pe.append(all_k_pe[ctx_offset + i])
        ctx_offset += ctx_len
        # Add generation tokens for this sequence
        # Generation tokens are at: total_ctx + step * num_seqs + seq_idx
        for step in range(num_generation_steps):
            for q in range(generation_seq_len_q):
                gen_idx = (
                    total_ctx_tokens
                    + step * num_seqs * generation_seq_len_q
                    + seq_idx * generation_seq_len_q
                    + q
                )
                seq_k_nope.append(all_k_nope[gen_idx])
                seq_k_pe.append(all_k_pe[gen_idx])
        reference_k_nope_per_seq.append(torch.stack(seq_k_nope))
        reference_k_pe_per_seq.append(torch.stack(seq_k_pe))

    request_ids = list(range(num_seqs))
    kv_cache_manager.add_dummy_requests(request_ids, context_sequence_lengths)

    # ===== Context Phase =====
    print("\n--- Context Phase ---")
    ctx_k_nope = all_k_nope[:total_ctx_tokens]
    ctx_k_pe = all_k_pe[:total_ctx_tokens]
    ctx_q = torch.randn(total_ctx_tokens, num_heads, kv_lora_rank, dtype=dtype, device=device)
    ctx_q_pe = torch.randn(
        total_ctx_tokens, num_heads, qk_rope_head_dim, dtype=dtype, device=device
    )
    ctx_fused_q = torch.cat([ctx_q, ctx_q_pe], dim=-1).view(-1, num_heads * head_dim)

    ctx_seq_lens = torch.tensor(context_sequence_lengths, dtype=torch.int)
    attn_metadata = AttentionCls.Metadata(
        seq_lens=ctx_seq_lens,
        request_ids=request_ids,
        max_num_requests=num_seqs,
        num_contexts=num_seqs,
        prompt_lens=context_sequence_lengths,
        max_num_tokens=total_ctx_tokens,
        kv_cache_manager=kv_cache_manager,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[0] * num_seqs,
        ),
        mapping=mapping,
        sparse_attention_config=sparse_config,
    )
    attn_metadata.prepare()

    # Build topk indices
    topk_indices = torch.full((total_ctx_tokens, SPARSE_TOPK), -1, dtype=torch.int32, device=device)
    offset = 0
    for seq_len in context_sequence_lengths:
        for i in range(seq_len):
            valid_len = min(i + 1, SPARSE_TOPK)
            topk_indices[offset + i, :valid_len] = torch.randperm(i + 1, device=device)[
                :valid_len
            ].to(torch.int32)
        offset += seq_len
    sparse_lens = (topk_indices >= 0).sum(dim=-1, dtype=torch.int32)

    latent_cache = torch.cat([ctx_k_nope, ctx_k_pe], dim=-1)

    _ = attention_layer.forward(
        ctx_fused_q.clone(),
        None,
        None,
        attn_metadata,
        attention_input_type=AttentionInputType.context_only,
        latent_cache=latent_cache,
        q_pe=ctx_q_pe,
        topk_indices=topk_indices,
        sparse_lens=sparse_lens,
        is_generation=False,
        attention_window_size=window_size,
    )

    torch.cuda.synchronize()

    # Validate context phase KV cache for each sequence
    for seq_idx, seq_len in enumerate(context_sequence_lengths):
        cache_len = min(seq_len, window_size)
        extracted_cache = extract_kv_cache_from_blocks(
            kv_cache_manager,
            seq_idx=seq_idx,
            cache_seq_len=cache_len,
            layer_idx=0,
            request_ids=[seq_idx],
            total_seq_len=seq_len,
            window_size=window_size,
        )
        print(f"\nSequence {seq_idx} (ctx_len={seq_len}):")

        k_pe_rotated = _apply_rope_to_k_pe(
            reference_k_pe_per_seq[seq_idx][:seq_len],
            rope_cos_sin,
            [seq_len],
        )

        validate_sliding_window_contents(
            extracted_cache,
            reference_k_nope_per_seq[seq_idx][:seq_len],
            k_pe_rotated,
            window_size,
            kv_lora_rank=kv_lora_rank,
        )

    # ===== Generation Phase =====
    if num_generation_steps > 0:
        print(f"\n--- Generation Phase ({num_generation_steps} steps) ---")

        gen_offset = total_ctx_tokens
        for step in range(num_generation_steps):
            _allocate_kv_cache_for_generation(kv_cache_manager, request_ids, generation_seq_len_q)

            gen_tokens = num_seqs * generation_seq_len_q
            gen_k_nope = all_k_nope[gen_offset : gen_offset + gen_tokens]
            gen_k_pe = all_k_pe[gen_offset : gen_offset + gen_tokens]
            gen_q = torch.randn(gen_tokens, num_heads, kv_lora_rank, dtype=dtype, device=device)
            gen_q_pe = torch.randn(
                gen_tokens, num_heads, qk_rope_head_dim, dtype=dtype, device=device
            )
            # Keep fused_q in 3D shape for mla_rope_generation: (num_tokens, num_heads, head_dim)
            gen_fused_q = torch.cat([gen_q, gen_q_pe], dim=-1)

            gen_seq_lens = torch.tensor([generation_seq_len_q] * num_seqs, dtype=torch.int)
            cached_lens = [
                ctx_len + step * generation_seq_len_q for ctx_len in context_sequence_lengths
            ]

            attn_metadata = AttentionCls.Metadata(
                seq_lens=gen_seq_lens,
                request_ids=request_ids,
                max_num_requests=num_seqs,
                num_contexts=0,
                prompt_lens=context_sequence_lengths,
                max_num_tokens=gen_tokens,
                kv_cache_manager=kv_cache_manager,
                kv_cache_params=KVCacheParams(
                    use_cache=True,
                    num_cached_tokens_per_seq=cached_lens,
                ),
                mapping=mapping,
                enable_flash_mla=torch.cuda.get_device_capability() == (9, 0),
                sparse_attention_config=sparse_config,
            )
            attn_metadata.prepare()

            gen_latent_cache = torch.cat([gen_k_nope, gen_k_pe], dim=-1)

            # Prepare generation seqlens for mla_rope_generation
            cu_q_seqlens, cu_kv_seqlens, fmha_scheduler_counter = _prepare_generation_seqlens(
                num_seqs=num_seqs,
                generation_seq_len_q=generation_seq_len_q,
                cached_lens=cached_lens,
                device=device,
            )

            # Reshape fused_q for mla_rope_generation: (num_tokens, num_heads * head_dim)
            gen_fused_q_flat = gen_fused_q.view(-1, num_heads * head_dim)

            # Call mla_rope_generation to apply RoPE and write to KV cache
            attention_layer.mla_rope_generation(
                fused_q=gen_fused_q_flat.clone(),
                q_pe=gen_q_pe.clone(),
                latent_cache=gen_latent_cache,
                metadata=attn_metadata,
                cu_q_seqlens=cu_q_seqlens,
                cu_kv_seqlens=cu_kv_seqlens,
                fmha_scheduler_counter=fmha_scheduler_counter,
                mla_bmm1_scale=None,
                mla_bmm2_scale=None,
                quant_q_buffer=None,
            )

            # Build topk indices for generation tokens
            gen_topk_indices = torch.full(
                (gen_tokens, SPARSE_TOPK), -1, dtype=torch.int32, device=device
            )
            gen_offset_idx = 0
            for seq_idx, ctx_len in enumerate(context_sequence_lengths):
                total_kv_len = cached_lens[seq_idx] + generation_seq_len_q
                for i in range(generation_seq_len_q):
                    valid_len = min(total_kv_len, SPARSE_TOPK)
                    gen_topk_indices[gen_offset_idx + i, :valid_len] = torch.randperm(
                        total_kv_len, device=device
                    )[:valid_len].to(torch.int32)
                gen_offset_idx += generation_seq_len_q
            gen_sparse_lens = (gen_topk_indices >= 0).sum(dim=-1, dtype=torch.int32)

            # Call full attention forward (without verifying accuracy, just for complete flow)
            _ = attention_layer.forward(
                gen_fused_q_flat.clone(),
                None,
                None,
                attn_metadata,
                attention_input_type=AttentionInputType.generation_only,
                latent_cache=gen_latent_cache,
                q_pe=gen_q_pe,
                cu_q_seqlens=cu_q_seqlens,
                cu_kv_seqlens=cu_kv_seqlens,
                fmha_scheduler_counter=fmha_scheduler_counter,
                mla_bmm1_scale=None,
                mla_bmm2_scale=None,
                quant_q_buffer=None,
                topk_indices=gen_topk_indices,
                sparse_lens=gen_sparse_lens,
                is_generation=True,
                attention_window_size=window_size,
            )

            gen_offset += gen_tokens

        torch.cuda.synchronize()

        # Validate final KV cache after generation
        for seq_idx, seq_len in enumerate(context_sequence_lengths):
            total_seq_len = seq_len + num_generation_steps * generation_seq_len_q
            cache_len = min(total_seq_len, window_size)
            extracted_cache = extract_kv_cache_from_blocks(
                kv_cache_manager,
                seq_idx=seq_idx,
                cache_seq_len=cache_len,
                layer_idx=0,
                request_ids=[seq_idx],
                total_seq_len=total_seq_len,
                window_size=window_size,
            )
            print(f"\nSequence {seq_idx} after generation (total_len={total_seq_len}):")

            full_k_pe = reference_k_pe_per_seq[seq_idx][:total_seq_len]
            k_pe_rotated = _apply_rope_to_k_pe(
                full_k_pe,
                rope_cos_sin,
                [total_seq_len],
            )

            validate_sliding_window_contents(
                extracted_cache,
                reference_k_nope_per_seq[seq_idx][:total_seq_len],
                k_pe_rotated,
                window_size,
                kv_lora_rank=kv_lora_rank,
            )

    print("\nTest PASSED")
    kv_cache_manager.shutdown()


@pytest.mark.skipif(get_sm_version() < 90, reason="Sparse MLA requires SM90 (Hopper) or later")
@pytest.mark.parametrize("scenario", scenarios, ids=lambda x: f"rope_append={x.rope_append}")
@pytest.mark.parametrize(
    "context_sequence_lengths", context_sequence_lengths_list, ids=lambda x: f"ctx_lens={x}"
)
@pytest.mark.parametrize("window_size", window_sizes, ids=lambda x: f"win{x}")
@pytest.mark.parametrize("num_generation_steps", num_generation_steps_list, ids=lambda x: f"gen{x}")
def test_swa_sparse_mla(
    scenario: Scenario,
    context_sequence_lengths: List[int],
    window_size: int,
    num_generation_steps: int,
):
    print(f"\n{'=' * 80}")
    print("Testing SWA Sparse MLA:")
    print(f"  Context lengths: {context_sequence_lengths}")
    print(f"  Window size: {window_size}")
    print(f"  Generation steps: {num_generation_steps}")
    print(f"{'=' * 80}")

    _run_swa_test(
        scenario=scenario,
        context_sequence_lengths=context_sequence_lengths,
        window_size=window_size,
        num_generation_steps=num_generation_steps,
    )


if __name__ == "__main__":
    _run_swa_test(
        scenario=scenarios[1],
        context_sequence_lengths=[256],
        window_size=128,
        num_generation_steps=3,
    )
