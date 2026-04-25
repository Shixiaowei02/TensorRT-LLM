import inspect
import weakref
from copy import deepcopy
from types import SimpleNamespace

import torch
from transformers import PretrainedConfig

# from utils.util import default_dtype
import tensorrt_llm
from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4.cache_manager import DeepseekV4CacheManager
from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4.compressor import Compressor
from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4.deepseek_v4 import (
    DeepseekV4Indexer,
    DeepseekV4TrtllmAttention,
    DeepseekV4TrtllmAttentionMetadata,
)
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv4 import (
    DeepseekV4ForCausalLM,
    DeepseekV4MoE,
    _make_deepseek_v4_pos_embd_params,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.utils import model_extra_attrs
from tensorrt_llm.functional import PositionEmbeddingType, RotaryScalingType
from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping

DEEPSEEK_V4_TINY_CONFIG = {
    "architectures": ["DeepseekV4ForCausalLM"],
    "model_type": "deepseek_v4",
    "hidden_size": 4096,
    "num_attention_heads": 64,
    "num_key_value_heads": 1,
    "qk_nope_head_dim": 448,
    "qk_rope_head_dim": 64,
    "v_head_dim": 512,
    "q_lora_rank": 1024,
    "kv_lora_rank": 448,
    "o_groups": 8,
    "o_lora_rank": 1024,
    "max_position_embeddings": 65536,
    "rms_norm_eps": 1e-6,
    "dtype": "bfloat16",
    "vocab_size": 129280,
    "num_hidden_layers": 7,
    "n_hash_layers": 3,
    "moe_intermediate_size": 2048,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_experts_per_tok": 6,
    "n_group": 1,
    "topk_group": 1,
    "routed_scaling_factor": 1.5,
    "score_func": "sqrtsoftplus",
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-6,
    "compress_rope_theta": 40000.0,
    "rope_theta": 10000.0,
    "rope_scaling": {
        "type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 65536,
        "beta_fast": 32,
        "beta_slow": 1,
    },
    "quantization_config": {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "scale_fmt": "ue8m0",
        "weight_block_size": [128, 128],
    },
}


# class TestDeepSeekV4(unittest.TestCase):


def _make_deepseek_v4_test_config():
    config = PretrainedConfig(**deepcopy(DEEPSEEK_V4_TINY_CONFIG))
    config.torch_dtype = torch.bfloat16
    return config


def _make_rope_test_model_config(compress_ratios):
    return SimpleNamespace(
        pretrained_config=_make_deepseek_v4_test_config(),
        sparse_attention_config=SimpleNamespace(compress_ratios=compress_ratios),
    )


def test_deepseek_v4_rope_params_use_compressed_theta_only_for_compressed_layers():
    model_config = _make_rope_test_model_config([128, 4, 1])
    config = model_config.pretrained_config

    for layer_idx in [0, 1]:
        pos_params = _make_deepseek_v4_pos_embd_params(model_config, layer_idx)
        assert pos_params.type == PositionEmbeddingType.yarn
        assert pos_params.is_neox is False
        assert pos_params.rope.theta == config.compress_rope_theta
        assert pos_params.rope.scale_type == RotaryScalingType.yarn
        assert pos_params.rope.scale == config.rope_scaling["factor"]
        assert (
            pos_params.rope.original_max_positions
            == config.rope_scaling["original_max_position_embeddings"]
        )

    pos_params = _make_deepseek_v4_pos_embd_params(model_config, 2)
    assert pos_params.type == PositionEmbeddingType.yarn
    assert pos_params.is_neox is False
    assert pos_params.rope.theta == config.rope_theta
    assert pos_params.rope.scale_type == RotaryScalingType.none
    assert pos_params.rope.scale == 1.0
    assert pos_params.rope.mscale == 1.0
    assert pos_params.rope.mscale_all_dim == 0.0
    assert pos_params.rope.original_max_positions == config.max_position_embeddings


def test_deepseek_v4_rope_params_fallback_to_base_rope_for_non_compressed_edge_cases():
    config = _make_deepseek_v4_test_config()
    model_configs = [
        SimpleNamespace(pretrained_config=config, sparse_attention_config=None),
        _make_rope_test_model_config(None),
        _make_rope_test_model_config([]),
        _make_rope_test_model_config([128]),
        _make_rope_test_model_config([128]),
        _make_rope_test_model_config([0]),
    ]
    layer_idxs = [None, 0, 0, 1, -1, 0]

    for model_config, layer_idx in zip(model_configs, layer_idxs):
        pos_params = _make_deepseek_v4_pos_embd_params(model_config, layer_idx)
        assert pos_params.rope.theta == config.rope_theta
        assert pos_params.rope.scale_type == RotaryScalingType.none
        assert pos_params.rope.scale == 1.0


def test_deepseek_v4_compressed_rope_falls_back_to_base_theta_if_missing_compress_theta():
    model_config = _make_rope_test_model_config([4])
    config = model_config.pretrained_config
    delattr(config, "compress_rope_theta")

    pos_params = _make_deepseek_v4_pos_embd_params(model_config, 0)
    assert pos_params.rope.theta == config.rope_theta
    assert pos_params.rope.scale_type == RotaryScalingType.yarn


def test_deepseek_v4_compressor_rotate_and_indexer_rope_contracts():
    assert inspect.signature(Compressor).parameters["rotate_activation"].default is False

    indexer_init = inspect.getsource(DeepseekV4Indexer.__init__)
    assert "is_neox=False" in indexer_init
    assert "rotate_activation=True" in indexer_init

    attention_init = inspect.getsource(DeepseekV4TrtllmAttention.__init__)
    assert "rotate_activation=False" in attention_init


def test_deepseek_v4_moe_swiglu_limit_is_routed_only():
    moe_init = inspect.getsource(DeepseekV4MoE.__init__)
    # Routed experts: swiglu_limit is built once and passed to create_moe.
    assert moe_init.count("swiglu_limit=self.swiglu_limit") == 1
    assert "torch.full" in moe_init  # per-local-expert fp32 tensor for the C++ op.

    # Shared-expert block must not propagate swiglu_limit (V4 reference: shared
    # expert has swiglu_limit=0, i.e., disabled).
    shared_expert_block = moe_init.split("self.shared_experts = GatedMLP", 1)[1].split(
        "self.allreduce", 1
    )[0]
    assert "swiglu_limit" not in shared_expert_block


def test_deepseek_v4_q_b_layernorm_matches_per_head_reference():
    """V4 reference Q post-q_b_proj normalization is per-head unweighted RMS:
        q = wq_b(q).unflatten(-1, (n_heads, head_dim))
        q *= rsqrt(q.square().mean(-1, keepdim=True) + eps)

    The deepseek_v4 MLA branch realizes this by calling the standard ``RMSNorm``
    op (so cuda_tile / flashinfer fast paths apply) on a ``[N*n_heads,
    head_dim]`` view. ``has_weights=False`` registers an all-ones buffer so
    no learnable scale is applied — matching the reference, which has no
    ``q_b_layernorm.weight`` key in the checkpoint.

    Tolerance accounts for fp32-internal RMSNorm vs bf16-direct reference
    rounding (one-bf16-ULP at the typical post-norm magnitude).
    """
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    if not torch.cuda.is_available():
        import pytest

        pytest.skip("RMSNorm fast paths require CUDA")

    n_heads, head_dim, eps = 8, 64, 1e-6
    torch.manual_seed(0)
    device = "cuda"
    q = torch.randn(4, n_heads * head_dim, dtype=torch.bfloat16, device=device)

    norm = RMSNorm(
        hidden_size=head_dim, eps=eps, dtype=torch.bfloat16, has_weights=False, device=device
    )
    out = norm(q.view(-1, head_dim)).view_as(q)

    ref = q.unflatten(-1, (n_heads, head_dim))
    ref = ref * torch.rsqrt(ref.square().float().mean(-1, keepdim=True) + eps).to(ref.dtype)
    ref = ref.reshape_as(q)

    assert out.shape == q.shape
    # bf16 ULP at magnitude ~1 is 2^-7 ≈ 7.8e-3; use 2e-2 to absorb worst case.
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=2e-2)


def test_deepseek_v4_q_b_layernorm_differs_from_joint_flat_rms():
    """Guard against regressing to flat-RMSNorm-over-(n_heads*head_dim).

    Per-head normalization treats heads independently, so when heads have
    different scales the output differs from a single joint RMS over the
    flattened dim. The old buggy q_b_layernorm did the joint version.
    """
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    if not torch.cuda.is_available():
        import pytest

        pytest.skip("RMSNorm fast paths require CUDA")

    head_dim, eps = 8, 1e-6
    head_scales = torch.tensor([1.0, 10.0, 0.1, 1.0], dtype=torch.bfloat16)
    n_heads = head_scales.numel()
    device = "cuda"
    torch.manual_seed(0)
    base = torch.randn(2, n_heads, head_dim, dtype=torch.bfloat16, device=device)
    q = (base * head_scales.to(device).view(1, n_heads, 1)).reshape(2, n_heads * head_dim)

    per_head_norm = RMSNorm(
        hidden_size=head_dim, eps=eps, dtype=torch.bfloat16, has_weights=False, device=device
    )
    per_head = per_head_norm(q.view(-1, head_dim)).view_as(q)

    joint_norm = RMSNorm(
        hidden_size=n_heads * head_dim,
        eps=eps,
        dtype=torch.bfloat16,
        has_weights=False,
        device=device,
    )
    joint = joint_norm(q)

    # Per-head and joint differ substantially when heads have wildly different
    # scales — they only agree when every head has the same RMS, which the
    # head_scales tensor breaks by construction.
    assert not torch.allclose(per_head, joint, atol=0.1)


def test_deepseek_v4_mla_q_b_layernorm_init_and_forward_shape():
    """MLA DeepSeek-V4 branch must use the standard RMSNorm op sized to
    ``qk_head_dim`` with ``has_weights=False`` (V4 ckpt has no
    ``q_b_layernorm.weight``), and call it on a ``[N*n_heads, head_dim]``
    view of q so the per-row reduction matches per-head reduction."""
    from tensorrt_llm._torch.modules.attention import MLA

    init_src = inspect.getsource(MLA.__init__)
    forward_src = inspect.getsource(MLA.forward_impl_with_deepseek_v4)

    assert "self.q_b_layernorm = RMSNorm(hidden_size=self.qk_head_dim" in init_src
    assert "has_weights=False" in init_src
    assert "self.q_b_layernorm(q.view(-1, self.qk_head_dim)).view_as(q)" in forward_src


def test_deepseek_v4_sanity():
    config_dict = deepcopy(DEEPSEEK_V4_TINY_CONFIG)
    config = PretrainedConfig(**config_dict)
    config.dtype = torch.bfloat16
    config.mapping = Mapping(world_size=1, tp_size=1, rank=0)
    config.tie_word_embeddings = False

    vocab_size = config.vocab_size
    max_batch_size = 4

    sparse_attn_config = DeepSeekV4SparseAttentionConfig(
        index_n_heads=64,
        index_head_dim=128,
        window_size=128,
        compress_ratios=[1, 1, 4, 128, 4, 128, 4],
        index_topk=512,
    )
    config.sparse_attention_config = sparse_attn_config

    device = torch.device("cuda")
    # with default_dtype(config.dtype):
    model_config = ModelConfig(
        pretrained_config=config, sparse_attention_config=sparse_attn_config, attn_backend="TRTLLM"
    )
    model = DeepseekV4ForCausalLM(model_config).to(device)

    context_sequence_length = [3, 2, 5]
    sequence_length = context_sequence_length + [1, 1]

    # Total tokens = sum(sequence_length) = 3+2+5+1+1 = 12
    input_ids = torch.randint(
        0, vocab_size, (sum(sequence_length),), dtype=torch.int32, device=device
    )
    past_seen_tokens = [0, 0, 0, 62, 75]
    request_ids = list(range(len(sequence_length)))
    token_nums = (torch.tensor(past_seen_tokens) + torch.tensor(sequence_length)).tolist()
    prompt_lens = token_nums[:3] + past_seen_tokens[3:]
    tokens_per_block = 128  # DeepSeek-V4 requirement
    max_new_tokens = 1024
    required_blocks = sum(
        (token_num + max_new_tokens + tokens_per_block - 1) // tokens_per_block
        for token_num in token_nums
    )
    num_blocks = max(10, required_blocks)
    head_dim = config.v_head_dim
    num_layers = config.num_hidden_layers
    max_seq_len = num_blocks * tokens_per_block
    batch_size = len(sequence_length)

    if config.dtype == torch.half:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF
    elif config.dtype == torch.bfloat16:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16
    else:
        raise ValueError("Invalid dtype")
    mapping = config.mapping
    kv_cache_config = KvCacheConfig(max_tokens=num_blocks * tokens_per_block)
    kv_cache_config.max_util_for_resume = 0.1

    kv_cache_manager = DeepseekV4CacheManager(
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=False,
            max_tokens=num_blocks * tokens_per_block,
            event_buffer_max_size=0,
        ),
        kv_cache_type=tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=num_layers,
        num_kv_heads=1,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
        mapping=mapping,
        dtype=kv_cache_dtype,
        compressor_dtype=tensorrt_llm.bindings.DataType.FLOAT,
        vocab_size=vocab_size,
        max_num_tokens=max_seq_len * max_batch_size,
        sparse_attn_config=sparse_attn_config,
        model_config=model_config,
    )
    # Register request IDs in KV cache via prepare_context / resize_context
    reqs = []
    for i, req_id in enumerate(request_ids):
        req = LlmRequest(
            request_id=req_id,
            max_new_tokens=max_new_tokens,
            input_tokens=list(range(token_nums[i])),
            sampling_config=SamplingConfig(),
            is_streaming=False,
        )
        success = kv_cache_manager.prepare_context(req)
        assert success, f"Failed to prepare context for request {req_id}"
        # Allocate enough capacity for context tokens plus generation headroom
        success = kv_cache_manager.resize_context(req, token_nums[i] + max_new_tokens)
        assert success, f"Failed to resize context for request {req_id}"
        reqs.append(req)

    attn_metadata = DeepseekV4TrtllmAttentionMetadata(
        seq_lens=torch.tensor(sequence_length, dtype=torch.int32),
        num_contexts=len(context_sequence_length),
        max_num_requests=len(sequence_length),
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=past_seen_tokens,
        ),
        kv_cache_manager=kv_cache_manager,
        request_ids=request_ids,
        prompt_lens=prompt_lens,
        max_num_tokens=8192,
        mapping=mapping,
        sparse_attention_config=sparse_attn_config,
    )

    position_ids = []
    seq_lens = []
    for i, tokens in enumerate(past_seen_tokens):
        seq_len = context_sequence_length[i] if i < len(context_sequence_length) else 1
        position_id = torch.arange(tokens, tokens + seq_len, device=input_ids.device)
        position_ids.append(position_id)
        seq_lens.append(seq_len)

    position_ids = torch.cat(position_ids).unsqueeze(0).to(torch.int32)

    extra_attrs = model_config.extra_attrs
    extra_attrs["attention_metadata"] = weakref.ref(attn_metadata)
    with torch.inference_mode(), model_extra_attrs(extra_attrs):
        scheduled_batch = ScheduledRequests()
        scheduled_batch.context_requests_last_chunk = reqs
        kv_cache_manager.prepare_resources(scheduled_batch)
        attn_metadata.prepare()

        logits = model.forward(
            input_ids=input_ids, position_ids=position_ids, attn_metadata=attn_metadata
        )

        for req in reqs:
            req.context_current_position = seq_lens[req.py_request_id]
            req.add_new_token(seq_lens[req.py_request_id], 0)
        kv_cache_manager.update_resources(scheduled_batch)
    assert len(past_seen_tokens) == logits.shape[0]

    extra_attrs["attention_metadata"] = weakref.ref(attn_metadata)
    with torch.inference_mode(), model_extra_attrs(extra_attrs):
        seq_lens = [seq_len + 1 for seq_len in seq_lens]
        scheduled_batch = ScheduledRequests()
        scheduled_batch.generation_requests = reqs
        kv_cache_manager.prepare_resources(scheduled_batch)
        attn_metadata.prepare()
        logits = model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attn_metadata=attn_metadata,
            return_context_logits=True,
        )
        for req in reqs:
            req.add_new_token(seq_lens[req.py_request_id], 0)
        kv_cache_manager.update_resources(scheduled_batch)
    assert input_ids.shape == logits.shape[:-1]

    for req in reqs:
        kv_cache_manager.free_resources(req)
    kv_cache_manager.shutdown()
