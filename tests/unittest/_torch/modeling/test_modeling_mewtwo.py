import weakref
from copy import deepcopy

import torch
from transformers import PretrainedConfig

# from utils.util import default_dtype
import tensorrt_llm
from tensorrt_llm._torch.attention_backend.sparse.mewtwo.cache_manager import MewtwoCacheManager
from tensorrt_llm._torch.attention_backend.sparse.mewtwo.mewtwo import MewtwoTrtllmAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_mewtwo import MewtwoForCausalLM
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.utils import model_extra_attrs
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MewtwoSparseAttentionConfig
from tensorrt_llm.mapping import Mapping

MEWTWO_TINY_CONFIG = {
    "architectures": ["MewtwoForCausalLM"],
    "model_type": "mewtwo",
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


# class TestMewtwo(unittest.TestCase):


def test_mewtwo_sanity():
    config_dict = deepcopy(MEWTWO_TINY_CONFIG)
    config = PretrainedConfig(**config_dict)
    config.dtype = torch.bfloat16
    config.mapping = Mapping(world_size=1, tp_size=1, rank=0)
    config.tie_word_embeddings = False

    vocab_size = config.vocab_size
    max_batch_size = 4

    sparse_attn_config = MewtwoSparseAttentionConfig(
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
    model = MewtwoForCausalLM(model_config).to(device)

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
    tokens_per_block = 128  # Mewtwo requirement
    required_blocks = sum(
        (token_num + tokens_per_block - 1) // tokens_per_block for token_num in token_nums
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

    kv_cache_manager = MewtwoCacheManager(
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
    # reqs = add_dummy_requests(kv_cache_manager, request_ids, token_nums)
    reqs = [
        LlmRequest(
            request_id=req_id,
            max_new_tokens=1024,
            input_tokens=list(range(token_nums[i])),
            sampling_config=SamplingConfig(),
            is_streaming=False,
        )
        for i, req_id in enumerate(request_ids)
    ]

    attn_metadata = MewtwoTrtllmAttentionMetadata(
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

    position_ids = torch.cat(position_ids).unsqueeze(0)

    extra_attrs = model_config.extra_attrs
    extra_attrs["attention_metadata"] = weakref.ref(attn_metadata)
    with torch.inference_mode(), model_extra_attrs(extra_attrs):
        scheduled_batch = ScheduledRequests()
        scheduled_batch.context_requests = reqs
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


test_mewtwo_sanity()
