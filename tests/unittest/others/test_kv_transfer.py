import random
import time
import uuid

import pytest
import torch

import tensorrt_llm
import tensorrt_llm.bindings
import tensorrt_llm.bindings.executor as trtllm
from tensorrt_llm import DisaggregatedParams, Mapping, SamplingParams
from tensorrt_llm._torch.disaggregation.base.kv_transfer import KVSlice, SessionStatus
from tensorrt_llm._torch.disaggregation.native.aux_buffer import AuxBuffer
from tensorrt_llm._torch.disaggregation.native.kv_transfer import (
    TransferAgentConfig,
    TransferWorker,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestType
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings import DataType


def create_transfer_worker_setup(
    ctx_tp: int,
    ctx_pp: int,
    ctx_enable_dp: bool,
    gen_tp: int,
    gen_pp: int,
    gen_enable_dp: bool,
    is_mla: bool = False,
):
    """Helper function to set up transfer workers for testing."""
    ctx_mappings = []
    for i in range(ctx_pp):
        for j in range(ctx_tp):
            ctx_mappings.append(
                Mapping(
                    world_size=ctx_tp * ctx_pp,
                    rank=i * ctx_tp + j,
                    tp_size=ctx_tp,
                    pp_size=ctx_pp,
                    enable_attention_dp=ctx_enable_dp,
                )
            )
    gen_mappings = []
    for i in range(gen_pp):
        for j in range(gen_tp):
            gen_mappings.append(
                Mapping(
                    world_size=gen_tp * gen_pp,
                    rank=i * gen_tp + j,
                    tp_size=gen_tp,
                    pp_size=gen_pp,
                    enable_attention_dp=gen_enable_dp,
                )
            )

    meta_max_batch_size = 32
    beam_width = 1
    max_draft_len = 4

    ctx_instance_num = ctx_tp * ctx_pp
    gen_instance_num = gen_tp * gen_pp
    num_layers = 4
    head_dim = 128
    num_kv_heads = 4 if not is_mla else 1
    tokens_per_block = 8
    max_seq_len = 256
    max_batch_size = 4
    dtype = DataType.FLOAT
    ctx_transfer_workers = []
    ctx_kv_cache_managers = []
    device_id = 0
    ctx_instance_name = "ctx_instance"
    gen_instance_name = "gen_instance"

    request_len = 16

    for i in range(ctx_instance_num):
        ctx_aux_buffer = AuxBuffer(meta_max_batch_size, beam_width, max_draft_len)
        transfer_agent_config = TransferAgentConfig()
        ctx_kv_cache_manager = KVCacheManager(
            trtllm.KvCacheConfig(
                max_tokens=2048,
                enable_block_reuse=False,
            ),
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF
            if not is_mla
            else tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            mapping=ctx_mappings[i],
            dtype=dtype,
        )
        random_seed = 0 if is_mla else None
        ctx_block_data_pool = ctx_kv_cache_manager.get_unique_primary_pool()
        if random_seed is not None:
            generator = torch.Generator(device=ctx_block_data_pool.device).manual_seed(random_seed)
        else:
            generator = None
        random_values = torch.rand(
            ctx_block_data_pool.shape,
            dtype=torch.float32,
            device=ctx_block_data_pool.device,
            generator=generator,
        )

        ctx_block_data_pool.copy_(random_values)
        ctx_kv_cache_managers.append(ctx_kv_cache_manager)
        ctx_transfer_workers.append(
            TransferWorker(
                kv_cache_manager=ctx_kv_cache_manager,
                mapping=ctx_mappings[i],
                device_id=device_id,
                instance_name=ctx_instance_name,
                transfer_agent_config=transfer_agent_config,
                aux_buffer=ctx_aux_buffer,
            )
        )

    ctx_info_endpoint = ctx_transfer_workers[0]._instance_info_server.endpoint
    ctx_endpoints = [
        ctx_transfer_worker._sender.endpoint for ctx_transfer_worker in ctx_transfer_workers
    ]
    ctx_layer_num_per_pp = []
    for pp_rank in range(ctx_pp):
        ctx_layer_num_per_pp.append(
            len(ctx_transfer_workers[pp_rank * ctx_tp]._kv_cache_manager.pp_layers)
        )

    for ctx_transfer_worker in ctx_transfer_workers:
        ctx_transfer_worker.populate_instance_and_rank_info(
            endpoints=ctx_endpoints, layer_num_per_pp=ctx_layer_num_per_pp
        )

    gen_transfer_workers = []
    gen_kv_cache_managers = []
    for i in range(gen_instance_num):
        gen_aux_buffer = AuxBuffer(meta_max_batch_size, beam_width, max_draft_len)
        gen_kv_cache_manager = KVCacheManager(
            trtllm.KvCacheConfig(
                max_tokens=2048,
                enable_block_reuse=False,
            ),
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF
            if not is_mla
            else tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            mapping=gen_mappings[i],
            dtype=dtype,
        )
        gen_kv_cache_managers.append(gen_kv_cache_manager)
        gen_transfer_workers.append(
            TransferWorker(
                kv_cache_manager=gen_kv_cache_manager,
                mapping=gen_mappings[i],
                device_id=device_id,
                instance_name=gen_instance_name,
                transfer_agent_config=transfer_agent_config,
                aux_buffer=gen_aux_buffer,
            )
        )
    _ = gen_transfer_workers[0]._instance_info_server.endpoint  # noqa: F841
    gen_endpoints = [
        gen_transfer_worker._sender.endpoint for gen_transfer_worker in gen_transfer_workers
    ]
    gen_layer_num_per_pp = []
    for pp_rank in range(gen_pp):
        gen_layer_num_per_pp.append(
            len(gen_transfer_workers[pp_rank * gen_tp]._kv_cache_manager.pp_layers)
        )
    for gen_transfer_worker in gen_transfer_workers:
        gen_transfer_worker.populate_instance_and_rank_info(
            endpoints=gen_endpoints, layer_num_per_pp=gen_layer_num_per_pp
        )

    return {
        "ctx_transfer_workers": ctx_transfer_workers,
        "ctx_kv_cache_managers": ctx_kv_cache_managers,
        "gen_transfer_workers": gen_transfer_workers,
        "gen_kv_cache_managers": gen_kv_cache_managers,
        "ctx_info_endpoint": ctx_info_endpoint,
        "ctx_layer_num_per_pp": ctx_layer_num_per_pp,
        "gen_layer_num_per_pp": gen_layer_num_per_pp,
        "ctx_tp": ctx_tp,
        "ctx_pp": ctx_pp,
        "ctx_enable_dp": ctx_enable_dp,
        "gen_tp": gen_tp,
        "gen_pp": gen_pp,
        "gen_enable_dp": gen_enable_dp,
        "is_mla": is_mla,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "tokens_per_block": tokens_per_block,
        "request_len": request_len,
    }


def add_and_verify_request(
    setup, ctx_request_id, gen_request_id, request_len, send_first: bool = True
):
    """Helper function to add and verify a request transfer."""
    ctx_transfer_workers = setup["ctx_transfer_workers"]
    ctx_kv_cache_managers = setup["ctx_kv_cache_managers"]
    gen_transfer_workers = setup["gen_transfer_workers"]
    gen_kv_cache_managers = setup["gen_kv_cache_managers"]
    ctx_info_endpoint = setup["ctx_info_endpoint"]
    ctx_layer_num_per_pp = setup["ctx_layer_num_per_pp"]
    gen_layer_num_per_pp = setup["gen_layer_num_per_pp"]
    ctx_tp = setup["ctx_tp"]
    ctx_pp = setup["ctx_pp"]
    ctx_enable_dp = setup["ctx_enable_dp"]
    gen_tp = setup["gen_tp"]
    gen_pp = setup["gen_pp"]
    gen_enable_dp = setup["gen_enable_dp"]
    is_mla = setup["is_mla"]
    num_layers = setup["num_layers"]
    num_kv_heads = setup["num_kv_heads"]
    head_dim = setup["head_dim"]
    tokens_per_block = setup["tokens_per_block"]

    sampling_params = SamplingParams()

    ctx_dp_rank = 0
    if ctx_enable_dp:
        ctx_dp_rank = ctx_request_id % ctx_tp
        valid_ctx_kv_cache_managers = []
        valid_ctx_transfer_workers = []
        for i in range(ctx_pp):
            valid_ctx_kv_cache_managers.append(ctx_kv_cache_managers[ctx_dp_rank + i * ctx_tp])
            valid_ctx_transfer_workers.append(ctx_transfer_workers[ctx_dp_rank + i * ctx_tp])
    else:
        valid_ctx_kv_cache_managers = ctx_kv_cache_managers
        valid_ctx_transfer_workers = ctx_transfer_workers
    gen_dp_rank = 0
    if gen_enable_dp:
        gen_dp_rank = gen_request_id % gen_tp
        valid_gen_kv_cache_managers = []
        valid_gen_transfer_workers = []
        for i in range(gen_pp):
            valid_gen_kv_cache_managers.append(gen_kv_cache_managers[gen_dp_rank + i * gen_tp])
            valid_gen_transfer_workers.append(gen_transfer_workers[gen_dp_rank + i * gen_tp])
    else:
        valid_gen_kv_cache_managers = gen_kv_cache_managers
        valid_gen_transfer_workers = gen_transfer_workers

    unique_rid = str(uuid.uuid4())
    ctx_request = LlmRequest(
        request_id=ctx_request_id,
        max_new_tokens=1,
        input_tokens=list(range(request_len)),
        sampling_config=tensorrt_llm.bindings.SamplingConfig(
            sampling_params._get_sampling_config()
        ),
        is_streaming=False,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY,
    )
    ctx_request.py_disaggregated_params = DisaggregatedParams(disagg_id=unique_rid)

    for ctx_kv_cache_manager in valid_ctx_kv_cache_managers:
        ctx_kv_cache_manager.impl.add_sequence(
            ctx_request.py_request_id, ctx_request.prompt_len, 1, ctx_request
        )

    gen_request = LlmRequest(
        request_id=gen_request_id,
        max_new_tokens=1,
        input_tokens=list(range(request_len)),
        sampling_config=tensorrt_llm.bindings.SamplingConfig(
            sampling_params._get_sampling_config()
        ),
        is_streaming=False,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY,
    )
    gen_request.py_disaggregated_params = DisaggregatedParams(
        ctx_request_id=ctx_request.py_request_id,
        ctx_dp_rank=ctx_dp_rank,
        ctx_info_endpoint=ctx_info_endpoint,
        disagg_id=unique_rid,
    )
    for gen_kv_cache_manager in valid_gen_kv_cache_managers:
        gen_kv_cache_manager.impl.add_sequence(
            gen_request.py_request_id, gen_request.prompt_len, 1, gen_request
        )

    ctx_blocks = [
        ctx_kv_cache_manager.get_batch_cache_indices([ctx_request.py_request_id])[0]
        for ctx_kv_cache_manager in valid_ctx_kv_cache_managers
    ]

    gen_blocks = [
        gen_kv_cache_manager.get_batch_cache_indices([gen_request.py_request_id])[0]
        for gen_kv_cache_manager in valid_gen_kv_cache_managers
    ]

    if send_first:
        sender_sessions = [
            ctx_transfer_worker.create_tx_session(ctx_request)
            for ctx_transfer_worker in valid_ctx_transfer_workers
        ]

        send_kv_slices = [
            KVSlice(is_last_slice=True, blocks=ctx_block_id) for ctx_block_id in ctx_blocks
        ]
        send_slice_tasks = [
            sender_session.slice_tasks[sender_session.send(send_kv_slice)]
            for sender_session, send_kv_slice in zip(sender_sessions, send_kv_slices)
        ]

        for sender_session in sender_sessions:
            assert sender_session.state.status == SessionStatus.INIT

        receiver_sessions = [
            gen_transfer_worker.create_rx_session(gen_request)
            for gen_transfer_worker in valid_gen_transfer_workers
        ]
        recv_kv_slices = [
            KVSlice(is_last_slice=True, blocks=gen_block_id) for gen_block_id in gen_blocks
        ]
        recv_slice_tasks = [
            receiver_session.slice_tasks[receiver_session.receive(recv_kv_slice)]
            for receiver_session, recv_kv_slice in zip(receiver_sessions, recv_kv_slices)
        ]

    else:
        receiver_sessions = [
            gen_transfer_worker.create_rx_session(gen_request)
            for gen_transfer_worker in valid_gen_transfer_workers
        ]
        recv_kv_slices = [
            KVSlice(is_last_slice=True, blocks=gen_block_id) for gen_block_id in gen_blocks
        ]
        recv_slice_tasks = [
            receiver_session.slice_tasks[receiver_session.receive(recv_kv_slice)]
            for receiver_session, recv_kv_slice in zip(receiver_sessions, recv_kv_slices)
        ]

        random_sleep_time = random.uniform(0.000001, 0.001)
        time.sleep(random_sleep_time)  # here , recv session may send or not send req_info
        sender_sessions = [
            ctx_transfer_worker.create_tx_session(ctx_request)
            for ctx_transfer_worker in valid_ctx_transfer_workers
        ]

        time.sleep(0.1)  # wait for recv session to send req_info

        for sender_session in sender_sessions:
            assert sender_session.state.status != SessionStatus.INIT

        send_kv_slices = [
            KVSlice(is_last_slice=True, blocks=ctx_block_id) for ctx_block_id in ctx_blocks
        ]
        send_slice_tasks = [
            sender_session.slice_tasks[sender_session.send(send_kv_slice)]
            for sender_session, send_kv_slice in zip(sender_sessions, send_kv_slices)
        ]

    for send_slice_task in send_slice_tasks:
        send_slice_task.future.result()
    for recv_slice_task in recv_slice_tasks:
        recv_slice_task.future.result()
    for sender_session in sender_sessions:
        assert sender_session.state.status == SessionStatus.TRANSFERRED
    for receiver_session in receiver_sessions:
        assert receiver_session.state.status == SessionStatus.TRANSFERRED

    ctx_block_datas = [
        ctx_kv_cache_manager.get_unique_primary_pool()[ctx_block_id]
        for ctx_kv_cache_manager, ctx_block_id in zip(valid_ctx_kv_cache_managers, ctx_blocks)
    ]
    gen_block_datas = [
        gen_kv_cache_manager.get_unique_primary_pool()[gen_block_id]
        for gen_kv_cache_manager, gen_block_id in zip(valid_gen_kv_cache_managers, gen_blocks)
    ]

    valid_ctx_tp = 1 if ctx_enable_dp else ctx_tp
    valid_gen_tp = 1 if gen_enable_dp else gen_tp
    if is_mla:
        valid_ctx_tp = 1
        valid_gen_tp = 1
    ctx_block_data_merge = torch.zeros(
        size=(
            ctx_block_datas[0].shape[0],
            num_layers,
            2,
            ctx_block_datas[0].shape[3] * valid_ctx_tp,
        )
    )
    for pp_rank in range(ctx_pp):
        for tp_rank in range(valid_ctx_tp):
            layer_start_idx = sum(ctx_layer_num_per_pp[:pp_rank])
            layer_end_idx = layer_start_idx + ctx_layer_num_per_pp[pp_rank]
            head_dim_per_rank = num_kv_heads // valid_ctx_tp * head_dim * tokens_per_block
            start_head_offset = tp_rank * head_dim_per_rank
            end_head_offset = start_head_offset + head_dim_per_rank
            block_id = pp_rank * valid_ctx_tp + tp_rank
            ctx_block_data_merge[
                :, layer_start_idx:layer_end_idx, :, start_head_offset:end_head_offset
            ] = ctx_block_datas[block_id]

    gen_block_data_merge = torch.zeros(
        size=(
            gen_block_datas[0].shape[0],
            num_layers,
            2,
            gen_block_datas[0].shape[3] * valid_gen_tp,
        )
    )
    for pp_rank in range(gen_pp):
        for tp_rank in range(valid_gen_tp):
            layer_start_idx = sum(gen_layer_num_per_pp[:pp_rank])
            layer_end_idx = layer_start_idx + gen_layer_num_per_pp[pp_rank]
            head_dim_per_rank = num_kv_heads // valid_gen_tp * head_dim * tokens_per_block
            start_head_offset = tp_rank * head_dim_per_rank
            end_head_offset = start_head_offset + head_dim_per_rank
            block_id = pp_rank * valid_gen_tp + tp_rank
            gen_block_data_merge[
                :, layer_start_idx:layer_end_idx, :, start_head_offset:end_head_offset
            ] = gen_block_datas[block_id]

    assert ctx_block_data_merge.equal(gen_block_data_merge)


# Test configurations as pytest parameters
PARALLEL_TEST_CONFIGS = [
    # (ctx_tp, ctx_pp, ctx_enable_dp, gen_tp, gen_pp, gen_enable_dp, is_mla, test_id)
    (1, 1, False, 1, 1, False, False, "tp1_pp1_to_tp1_pp1"),
    (1, 1, False, 1, 2, False, False, "tp1_pp1_to_tp1_pp2"),
    (1, 2, False, 1, 1, False, False, "tp1_pp2_to_tp1_pp1"),
    (1, 2, False, 1, 2, False, False, "tp1_pp2_to_tp1_pp2"),
    (1, 2, False, 2, 1, False, False, "tp1_pp2_to_tp2_pp1"),
    (2, 1, False, 1, 2, False, False, "tp2_pp1_to_tp1_pp2"),
    (4, 1, False, 2, 2, False, False, "tp4_pp1_to_tp2_pp2"),
    (1, 4, False, 2, 2, False, False, "tp1_pp4_to_tp2_pp2"),
    (2, 1, True, 2, 1, True, False, "tp2_pp1_dp_to_tp2_pp1_dp"),
    (2, 1, True, 1, 2, False, False, "tp2_pp1_dp_to_tp1_pp2"),
    (1, 4, False, 2, 2, True, False, "tp1_pp4_to_tp2_pp2_dp"),
    (2, 1, False, 2, 1, True, True, "tp2_pp1_to_tp2_pp1_dp_mla"),
    (2, 1, True, 2, 1, False, True, "tp2_pp1_dp_to_tp2_pp1_mla"),
]


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "ctx_tp,ctx_pp,ctx_enable_dp,gen_tp,gen_pp,gen_enable_dp,is_mla",
    [(c[0], c[1], c[2], c[3], c[4], c[5], c[6]) for c in PARALLEL_TEST_CONFIGS],
    ids=[c[7] for c in PARALLEL_TEST_CONFIGS],
)
def test_transfer_worker_with_parallel(
    ctx_tp, ctx_pp, ctx_enable_dp, gen_tp, gen_pp, gen_enable_dp, is_mla
):
    """Test transfer worker with various parallel configurations."""
    setup = create_transfer_worker_setup(
        ctx_tp=ctx_tp,
        ctx_pp=ctx_pp,
        ctx_enable_dp=ctx_enable_dp,
        gen_tp=gen_tp,
        gen_pp=gen_pp,
        gen_enable_dp=gen_enable_dp,
        is_mla=is_mla,
    )

    request_len = setup["request_len"]
    add_and_verify_request(setup, 0, 1, request_len, send_first=True)
    add_and_verify_request(setup, 2, 3, request_len, send_first=True)
    add_and_verify_request(setup, 4, 5, request_len * 2, send_first=False)
