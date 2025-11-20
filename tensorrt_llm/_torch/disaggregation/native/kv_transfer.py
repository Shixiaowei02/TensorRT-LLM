import concurrent
import pickle
import threading
import time
import uuid
import weakref
from dataclasses import dataclass
from typing import List, Optional

import torch
import zmq

import tensorrt_llm.bindings
import tensorrt_llm.bindings.executor as trtllm
from tensorrt_llm import Mapping, SamplingParams
from tensorrt_llm._torch.disaggregation.base.agent import (
    BaseTransferAgent,
    MemoryDescs,
    RegMemoryDescs,
    TransferOp,
    TransferRequest,
)
from tensorrt_llm._torch.disaggregation.base.kv_transfer import (
    KVSlice,
    RxSessionBase,
    SessionArgsBase,
    SessionState,
    State,
    TaskIdType,
    TxSessionBase,
)
from tensorrt_llm._torch.disaggregation.native.kv_mapper import (
    InstanceInfo,
    PeerMapper,
    PeerOverlapTargets,
    PeerRegistrar,
    RankInfo,
)
from tensorrt_llm._torch.disaggregation.nixl.agent import NixlTransferAgent
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import get_size_in_bytes
from tensorrt_llm.bindings import DataType
from tensorrt_llm.disaggregated_params import DisaggregatedParams

AttentionTypeCpp = tensorrt_llm.bindings.internal.batch_manager.AttentionType
LlmRequestType = tensorrt_llm.bindings.internal.batch_manager.LlmRequestType


@dataclass
class GenReqInfo:
    ctx_req_id: int
    instance_name: str
    instance_rank: int
    block_ids: list[int]
    disagg_id: str
    start_token_id: Optional[int] = None


@dataclass
class AgentRecvArgs:
    disagg_id: str
    futrure_for_task: concurrent.futures.Future
    expect_count: int
    remote_name: str
    slice_id: int


@dataclass
class AgentSendArgs:
    future_for_task: concurrent.futures.Future
    src_kv_ptrs: List[int]
    dst_kv_ptrs: List[int]
    kv_sizes: List[int]
    expect_count: int
    remote_name: str
    src_aux_ptrs: List[int] = None
    dst_aux_ptrs: List[int] = None
    aux_sizes: List[int] = None
    peer_endpoint: Optional[str] = None  # used for send state
    disagg_id: Optional[str] = None
    slice_id: Optional[int] = None
    is_last_slice: Optional[bool] = False


def get_local_ip():
    """
    Lupin-style local IP detection - smart and reliable approach
    """
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        if not local_ip.startswith("127."):
            return local_ip
    except OSError:
        pass

    try:
        import netifaces

        for interface in netifaces.interfaces():
            addrs = netifaces.ifaddresses(interface)
            if netifaces.AF_INET in addrs:
                for addr in addrs[netifaces.AF_INET]:
                    ip = addr["addr"]
                    if not ip.startswith("127.") and not ip.startswith("169.254"):
                        return ip
    except ImportError:
        pass

    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        if not local_ip.startswith("127."):
            return local_ip
    except OSError:
        pass
    return "127.0.0.1"


class MessageType:
    TERMINATION = "TERMINATION"
    TASK_STATE = "TASK_STATE"
    PEER_INFO = "PEER_INFO"
    REQUEST_DATA = "REQUEST_DATA"
    REQUEST_INSTANCE_INFO = "REQUEST_INSTANCE_INFO"
    REGISTER_RANK_INFO = "REQUEST_RANK_INFO"


class SliceSenderTask:
    def __init__(
        self,
        kv_slice: KVSlice,
        disagg_params: DisaggregatedParams,
        slice_id: int,
        peer_mapper: PeerMapper,
    ):
        self.kv_ptr_extractor = peer_mapper.kv_block_ptr_extractor
        self.peer_mapper = peer_mapper
        self.future = concurrent.futures.Future()
        self.first_extracted = False
        self.encountered_count = 0
        self.expect_count = 0
        self.slice = kv_slice
        self.disagg_params = disagg_params
        self.disagg_id = disagg_params.disagg_id
        self.slice_id = slice_id
        self.state = State.INIT
        self.transferred_count = 0

    def get_state(self) -> State:
        return self.state

    def get_future_for_task(self) -> concurrent.futures.Future:
        return self.future

    def extract_trans_meta(self, dst_info: GenReqInfo) -> AgentSendArgs:
        peer_instance_rank_info = self.peer_mapper.get_peer_registrar().get_peer_rank_info(
            dst_info.instance_name, dst_info.instance_rank
        )
        peer_domain_targets = self.peer_mapper.get_peer_overlap_targets(
            peer_instance_rank_info, peer_instance_rank_info.dp_rank
        )
        expect_count = len(peer_domain_targets.ranks)
        if not self.first_extracted:
            self.first_extracted = True
            self.expect_count = expect_count
        self.encountered_count = self.encountered_count + 1
        if not self._need_send_transfer(peer_domain_targets, peer_instance_rank_info):
            return AgentSendArgs(
                future_for_task=self.future,
                src_kv_ptrs=[],
                dst_kv_ptrs=[],
                kv_sizes=[],
                expect_count=expect_count,
                remote_name=peer_instance_rank_info.instance_name
                + str(peer_instance_rank_info.instance_rank),
                peer_endpoint=peer_instance_rank_info.recv_endpoint,
                disagg_id=self.disagg_id,
                slice_id=self.slice_id,
                is_last_slice=self.slice.is_last_slice,
            )

        dst_block_ids = dst_info.block_ids
        src_block_ids = self.slice.block_ids

        src_block_ids, dst_block_ids = self._filter_kv_blocks(src_block_ids, dst_block_ids)

        src_kv_blocks_ptrs = self.kv_ptr_extractor.extract_kv_block_ptrs(src_block_ids)
        self_kv_block_size = self.kv_ptr_extractor.kv_pool_attributes.kv_cache_block_sizes[0]
        peer_kv_ptr_extractor = self.peer_mapper.get_kv_extractor(
            peer_instance_rank_info.instance_name, peer_instance_rank_info.instance_rank
        )
        dst_kv_blocks_ptrs = peer_kv_ptr_extractor.extract_kv_block_ptrs(dst_block_ids)
        peer_kv_block_size = peer_kv_ptr_extractor.kv_pool_attributes.kv_cache_block_sizes[0]
        print(
            f"call extract_trans_meta src_kv_blocks_ptrs: {src_kv_blocks_ptrs} "
            f"self_kv_block_size: {self_kv_block_size} "
            f"dst_kv_blocks_ptrs: {dst_kv_blocks_ptrs} "
            f"peer_kv_block_size: {peer_kv_block_size}"
        )
        transformer = self.peer_mapper.get_kv_ptrs_mapper(peer_instance_rank_info)
        (
            src_kv_blocks_transfer_ptrs,
            dst_kv_blocks_transfer_ptrs,
            dst_kv_blocks_size,
        ) = transformer(
            src_kv_blocks_ptrs, self_kv_block_size, dst_kv_blocks_ptrs, peer_kv_block_size
        )

        src_kv_blocks_size = dst_kv_blocks_size
        print(
            f"src_kv_blocks_transfer_ptrs: {src_kv_blocks_transfer_ptrs} "
            f"src_kv_blocks_size: {src_kv_blocks_size} "
            f"dst_kv_blocks_transfer_ptrs: {dst_kv_blocks_transfer_ptrs} "
            f"dst_kv_blocks_size: {dst_kv_blocks_size}"
        )

        return AgentSendArgs(
            future_for_task=self.future,
            src_kv_ptrs=src_kv_blocks_transfer_ptrs,
            dst_kv_ptrs=dst_kv_blocks_transfer_ptrs,
            kv_sizes=[src_kv_blocks_size] * len(src_kv_blocks_transfer_ptrs),
            expect_count=expect_count,
            remote_name=peer_instance_rank_info.instance_name
            + str(peer_instance_rank_info.instance_rank),
            peer_endpoint=peer_instance_rank_info.recv_endpoint,
            disagg_id=self.disagg_id,
            slice_id=self.slice_id,
            is_last_slice=self.slice.is_last_slice,
        )

    def is_active(self) -> bool:
        if self.first_extracted:
            return self.encountered_count < self.expect_count
        else:
            return True

    def _need_send_transfer(
        self, peer_domain_targets: PeerOverlapTargets, peer_instance_rank_info: RankInfo
    ) -> bool:
        if peer_domain_targets.duplicate_head_factor <= 1:
            return True
        peer_dp_rank = (
            peer_instance_rank_info.dp_rank if peer_instance_rank_info.enable_attention_dp else 0
        )
        self_tp_size_per_dp_group = (
            self.peer_mapper.get_peer_registrar().get_self_rank_info().tp_size
            // self.peer_mapper.get_peer_registrar().get_self_rank_info().dp_size
            if self.peer_mapper.get_peer_registrar().get_self_rank_info().enable_attention_dp
            else self.peer_mapper.get_peer_registrar().get_self_rank_info().tp_size
        )
        self_tprank_in_dp_group = (
            self.peer_mapper.get_peer_registrar().get_self_rank_info().tp_rank
            % self_tp_size_per_dp_group
        )
        return (peer_dp_rank % peer_domain_targets.duplicate_head_factor) == (
            self_tprank_in_dp_group % peer_domain_targets.duplicate_head_factor
        )

    def _filter_kv_blocks(self, src_block_ids, dst_block_ids) -> tuple[list[int], list[int]]:
        # TODO: filter the kv blocks according to the peer_domain_targets
        return src_block_ids, dst_block_ids


class Sender:
    def __init__(
        self,
        peer_mapper: PeerMapper,
        device_id: int,
        transfer_agent: BaseTransferAgent,
    ):
        self.peer_mapper = peer_mapper
        self.device_id = device_id
        self.transfer_agent = transfer_agent
        self.send_session_cache_lock = threading.Lock()

        self._peer_transfer_req_info_cache = {}
        self._peer_transfer_req_info_lock = threading.Lock()

        self._zmq_context = zmq.Context()
        self.server_socket = self._zmq_context.socket(zmq.ROUTER)
        self.server_socket.bind(f"tcp://{get_local_ip()}:*")
        self.server_endpoint = self.server_socket.getsockopt(zmq.LAST_ENDPOINT).decode()

        self.socket_cache = {}

        self.slice_tasks = {}  # disagg_id -> list[KVSlice]
        print(f" Sender init end with server_endpoint: {self.server_endpoint}")

        background_thread = threading.Thread(target=self._handle_sender_loop, daemon=True)
        background_thread.start()

        self.tx_sessions = {}  # disagg_id -> TxSession

    def get_endpoint(self):
        return self.server_endpoint

    def init_session_resource(self, tx_session: TxSessionBase):
        self.tx_sessions[tx_session.session_args.disagg_params.disagg_id] = weakref.ref(tx_session)
        return

    def clear_sender_session_resource(self, disagg_id: str):
        del self.tx_sessions[disagg_id]
        del self.slice_tasks[disagg_id]

    def async_send_slice(
        self, disagg_params: DisaggregatedParams, slice: KVSlice
    ) -> SliceSenderTask:
        if disagg_params.disagg_id not in self.slice_tasks:
            self.slice_tasks[disagg_params.disagg_id] = []
        slice_id = len(self.slice_tasks[disagg_params.disagg_id])
        new_slice_task = SliceSenderTask(slice, disagg_params, slice_id, self.peer_mapper)
        self.slice_tasks[disagg_params.disagg_id].append(new_slice_task)

        self._handle_send_slice_task(new_slice_task)
        return new_slice_task

    def submit_send_task(self, agent_send_args: AgentSendArgs):
        if not hasattr(self, "_send_task_queue"):
            import queue
            import threading

            self._send_task_queue = queue.Queue()
            self._background_thread = threading.Thread(
                target=self._handle_send_task_loop, daemon=True
            )
            self._background_thread.start()
        self._send_task_queue.put(agent_send_args)

    def _handle_send_task_loop(self):
        while True:
            agent_send_args = self._send_task_queue.get()
            if agent_send_args is None:
                break
            self._handle_send_task(agent_send_args)

    def _handle_send_task(self, agent_send_args: AgentSendArgs):
        assert len(agent_send_args.src_kv_ptrs) == len(agent_send_args.dst_kv_ptrs)
        assert len(agent_send_args.kv_sizes) == len(agent_send_args.src_kv_ptrs)
        src_kv_list = [
            (src_ptr, size, self.device_id)
            for src_ptr, size in zip(agent_send_args.src_kv_ptrs, agent_send_args.kv_sizes)
        ]
        dst_kv_list = [
            (dst_ptr, size, self.device_id)
            for dst_ptr, size in zip(agent_send_args.dst_kv_ptrs, agent_send_args.kv_sizes)
        ]
        if self.tx_sessions[agent_send_args.disagg_id]().get_state().state != State.ERR:
            self.tx_sessions[agent_send_args.disagg_id]().get_state().state = State.TRANSFERRING

        src_memory_descs = MemoryDescs("VRAM", src_kv_list)
        dst_memory_descs = MemoryDescs("VRAM", dst_kv_list)
        request = TransferRequest(
            TransferOp.WRITE, src_memory_descs, dst_memory_descs, agent_send_args.remote_name, ""
        )

        print(f"  transfer agent submit transfer requests: {request}")
        status = self.transfer_agent.submit_transfer_requests(request)
        sync_status = "SUCCESS"
        if not status.wait():
            sync_status = "FAILED"
            agent_send_args.future_for_task.set_exception(RuntimeError("Transfer failed"))

            self.slice_tasks[agent_send_args.disagg_id][agent_send_args.slice_id].state = State.ERR
            self.tx_sessions[agent_send_args.disagg_id]().get_state().state = State.ERR
        socket = self._get_socket(agent_send_args.peer_endpoint)

        ## TODO: just last slice need to send task state ?
        socket.send_multipart(
            [
                str(MessageType.TASK_STATE).encode("ascii"),
                agent_send_args.disagg_id.encode("ascii"),
                str(agent_send_args.slice_id).encode("ascii"),
                str(agent_send_args.is_last_slice).encode("ascii"),
                sync_status.encode("ascii"),
            ]
        )
        self.slice_tasks[agent_send_args.disagg_id][agent_send_args.slice_id].transferred_count += 1
        if (
            self.slice_tasks[agent_send_args.disagg_id][agent_send_args.slice_id].transferred_count
            > agent_send_args.expect_count
        ):
            agent_send_args.future_for_task.set_exception(
                RuntimeError(
                    f"Session {agent_send_args.disagg_id} has more than {agent_send_args.expect_count} transfers"
                )
            )
            # TODO: set exception for the session ?
            self.tx_sessions[agent_send_args.disagg_id]().get_state().state = State.ERR
        elif (
            self.slice_tasks[agent_send_args.disagg_id][agent_send_args.slice_id].transferred_count
            == agent_send_args.expect_count
        ):
            agent_send_args.future_for_task.set_result(sync_status)
            self.slice_tasks[agent_send_args.disagg_id][
                agent_send_args.slice_id
            ].state = State.FINISHED
            self.tx_sessions[agent_send_args.disagg_id]().get_state().finished_tasks.append(
                agent_send_args.slice_id
            )
            if agent_send_args.is_last_slice:
                self.tx_sessions[agent_send_args.disagg_id]().get_state().state = State.FINISHED

    def _handle_send_slice_task(self, send_slice_task: SliceSenderTask):
        # assert send_slice_session.disagg_id in self.disagg_id_state
        # assert self.disagg_id_state[send_slice_session.disagg_id] == \
        #     SessionState.WAITING_FOR_SEND OR SessionState.TRANSFERRING
        send_slice_task.state = State.TRANSFERRING
        transfer_recv_req_info_dict = {}
        with self._peer_transfer_req_info_lock:
            if send_slice_task.disagg_id in self._peer_transfer_req_info_cache:
                transfer_recv_req_info_dict = self._peer_transfer_req_info_cache[
                    send_slice_task.disagg_id
                ]
        for transfer_recv_req_info in transfer_recv_req_info_dict.values():
            trans_meta = send_slice_task.extract_trans_meta(transfer_recv_req_info)
            # print(f" call submit_send_task trans_meta: {trans_meta}")
            self.submit_send_task(trans_meta)

    def _handle_sender_loop(self):
        while True:
            message = self.server_socket.recv_multipart()
            send_id = message[0]
            recv_message = message[1:]
            if self._message_is_termination(recv_message):
                break
            elif self._message_is_request_data(recv_message):
                self._handle_request_data(send_id, recv_message)
            # elif self._message_is_request_instance_info(recv_message):
            #     self._handle_request_instance_info(send_id, recv_message)

            elif self._message_is_register_rank_info(recv_message):
                self._handle_register_rank_info(send_id, recv_message)
            else:
                raise ValueError(
                    f"transceiver sender loop received unknown message type: {recv_message[0]}"
                )

    def _message_is_termination(self, message: list[bytes]):
        return message[0] == str(MessageType.TERMINATION).encode("ascii")

    def _message_is_request_data(self, message: list[bytes]):
        # mapping_info = self._convert_message_to_mapping_info(message)
        return message[0] == str(MessageType.REQUEST_DATA).encode("ascii")

    def _message_is_register_rank_info(self, message: list[bytes]):
        return message[0] == str(MessageType.REGISTER_RANK_INFO).encode("ascii")

    def _handle_register_rank_info(self, send_id: bytes, message: list[bytes]):
        instance_rank_info: RankInfo = pickle.loads(message[1])

        self.peer_mapper.get_peer_registrar().register(
            instance_rank_info.instance_name, instance_rank_info.instance_rank, instance_rank_info
        )

        agent_name = instance_rank_info.instance_name + str(instance_rank_info.instance_rank)
        print(f"  transfer agent load remote agent: {agent_name}")
        self.transfer_agent.load_remote_agent(
            instance_rank_info.instance_name + str(instance_rank_info.instance_rank),
            instance_rank_info.transfer_engine_info,
        )
        print(
            f"_handle_register_rank_info end with instance_name, "
            f"instance_rank: {instance_rank_info.instance_name}, "
            f"{instance_rank_info.instance_rank}"
        )

    def _handle_request_data(self, send_id: bytes, message: list[bytes]):
        transfer_gen_side_req_info: GenReqInfo = pickle.loads(message[1])

        send_slice_tasks: list[SliceSenderTask] = self._get_send_slice_tasks(
            transfer_gen_side_req_info.disagg_id
        )
        if send_slice_tasks is None:
            print(" _handle_request_data, send_slice_tasks is None")
            self._save_peer_transfer_req_info(transfer_gen_side_req_info)
        else:
            for send_slice_task in send_slice_tasks:
                trans_meta = send_slice_task.extract_trans_meta(transfer_gen_side_req_info)
                self.submit_send_task(trans_meta)

    def _get_send_slice_tasks(self, disagg_id: str):
        if disagg_id not in self.slice_tasks:
            return None
        return self.slice_tasks[disagg_id]

    def _get_socket(self, endpoint: str):
        if endpoint not in self.socket_cache:
            self.socket_cache[endpoint] = self._zmq_context.socket(zmq.DEALER)
            self.socket_cache[endpoint].connect(endpoint)
        return self.socket_cache[endpoint]

    def _save_peer_transfer_req_info(self, peer_transfer_req_info: GenReqInfo):
        with self._peer_transfer_req_info_lock:
            if peer_transfer_req_info.ctx_req_id not in self._peer_transfer_req_info_cache:
                self._peer_transfer_req_info_cache[peer_transfer_req_info.ctx_req_id] = {}
            self._peer_transfer_req_info_cache[peer_transfer_req_info.ctx_req_id][
                peer_transfer_req_info.instance_rank
            ] = peer_transfer_req_info
            expect_count = len(
                self.peer_mapper.get_peer_overlap_targets(
                    peer_transfer_req_info.instance_name, peer_transfer_req_info.instance_rank
                ).ranks
            )
            if expect_count == len(
                self._peer_transfer_req_info_cache[peer_transfer_req_info.disagg_id]
            ):
                self.tx_sessions[peer_transfer_req_info.disagg_id]().get_state().state = State.READY


class TxSession(TxSessionBase):
    def __init__(self, request_id: int, disagg_params: DisaggregatedParams, sender: Sender):
        super().__init__(sender, SessionArgsBase(request_id, disagg_params))
        self.request_id = request_id
        self.sender = sender
        self.session_state = SessionState(state=State.INIT, finished_tasks=[])
        self.exception = None
        self.slice_tasks = []  # slice_id -> SliceTxSession
        self.sender.init_session_resource(self)

    def send(self, slice: KVSlice) -> TaskIdType:
        slice_sender_task = self.sender.async_send_slice(self.session_args.disagg_params, slice)
        self.slice_tasks.append(slice_sender_task)
        return slice_sender_task.slice_id

    def get_state(self) -> SessionState:
        return self.session_state

    def poll_task(self, id: TaskIdType) -> State:
        return self.slice_tasks[id].get_state()

    def get_exception(self) -> Optional[Exception]:
        return self.exception

    def __del__(self):
        self.sender.clear_sender_session_resource(self.session_args.disagg_params.disagg_id)


class SliceReceiverTask:
    def __init__(
        self,
        disagg_id: str,
        kv_slice: KVSlice,
        slice_id: int,
        disagg_params: DisaggregatedParams,
        peer_mapper: PeerMapper,
    ):
        self.disagg_id = disagg_id
        self.kv_slice = kv_slice
        self.slice_id = slice_id
        self.disagg_params = disagg_params
        self.peer_mapper = peer_mapper
        self.state = State.INIT
        self.exception = None
        self.future = concurrent.futures.Future()
        self.first_extracted = False
        self.expect_count = 0

    def get_state(self) -> SessionState:
        return self.state

    def get_exception(self) -> Optional[Exception]:
        return self.exception

    def get_future_for_task(self) -> concurrent.futures.Future:
        return self.future

    def create_gen_side_transfer_req_info(self) -> GenReqInfo:
        return GenReqInfo(
            ctx_req_id=self.disagg_params.ctx_request_id,
            instance_name=self.peer_mapper.get_peer_registrar().get_self_rank_info().instance_name,
            instance_rank=self.peer_mapper.get_peer_registrar().get_self_rank_info().instance_rank,
            block_ids=self.kv_slice.block_ids,
            disagg_id=self.disagg_id,
        )

    def extract_trans_meta(self, peer_instance_info, peer_dp_rank) -> AgentRecvArgs:
        peer_domain_targets = self.peer_mapper.get_peer_overlap_targets(
            peer_instance_info, peer_dp_rank
        )
        expect_count = len(peer_domain_targets.ranks)
        if not self.first_extracted:
            self.first_extracted = True
            self.expect_count = expect_count
        return AgentRecvArgs(
            futrure_for_task=self.future,
            expect_count=expect_count,
            remote_name=None,
            slice_id=self.slice_id,
            disagg_id=self.disagg_id,
        ), peer_domain_targets.ranks


class Receiver:
    def __init__(
        self,
        peer_mapper: PeerMapper,
        device_id: int,
        transfer_agent: BaseTransferAgent,
    ):
        self.peer_mapper = peer_mapper
        self.device_id = device_id
        self.transfer_agent = transfer_agent
        self.receive_session_cache = {}
        self.receive_session_cache_lock = threading.Lock()

        self._zmq_context = zmq.Context()
        self.socket_cache = {}
        self.context_endpoint_to_instance_info_cache = {}

        self.receiver_server_socket = self._zmq_context.socket(zmq.ROUTER)
        self.receiver_server_socket.bind(f"tcp://{get_local_ip()}:*")
        self.receiver_server_endpoint = self.receiver_server_socket.getsockopt(
            zmq.LAST_ENDPOINT
        ).decode()
        self._receiver_background_thread = threading.Thread(
            target=self._handle_receiver_loop, daemon=True
        )
        self._receiver_background_thread.start()

        self.slice_tasks = {}  # disagg_id -> list[SliceReceiverTask]

        self.rx_sessions = {}  # disagg_id -> RxSession
        self.last_slice_counts = {}  # disagg_id -> int
        print(f" Receiver init end with receiver_server_endpoint: {self.receiver_server_endpoint}")

    def get_endpoint(self):
        return self.receiver_server_endpoint

    def async_receive_slice(
        self, disagg_params: DisaggregatedParams, slice: KVSlice
    ) -> SliceReceiverTask:
        disagg_id = disagg_params.disagg_id
        if disagg_id not in self.slice_tasks:
            self.slice_tasks[disagg_id] = []

        slice_id = len(self.slice_tasks[disagg_id])
        slice_receiver_task = SliceReceiverTask(
            disagg_params.disagg_id, slice, slice_id, disagg_params, self.peer_mapper
        )
        self.slice_tasks[disagg_id].append(slice_receiver_task)

        self._async_request_data_transfer(slice_receiver_task)

        return slice_receiver_task

    def init_session_resource(self, rx_session: RxSessionBase):
        self.rx_sessions[rx_session.session_args.disagg_params.disagg_id] = weakref.ref(rx_session)

    def clear_session_resource(self, disagg_id: str):
        del self.rx_sessions[disagg_id]
        del self.slice_tasks[disagg_id]

    def _async_request_data_transfer(self, slice_receiver_task: SliceReceiverTask):
        disagg_params = slice_receiver_task.disagg_params
        print(f" _async_request_data_transfer disagg_params: {disagg_params}")
        context_peer_infos: InstanceInfo = self._get_context_info(disagg_params)
        print(f" _async_request_data_transfer context_peer_infos: {context_peer_infos}")
        transfer_gen_side_req_info = slice_receiver_task.create_gen_side_transfer_req_info()

        agent_recv_args, target_ranks = slice_receiver_task.extract_trans_meta(
            context_peer_infos, disagg_params.ctx_dp_rank
        )

        for rank in target_ranks:
            self._send_data_request(
                context_peer_infos.ctx_server_endpoints[rank], transfer_gen_side_req_info
            )
        self.submit_receive_task(agent_recv_args)
        return

    def _need_register_peer_in_first_request(self, disagg_params: DisaggregatedParams) -> bool:
        return disagg_params.ctx_info_endpoint not in self.context_endpoint_to_instance_info_cache

    def _get_socket(self, endpoint: str):
        if endpoint not in self.socket_cache:
            self.socket_cache[endpoint] = self._zmq_context.socket(zmq.DEALER)
            self.socket_cache[endpoint].connect(endpoint)
        return self.socket_cache[endpoint]

    def _get_context_info(self, disagg_params: DisaggregatedParams) -> InstanceInfo:
        if self._need_register_peer_in_first_request(disagg_params):
            socket = self._zmq_context.socket(zmq.DEALER)
            socket.connect(disagg_params.ctx_info_endpoint)
            message = [str(MessageType.REQUEST_INSTANCE_INFO).encode("ascii")]
            socket.send_multipart(message)
            message = socket.recv_multipart()
            instance_info = pickle.loads(message[0])
            print(f" _get_context_info instance_info: {instance_info}")
            socket.close()

            for endpoint in instance_info.ctx_server_endpoints:
                socket = self._get_socket(endpoint)
                send_message = []
                send_message.append(str(MessageType.REGISTER_RANK_INFO).encode("ascii"))
                send_message.append(
                    pickle.dumps(self.peer_mapper.get_peer_registrar().get_self_rank_info())
                )
                socket.send_multipart(send_message)

            self.context_endpoint_to_instance_info_cache[disagg_params.ctx_info_endpoint] = (
                instance_info
            )
            return instance_info

        else:
            return self.context_endpoint_to_instance_info_cache[disagg_params.ctx_info_endpoint]

    def submit_receive_task(self, trans_recv_meta: AgentRecvArgs):
        print(f" call submit_receive_task trans_recv_meta: {trans_recv_meta}")
        if trans_recv_meta.disagg_id not in self.last_slice_counts:
            self.last_slice_counts[trans_recv_meta.disagg_id] = 0
        self.slice_tasks[trans_recv_meta.disagg_id][
            trans_recv_meta.slice_id
        ].state = State.TRANSFERRING

    def _handle_receiver_loop(self):
        while True:
            message = self.receiver_server_socket.recv_multipart()
            send_id = message[0]
            recv_message = message[1:]
            if self._message_is_termination(recv_message):
                break
            elif self._message_is_task_state(recv_message):
                self._handle_task_state(send_id, recv_message)
            else:
                raise ValueError(
                    f"transceiver receiver loop received unknown message type: {recv_message[0]}"
                )

    def _message_is_termination(self, message: list[bytes]):
        return message[0] == str(MessageType.TERMINATION).encode("ascii")

    def _message_is_task_state(self, message: list[bytes]):
        return message[0] == str(MessageType.TASK_STATE).encode("ascii")

    def _handle_task_state(self, send_id: bytes, message: list[bytes]):
        assert len(message) == 5
        assert message[0].decode("ascii") == str(MessageType.TASK_STATE)
        disagg_id = message[1].decode("ascii")
        # peer_slice_id = int(message[2].decode("ascii"))  # Not used
        is_last_slice = bool(message[3].decode("ascii"))
        task_state = message[4].decode("ascii")
        if task_state == "SUCCESS":
            if is_last_slice:
                self.last_slice_counts[disagg_id] += 1
                if self.last_slice_counts[disagg_id] == self.slice_tasks[disagg_id][0].expect_count:
                    self.slice_tasks[disagg_id][0].get_future_for_task().set_result("SUCCESS")
                    self.slice_tasks[disagg_id][0].state = State.FINISHED
                    self.rx_sessions[disagg_id]().session_state.state = State.FINISHED
                    self.rx_sessions[disagg_id]().session_state.finished_tasks.append(
                        0
                    )  # receive task slice only support slice 0
        elif task_state == "FAILED":
            self.slice_tasks[disagg_id][0].get_future_for_task().set_exception(
                RuntimeError(f"Task state: {task_state}")
            )
            self.slice_tasks[disagg_id][0].state = State.ERR
            self.rx_sessions[disagg_id]().session_state.state = State.ERR
        else:
            raise ValueError(f" session {disagg_id} received unknown task state: {task_state}")

    def _send_data_request(self, endpoint: str, transfer_gen_side_req_info: GenReqInfo):
        print(
            f" call _send_data_request endpoint: {endpoint} transfer_gen_side_req_info: {transfer_gen_side_req_info}"
        )
        socket = self._get_socket(endpoint)
        send_message = []
        send_message.append(str(MessageType.REQUEST_DATA).encode("ascii"))
        send_message.append(pickle.dumps(transfer_gen_side_req_info))
        socket.send_multipart(send_message)


class RxSession(RxSessionBase):
    def __init__(self, request_id: int, disagg_params: DisaggregatedParams, receiver: Receiver):
        super().__init__(receiver, SessionArgsBase(request_id, disagg_params))
        self.request_id = request_id

        self.disagg_params = disagg_params
        self.disagg_id = disagg_params.disagg_id
        self.receiver = receiver
        self.receiver.init_session_resource(self)
        self.exception = None
        self.slice_tasks = []
        self.session_state = SessionState(state=State.INIT, finished_tasks=[])

    def receive(self, slice: KVSlice) -> TaskIdType:
        self.slice_tasks.append(self.receiver.async_receive_slice(self.disagg_params, slice))
        task_id = self.slice_tasks[-1].slice_id
        return task_id

    def get_state(self) -> SessionState:
        return self.session_state

    def poll_task(self, id: TaskIdType) -> State:
        return self.slice_tasks[id].get_state()

    def get_exception(self) -> Optional[Exception]:
        return self.exception

    def __del__(self):
        self.receiver.clear_session_resource(self.disagg_id)


class TransferAgentConfig:
    pass


class InstanceInfoServer:
    def __init__(self, instance_info: InstanceInfo, addr: str = None, port: int = None):
        self.instance_info = instance_info

        self._zmq_context = zmq.Context()
        self.server_socket = self._zmq_context.socket(zmq.ROUTER)
        if addr is None and port is None:
            self.server_socket.bind(f"tcp://{get_local_ip()}:*")
        else:
            self.server_socket.bind(f"tcp://{addr}:{port}")
        self.server_endpoint = self.server_socket.getsockopt(zmq.LAST_ENDPOINT).decode()
        self._instance_info_server_thread = threading.Thread(
            target=self._loop_handle_request, daemon=True
        )
        self._instance_info_server_thread.start()

    def get_instance_info(self) -> InstanceInfo:
        return self.instance_info

    def get_endpoint(self) -> str:
        return self.server_endpoint

    def _loop_handle_request(self):
        while True:
            message = self.server_socket.recv_multipart()
            send_id = message[0]
            recv_message = message[1:]
            if self._message_is_termination(recv_message):
                break
            elif self._message_is_request_instance_info(recv_message):
                self._handle_request_instance_info(send_id, recv_message)
            else:
                raise ValueError(
                    f" instance info server received unknown message type: {recv_message[0]}"
                )

    def _message_is_termination(self, message: list[bytes]):
        return message[0] == str(MessageType.TERMINATION).encode("ascii")

    def _message_is_request_instance_info(self, message: list[bytes]):
        return message[0] == str(MessageType.REQUEST_INSTANCE_INFO).encode("ascii")

    def _handle_request_instance_info(self, send_id: bytes, message: list[bytes]):
        self.server_socket.send_multipart([send_id, pickle.dumps(self.instance_info)])


class TransferWorker:
    def __init__(
        self,
        kv_cache_manager: KVCacheManager,
        mapping: Mapping,
        device_id: int,
        instance_name: str,
        transfer_agent_config: TransferAgentConfig,
    ):
        self.mapping = mapping

        self.instance_info: InstanceInfo = None
        self.instance_rank_info: RankInfo = None
        self.kv_cache_manager = kv_cache_manager
        self.init_instance_info(instance_name)

        self.instance_info_server = InstanceInfoServer(self.instance_info)
        self.peer_registrar = PeerRegistrar(self.instance_rank_info, self.instance_info)
        self.peer_mapper = PeerMapper(self.peer_registrar, self.kv_cache_manager)
        self.device_id = device_id
        self.transfer_agent = NixlTransferAgent(
            self.instance_rank_info.instance_name + str(self.instance_rank_info.instance_rank), True
        )

        self._register_kv_cache()

        self.sender = Sender(self.peer_mapper, device_id, self.transfer_agent)
        self.receiver = Receiver(self.peer_mapper, device_id, self.transfer_agent)
        self.instance_rank_info.transfer_engine_info = bytes(
            self.transfer_agent.get_local_agent_desc()
        )
        self.instance_rank_info.server_endpoint = self.sender.get_endpoint()
        self.instance_rank_info.recv_endpoint = self.receiver.get_endpoint()

    def update_instance_info_with_collective_info(
        self, update_endpoints: list[str], update_layer_num_per_pp: list[int]
    ):
        self.instance_info.ctx_server_endpoints = update_endpoints
        self.instance_info.layer_num_per_pp = update_layer_num_per_pp
        self.instance_rank_info.layer_num_per_pp = update_layer_num_per_pp

    def create_sender_session(self, request: LlmRequest) -> TxSession:
        return TxSession(
            request_id=request.py_request_id,
            disagg_params=request.py_disaggregated_params,
            sender=self.sender,
        )

    def create_receiver_session(self, request: LlmRequest) -> RxSession:
        return RxSession(
            request_id=request.py_request_id,
            disagg_params=request.py_disaggregated_params,
            receiver=self.receiver,
        )

    def init_instance_info(self, instance_name):
        rank = self.mapping.rank

        tp_size = self.mapping.tp_size
        pp_size = self.mapping.pp_size
        dp_size = self.mapping.dp_size
        cp_size = self.mapping.cp_size
        tp_rank = self.mapping.tp_rank
        pp_rank = self.mapping.pp_rank
        enable_attention_dp = self.mapping.enable_attention_dp
        dp_rank = 0
        if enable_attention_dp:
            dp_size = self.mapping.tp_size
            dp_rank = tp_rank
        cp_rank = self.mapping.cp_rank
        is_mla = self.kv_cache_manager.kv_factor == 1
        self.kv_cache_manager.kv_factor
        heads_num_per_rank = self.kv_cache_manager.num_kv_heads
        tokens_per_block = self.kv_cache_manager.tokens_per_block
        dims_per_head = self.kv_cache_manager.head_dim
        element_size = get_size_in_bytes(1, self.kv_cache_manager.dtype)
        layer_num_per_pp = [self.kv_cache_manager.num_layers]
        ctx_server_endpoints = []
        self.instance_info = InstanceInfo(
            instance_name=instance_name,
            tp_size=tp_size,
            pp_size=pp_size,
            dp_size=dp_size,
            cp_size=cp_size,
            kv_head_num_per_rank=heads_num_per_rank,
            tokens_per_block=tokens_per_block,
            dims_per_head=dims_per_head,
            element_size=element_size,
            enable_attention_dp=enable_attention_dp,
            is_mla=is_mla,
            layer_num_per_pp=layer_num_per_pp,
            ctx_server_endpoints=ctx_server_endpoints,
        )
        self.instance_rank_info = RankInfo(
            instance_name=instance_name,
            instance_rank=rank,
            tp_size=tp_size,
            tp_rank=tp_rank,
            pp_size=pp_size,
            pp_rank=pp_rank,
            dp_size=dp_size,
            dp_rank=dp_rank,
            cp_size=cp_size,
            cp_rank=cp_rank,
            kv_head_num_per_rank=heads_num_per_rank,
            tokens_per_block=tokens_per_block,
            dims_per_head=dims_per_head,
            element_size=element_size,
            enable_attention_dp=enable_attention_dp,
            is_mla=is_mla,
            layer_num_per_pp=layer_num_per_pp,
            kvcache_ptrs=[self.kv_cache_manager.get_unique_primary_pool().data_ptr()],
            aux_ptrs=[],
            server_endpoint="",
            recv_endpoint="",
            transfer_engine_info=bytes(),
        )

    def _register_kv_cache(self):
        memory_pool = self.kv_cache_manager.get_unique_primary_pool()
        memory_desc = (
            memory_pool.data_ptr(),
            memory_pool.numel() * memory_pool.element_size(),
            self.device_id,
            "kv_cache_memory",
        )
        reg_memory_desc = RegMemoryDescs("VRAM", [memory_desc])
        self.transfer_agent.register_memory(reg_memory_desc)
        print(f"  transfer agent register kv cache memory: {memory_desc}")


def test_transfer_worker():
    mapping = Mapping(world_size=1, rank=0)
    num_layers = 2
    head_dim = 128
    num_kv_heads = 4
    tokens_per_block = 8
    max_seq_len = 256
    max_batch_size = 1
    dtype = DataType.FLOAT

    ctx_kv_cache_manager = KVCacheManager(
        trtllm.KvCacheConfig(
            max_tokens=2048,
            enable_block_reuse=False,
        ),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        mapping=mapping,
        dtype=dtype,
    )

    ctx_instance_name = "ctx_instance"
    transfer_agent_config = TransferAgentConfig()
    ctx_transfer_worker = TransferWorker(
        kv_cache_manager=ctx_kv_cache_manager,
        mapping=mapping,
        device_id=0,
        instance_name=ctx_instance_name,
        transfer_agent_config=transfer_agent_config,
    )
    ctx_enpoint = ctx_transfer_worker.sender.server_endpoint
    ctx_layer_num_per_pp = [num_layers]
    ctx_transfer_worker.update_instance_info_with_collective_info(
        update_endpoints=[ctx_enpoint], update_layer_num_per_pp=ctx_layer_num_per_pp
    )
    gen_kv_cache_manager = KVCacheManager(
        trtllm.KvCacheConfig(
            max_tokens=2048,
            enable_block_reuse=False,
        ),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        mapping=mapping,
        dtype=dtype,
    )

    gen_instance_name = "gen_instance"

    gen_transfer_worker = TransferWorker(
        kv_cache_manager=gen_kv_cache_manager,
        mapping=mapping,
        device_id=0,
        instance_name=gen_instance_name,
        transfer_agent_config=transfer_agent_config,
    )

    gen_enpoint = gen_transfer_worker.sender.server_endpoint
    gen_layer_num_per_pp = [num_layers]
    gen_transfer_worker.update_instance_info_with_collective_info(
        update_endpoints=[gen_enpoint], update_layer_num_per_pp=gen_layer_num_per_pp
    )

    sampling_params = SamplingParams()

    request_len = 32
    disagg_id = str(uuid.uuid4())
    ctx_request = LlmRequest(
        request_id=0,
        max_new_tokens=1,
        input_tokens=list(range(request_len)),
        sampling_config=tensorrt_llm.bindings.SamplingConfig(
            sampling_params._get_sampling_config()
        ),
        is_streaming=False,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY,
    )
    ctx_request.py_disaggregated_params = DisaggregatedParams(disagg_id=disagg_id)

    ctx_kv_cache_manager.impl.add_sequence(
        ctx_request.py_request_id, ctx_request.prompt_len, 1, ctx_request
    )
    ctx_block_ids = ctx_kv_cache_manager.get_batch_cache_indices([ctx_request.py_request_id])[0]

    ctx_block_data_pools = ctx_kv_cache_manager.get_unique_primary_pool()

    random_values = torch.rand(
        ctx_block_data_pools.shape, dtype=torch.float32, device=ctx_block_data_pools.device
    )
    ctx_block_data_pools.copy_(random_values)

    gen_request = LlmRequest(
        request_id=1,
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
        ctx_dp_rank=0,
        ctx_info_endpoint=ctx_transfer_worker.instance_info_server.get_endpoint(),
        disagg_id=disagg_id,
    )
    gen_kv_cache_manager.impl.add_sequence(
        gen_request.py_request_id, gen_request.prompt_len, 1, gen_request
    )
    print("gen async_receive before")
    recv_session = gen_transfer_worker.create_receiver_session(gen_request)

    time.sleep(0.1)
    print("gen async_receive after")
    print("ctx async_send before")
    send_session = ctx_transfer_worker.create_sender_session(ctx_request)
    print("ctx async_send after")

    context_block_ids = ctx_kv_cache_manager.get_batch_cache_indices([ctx_request.py_request_id])[0]
    print(f"context_block_ids: {context_block_ids}")
    gen_block_ids = gen_kv_cache_manager.get_batch_cache_indices([gen_request.py_request_id])[0]
    print(f"gen_block_ids: {gen_block_ids}")

    recv_kv_slice = KVSlice(is_last_slice=True, block_ids=gen_block_ids)
    recv_task_id = recv_session.receive(recv_kv_slice)

    send_kv_slice = KVSlice(is_last_slice=True, block_ids=ctx_block_ids)
    send_task_id = send_session.send(send_kv_slice)

    send_slice_task = send_session.slice_tasks[send_task_id]
    recv_slice_task = recv_session.slice_tasks[recv_task_id]

    print(
        f" gen kvcachemanager pool ptr: {gen_kv_cache_manager.get_unique_primary_pool().data_ptr()}"
    )
    print(
        f" ctx kvcachemanager pool ptr: {ctx_kv_cache_manager.get_unique_primary_pool().data_ptr()}"
    )
    print(
        f"send_slice_task future_result: {send_slice_task.get_future_for_task().result()}, "
        f"state: {send_slice_task.get_state()}"
    )
    print(f"send session state: {send_session.get_state()}")
    print(
        f"recv_slice_task future_result: {recv_slice_task.get_future_for_task().result()}, "
        f"state: {recv_slice_task.get_state()}"
    )
    print(f"recv session state: {recv_session.get_state()}")
    ctx_block_datas = ctx_kv_cache_manager.get_unique_primary_pool()[ctx_block_ids]
    print(
        f"ctx_block_datas: {ctx_block_datas}, ctx_block_datas.shape: {ctx_block_datas.shape}, "
        f"ctx_block_datas.data_ptr: {ctx_block_datas.data_ptr()}"
    )

    gen_block_datas = gen_kv_cache_manager.get_unique_primary_pool()[gen_block_ids]
    print(
        f"gen_block_datas: {gen_block_datas}, gen_block_datas.shape: {gen_block_datas.shape}, "
        f"gen_block_datas.data_ptr: {gen_block_datas.data_ptr()}"
    )
    assert ctx_block_datas.equal(gen_block_datas)


if __name__ == "__main__":
    test_transfer_worker()
