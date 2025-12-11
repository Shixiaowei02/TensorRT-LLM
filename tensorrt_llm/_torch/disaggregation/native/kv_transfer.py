import concurrent
import threading
import weakref
from dataclasses import asdict, dataclass
from typing import List, Optional

import msgpack
import zmq

import tensorrt_llm.bindings
from tensorrt_llm import Mapping, logger
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
from tensorrt_llm._torch.disaggregation.native.aux_buffer import AuxBuffer
from tensorrt_llm._torch.disaggregation.native.kv_mapper import (
    InstanceInfo,
    PeerMapper,
    PeerOverlapTargets,
    PeerRegistrar,
    RankInfo,
)
from tensorrt_llm._torch.disaggregation.native.utils import get_local_ip
from tensorrt_llm._torch.disaggregation.nixl.agent import BindingsNixlTransferAgent
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import get_size_in_bytes, nvtx_range
from tensorrt_llm.disaggregated_params import DisaggregatedParams

AttentionTypeCpp = tensorrt_llm.bindings.internal.batch_manager.AttentionType
LlmRequestType = tensorrt_llm.bindings.internal.batch_manager.LlmRequestType


@dataclass
class GenReqInfo:
    ctx_req_id: int
    instance_name: str
    instance_rank: int
    blocks: list[int]
    disagg_id: str
    start_token_idx: Optional[int] = None
    aux_slot: Optional[int] = None

    def to_bytes(self) -> bytes:
        return msgpack.packb(asdict(self))

    @classmethod
    def from_bytes(cls, data: bytes) -> "GenReqInfo":
        return cls(**msgpack.unpackb(data, raw=False))


@dataclass
class AgentRecvArgs:
    disagg_id: str
    future_for_task: concurrent.futures.Future
    expected_count: int
    peer_name: str
    slice_id: int


@dataclass
class AgentSendArgs:
    future_for_task: concurrent.futures.Future

    expected_count: int
    peer_name: str
    src_kv_ptrs: List[int] = None
    dst_kv_ptrs: List[int] = None
    kv_sizes: List[int] = None
    dst_device_id: int = None
    src_aux_ptrs: List[int] = None
    dst_aux_ptrs: List[int] = None
    aux_sizes: List[int] = None
    peer_endpoint: Optional[str] = None  # used for send state
    disagg_id: Optional[str] = None
    slice_id: Optional[int] = None
    is_last_slice: Optional[bool] = False
    is_only_meta_data: Optional[bool] = False


class MessageType:
    TERMINATION = "TERMINATION"
    TASK_STATE = "TASK_STATE"
    PEER_INFO = "PEER_INFO"
    REQUEST_DATA = "REQUEST_DATA"
    REQUEST_INSTANCE_INFO = "REQUEST_INSTANCE_INFO"
    REGISTER_RANK_INFO = "REQUEST_RANK_INFO"
    META_SEND_STATE = "META_SEND_STATE"


class AuxSendTask:
    def __init__(self, params: DisaggregatedParams, slot: int, mapper: PeerMapper):
        self._params = params
        self._disagg_id = params.disagg_id
        self._slot = slot
        self._mapper = mapper
        self._state = State.INIT
        self._future = concurrent.futures.Future()

    @property
    def state(self) -> State:
        return self._state

    @state.setter
    def state(self, s: State):
        self._state = s

    @property
    def future(self) -> concurrent.futures.Future:
        return self._future

    def extract_trans_meta(self, req_info: GenReqInfo) -> AgentSendArgs:
        peer_aux_meta = (
            self._mapper.get_peer_registrar()
            .get_peer_rank_info(req_info.instance_name, req_info.instance_rank)
            .aux_meta
        )

        peer_slot = req_info.aux_slot

        src_aux_meta = self._mapper.get_peer_registrar().get_self_rank_info().aux_meta

        src_ptrs = [
            ptr + item_size * self._slot
            for ptr, item_size in zip(src_aux_meta.ptrs, src_aux_meta.item_sizes)
        ]
        dst_ptrs = [
            ptr + item_size * peer_slot
            for ptr, item_size in zip(peer_aux_meta.ptrs, peer_aux_meta.item_sizes)
        ]
        size = [item_size for item_size in src_aux_meta.item_sizes]
        return AgentSendArgs(
            future_for_task=self._future,
            src_aux_ptrs=src_ptrs,
            dst_aux_ptrs=dst_ptrs,
            aux_sizes=size,
            expected_count=1,
            is_only_meta_data=True,
            peer_name=req_info.instance_name + str(req_info.instance_rank),
            peer_endpoint=self._mapper.get_peer_registrar()
            .get_peer_rank_info(req_info.instance_name, req_info.instance_rank)
            .recv_endpoint,
            disagg_id=self._disagg_id,
        )


class KVSendTask:
    def __init__(
        self,
        kv_slice: KVSlice,
        params: DisaggregatedParams,
        slice_id: int,
        mapper: PeerMapper,
    ):
        self._mapper = mapper
        self._future = concurrent.futures.Future()
        self._first = False
        self._enc = 0
        self._expected = 0
        self._slice = kv_slice
        self._params = params
        self._disagg_id = params.disagg_id
        self._slice_id = slice_id
        self._state = State.INIT
        self._transferred = 0

    @property
    def state(self) -> State:
        return self._state

    @state.setter
    def state(self, s: State):
        self._state = s

    @property
    def future(self) -> concurrent.futures.Future:
        return self._future

    @property
    def slice_id(self) -> int:
        return self._slice_id

    @property
    def transferred_count(self) -> int:
        return self._transferred

    @transferred_count.setter
    def transferred_count(self, v: int):
        self._transferred = v

    @nvtx_range("extract_trans_meta")
    def extract_trans_meta(self, req_info: GenReqInfo) -> AgentSendArgs:
        peer_ri = self._mapper.get_peer_registrar().get_peer_rank_info(
            req_info.instance_name, req_info.instance_rank
        )
        peer_overlap_targets = self._mapper.get_peer_overlap_targets(peer_ri, peer_ri.dp_rank)
        expected_count = len(peer_overlap_targets.ranks)
        if not self._first:
            self._first = True
            self._expected = expected_count
        self._enc = self._enc + 1
        if not self._need_send_transfer(peer_overlap_targets, peer_ri):
            return AgentSendArgs(
                future_for_task=self._future,
                src_kv_ptrs=[],
                dst_kv_ptrs=[],
                kv_sizes=[],
                expected_count=expected_count,
                peer_name=peer_ri.instance_name + str(peer_ri.instance_rank),
                peer_endpoint=peer_ri.recv_endpoint,
                disagg_id=self._disagg_id,
                slice_id=self._slice_id,
                is_last_slice=self._slice.is_last_slice,
            )

        dst_device_id = peer_ri.device_id
        dst_blocks = req_info.blocks
        src_blocks = self._slice.blocks

        if len(src_blocks) + 1 == len(dst_blocks):
            # FIXME: this is a temporary solution, need to be fixed for the draft tokens
            logger.warning(
                " src_block_num is one less than dst_block_num, maybe it is due to draft tokens,"
                " remove the last block from dst_block_ids "
            )
            dst_blocks = dst_blocks[:-1]
        src_blocks, dst_blocks = self._filter_kv_blocks(src_blocks, dst_blocks)

        kv_ptr_extractor = self._mapper.kv_ptr_extractor
        src_kv_blocks_ptrs = kv_ptr_extractor.extract_kv_block_ptrs(src_blocks)
        self_kv_block_size = kv_ptr_extractor.kv_pool_attributes.kv_cache_block_sizes[0]
        peer_kv_ptr_extractor = self._mapper.get_kv_extractor(
            peer_ri.instance_name, peer_ri.instance_rank
        )
        dst_kv_blocks_ptrs = peer_kv_ptr_extractor.extract_kv_block_ptrs(dst_blocks)
        peer_kv_block_size = peer_kv_ptr_extractor.kv_pool_attributes.kv_cache_block_sizes[0]
        logger.debug(
            f"KVSendTask.extract_trans_meta: extracted KV block pointers -> "
            f"src_ptrs={src_kv_blocks_ptrs}, src_block_size={self_kv_block_size}, "
            f"dst_ptrs={dst_kv_blocks_ptrs}, dst_block_size={peer_kv_block_size}"
        )
        transformer = self._mapper.get_kv_ptrs_mapper(peer_ri)
        (
            src_kv_blocks_transfer_ptrs,
            dst_kv_blocks_transfer_ptrs,
            dst_kv_blocks_size,
        ) = transformer(
            src_kv_blocks_ptrs, self_kv_block_size, dst_kv_blocks_ptrs, peer_kv_block_size
        )

        src_kv_blocks_size = dst_kv_blocks_size
        logger.debug(
            f"KVSendTask.extract_trans_meta: mapped KV pointers for transfer -> "
            f"src_transfer_ptrs={src_kv_blocks_transfer_ptrs}, src_transfer_block_size={src_kv_blocks_size}, "
            f"dst_transfer_ptrs={dst_kv_blocks_transfer_ptrs}, dst_transfer_block_size={dst_kv_blocks_size}"
        )

        return AgentSendArgs(
            future_for_task=self._future,
            src_kv_ptrs=src_kv_blocks_transfer_ptrs,
            dst_kv_ptrs=dst_kv_blocks_transfer_ptrs,
            kv_sizes=[src_kv_blocks_size] * len(src_kv_blocks_transfer_ptrs),
            dst_device_id=dst_device_id,
            expected_count=expected_count,
            peer_name=peer_ri.instance_name + str(peer_ri.instance_rank),
            peer_endpoint=peer_ri.recv_endpoint,
            disagg_id=self._disagg_id,
            slice_id=self._slice_id,
            is_last_slice=self._slice.is_last_slice,
        )

    def is_active(self) -> bool:
        if self._first:
            return self._enc < self._expected
        else:
            return True

    def _need_send_transfer(
        self, peer_overlap_targets: PeerOverlapTargets, peer_rank_info: RankInfo
    ) -> bool:
        if peer_overlap_targets.duplicate_head_factor <= 1:
            return True
        peer_ri = peer_rank_info
        peer_dp_rank = peer_ri.dp_rank if peer_ri.enable_attention_dp else 0
        self_tp_size_per_dp_group = (
            self._mapper.get_peer_registrar().get_self_rank_info().tp_size
            // self._mapper.get_peer_registrar().get_self_rank_info().dp_size
            if self._mapper.get_peer_registrar().get_self_rank_info().enable_attention_dp
            else self._mapper.get_peer_registrar().get_self_rank_info().tp_size
        )
        self_tprank_in_dp_group = (
            self._mapper.get_peer_registrar().get_self_rank_info().tp_rank
            % self_tp_size_per_dp_group
        )
        return (peer_dp_rank % peer_overlap_targets.duplicate_head_factor) == (
            self_tprank_in_dp_group % peer_overlap_targets.duplicate_head_factor
        )

    def _filter_kv_blocks(self, src_blocks, dst_blocks) -> tuple[list[int], list[int]]:
        # TODO: filter the kv blocks according to the peer_overlap_targets
        return src_blocks, dst_blocks


class Sender:
    def __init__(
        self,
        mapper: PeerMapper,
        device_id: int,
        agent: BaseTransferAgent,
    ):
        self._mapper = mapper
        self._device_id = device_id
        self._agent = agent
        self._send_session_lock = threading.Lock()

        self._peer_req_cache = {}
        self._peer_req_lock = threading.Lock()

        self._zmq_context = zmq.Context()
        self._socket = self._zmq_context.socket(zmq.ROUTER)
        self._socket.bind(f"tcp://{get_local_ip()}:*")
        self._endpoint = self._socket.getsockopt(zmq.LAST_ENDPOINT).decode()

        self._socket_cache = {}

        self._slice_tasks = {}  # disagg_id -> list[KVSlice]

        self._meta_tasks = {}  # disagg_id -> AuxSendTask
        logger.info(f" Sender init end with endpoint: {self.endpoint}")

        background_thread = threading.Thread(target=self._handle_sender_loop, daemon=True)
        background_thread.start()

        self._tx_sessions = {}  # disagg_id -> TxSession

    @property
    def endpoint(self):
        return self._endpoint

    def init_session_resource(self, tx_session: TxSessionBase):
        self._tx_sessions[tx_session.session_args.disagg_params.disagg_id] = weakref.ref(tx_session)
        return

    def _get_tx_session(self, disagg_id: str) -> TxSessionBase:
        session_ref = self._tx_sessions[disagg_id]
        if session_ref is None:
            raise RuntimeError(f"TxSession {disagg_id} not found")
        session = session_ref()
        if session is None:
            raise RuntimeError(f"TxSession {disagg_id} has been cleared")
        return session

    def clear_session_resource(self, disagg_id: str):
        logger.debug(f"Clearing session resources for disagg_id={disagg_id}")
        self._slice_tasks.pop(disagg_id, None)
        self._peer_req_cache.pop(disagg_id, None)
        return

    def async_send_slice(self, params: DisaggregatedParams, slice: KVSlice) -> KVSendTask:
        if params.disagg_id not in self._slice_tasks:
            self._slice_tasks[params.disagg_id] = []
        slice_id = len(self._slice_tasks[params.disagg_id])
        new_slice_task = KVSendTask(slice, params, slice_id, self._mapper)
        self._slice_tasks[params.disagg_id].append(new_slice_task)

        self._handle_send_task(new_slice_task)
        return new_slice_task

    def async_send_meta_data(self, params: DisaggregatedParams, slot: int):
        meta_sender_task = AuxSendTask(params, slot, self._mapper)

        self._meta_tasks[params.disagg_id] = meta_sender_task
        self._handle_send_task(meta_sender_task)
        return meta_sender_task

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
            if agent_send_args.is_only_meta_data:
                self._submit_send_meta_to_agent(agent_send_args)
            else:
                self._submit_send_slice_to_agent(agent_send_args)

    @nvtx_range("submit_send_slice_to_agent")
    def _submit_send_slice_to_agent(self, agent_send_args: AgentSendArgs):
        assert len(agent_send_args.src_kv_ptrs) == len(agent_send_args.dst_kv_ptrs)
        assert len(agent_send_args.kv_sizes) == len(agent_send_args.src_kv_ptrs)
        src_kv_list = [
            (src_ptr, size, self._device_id)
            for src_ptr, size in zip(agent_send_args.src_kv_ptrs, agent_send_args.kv_sizes)
        ]

        # TODO : device_id should be the device id of the destination
        dst_kv_list = [
            (dst_ptr, size, agent_send_args.dst_device_id)
            for dst_ptr, size in zip(agent_send_args.dst_kv_ptrs, agent_send_args.kv_sizes)
        ]
        if self._get_tx_session(agent_send_args.disagg_id).state.state != State.ERR:
            self._get_tx_session(agent_send_args.disagg_id).state.state = State.TRANSFERRING

        src_memory_descs = MemoryDescs("VRAM", src_kv_list)
        dst_memory_descs = MemoryDescs("VRAM", dst_kv_list)
        request = TransferRequest(
            TransferOp.WRITE, src_memory_descs, dst_memory_descs, agent_send_args.peer_name, None
        )

        skip_send = len(src_kv_list) == 0
        logger.debug(f"Submitting transfer request to transfer agent: {request}")
        if not skip_send:
            src_memory_descs = MemoryDescs("VRAM", src_kv_list)
            dst_memory_descs = MemoryDescs("VRAM", dst_kv_list)
            request = TransferRequest(
                TransferOp.WRITE,
                src_memory_descs,
                dst_memory_descs,
                agent_send_args.peer_name,
                None,
            )
            status = self._agent.submit_transfer_requests(request)
        sync_status = "SUCCESS"
        if not skip_send and not status.wait():
            sync_status = "FAILED"
            agent_send_args.future_for_task.set_exception(RuntimeError("Transfer failed"))

            self._slice_tasks[agent_send_args.disagg_id][agent_send_args.slice_id].state = State.ERR
            self._get_tx_session(agent_send_args.disagg_id).state.state = State.ERR
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
        # update transferred_count via property
        curr = self._slice_tasks[agent_send_args.disagg_id][
            agent_send_args.slice_id
        ].transferred_count
        self._slice_tasks[agent_send_args.disagg_id][agent_send_args.slice_id].transferred_count = (
            curr + 1
        )

        if (
            self._slice_tasks[agent_send_args.disagg_id][agent_send_args.slice_id].transferred_count
            > agent_send_args.expected_count
        ):
            agent_send_args.future_for_task.set_exception(
                RuntimeError(
                    f"Session {agent_send_args.disagg_id} has more than {agent_send_args.expected_count} transfers"
                )
            )
            # TODO: set exception for the session ?
            self._get_tx_session(agent_send_args.disagg_id).state.state = State.ERR
        elif (
            self._slice_tasks[agent_send_args.disagg_id][agent_send_args.slice_id].transferred_count
            == agent_send_args.expected_count
        ):
            # TODO avoid set_result if tranfser failed since it has been set exception
            agent_send_args.future_for_task.set_result(sync_status)
            self._slice_tasks[agent_send_args.disagg_id][
                agent_send_args.slice_id
            ].state = State.FINISHED
            self._get_tx_session(agent_send_args.disagg_id).state.finished_tasks.append(
                agent_send_args.slice_id
            )
            if agent_send_args.is_last_slice:
                self._get_tx_session(agent_send_args.disagg_id).state.state = State.FINISHED

    @nvtx_range("submit_send_meta_to_agent")
    def _submit_send_meta_to_agent(self, agent_send_args: AgentSendArgs):
        # TODO:  submit the meta data task to the transfer agent
        assert agent_send_args.is_only_meta_data is True
        assert agent_send_args.src_aux_ptrs is not None

        src_aux_list = [
            (src_ptr, size, 0)
            for src_ptr, size in zip(agent_send_args.src_aux_ptrs, agent_send_args.aux_sizes)
        ]
        dst_aux_list = [
            (dst_ptr, size, 0)
            for dst_ptr, size in zip(agent_send_args.dst_aux_ptrs, agent_send_args.aux_sizes)
        ]
        src_memory_descs = MemoryDescs("DRAM", src_aux_list)
        dst_memory_descs = MemoryDescs("DRAM", dst_aux_list)
        request = TransferRequest(
            TransferOp.WRITE, src_memory_descs, dst_memory_descs, agent_send_args.peer_name, None
        )

        logger.debug(f"Submitting metadata transfer request to transfer agent: {request}")

        status = self._agent.submit_transfer_requests(request)
        sync_status = "SUCCESS"
        if not status.wait():
            sync_status = "FAILED"
            agent_send_args.future_for_task.set_exception(RuntimeError("Transfer failed"))
            self._get_tx_session(agent_send_args.disagg_id).state.state = State.ERR
        socket = self._get_socket(agent_send_args.peer_endpoint)
        socket.send_multipart(
            [
                str(MessageType.META_SEND_STATE).encode("ascii"),
                agent_send_args.disagg_id.encode("ascii"),
                sync_status.encode("ascii"),
            ]
        )
        self._get_tx_session(agent_send_args.disagg_id).state.state = State.META_DATA_SENT

    def _handle_send_task(self, send_slice_task: KVSendTask | AuxSendTask):
        send_slice_task.state = State.TRANSFERRING
        transfer_recv_req_info_dict = {}
        with self._peer_req_lock:
            if send_slice_task._disagg_id in self._peer_req_cache:
                transfer_recv_req_info_dict = self._peer_req_cache[send_slice_task._disagg_id]
        for transfer_recv_req_info in transfer_recv_req_info_dict.values():
            trans_meta = send_slice_task.extract_trans_meta(transfer_recv_req_info)
            self.submit_send_task(trans_meta)

    def _handle_sender_loop(self):
        while True:
            message = self._socket.recv_multipart()
            send_id = message[0]
            recv_message = message[1:]
            if self._message_is_termination(recv_message):
                break
            elif self._message_is_request_data(recv_message):
                self._handle_request_data(send_id, recv_message)
            elif self._message_is_register_rank_info(recv_message):
                self._handle_register_rank_info(send_id, recv_message)
            else:
                raise ValueError(
                    f"transceiver sender loop received unknown message type: {recv_message[0]}"
                )

    def _message_is_termination(self, message: list[bytes]):
        return message[0] == str(MessageType.TERMINATION).encode("ascii")

    def _message_is_request_data(self, message: list[bytes]):
        return message[0] == str(MessageType.REQUEST_DATA).encode("ascii")

    def _message_is_register_rank_info(self, message: list[bytes]):
        return message[0] == str(MessageType.REGISTER_RANK_INFO).encode("ascii")

    def _handle_register_rank_info(self, send_id: bytes, message: list[bytes]):
        ri: RankInfo = RankInfo.from_bytes(message[1])

        self._mapper.get_peer_registrar().register(ri.instance_name, ri.instance_rank, ri)

        agent_name = ri.instance_name + str(ri.instance_rank)
        logger.debug(f"Loading remote transfer agent descriptor for peer '{agent_name}'")
        self._agent.load_remote_agent(
            ri.instance_name + str(ri.instance_rank),
            ri.transfer_engine_info,
        )
        logger.debug(
            f"Completed handling REGISTER_RANK_INFO for instance='{ri.instance_name}', rank={ri.instance_rank}"
        )

    def _handle_request_data(self, send_id: bytes, message: list[bytes]):
        gen_req_info: GenReqInfo = GenReqInfo.from_bytes(message[1])

        send_slice_tasks: list[KVSendTask] = self._get_send_slice_tasks(gen_req_info.disagg_id)
        self._save_peer_transfer_req_info(gen_req_info)
        if send_slice_tasks is None:
            pass
        else:
            for send_slice_task in send_slice_tasks:
                trans_meta = send_slice_task.extract_trans_meta(gen_req_info)
                self.submit_send_task(trans_meta)

    def _get_send_slice_tasks(self, disagg_id: str):
        if disagg_id not in self._slice_tasks:
            return None
        return self._slice_tasks[disagg_id]

    def _get_socket(self, endpoint: str):
        if endpoint is None:
            raise ValueError("endpoint is None")
        if endpoint not in self._socket_cache:
            self._socket_cache[endpoint] = self._zmq_context.socket(zmq.DEALER)
            self._socket_cache[endpoint].connect(endpoint)
        return self._socket_cache[endpoint]

    def _save_peer_transfer_req_info(self, peer_transfer_req_info: GenReqInfo):
        with self._peer_req_lock:
            if peer_transfer_req_info.disagg_id not in self._peer_req_cache:
                self._peer_req_cache[peer_transfer_req_info.disagg_id] = {}
            self._peer_req_cache[peer_transfer_req_info.disagg_id][
                peer_transfer_req_info.instance_rank
            ] = peer_transfer_req_info
            peer_ri = self._mapper.get_peer_registrar().get_peer_rank_info(
                peer_transfer_req_info.instance_name, peer_transfer_req_info.instance_rank
            )
            expected_count = len(
                self._mapper.get_peer_overlap_targets(
                    peer_ri, peer_transfer_req_info.instance_rank
                ).ranks
            )
            if expected_count == len(self._peer_req_cache[peer_transfer_req_info.disagg_id]):
                if peer_transfer_req_info.disagg_id in self._tx_sessions:
                    self._get_tx_session(peer_transfer_req_info.disagg_id).state.state = State.READY


class TxSession(TxSessionBase):
    def __init__(
        self, request_id: int, disagg_params: DisaggregatedParams, sender: Sender, aux_slot: int
    ):
        super().__init__(sender, SessionArgsBase(request_id, disagg_params))
        self.request_id = request_id
        self._sender = sender
        self._state = SessionState(state=State.INIT, finished_tasks=[])
        self._exception = None
        self.slice_tasks = []  # slice_id -> SliceTxSession
        self._sender.init_session_resource(self)
        self.aux_slot = aux_slot
        self._closed = False

    @property
    def state(self) -> SessionState:
        return self._state

    @state.setter
    def state(self, s: SessionState):
        self._state = s

    def send(self, slice: KVSlice) -> TaskIdType:
        slice_sender_task = self._sender.async_send_slice(self.session_args.disagg_params, slice)
        self.slice_tasks.append(slice_sender_task)
        return slice_sender_task.slice_id

    def send_meta_data(self):
        return self._sender.async_send_meta_data(self.session_args.disagg_params, self.aux_slot)

    def poll_task(self, id: TaskIdType) -> State:
        return self.slice_tasks[id].state

    @property
    def exception(self) -> Optional[Exception]:
        return self._exception

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            if hasattr(self, "_sender"):
                try:
                    self._sender.clear_session_resource(self.session_args.disagg_params.disagg_id)
                except Exception:
                    raise RuntimeError("Sender clear_session_resource failed")
        except Exception:
            raise RuntimeError("TxSession close failed")
        finally:
            self.slice_tasks = []
            self._sender = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            logger.error("TxSession.__del__ failed")


class KVRecvTask:
    def __init__(
        self,
        disagg_id: str,
        kv_slice: KVSlice,
        slice_id: int,
        params: DisaggregatedParams,
        mapper: PeerMapper,
        aux_slot: int,
    ):
        self._disagg_id = disagg_id
        self._kv_slice = kv_slice
        self._slice_id = slice_id
        self._params = params
        self._mapper = mapper
        self._state = State.INIT
        self._exception = None
        self._future = concurrent.futures.Future()
        self._first = False
        self._expected = 0
        self._aux_slot = aux_slot

    @property
    def state(self) -> State:
        return self._state

    @state.setter
    def state(self, s: State):
        self._state = s

    @property
    def future(self) -> concurrent.futures.Future:
        return self._future

    @property
    def slice_id(self) -> int:
        return self._slice_id

    @property
    def expected_count(self) -> int:
        return self._expected

    @expected_count.setter
    def expected_count(self, v: int):
        self._expected = v

    def create_transfer_req_info(self) -> GenReqInfo:
        return GenReqInfo(
            ctx_req_id=self._params.ctx_request_id,
            instance_name=self._mapper.get_peer_registrar().get_self_rank_info().instance_name,
            instance_rank=self._mapper.get_peer_registrar().get_self_rank_info().instance_rank,
            blocks=self._kv_slice.blocks,
            disagg_id=self._disagg_id,
            aux_slot=self._aux_slot,
        )

    def extract_trans_meta(self, peer_ii, peer_dp_rank) -> AgentRecvArgs:
        peer_overlap_targets = self._mapper.get_peer_overlap_targets(peer_ii, peer_dp_rank)
        expected_count = len(peer_overlap_targets.ranks)
        if not self._first:
            self._first = True
            self._expected = expected_count
        return AgentRecvArgs(
            future_for_task=self._future,
            expected_count=expected_count,
            peer_name=None,
            slice_id=self._slice_id,
            disagg_id=self._disagg_id,
        ), peer_overlap_targets.ranks


class Receiver:
    def __init__(
        self,
        mapper: PeerMapper,
        device_id: int,
        agent: BaseTransferAgent,
    ):
        self._mapper = mapper
        self._device_id = device_id
        self._agent = agent
        self._receive_cache = {}
        self._receive_cache_lock = threading.Lock()

        self._zmq_context = zmq.Context()
        self._socket_cache = {}
        self._ctx_ep_ii = {}

        self._socket = self._zmq_context.socket(zmq.ROUTER)
        self._socket.bind(f"tcp://{get_local_ip()}:*")
        self._endpoint = self._socket.getsockopt(zmq.LAST_ENDPOINT).decode()
        self._receiver_background_thread = threading.Thread(
            target=self._handle_receiver_loop, daemon=True
        )
        self._receiver_background_thread.start()

        self._slice_tasks = {}  # disagg_id -> list[SliceReceiverTask]

        self._rx_sessions = {}  # disagg_id -> RxSession
        self._last_slice_counts = {}  # disagg_id -> int
        logger.info(f" Receiver init  with endpoint: {self._endpoint}")

    @property
    def endpoint(self):
        return self._endpoint

    def async_receive_slice(
        self, params: DisaggregatedParams, slice: KVSlice, aux_slot: int
    ) -> KVRecvTask:
        disagg_id = params.disagg_id
        if disagg_id not in self._slice_tasks:
            self._slice_tasks[disagg_id] = []

        slice_id = len(self._slice_tasks[disagg_id])
        slice_receiver_task = KVRecvTask(
            params.disagg_id,
            slice,
            slice_id,
            params,
            self._mapper,
            aux_slot=aux_slot,
        )
        self._slice_tasks[disagg_id].append(slice_receiver_task)

        self._async_request_data_transfer(slice_receiver_task)

        return slice_receiver_task

    def init_session_resource(self, rx_session: RxSessionBase):
        self._rx_sessions[rx_session.session_args.disagg_params.disagg_id] = weakref.ref(rx_session)

    def _get_rx_sessions(self, disagg_id: str) -> RxSessionBase:
        session_ref = self._rx_sessions[disagg_id]
        if session_ref is None:
            raise RuntimeError(f"RxSession {disagg_id} not found")
        session = session_ref()
        if session is None:
            raise RuntimeError(f"RxSession {disagg_id} has been cleared")
        return session

    def clear_session_resource(self, disagg_id: str):
        logger.debug(f"Clearing RX session resources for disagg_id={disagg_id}")
        self._rx_sessions.pop(disagg_id, None)
        self._slice_tasks.pop(disagg_id, None)

    def _async_request_data_transfer(self, slice_receiver_task: KVRecvTask):
        params = slice_receiver_task._params
        logger.debug(f"Preparing async data transfer request for disagg_params={params}")
        context_peer_infos: InstanceInfo = self._get_context_info(params)
        gen_req = slice_receiver_task.create_transfer_req_info()
        if params.ctx_dp_rank is None:
            raise ValueError("ctx_dp_rank is None")
        ctx_dp_rank = params.ctx_dp_rank
        agent_recv_args, target_ranks = slice_receiver_task.extract_trans_meta(
            context_peer_infos, ctx_dp_rank
        )

        self.submit_receive_task(agent_recv_args)
        for rank in target_ranks:
            self._send_data_request(context_peer_infos.ctx_endpoints[rank], gen_req)

        return

    def _need_register_peer_in_first_request(self, params: DisaggregatedParams) -> bool:
        return params.ctx_info_endpoint not in self._ctx_ep_ii

    def _get_socket(self, endpoint: str):
        if endpoint not in self._socket_cache:
            self._socket_cache[endpoint] = self._zmq_context.socket(zmq.DEALER)
            self._socket_cache[endpoint].connect(endpoint)
        return self._socket_cache[endpoint]

    def _get_context_info(self, params: DisaggregatedParams) -> InstanceInfo:
        if self._need_register_peer_in_first_request(params):
            socket = self._zmq_context.socket(zmq.DEALER)
            socket.connect(params.ctx_info_endpoint)
            message = [str(MessageType.REQUEST_INSTANCE_INFO).encode("ascii")]
            socket.send_multipart(message)
            message = socket.recv_multipart()
            ii = InstanceInfo.from_bytes(message[0])
            logger.debug(f"Fetched InstanceInfo from context service: {ii}")
            socket.close()

            for endpoint in ii.ctx_endpoints:
                socket = self._get_socket(endpoint)
                send_message = []
                send_message.append(str(MessageType.REGISTER_RANK_INFO).encode("ascii"))
                send_message.append(
                    self._mapper.get_peer_registrar().get_self_rank_info().to_bytes()
                )
                socket.send_multipart(send_message)

            self._ctx_ep_ii[params.ctx_info_endpoint] = ii
            return ii

        else:
            return self._ctx_ep_ii[params.ctx_info_endpoint]

    def submit_receive_task(self, trans_recv_meta: AgentRecvArgs):
        logger.debug(f"Registering receive task: {trans_recv_meta}")
        if trans_recv_meta.disagg_id not in self._last_slice_counts:
            self._last_slice_counts[trans_recv_meta.disagg_id] = 0
        # set state via property
        self._slice_tasks[trans_recv_meta.disagg_id][
            trans_recv_meta.slice_id
        ].state = State.TRANSFERRING

    def _handle_receiver_loop(self):
        while True:
            message = self._socket.recv_multipart()
            send_id = message[0]
            recv_message = message[1:]
            if self._message_is_termination(recv_message):
                break
            elif self._message_is_task_state(recv_message):
                self._handle_task_state(send_id, recv_message)
            elif self._message_is_meta_send_state(recv_message):
                self._handle_meta_send_state(send_id, recv_message)
            else:
                raise ValueError(
                    f"transceiver receiver loop received unknown message type: {recv_message[0]}"
                )

    def _message_is_termination(self, message: list[bytes]):
        return message[0] == str(MessageType.TERMINATION).encode("ascii")

    def _message_is_task_state(self, message: list[bytes]):
        return message[0] == str(MessageType.TASK_STATE).encode("ascii")

    def _message_is_meta_send_state(self, message: list[bytes]):
        return message[0] == str(MessageType.META_SEND_STATE).encode("ascii")

    def _handle_task_state(self, send_id: bytes, message: list[bytes]):
        assert len(message) == 5
        assert message[0].decode("ascii") == str(MessageType.TASK_STATE)
        disagg_id = message[1].decode("ascii")
        # peer_slice_id = int(message[2].decode("ascii"))  # Not used
        is_last_slice = message[3].decode("ascii") == "True"
        task_state = message[4].decode("ascii")
        if task_state == "SUCCESS":
            if is_last_slice:
                self._last_slice_counts[disagg_id] += 1
                if (
                    self._last_slice_counts[disagg_id]
                    == self._slice_tasks[disagg_id][0].expected_count
                ):
                    # use future property
                    self._slice_tasks[disagg_id][0].future.set_result("SUCCESS")
                    self._slice_tasks[disagg_id][0].state = State.FINISHED
                    self._get_rx_sessions(disagg_id).state.state = State.FINISHED
                    self._get_rx_sessions(disagg_id).state.finished_tasks.append(
                        0
                    )  # receive task slice only support slice 0

                    logger.debug(
                        f"Task state handled successfully; current rx_sessions keys: {list(self._rx_sessions.keys())}"
                    )
        elif task_state == "FAILED":
            self._slice_tasks[disagg_id][0].future.set_exception(
                RuntimeError(f"Task state: {task_state}")
            )
            self._slice_tasks[disagg_id][0].state = State.ERR
            self._get_rx_sessions(disagg_id).state.state = State.ERR
        else:
            raise ValueError(f" session {disagg_id} received unknown task state: {task_state}")

    def _handle_meta_send_state(self, send_id: bytes, message: list[bytes]):
        assert len(message) == 3
        assert message[0].decode("ascii") == str(MessageType.META_SEND_STATE)
        disagg_id = message[1].decode("ascii")
        sync_status = message[2].decode("ascii")
        if sync_status == "SUCCESS":
            self._get_rx_sessions(disagg_id).state.state = State.META_DATA_SENT
        elif sync_status == "FAILED":
            self._get_rx_sessions(disagg_id).state.state = State.ERR
        else:
            raise ValueError(
                f" session {disagg_id} received unknown meta send state: {sync_status}"
            )

    def _send_data_request(self, endpoint: str, transfer_gen_side_req_info: GenReqInfo):
        logger.debug(
            f"Sending data request to endpoint '{endpoint}' with request info: {transfer_gen_side_req_info}"
        )
        socket = self._get_socket(endpoint)
        send_message = []
        send_message.append(str(MessageType.REQUEST_DATA).encode("ascii"))
        send_message.append(transfer_gen_side_req_info.to_bytes())
        socket.send_multipart(send_message)


class RxSession(RxSessionBase):
    def __init__(
        self,
        request_id: int,
        disagg_params: DisaggregatedParams,
        receiver: Receiver,
        aux_slot: int,
    ):
        super().__init__(receiver, SessionArgsBase(request_id, disagg_params))
        self.request_id = request_id
        self.aux_slot = aux_slot
        self._params = disagg_params
        self._disagg_id = disagg_params.disagg_id
        self._receiver = receiver
        self._receiver.init_session_resource(self)
        self._exception = None
        self.slice_tasks = []
        self._state = SessionState(state=State.INIT, finished_tasks=[])

    @property
    def state(self) -> SessionState:
        return self._state

    @state.setter
    def state(self, s: SessionState):
        self._state = s

    def receive(self, slice: KVSlice) -> TaskIdType:
        self.slice_tasks.append(
            self._receiver.async_receive_slice(self._params, slice, self.aux_slot)
        )

        task_id = self.slice_tasks[-1].slice_id
        return task_id

    def poll_task(self, id: TaskIdType) -> State:
        return self.slice_tasks[id].state

    @property
    def exception(self) -> Optional[Exception]:
        return self._exception

    def close(self):
        try:
            if hasattr(self, "_receiver"):
                try:
                    self._receiver.clear_session_resource(self._params.disagg_id)
                except Exception:
                    raise RuntimeError("Receiver clear_session_resource failed")
        except Exception:
            raise RuntimeError("RxSession close failed")
        finally:
            self.slice_tasks = []
            self._receiver = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            logger.error("RxSession.__del__ failed")


class TransferAgentConfig:
    pass


class InstanceInfoServer:
    def __init__(self, instance_info: InstanceInfo, addr: str = None, port: int = None):
        self._instance_info = instance_info

        self._zmq_context = zmq.Context()
        self._socket = self._zmq_context.socket(zmq.ROUTER)
        if addr is None and port is None:
            self._socket.bind(f"tcp://{get_local_ip()}:*")
        else:
            self._socket.bind(f"tcp://{addr}:{port}")
        self._endpoint = self._socket.getsockopt(zmq.LAST_ENDPOINT).decode()
        self._thread = threading.Thread(target=self._loop_handle_request, daemon=True)
        self._thread.start()

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @endpoint.setter
    def endpoint(self, value) -> None:
        self._endpoint = value

    def _loop_handle_request(self):
        while True:
            message = self._socket.recv_multipart()
            send_id = message[0]
            recv_message = message[1:]
            if self._is_termination(recv_message):
                break
            elif self._is_request_instance_info(recv_message):
                self._handle_request_instance_info(send_id, recv_message)
            else:
                raise ValueError(
                    f" instance info server received unknown message type: {recv_message[0]}"
                )

    def _is_termination(self, message: list[bytes]):
        return message[0] == str(MessageType.TERMINATION).encode("ascii")

    def _is_request_instance_info(self, message: list[bytes]):
        return message[0] == str(MessageType.REQUEST_INSTANCE_INFO).encode("ascii")

    def _handle_request_instance_info(self, send_id: bytes, message: list[bytes]):
        self._socket.send_multipart([send_id, self._instance_info.to_bytes()])


def _deregister_registered_memory(transfer_agent, registered_memorys):
    try:
        if transfer_agent is None or not registered_memorys:
            return
        for register_memory in registered_memorys:
            try:
                transfer_agent.deregister_memory(register_memory)
            except Exception:
                logger.error("deregister memory failed in finalizer")
    except Exception:
        logger.error("unexpected error in _deregister_registered_memory finalizer")


class TransferWorker:
    def __init__(
        self,
        kv_cache_manager: KVCacheManager,
        mapping: Mapping,
        device_id: int,
        instance_name: str,
        transfer_agent_config: TransferAgentConfig,
        aux_buffer: Optional[AuxBuffer] = None,
    ):
        self._mapping = mapping

        self._instance_info: InstanceInfo = None
        self._rank_info: RankInfo = None
        self._kv_cache_manager = kv_cache_manager
        self._aux_buffer = aux_buffer
        self._device_id = device_id
        self._finalizer = None

        self.init_instance_info(instance_name)
        need_info_server = self._mapping.rank == 0
        if need_info_server:
            self._instance_info_server = InstanceInfoServer(self._instance_info)
        else:
            self._instance_info_server = None
        self._peer_registrar = PeerRegistrar(self._rank_info, self._instance_info)
        self._peer_mapper = PeerMapper(self._peer_registrar, self._kv_cache_manager)
        self._agent = BindingsNixlTransferAgent(
            self._rank_info.instance_name + str(self._rank_info.instance_rank), True
        )
        self._registered_mem = []
        self._register_kv_cache()
        if self._aux_buffer is not None:
            self._register_aux_buffer()

        self._sender = Sender(self._peer_mapper, device_id, self._agent)
        self._receiver = Receiver(self._peer_mapper, device_id, self._agent)
        self._rank_info.transfer_engine_info = bytes(self._agent.get_local_agent_desc())
        self._rank_info.endpoint = self._sender.endpoint
        self._rank_info.recv_endpoint = self._receiver.endpoint

        reg_snapshot = list(self._registered_mem) if self._registered_mem is not None else []
        self._finalizer = weakref.finalize(
            self, _deregister_registered_memory, self._agent, reg_snapshot
        )

    def refresh_instance_info(
        self, update_endpoints: list[str], update_layer_num_per_pp: list[int]
    ):
        self._instance_info.ctx_endpoints = update_endpoints
        self._instance_info.layer_num_per_pp = update_layer_num_per_pp
        self._rank_info.layer_num_per_pp = update_layer_num_per_pp

    def create_tx_session(self, request: LlmRequest) -> TxSession:
        """
        Create a txSession for the request.
        """
        if self._aux_buffer is not None:
            aux_slot = self._aux_buffer.alloc_slot()
        else:
            aux_slot = None
        return TxSession(
            request_id=request.py_request_id,
            disagg_params=request.py_disaggregated_params,
            sender=self._sender,
            aux_slot=aux_slot,
        )

    def create_rx_session(self, request: LlmRequest) -> RxSession:
        """
        Create a rxSession for the request.
        """
        if self._aux_buffer is not None:
            aux_slot = self._aux_buffer.alloc_slot()
        else:
            aux_slot = None
        return RxSession(
            request_id=request.py_request_id,
            disagg_params=request.py_disaggregated_params,
            receiver=self._receiver,
            aux_slot=aux_slot,
        )

    def clear_session(self, session: TxSession | RxSession):
        aux_slot = session.aux_slot
        if self._aux_buffer is not None:
            self._aux_buffer.free_slot(aux_slot)

    def init_instance_info(self, instance_name):
        rank = self._mapping.rank

        tp_size = self._mapping.tp_size
        pp_size = self._mapping.pp_size
        dp_size = self._mapping.dp_size
        cp_size = self._mapping.cp_size
        tp_rank = self._mapping.tp_rank
        pp_rank = self._mapping.pp_rank
        enable_attention_dp = self._mapping.enable_attention_dp
        dp_rank = 0
        if enable_attention_dp:
            dp_size = self._mapping.tp_size
            dp_rank = tp_rank
        cp_rank = self._mapping.cp_rank
        is_mla = self._kv_cache_manager.kv_factor == 1
        self._kv_cache_manager.kv_factor
        heads_num_per_rank = self._kv_cache_manager.num_kv_heads_per_layer[0]
        tokens_per_block = self._kv_cache_manager.tokens_per_block
        dims_per_head = self._kv_cache_manager.head_dim
        element_size = get_size_in_bytes(1, self._kv_cache_manager.dtype)
        layer_num_per_pp = [len(self._kv_cache_manager.pp_layers)]
        ctx_endpoints = []
        self._instance_info = InstanceInfo(
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
            ctx_endpoints=ctx_endpoints,
        )
        self._rank_info = RankInfo(
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
            device_id=self._device_id,
            kv_head_num_per_rank=heads_num_per_rank,
            tokens_per_block=tokens_per_block,
            dims_per_head=dims_per_head,
            element_size=element_size,
            enable_attention_dp=enable_attention_dp,
            is_mla=is_mla,
            layer_num_per_pp=layer_num_per_pp,
            kvcache_ptrs=[self._kv_cache_manager.get_unique_primary_pool().data_ptr()],
            aux_ptrs=[],
            server_endpoint="",
            recv_endpoint="",
            transfer_engine_info=bytes(),
            aux_meta=self._aux_buffer.meta if self._aux_buffer is not None else None,
        )

    def _register_kv_cache(self):
        memory_pool = self._kv_cache_manager.get_unique_primary_pool()
        memory_desc = (
            memory_pool.data_ptr(),
            memory_pool.numel() * memory_pool.element_size(),
            self._device_id,
            "kv_cache_memory",
        )
        reg_memory_desc = RegMemoryDescs("VRAM", [memory_desc])
        self._agent.register_memory(reg_memory_desc)
        logger.debug(f"Registered KV cache memory with transfer agent: {memory_desc}")
        self._registered_mem.append(reg_memory_desc)

    def _register_aux_buffer(self):
        aux_meta = self._aux_buffer.meta
        ptr_num = len(aux_meta.ptrs)
        ptr_descs = []
        for i in range(ptr_num):
            ptr_descs.append((aux_meta.ptrs[i], aux_meta.size[i], 0, f"aux_buffer_ptr_{i}"))
        reg_memory_desc = RegMemoryDescs("DRAM", ptr_descs)
        self._agent.register_memory(reg_memory_desc)
        logger.debug(f"Registered auxiliary buffer memory with transfer agent: {reg_memory_desc}")
        self._registered_mem.append(reg_memory_desc)

    # pack the meta data to the meta buffer

    def pack_meta_data(self, tx_session: TxSession, request: LlmRequest):
        self._aux_buffer.fill_slot(tx_session.aux_slot, request)

    def unpack_meta_data(self, rx_session: RxSession, request: LlmRequest):
        first_gen_tokens, draft_tokens = self._aux_buffer.get_slot_tokens(rx_session.aux_slot)

        # TODO: not first gen ,but add_tokens?
        request.py_first_gen_tokens = first_gen_tokens
        request.py_draft_tokens = draft_tokens
        return request

    def __del__(self):
        try:
            if (
                hasattr(self, "_finalizer")
                and self._finalizer is not None
                and self._finalizer.alive
            ):
                try:
                    pass
                except Exception:
                    logger.error("TransferWorker.__del__: finalizer invocation failed")
        except Exception:
            logger.error("Exception in TransferWorker.__del__")
