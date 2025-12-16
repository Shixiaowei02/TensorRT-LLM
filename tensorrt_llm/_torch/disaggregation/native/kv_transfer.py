import concurrent
import queue
import threading
import weakref
from dataclasses import asdict, dataclass
from enum import Enum
from typing import List, Optional

import msgpack

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
    SessionStatus,
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
from tensorrt_llm._torch.disaggregation.native.messenger import ZMQMessenger
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
    unique_rid: str
    start_token_idx: Optional[int] = None
    aux_slot: Optional[int] = None

    def to_bytes(self) -> bytes:
        return msgpack.packb(asdict(self))

    @classmethod
    def from_bytes(cls, data: bytes) -> "GenReqInfo":
        return cls(**msgpack.unpackb(data, raw=False))


@dataclass
class AgentRecvArgs:
    unique_rid: str
    future_for_task: concurrent.futures.Future
    expected_transfers: int
    peer_name: str
    slice_id: int


@dataclass
class AgentSendArgs:
    future_for_task: concurrent.futures.Future

    expected_transfers: int
    peer_name: str
    src_kv_ptrs: List[int] = None
    dst_kv_ptrs: List[int] = None
    kv_sizes: List[int] = None
    dst_device_id: int = None
    src_aux_ptrs: List[int] = None
    dst_aux_ptrs: List[int] = None
    aux_sizes: List[int] = None
    peer_endpoint: Optional[str] = None  # used for send state
    unique_rid: Optional[str] = None
    slice_id: Optional[int] = None
    is_last_slice: Optional[bool] = False
    is_only_aux: Optional[bool] = False


class MessageType:
    TERMINATION = "TERMINATION"
    TASK_STATE = "TASK_STATE"
    PEER_INFO = "PEER_INFO"
    REQUEST_DATA = "REQUEST_DATA"
    REQUEST_INSTANCE_INFO = "REQUEST_INSTANCE_INFO"
    REGISTER_RANK_INFO = "REGISTER_RANK_INFO"
    AUX_SEND_STATE = "AUX_SEND_STATE"


def _message_is_termination(message: list[bytes]):
    return message[0] == str(MessageType.TERMINATION).encode("ascii")


def _message_is_request_data(message: list[bytes]):
    return message[0] == str(MessageType.REQUEST_DATA).encode("ascii")


def _message_is_register_rank_info(message: list[bytes]):
    return message[0] == str(MessageType.REGISTER_RANK_INFO).encode("ascii")


def _message_is_task_state(message: list[bytes]):
    return message[0] == str(MessageType.TASK_STATE).encode("ascii")


def _message_is_aux_send_state(message: list[bytes]):
    return message[0] == str(MessageType.AUX_SEND_STATE).encode("ascii")


def _message_is_request_instance_info(message: list[bytes]):
    return message[0] == str(MessageType.REQUEST_INSTANCE_INFO).encode("ascii")


class TaskStatus(Enum):
    INIT = "INIT"
    TRANSFERRING = "TRANSFERRING"
    TRANSFERRED = "TRANSFERRED"
    COMPLETED = "COMPLETED"
    CANCELED = "CANCELED"
    ERROR = "ERROR"
    AUX_TRANSMITTED = "AUX_TRANSMITTED"


class AuxSendTask:
    def __init__(self, params: DisaggregatedParams, slot: int, mapper: PeerMapper):
        self._params = params
        self._unique_rid = params.disagg_id
        self._slot = slot
        self._mapper = mapper
        self._status = TaskStatus.INIT
        self._future = concurrent.futures.Future()

    @property
    def status(self) -> TaskStatus:
        return self._status

    @status.setter
    def status(self, s: TaskStatus):
        self._status = s

    @property
    def future(self) -> concurrent.futures.Future:
        return self._future

    def extract_transfer_metadata(self, req_info: GenReqInfo) -> AgentSendArgs:
        peer_aux_meta = self._mapper.peer_registrar.get_peer_rank_info(
            req_info.instance_name, req_info.instance_rank
        ).aux_meta

        peer_slot = req_info.aux_slot

        src_aux_meta = self._mapper.peer_registrar.rank_info.aux_meta

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
            expected_transfers=1,
            is_only_aux=True,
            peer_name=req_info.instance_name + str(req_info.instance_rank),
            peer_endpoint=self._mapper.peer_registrar.get_peer_rank_info(
                req_info.instance_name, req_info.instance_rank
            ).recv_endpoint,
            unique_rid=self._unique_rid,
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
        self._first_transfer = False
        self._extraction_count = 0
        self._expected_transfers = 0
        self._slice = kv_slice
        self._params = params
        self._unique_rid = params.disagg_id
        self._slice_id = slice_id
        self._status = TaskStatus.INIT
        self._transferred_count = 0

    @property
    def status(self) -> TaskStatus:
        return self._status

    @status.setter
    def status(self, s: TaskStatus):
        self._status = s

    @property
    def future(self) -> concurrent.futures.Future:
        return self._future

    @property
    def slice_id(self) -> int:
        return self._slice_id

    @property
    def transferred_count(self) -> int:
        return self._transferred_count

    @transferred_count.setter
    def transferred_count(self, v: int):
        self._transferred_count = v

    @nvtx_range("extract_transfer_metadata")
    def extract_transfer_metadata(self, req_info: GenReqInfo) -> AgentSendArgs:
        peer_ri = self._mapper.peer_registrar.get_peer_rank_info(
            req_info.instance_name, req_info.instance_rank
        )
        targets = self._mapper.get_peer_overlap_targets(peer_ri, peer_ri.dp_rank)
        expected_transfers = len(targets.ranks)
        if not self._first_transfer:
            self._first_transfer = True
            self._expected_transfers = expected_transfers
        self._extraction_count = self._extraction_count + 1
        if not self._should_send(targets, peer_ri):
            return AgentSendArgs(
                future_for_task=self._future,
                src_kv_ptrs=[],
                dst_kv_ptrs=[],
                kv_sizes=[],
                expected_transfers=expected_transfers,
                peer_name=peer_ri.instance_name + str(peer_ri.instance_rank),
                peer_endpoint=peer_ri.recv_endpoint,
                unique_rid=self._unique_rid,
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

        extractor = self._mapper._kv_ptr_extractor
        src_ptrs = extractor.extract_kv_block_ptrs(src_blocks)
        self_block_size = extractor.kv_pool_attributes.kv_cache_block_sizes[0]
        peer_extractor = self._mapper.get_kv_extractor(peer_ri.instance_name, peer_ri.instance_rank)
        dst_ptrs = peer_extractor.extract_kv_block_ptrs(dst_blocks)
        dst_block_size = peer_extractor.kv_pool_attributes.kv_cache_block_sizes[0]
        logger.debug(
            f"KVSendTask.extract_transfer_metadata: extracted KV block pointers -> "
            f"src_ptrs={src_ptrs}, src_block_size={self_block_size}, "
            f"dst_ptrs={dst_ptrs}, dst_block_size={dst_block_size}"
        )
        transformer = self._mapper.get_kv_ptrs_mapper(peer_ri)
        (
            src_ptrs,
            dst_ptrs,
            dst_blocks_size,
        ) = transformer(src_ptrs, self_block_size, dst_ptrs, dst_block_size)

        src_blocks_size = dst_blocks_size
        logger.debug(
            f"KVSendTask.extract_transfer_metadata: mapped KV pointers for transfer -> "
            f"src_transfer_ptrs={src_ptrs}, src_transfer_block_size={src_blocks_size}, "
            f"dst_transfer_ptrs={dst_ptrs}, dst_transfer_block_size={dst_blocks_size}"
        )

        return AgentSendArgs(
            future_for_task=self._future,
            src_kv_ptrs=src_ptrs,
            dst_kv_ptrs=dst_ptrs,
            kv_sizes=[src_blocks_size] * len(src_ptrs),
            dst_device_id=dst_device_id,
            expected_transfers=expected_transfers,
            peer_name=peer_ri.instance_name + str(peer_ri.instance_rank),
            peer_endpoint=peer_ri.recv_endpoint,
            unique_rid=self._unique_rid,
            slice_id=self._slice_id,
            is_last_slice=self._slice.is_last_slice,
        )

    def is_active(self) -> bool:
        if self._first_transfer:
            return self._extraction_count < self._expected_transfers
        else:
            return True

    def _should_send(
        self, peer_overlap_targets: PeerOverlapTargets, peer_rank_info: RankInfo
    ) -> bool:
        dup_head_factor = peer_overlap_targets.duplicate_head_factor
        if dup_head_factor <= 1:
            return True
        peer_ri = peer_rank_info
        peer_dp_rank = peer_ri.dp_rank if peer_ri.enable_attention_dp else 0
        self_ri = self._mapper.peer_registrar.rank_info
        self_tp_size_per_dp_group = (
            self_ri.tp_size // self_ri.dp_size if self_ri.enable_attention_dp else self_ri.tp_size
        )
        self_tp_rank_in_dp_group = self_ri.tp_rank % self_tp_size_per_dp_group
        return (peer_dp_rank % dup_head_factor) == (self_tp_rank_in_dp_group % dup_head_factor)

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
        self._peer_reqs = {}
        self._peer_req_lock = threading.Lock()

        self._messenger = ZMQMessenger(mode="ROUTER")
        self._start_listener()

        self._socket_cache = {}
        self._kv_tasks = {}  # unique_rid -> list[KVSlice]
        self._aux_tasks = {}  # unique_rid -> AuxSendTask
        self._tx_sessions = {}  # unique_rid -> TxSession

        logger.info(f" Sender init end with endpoint: {self._messenger.endpoint}")
        self._closed = False

    @property
    def endpoint(self):
        return self._messenger.endpoint

    def init_session_resource(self, tx_session: TxSessionBase):
        self._tx_sessions[tx_session.session_args.params.disagg_id] = weakref.ref(tx_session)

        req_info_num = 0
        req_info = None
        with self._peer_req_lock:
            if tx_session.session_args.params.disagg_id in self._peer_reqs:
                req_info_num = len(self._peer_reqs[tx_session.session_args.params.disagg_id])
                if req_info_num > 0:
                    req_info = list(
                        self._peer_reqs[tx_session.session_args.params.disagg_id].values()
                    )[0]
        if req_info is not None:
            peer_ri = self._mapper.peer_registrar.get_peer_rank_info(
                req_info.instance_name, req_info.instance_rank
            )
            expected_transfers = len(
                self._mapper.get_peer_overlap_targets(peer_ri, peer_ri.instance_rank).ranks
            )
            if expected_transfers == req_info_num:
                tx_session.state.status = SessionStatus.READY
        return

    def _get_tx_session(self, unique_rid: str) -> TxSessionBase:
        session_ref = self._tx_sessions[unique_rid]
        if session_ref is None:
            raise RuntimeError(f"TxSession {unique_rid} not found")
        session = session_ref()
        if session is None:
            raise RuntimeError(f"TxSession {unique_rid} has been cleared")
        return session

    def clear_session_resource(self, unique_rid: str):
        logger.debug(f"Clearing session resources for unique_rid={unique_rid}")
        self._kv_tasks.pop(unique_rid, None)
        self._peer_reqs.pop(unique_rid, None)
        return

    def async_send_kv(self, params: DisaggregatedParams, slice: KVSlice) -> KVSendTask:
        unique_rid = params.disagg_id
        if unique_rid not in self._kv_tasks:
            self._kv_tasks[unique_rid] = []
        slice_id = len(self._kv_tasks[unique_rid])
        task = KVSendTask(slice, params, slice_id, self._mapper)
        self._kv_tasks[unique_rid].append(task)

        self._dispatch_task(task)
        return task

    def async_send_aux(self, params: DisaggregatedParams, slot: int):
        task = AuxSendTask(params, slot, self._mapper)

        self._aux_tasks[params.disagg_id] = task
        self._dispatch_task(task)
        return task

    def submit_task(self, agent_args: AgentSendArgs):
        if not hasattr(self, "_send_task_queue"):
            self._send_task_queue = queue.Queue()
            self._background_thread = threading.Thread(target=self._process_task_queue, daemon=True)
            self._background_thread.start()
        self._send_task_queue.put(agent_args)

    def _process_task_queue(self):
        while True:
            agent_args = self._send_task_queue.get()
            if agent_args is None:
                break
            if agent_args.is_only_aux:
                self._deliver_aux_to_agent(agent_args)
            else:
                self._deliver_kv_to_agent(agent_args)

    @staticmethod
    def _prepare_transfer_request(agent_args: AgentSendArgs, is_aux: bool, device_id: int):
        if is_aux:
            assert agent_args.src_aux_ptrs is not None and agent_args.dst_aux_ptrs is not None
            src_list = [
                (src_ptr, size, 0)
                for src_ptr, size in zip(agent_args.src_aux_ptrs, agent_args.aux_sizes)
            ]
            dst_list = [
                (dst_ptr, size, 0)
                for dst_ptr, size in zip(agent_args.dst_aux_ptrs, agent_args.aux_sizes)
            ]
            src_mem_type = "DRAM"
            dst_mem_type = "DRAM"
            peer_name = agent_args.peer_name
        else:
            assert agent_args.src_kv_ptrs is not None and agent_args.dst_kv_ptrs is not None
            src_list = [
                (src_ptr, size, device_id)
                for src_ptr, size in zip(agent_args.src_kv_ptrs, agent_args.kv_sizes)
            ]
            dst_list = [
                (dst_ptr, size, agent_args.dst_device_id)
                for dst_ptr, size in zip(agent_args.dst_kv_ptrs, agent_args.kv_sizes)
            ]
            src_mem_type = "VRAM"
            dst_mem_type = "VRAM"
            peer_name = agent_args.peer_name

        src_memory_descs = MemoryDescs(src_mem_type, src_list)
        dst_memory_descs = MemoryDescs(dst_mem_type, dst_list)
        request = TransferRequest(
            TransferOp.WRITE, src_memory_descs, dst_memory_descs, peer_name, None
        )
        return request, src_list, dst_list

    @nvtx_range("_deliver_kv_to_agent")
    def _deliver_kv_to_agent(self, agent_args: AgentSendArgs):
        assert len(agent_args.src_kv_ptrs) == len(agent_args.dst_kv_ptrs)
        assert len(agent_args.kv_sizes) == len(agent_args.src_kv_ptrs)
        assert agent_args.is_only_aux is False

        unique_rid = agent_args.unique_rid
        slice_id = agent_args.slice_id
        peer_endpoint = agent_args.peer_endpoint

        session = self._get_tx_session(unique_rid)
        assert session.state.status != SessionStatus.ERROR
        session.state.status = SessionStatus.TRANSFERRING
        self._kv_tasks[unique_rid][slice_id].status = TaskStatus.TRANSFERRING

        request, src_kv_list, _ = Sender._prepare_transfer_request(
            agent_args, is_aux=False, device_id=self._device_id
        )

        skip_send = len(src_kv_list) == 0
        logger.debug(f"Submitting transfer request to transfer agent: {request}")
        agent_handler = None
        if not skip_send:
            agent_handler = self._agent.submit_transfer_requests(request)

        sync_status = "SUCCESS"
        if not skip_send and not agent_handler.wait():
            sync_status = "FAILED"
            agent_args.future_for_task.set_exception(RuntimeError("Transfer failed"))
            self._kv_tasks[unique_rid][slice_id].status = TaskStatus.ERROR
            session.state.status = SessionStatus.ERROR

        messenger = self._get_or_connect_dealer(peer_endpoint)

        ## TODO: just last slice need to send task state?
        messenger.send(
            [
                str(MessageType.TASK_STATE).encode("ascii"),
                unique_rid.encode("ascii"),
                str(slice_id).encode("ascii"),
                str(agent_args.is_last_slice).encode("ascii"),
                sync_status.encode("ascii"),
            ]
        )

        task = self._kv_tasks[unique_rid][slice_id]
        curr = task.transferred_count
        task.transferred_count = curr + 1

        if task.transferred_count > agent_args.expected_transfers:
            agent_args.future_for_task.set_exception(
                RuntimeError(
                    f"Session {unique_rid} has more than {agent_args.expected_transfers} transfers"
                )
            )
            # TODO: set exception for the session ?
            session.state.status = SessionStatus.ERROR
        elif task.transferred_count == agent_args.expected_transfers:
            # TODO avoid set_result if tranfser failed since it has been set exception
            agent_args.future_for_task.set_result(sync_status)
            task.status = TaskStatus.TRANSFERRED
            session.state.finished_tasks.append(slice_id)
            if agent_args.is_last_slice:
                session.state.status = SessionStatus.TRANSFERRED

    @nvtx_range("submit_send_aux_to_agent")
    def _deliver_aux_to_agent(self, agent_args: AgentSendArgs):
        # TODO: submit the aux data task to the transfer agent
        assert agent_args.is_only_aux is True
        assert agent_args.src_aux_ptrs is not None

        request, _, _ = Sender._prepare_transfer_request(
            agent_args, is_aux=True, device_id=self._device_id
        )

        logger.debug(f"Submitting aux data transfer request to transfer agent: {request}")

        status = self._agent.submit_transfer_requests(request)
        sync_status = "SUCCESS"
        if not status.wait():
            sync_status = "FAILED"
            agent_args.future_for_task.set_exception(RuntimeError("Transfer failed"))
            self._get_tx_session(agent_args.unique_rid).state.status = SessionStatus.ERROR
        messenger = self._get_or_connect_dealer(agent_args.peer_endpoint)
        messenger.send(
            [
                str(MessageType.AUX_SEND_STATE).encode("ascii"),
                agent_args.unique_rid.encode("ascii"),
                sync_status.encode("ascii"),
            ]
        )
        self._get_tx_session(agent_args.unique_rid).state.status = SessionStatus.AUX_TRANSMITTED

    def _dispatch_task(self, task: KVSendTask | AuxSendTask):
        req_info_dict = {}
        with self._peer_req_lock:
            if task._unique_rid in self._peer_reqs:
                req_info_dict = self._peer_reqs[task._unique_rid]
        for info in req_info_dict.values():
            trans_meta = task.extract_transfer_metadata(info)
            self.submit_task(trans_meta)

    def _start_listener(self):
        def handle_message(messages: list[bytes]):
            send_id = messages[0]
            recv_message = messages[1:]
            if _message_is_termination(recv_message):
                return False
            elif _message_is_request_data(recv_message):
                self._respond_with_kv(send_id, recv_message)
            elif _message_is_register_rank_info(recv_message):
                self._register_peer_rank(send_id, recv_message)
            else:
                raise ValueError(f"Sender received unknown message type: {recv_message[0]}")

        self._messenger.start_listener(handle_message)

    def _register_peer_rank(self, send_id: bytes, message: list[bytes]):
        ri: RankInfo = RankInfo.from_bytes(message[1])

        self._mapper.peer_registrar.register(ri.instance_name, ri.instance_rank, ri)

        agent_name = ri.instance_name + str(ri.instance_rank)
        logger.debug(f"Loading remote transfer agent descriptor for peer '{agent_name}'")
        self._agent.load_remote_agent(
            ri.instance_name + str(ri.instance_rank),
            ri.transfer_engine_info,
        )
        logger.debug(
            f"Completed handling REGISTER_RANK_INFO for instance='{ri.instance_name}', rank={ri.instance_rank}"
        )

    def _respond_with_kv(self, send_id: bytes, message: list[bytes]):
        info: GenReqInfo = GenReqInfo.from_bytes(message[1])

        tasks: list[KVSendTask] = self._get_kv_tasks(info.unique_rid)
        self._save_peer_req_info(info)
        if tasks is None:
            pass
        else:
            for task in tasks:
                trans_meta = task.extract_transfer_metadata(info)
                self.submit_task(trans_meta)

    def _get_kv_tasks(self, unique_rid: str):
        if unique_rid not in self._kv_tasks:
            return None
        return self._kv_tasks[unique_rid]

    def _get_or_connect_dealer(self, endpoint: str):
        if endpoint is None:
            raise ValueError("endpoint is None")
        if endpoint not in self._socket_cache:
            self._socket_cache[endpoint] = ZMQMessenger(mode="DEALER", endpoint=endpoint)
        return self._socket_cache[endpoint]

    def _save_peer_req_info(self, peer_transfer_req_info: GenReqInfo):
        req_info = peer_transfer_req_info
        with self._peer_req_lock:
            if req_info.unique_rid not in self._peer_reqs:
                self._peer_reqs[req_info.unique_rid] = {}
            self._peer_reqs[req_info.unique_rid][req_info.instance_rank] = req_info
        peer_ri = self._mapper.peer_registrar.get_peer_rank_info(
            req_info.instance_name, req_info.instance_rank
        )
        expected_transfers = len(
            self._mapper.get_peer_overlap_targets(peer_ri, req_info.instance_rank).ranks
        )
        if expected_transfers == len(self._peer_reqs[req_info.unique_rid]):
            if req_info.unique_rid in self._tx_sessions:
                session = self._get_tx_session(req_info.unique_rid)
                if session.state.status == SessionStatus.INIT:
                    session.state.status = SessionStatus.READY

    def close(self):
        if self._closed:
            return
        self._closed = True

        # Stop the _process_task_queue thread if it exists
        if hasattr(self, "_send_task_queue"):
            self._send_task_queue.put(None)
            if hasattr(self, "_background_thread"):
                self._background_thread.join(timeout=5)

        # Send termination message to stop _start_loop
        stopper = ZMQMessenger(mode="DEALER", endpoint=self._messenger.endpoint)
        stopper.send([str(MessageType.TERMINATION).encode("ascii")])
        stopper.stop()
        self._messenger.stop()

    def __del__(self):
        try:
            self.close()
        except Exception as e:
            logger.error(f"Exception in Sender.__del__: {e}")


class TxSession(TxSessionBase):
    def __init__(self, request_id: int, params: DisaggregatedParams, sender: Sender, aux_slot: int):
        super().__init__(sender, SessionArgsBase(request_id, params))
        self.request_id = request_id
        self._sender = sender
        self._state = SessionState(status=SessionStatus.INIT, finished_tasks=[])
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
        slice_sender_task = self._sender.async_send_kv(self.session_args.params, slice)
        self.slice_tasks.append(slice_sender_task)
        return slice_sender_task.slice_id

    def send_aux(self):
        return self._sender.async_send_aux(self.session_args.params, self.aux_slot)

    def poll_task(self, id: TaskIdType) -> SessionStatus:
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
                    self._sender.clear_session_resource(self.session_args.params.disagg_id)
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
        unique_rid: str,
        kv_slice: KVSlice,
        slice_id: int,
        params: DisaggregatedParams,
        mapper: PeerMapper,
        aux_slot: int,
    ):
        self._unique_rid = unique_rid
        self._kv_slice = kv_slice
        self._slice_id = slice_id
        self._params = params
        self._mapper = mapper
        self._status = TaskStatus.INIT
        self._exception = None
        self._future = concurrent.futures.Future()
        self._first_transfer = False
        self._expected_transfers = 0
        self._aux_slot = aux_slot

    @property
    def status(self) -> TaskStatus:
        return self._status

    @status.setter
    def status(self, s: TaskStatus):
        self._status = s

    @property
    def future(self) -> concurrent.futures.Future:
        return self._future

    @property
    def slice_id(self) -> int:
        return self._slice_id

    @property
    def expected_transfers(self) -> int:
        return self._expected_transfers

    @expected_transfers.setter
    def expected_transfers(self, v: int):
        self._expected_transfers = v

    def create_transfer_req_info(self) -> GenReqInfo:
        return GenReqInfo(
            ctx_req_id=self._params.ctx_request_id,
            instance_name=self._mapper.peer_registrar.rank_info.instance_name,
            instance_rank=self._mapper.peer_registrar.rank_info.instance_rank,
            blocks=self._kv_slice.blocks,
            unique_rid=self._unique_rid,
            aux_slot=self._aux_slot,
        )

    def extract_transfer_metadata(self, peer_ii, peer_dp_rank) -> AgentRecvArgs:
        peer_overlap_targets = self._mapper.get_peer_overlap_targets(peer_ii, peer_dp_rank)
        expected_transfers = len(peer_overlap_targets.ranks)
        if not self._first_transfer:
            self._first_transfer = True
            self._expected_transfers = expected_transfers
        return AgentRecvArgs(
            future_for_task=self._future,
            expected_transfers=expected_transfers,
            peer_name=None,
            slice_id=self._slice_id,
            unique_rid=self._unique_rid,
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

        self._socket_cache = {}
        self._ctx_ep_instance_map = {}

        self._messenger = ZMQMessenger(mode="ROUTER")
        self._start_listener()

        self._kv_tasks = {}  # unique_rid -> list[SliceReceiverTask]

        self._rx_sessions = {}  # unique_rid -> RxSession
        self._last_slice_counts = {}  # unique_rid -> int
        self._closed = False
        logger.info(f" Receiver init with endpoint: {self._messenger.endpoint}")

    @property
    def endpoint(self):
        return self._messenger.endpoint

    def close(self):
        if not getattr(self, "_closed", False) or self._closed:
            return
        self._closed = True

        stopper = ZMQMessenger(mode="DEALER", endpoint=self._messenger.endpoint)
        stopper.send([str(MessageType.TERMINATION).encode("ascii")])
        stopper.stop()
        self._messenger.stop()

    def async_receive_kv_slice(
        self, params: DisaggregatedParams, slice: KVSlice, aux_slot: int
    ) -> KVRecvTask:
        unique_rid = params.disagg_id
        if unique_rid not in self._kv_tasks:
            self._kv_tasks[unique_rid] = []

        slice_id = len(self._kv_tasks[unique_rid])
        slice_receiver_task = KVRecvTask(
            params.disagg_id,
            slice,
            slice_id,
            params,
            self._mapper,
            aux_slot=aux_slot,
        )
        self._kv_tasks[unique_rid].append(slice_receiver_task)

        self._async_request_data_transfer(slice_receiver_task)

        return slice_receiver_task

    def init_session_resource(self, rx_session: RxSessionBase):
        self._rx_sessions[rx_session.session_args.params.disagg_id] = weakref.ref(rx_session)

    def _get_rx_sessions(self, unique_rid: str) -> RxSessionBase:
        session_ref = self._rx_sessions[unique_rid]
        if session_ref is None:
            raise RuntimeError(f"RxSession {unique_rid} not found")
        session = session_ref()
        if session is None:
            raise RuntimeError(f"RxSession {unique_rid} has been cleared")
        return session

    def clear_session_resource(self, unique_rid: str):
        logger.debug(f"Clearing RX session resources for unique_rid={unique_rid}")
        self._rx_sessions.pop(unique_rid, None)
        self._kv_tasks.pop(unique_rid, None)

    def _async_request_data_transfer(self, slice_receiver_task: KVRecvTask):
        params = slice_receiver_task._params
        logger.debug(f"Preparing async data transfer request for disagg_params={params}")
        context_peer_infos: InstanceInfo = self._get_sender_info(params)
        gen_req = slice_receiver_task.create_transfer_req_info()
        if params.ctx_dp_rank is None:
            raise ValueError("ctx_dp_rank is None")
        ctx_dp_rank = params.ctx_dp_rank
        agent_args, target_ranks = slice_receiver_task.extract_transfer_metadata(
            context_peer_infos, ctx_dp_rank
        )
        self.submit_receive_task(agent_args)
        for rank in target_ranks:
            self._send_data_request(context_peer_infos.ctx_endpoints[rank], gen_req)
        return

    def _need_register_peer_in_first_request(self, params: DisaggregatedParams) -> bool:
        return params.ctx_info_endpoint not in self._ctx_ep_instance_map

    def _get_or_connect_dealer(self, endpoint: str):
        if endpoint is None:
            raise ValueError("endpoint is None")
        if endpoint not in self._socket_cache:
            self._socket_cache[endpoint] = ZMQMessenger(mode="DEALER", endpoint=endpoint)
        return self._socket_cache[endpoint]

    def _get_sender_info(self, params: DisaggregatedParams) -> InstanceInfo:
        if self._need_register_peer_in_first_request(params):
            messenger = ZMQMessenger(mode="DEALER", endpoint=params.ctx_info_endpoint)
            messenger.send([str(MessageType.REQUEST_INSTANCE_INFO).encode("ascii")])
            message = messenger.receive()
            sender_info = InstanceInfo.from_bytes(message[0])
            messenger.stop()

            for endpoint in sender_info.ctx_endpoints:
                messenger = self._get_or_connect_dealer(endpoint)
                msg = []
                msg.append(str(MessageType.REGISTER_RANK_INFO).encode("ascii"))
                msg.append(self._mapper.peer_registrar.rank_info.to_bytes())
                messenger.send(msg)

            self._ctx_ep_instance_map[params.ctx_info_endpoint] = sender_info
            return sender_info

        else:
            return self._ctx_ep_instance_map[params.ctx_info_endpoint]

    def submit_receive_task(self, recv_meta: AgentRecvArgs):
        logger.debug(f"Registering receive task: {recv_meta}")
        if recv_meta.unique_rid not in self._last_slice_counts:
            self._last_slice_counts[recv_meta.unique_rid] = 0
        self._kv_tasks[recv_meta.unique_rid][recv_meta.slice_id].status = TaskStatus.TRANSFERRING

    def _start_listener(self):
        def handle_message(messages: list[bytes]) -> bool:
            send_id = messages[0]
            msg = messages[1:]
            if _message_is_termination(msg):
                return False
            elif _message_is_task_state(msg):
                self._handle_task_state(send_id, msg)
            elif _message_is_aux_send_state(msg):
                self._handle_aux_send_state(send_id, msg)
            else:
                raise ValueError(f"Sender received unknown message type: {msg[0]}")

        self._messenger.start_listener(handle_message)

    def _handle_task_state(self, send_id: bytes, message: list[bytes]):
        assert len(message) == 5
        assert message[0].decode("ascii") == str(MessageType.TASK_STATE)
        unique_rid = message[1].decode("ascii")
        is_last_slice = message[3].decode("ascii") == "True"
        task_state = message[4].decode("ascii")
        if task_state == "SUCCESS":
            if is_last_slice:
                self._last_slice_counts[unique_rid] += 1
                if (
                    self._last_slice_counts[unique_rid]
                    == self._kv_tasks[unique_rid][0].expected_transfers
                ):
                    # use future property
                    self._kv_tasks[unique_rid][0].future.set_result("SUCCESS")
                    self._kv_tasks[unique_rid][0].status = TaskStatus.TRANSFERRED
                    self._get_rx_sessions(unique_rid).state.status = SessionStatus.TRANSFERRED
                    self._get_rx_sessions(unique_rid).state.finished_tasks.append(
                        0
                    )  # receive task slice only support slice 0

                    logger.debug(
                        f"Task state handled successfully; current rx_sessions keys: {list(self._rx_sessions.keys())}"
                    )
        elif task_state == "FAILED":
            self._kv_tasks[unique_rid][0].future.set_exception(
                RuntimeError(f"Task state: {task_state}")
            )
            self._kv_tasks[unique_rid][0].status = TaskStatus.ERROR
            self._get_rx_sessions(unique_rid).state.status = SessionStatus.ERROR
        else:
            raise ValueError(f" session {unique_rid} received unknown task state: {task_state}")

    def _handle_aux_send_state(self, send_id: bytes, message: list[bytes]):
        assert len(message) == 3
        assert message[0].decode("ascii") == str(MessageType.AUX_SEND_STATE)
        unique_rid = message[1].decode("ascii")
        sync_status = message[2].decode("ascii")
        if sync_status == "SUCCESS":
            self._get_rx_sessions(unique_rid).state.status = SessionStatus.AUX_TRANSMITTED
        elif sync_status == "FAILED":
            self._get_rx_sessions(unique_rid).state.status = SessionStatus.ERROR
        else:
            raise ValueError(
                f" session {unique_rid} received unknown aux send state: {sync_status}"
            )

    def _send_data_request(self, endpoint: str, transfer_gen_side_req_info: GenReqInfo):
        logger.debug(
            f"Sending data request to endpoint '{endpoint}' with request info: {transfer_gen_side_req_info}"
        )
        messenger = self._get_or_connect_dealer(endpoint)
        msg = []
        msg.append(str(MessageType.REQUEST_DATA).encode("ascii"))
        msg.append(transfer_gen_side_req_info.to_bytes())
        messenger.send(msg)

    def __del__(self):
        try:
            self.close()
        except Exception as e:
            logger.error(f"Exception in Receiver.__del__: {e}")


class RxSession(RxSessionBase):
    def __init__(
        self,
        request_id: int,
        params: DisaggregatedParams,
        receiver: Receiver,
        aux_slot: int,
    ):
        super().__init__(receiver, SessionArgsBase(request_id, params))
        self.request_id = request_id
        self.aux_slot = aux_slot
        self._params = params
        self._unique_rid = params.disagg_id
        self._receiver = receiver
        self._receiver.init_session_resource(self)
        self._exception = None
        self.slice_tasks = []
        self._state = SessionState(status=SessionStatus.INIT, finished_tasks=[])

    @property
    def state(self) -> SessionState:
        return self._state

    @state.setter
    def state(self, s: SessionState):
        self._state = s

    def receive(self, slice: KVSlice) -> TaskIdType:
        self.slice_tasks.append(
            self._receiver.async_receive_kv_slice(self._params, slice, self.aux_slot)
        )

        task_id = self.slice_tasks[-1].slice_id
        return task_id

    def poll_task(self, id: TaskIdType) -> SessionStatus:
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
        if addr is None and port is None:
            endpoint = f"tcp://{get_local_ip()}:*"
        else:
            endpoint = f"tcp://{addr}:{port}"
        self._messenger = ZMQMessenger(mode="ROUTER", endpoint=endpoint)
        self._start_listener()
        self._closed = False

    @property
    def endpoint(self) -> str:
        return self._messenger.endpoint

    """
    @endpoint.setter
    def endpoint(self, value) -> None:
        self._messenger.endpoint = value
    """

    def close(self):
        if self._closed:
            return
        self._closed = True
        logger.debug("InstanceInfoServer.close() called")

        stopper = ZMQMessenger(mode="DEALER", endpoint=self._messenger.endpoint)
        stopper.send([str(MessageType.TERMINATION).encode("ascii")])
        stopper.stop()
        self._messenger.stop()

    def _start_listener(self):
        def handle_message(messages: list[bytes]) -> bool:
            send_id = messages[0]
            msg = messages[1:]
            if _message_is_termination(msg):
                return False
            elif _message_is_request_instance_info(msg):
                self._handle_request_instance_info(send_id, msg)
            else:
                raise ValueError(
                    f" instance info server received unknown message type: {messages[0]}"
                )

        self._messenger.start_listener(handle_message)

    def _handle_request_instance_info(self, send_id: bytes, message: list[bytes]):
        self._messenger.send([send_id, self._instance_info.to_bytes()])

    def __del__(self):
        try:
            self.close()
        except Exception as e:
            logger.error(f"Exception in InstanceInfoServer.__del__: {e}")


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
        is_leader = self._mapping.rank == 0
        if is_leader:
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

    def populate_instance_and_rank_info(self, endpoints: list[str], layer_num_per_pp: list[int]):
        self._instance_info.ctx_endpoints = endpoints
        self._instance_info.layer_num_per_pp = layer_num_per_pp
        self._rank_info.layer_num_per_pp = layer_num_per_pp

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
            params=request.py_disaggregated_params,
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
            params=request.py_disaggregated_params,
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
            ptr_descs.append((aux_meta.ptrs[i], aux_meta.item_sizes[i], 0, f"aux_buffer_ptr_{i}"))
        reg_memory_desc = RegMemoryDescs("DRAM", ptr_descs)
        self._agent.register_memory(reg_memory_desc)
        logger.debug(f"Registered auxiliary buffer memory with transfer agent: {reg_memory_desc}")
        self._registered_mem.append(reg_memory_desc)

    # pack the aux data to the meta buffer

    def pack_aux(self, tx_session: TxSession, request: LlmRequest):
        self._aux_buffer.fill_slot(tx_session.aux_slot, request)

    def unpack_aux(self, rx_session: RxSession, request: LlmRequest):
        first_gen_tokens, draft_tokens = self._aux_buffer.get_slot_tokens(rx_session.aux_slot)

        # TODO: not first gen ,but add_tokens?
        request.py_first_gen_tokens = first_gen_tokens
        request.py_draft_tokens = draft_tokens
        return request

    def __del__(self):
        try:
            if self._instance_info_server is not None:
                self._instance_info_server.close()
            self._sender.close()
            self._receiver.close()
        except Exception as e:
            logger.error(f"Exception in TransferWorker.__del__ error: {e}")
