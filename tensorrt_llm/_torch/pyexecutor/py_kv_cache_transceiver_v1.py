import concurrent
import pickle
import sys
import threading
import uuid
from abc import ABC
from dataclasses import dataclass
from typing import Optional

import zmq

import tensorrt_llm.bindings
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings import DataType
from tensorrt_llm.disaggregated_params import DisaggregatedParams

AttentionTypeCpp = tensorrt_llm.bindings.internal.batch_manager.AttentionType
LlmRequestType = tensorrt_llm.bindings.internal.batch_manager.LlmRequestType
DataType = tensorrt_llm.bindings.DataType

nixl_path = "/opt/nvidia/nvda_nixl/lib/python3/dist-packages"
if nixl_path not in sys.path:
    sys.path.insert(0, nixl_path)

from nixl._api import nixl_agent, nixl_agent_config, nixl_xfer_handle


class TransferOp:
    READ = "READ"
    WRITE = "WRITE"


class MemoryDesc:

    def __init__(self, ptr: int, size: int, device_id: int):
        self.ptr = ptr
        self.size = size
        self.device_id = device_id


class MemoryDescs:

    def __init__(self, type, descs: list[tuple[int, int, int] | MemoryDesc]):
        self.type = type
        self.descs = descs


class RegMemoryDescs:

    def __init__(self, type, descs: list[tuple[int, int, int, str]]):
        self.type = type
        self.descs = descs


class TransferRequest:

    def __init__(self, op: TransferOp, src_descs: MemoryDescs,
                 dst_descs: MemoryDescs, remote_name: str, sync_message: str):
        self.op = op
        self.src_descs = src_descs
        self.dst_descs = dst_descs
        self.remote_name = remote_name
        self.sync_message = sync_message


class TransferStatus:

    def __init__(self):
        pass

    def is_completed(self):
        raise NotImplementedError

    def wait(self):
        raise NotImplementedError


class BaseTransferAgent(ABC):

    def register_memory(self, descs: MemoryDescs):
        raise NotImplementedError

    def deregister_memory(self, descs: MemoryDescs):
        raise NotImplementedError

    def load_remote_agent(self, name: str, agent_desc: str):
        raise NotImplementedError

    def get_local_agent_desc(self):
        raise NotImplementedError

    def invalidate_remote_agent(self, name: str):
        raise NotImplementedError

    def submit_transfer_requests(self,
                                 request: TransferRequest) -> TransferStatus:
        raise NotImplementedError

    def notify_sync_message(self, name: str, sync_message: str):
        raise NotImplementedError

    def check_remote_descs(self, name: str, memory_descs: list[int]):
        raise NotImplementedError


class NixlTransferStatus(TransferStatus):

    def __init__(self, agent: nixl_agent, handle: nixl_xfer_handle):
        self.agent = agent
        self.handle = handle

    def is_completed(self):
        status = self.agent.check_xfer_state(self.handle)
        return status == "DONE"

    def wait(self):
        status = "PROC"
        while status == "PROC":
            status = self.agent.check_xfer_state(self.handle)
            if status == "ERR":
                return False  # transfer failed
            # sleep(0.1)
        return True


class NixlTransferAgent(BaseTransferAgent):

    def __init__(self, name: str, use_prog_thread: bool):
        self.name = name
        agent_config = nixl_agent_config(enable_prog_thread=use_prog_thread,
                                         backends=["UCX"])
        self.agent = nixl_agent(name, agent_config)

    def register_memory(self, descs: RegMemoryDescs):
        reg_descs = self.agent.get_reg_descs(descs.descs, descs.type)
        self.agent.register_memory(reg_descs)

    def deregister_memory(self, descs: RegMemoryDescs):
        self.agent.deregister_memory(descs.descs, descs.type)

    def load_remote_agent(self, name: str, agent_desc: bytes):
        self.agent.add_remote_agent(agent_desc)

    def get_local_agent_desc(self):
        return self.agent.get_agent_metadata()

    def invalidate_remote_agent(self, name: str):
        self.agent.remove_remote_agent(name)

    def submit_transfer_requests(self,
                                 request: TransferRequest) -> TransferStatus:
        src_xfer_descs = self.agent.get_xfer_descs(request.src_descs.descs,
                                                   request.src_descs.type)
        dst_xfer_descs = self.agent.get_xfer_descs(request.dst_descs.descs,
                                                   request.dst_descs.type)
        handle = self.agent.initialize_xfer(request.op, src_xfer_descs,
                                            dst_xfer_descs, request.remote_name,
                                            request.sync_message)
        status = self.agent.transfer(handle)
        assert status != "ERR"
        return NixlTransferStatus(self.agent, handle)


@dataclass
class InstanceRankInfo:
    instance_name: str
    instance_rank: int
    tp_size: int
    tp_rank: int
    pp_size: int
    pp_rank: int
    dp_size: int
    dp_rank: int
    cp_size: int
    cp_rank: int
    kv_head_num_per_rank: int
    #  [numLayers,kv_factor,heads,tokens,dimsPerHead]
    tokens_per_block: int
    dims_per_head: int
    element_size: int
    enable_attention_dp: bool
    is_mla: bool
    layer_num_per_pp: list[int]
    kvcache_ptrs: list[int]
    aux_ptrs: list[int]
    server_endpoint: str
    recv_endpoint: str
    transfer_engine_info: bytes


@dataclass
class InstanceInfo:
    instance_name: str
    tp_size: int
    pp_size: int
    dp_size: int
    cp_size: int
    kv_head_num_per_rank: int
    tokens_per_block: int
    dims_per_head: int
    element_size: int
    enable_attention_dp: bool
    is_mla: bool
    layer_num_per_pp: list[int]
    ctx_server_endpoints: list[str]


@dataclass
class peerDomainRanks:
    domain_pp_size: int
    domain_tp_size: int
    domain_cp_size: int
    duplicate_head_factor: int
    peer_duplicate_head_factor: int
    target_peer_pp_layer_num: list[int]
    ranks: list[int]


class ResourceRegister:

    def __init__(self, instance_rank_info: InstanceRankInfo,
                 instance_info: InstanceInfo):
        self.instance_rank_info = instance_rank_info
        self.instance_info = instance_info
        self.peer_instance_rank_info_cache = {}
        self.peer_domain_ranks_cache = {}

    def register_peer_info(self, peer_instance_name: str, peer_rank: int,
                           peer_instance_rank_info: InstanceRankInfo):
        self.peer_instance_rank_info_cache[
            peer_instance_name + str(peer_rank)] = peer_instance_rank_info

    def unregister_peer_info(self, peer_instance_name: str, peer_rank: int):
        del self.peer_instance_rank_info_cache[peer_instance_name +
                                               str(peer_rank)]

    def get_peer_instance_rank_info(self, peer_instance_name: str,
                                    peer_rank: int):
        return self.peer_instance_rank_info_cache[peer_instance_name +
                                                  str(peer_rank)]

    def get_instance_info(self):
        return self.instance_info

    def get_instance_rank_info(self):
        return self.instance_rank_info

    def get_peer_domain_ranks(self, peer_instance_info: InstanceInfo,
                              peer_dp_rank: int):

        # return the mapping rank relationship between self and peer in asymmetric parallelism case

        if peer_instance_info.instance_name + str(
                peer_dp_rank) in self.peer_domain_ranks_cache:
            return self.peer_domain_ranks_cache[peer_instance_info.instance_name
                                                + str(peer_dp_rank)]
        peer_pp_rank_start = 0
        peer_pp_rank_end = 0
        self_start_layer_id = sum(
            self.instance_rank_info.layer_num_per_pp[:self.instance_rank_info.
                                                     pp_rank])
        self_end_layer_id = self_start_layer_id + self.instance_rank_info.layer_num_per_pp[
            self.instance_rank_info.pp_rank]
        pre_peer_pp_layer_id = 0
        target_peer_pp_ranks = []
        target_peer_pp_layer_num = []
        for pp_rank in range(peer_instance_info.pp_size):
            peer_pp_start_layer_id = pre_peer_pp_layer_id
            peer_pp_end_layer_id = peer_pp_start_layer_id + peer_instance_info.layer_num_per_pp[
                pp_rank]
            if self_start_layer_id < peer_pp_end_layer_id and self_end_layer_id > peer_pp_start_layer_id:
                target_peer_pp_ranks.append(pp_rank)
                target_peer_pp_layer_num.append(
                    min(peer_pp_end_layer_id, self_end_layer_id) -
                    max(peer_pp_start_layer_id, self_start_layer_id))
            pre_peer_pp_layer_id += peer_instance_info.layer_num_per_pp[pp_rank]

        peer_pp_rank_start = target_peer_pp_ranks[0]
        domain_pp_size = len(target_peer_pp_ranks)
        peer_pp_rank_end = peer_pp_rank_start + len(target_peer_pp_ranks)

        # dp
        self_tp_size_per_dp_group = self.instance_rank_info.tp_size / self.instance_rank_info.dp_size if self.instance_rank_info.enable_attention_dp else self.instance_rank_info.tp_size
        peer_tp_size_per_dp_group = peer_instance_info.tp_size / peer_instance_info.dp_size if peer_instance_info.enable_attention_dp else peer_instance_info.tp_size
        self_tprank_in_dp_group = self.instance_rank_info.tp_rank % self_tp_size_per_dp_group
        peer_tp_rank_start = 0
        peer_tp_rank_end = 0
        domain_tp_size = 1
        if (self_tp_size_per_dp_group <= peer_tp_size_per_dp_group):
            domain_tp_size = peer_tp_size_per_dp_group // self_tp_size_per_dp_group
            peer_tp_rank_start = self_tprank_in_dp_group * domain_tp_size + peer_dp_rank * peer_tp_size_per_dp_group
            peer_tp_rank_end = peer_tp_rank_start + domain_tp_size
        else:
            peer_tp_rank_start = self_tprank_in_dp_group // (
                self_tp_size_per_dp_group // peer_tp_size_per_dp_group
            ) + peer_dp_rank * peer_tp_size_per_dp_group
            peer_tp_rank_end = peer_tp_rank_start + domain_tp_size
        # peer_tp_rank_start = self_tprank_in_dp_group * domain_tp_size + peer_dp_rank * peer_tp_size_per_dp_group
        # peer_tp_rank_end = peer_tp_rank_start + domain_tp_size
        peer_cp_rank_start = 0
        peer_cp_rank_end = 0
        domain_cp_size = 1
        if (self.instance_rank_info.cp_size <= peer_instance_info.cp_size):
            domain_cp_size = peer_instance_info.cp_size // self.instance_rank_info.cp_size
            peer_cp_rank_start = self.instance_rank_info.cp_rank * domain_cp_size
            peer_cp_rank_end = peer_cp_rank_start + domain_cp_size
        else:
            peer_cp_rank_start = self.instance_rank_info.cp_rank // (
                self.instance_rank_info.cp_size // peer_instance_info.cp_size)
            peer_cp_rank_end = peer_cp_rank_start + domain_cp_size
        ranks = []
        for pp_rank in range(peer_pp_rank_start, peer_pp_rank_end):

            for cp_rank in range(peer_cp_rank_start, peer_cp_rank_end):
                for tp_rank in range(peer_tp_rank_start, peer_tp_rank_end):
                    ranks.append(pp_rank * peer_instance_info.tp_size *
                                 peer_instance_info.cp_size +
                                 cp_rank * peer_instance_info.tp_size + tp_rank)
        duplicate_head_factor = max(
            1, self.instance_rank_info.kv_head_num_per_rank *
            self_tp_size_per_dp_group //
            (peer_instance_info.kv_head_num_per_rank *
             peer_tp_size_per_dp_group))
        peer_duplicate_head_factor = max(
            1, peer_instance_info.kv_head_num_per_rank *
            peer_tp_size_per_dp_group //
            (self.instance_rank_info.kv_head_num_per_rank *
             self_tp_size_per_dp_group))

        peer_domain_ranks = peerDomainRanks(
            domain_pp_size=domain_pp_size,
            domain_tp_size=domain_tp_size,
            domain_cp_size=domain_cp_size,
            duplicate_head_factor=duplicate_head_factor,
            peer_duplicate_head_factor=peer_duplicate_head_factor,
            target_peer_pp_layer_num=target_peer_pp_layer_num,
            ranks=ranks,
        )
        self.peer_domain_ranks_cache[peer_instance_info.instance_name +
                                     str(peer_dp_rank)] = peer_domain_ranks
        return peer_domain_ranks

    def get_kv_block_ptrs_extractor(self,
                                    peer_instance_rank_info: InstanceRankInfo):

        # return the kv block ptrs extractor for the peer. the extractor will be used to extract the kv block ptrs and submit to transfer agent.
        pass

        return


class TransferSession:

    def __init__(self, kv_cache_manager: KVCacheManager, request_id: int,
                 resource_register: ResourceRegister):
        self.kv_cache_manager = kv_cache_manager
        self.request_id = request_id
        self.resource_register = resource_register
        self.future = concurrent.futures.Future()
        self.first_extracted = False
        self.encountered_count = 0
        self.expect_count = 0
        self.session_id = str(uuid.uuid4())

    def get_future_for_session(self):
        return self.future

    def extra_trams_meta(self):
        pass


class TransferGenSideReqInfo:
    ctx_req_id: int
    instance_name: str
    instance_rank: int
    block_ids: list[int]
    gen_req_id: int
    session_id: str


@dataclass
class TransSendMeta:
    session_id: str
    future_for_session: concurrent.futures.Future
    src_kv_ptrs: list[int]
    dst_kv_ptrs: list[int]
    kv_sizes: list[int]
    expect_count: int
    remote_name: str
    src_aux_ptrs: list[int] = None
    dst_aux_ptrs: list[int] = None
    aux_sizes: list[int] = None
    peer_endpoint: Optional[str] = None  # used for send state
    peer_session_id: Optional[str] = None


@dataclass
class TransRecvMeta:
    session_id: str
    future_for_session: concurrent.futures.Future
    expect_count: int
    remote_name: str


class SessionState:

    INIT = "INIT"
    WAITING_FOR_SEND = "WAITING_FOR_SEND"
    TRANSFERRING = "TRANSFERRING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class MessageType:
    TERMINATION = "TERMINATION"
    TASK_STATE = "TASK_STATE"
    PEER_INFO = "PEER_INFO"
    REQUEST_DATA = "REQUEST_DATA"
    REQUEST_INSTANCE_INFO = "REQUEST_INSTANCE_INFO"
    REGISTER_RANK_INFO = "REQUEST_RANK_INFO"


class SendSession(TransferSession):

    def __init__(self, kv_cache_manager: KVCacheManager, request_id: int,
                 resource_register: ResourceRegister):
        super().__init__(kv_cache_manager, request_id, resource_register)

    def extract_trans_meta(self,
                           dst_info: TransferGenSideReqInfo) -> TransSendMeta:
        pass

    def get_state(self) -> SessionState:
        pass

    def trigger_send_chunk(self, chunk_id, chunk_num):
        # call sender_submit_send_task
        pass


class ReceiveSession(TransferSession):

    def __init__(self, kv_cache_manager: KVCacheManager, request_id: int,
                 resource_register: ResourceRegister):
        super().__init__(kv_cache_manager, request_id, resource_register)

    def extract_trans_meta(
            self, peer_instance_info: InstanceInfo,
            peer_dp_rank: int) -> tuple[TransRecvMeta, list[int]]:
        peer_domain_ranks = self.resource_register.get_peer_domain_ranks(
            peer_instance_info, peer_dp_rank)
        expect_count = len(peer_domain_ranks.ranks)
        if not self.first_extracted:
            self.first_extracted = True
            self.expect_count = len(peer_domain_ranks.ranks)
        self.encountered_count = self.encountered_count + 1
        return TransRecvMeta(session_id=self.session_id,
                             future_for_session=self.future,
                             expect_count=expect_count,
                             remote_name=None), peer_domain_ranks.ranks

    def get_state(self) -> SessionState:
        pass


class Sender:

    def __init__(self, kv_cache_manager: KVCacheManager,
                 resource_register: ResourceRegister, device_id: int,
                 transfer_agent: BaseTransferAgent):
        self.kv_cache_manager = kv_cache_manager
        self.resource_register = resource_register
        self.device_id = device_id
        self.transfer_agent = transfer_agent
        self.send_session_cache = {}
        self.send_session_cache_lock = threading.Lock()

        self._peer_transfer_recv_req_info_cache = {}
        self._peer_transfer_recv_req_info_lock = threading.Lock()

        self._zmq_context = zmq.Context()
        self.server_socket = self._zmq_context.socket(zmq.ROUTER)
        self.server_socket.bind(f"tcp://*:5555")
        self.server_endpoint = self.server_socket.getsockopt(
            zmq.LAST_ENDPOINT).decode()

        self.socket_cache = {}

    #  return a session for a request.
    #  upper layer can check the state of the session to know the progress of the request.
    #
    def async_send(self, request: LlmRequest) -> SendSession:
        request_id = request.py_request_id

        if request_id in self.send_session_cache:
            return self.send_session_cache[request_id]
        send_session = SendSession(self.kv_cache_manager, request_id,
                                   self.resource_register)
        self.send_session_cache[request_id] = send_session

        self._handel_send_session(send_session)

        return

    #  for upper layer to create a send session for a request ,only used for pre-allocate flow
    def create_send_session(self, request: LlmRequest) -> SendSession:

        #  for upper layer to create a send session for a request ,only used for pre-allocate flow. the kvcache will not send until session.trigger_send_chunk() is called.

        pass

    def cancel_request(self, request: LlmRequest):
        pass

    def submit_send_task(self, trans_send_meta: TransSendMeta):
        pass

    def _handel_send_session(self, send_session: SendSession):
        pass

    def _handle_sender_loop(self):
        while True:
            message = self.server_socket.recv_multipart()
            send_id = message[0]
            recv_message = message[1:]
            if self._message_is_termination(recv_message):
                break
            elif self._message_is_request_data(recv_message):
                self._handle_request_data(send_id, recv_message)
            elif self._message_is_request_instance_info(recv_message):
                self._handle_request_instance_info(send_id, recv_message)

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

    def _message_is_request_instance_info(self, message: list[bytes]):
        return message[0] == str(
            MessageType.REQUEST_INSTANCE_INFO).encode("ascii")

    def _message_is_register_rank_info(self, message: list[bytes]):
        return message[0] == str(MessageType.REGISTER_RANK_INFO).encode("ascii")

    def _handle_request_instance_info(self, send_id: bytes,
                                      message: list[bytes]):
        assert len(message) == 1
        send_message = [send_id, pickle.dumps(self.instance_info)]
        self.server_socket.send_multipart(send_message)

    def _handle_register_rank_info(self, send_id: bytes, message: list[bytes]):
        instance_rank_info: InstanceRankInfo = pickle.loads(message[1])
        self.cache_transfer_manager.register_peer(
            instance_rank_info.instance_name, instance_rank_info.instance_rank,
            instance_rank_info)
        self.transfer_agent.load_remote_agent(
            instance_rank_info.instance_name +
            str(instance_rank_info.instance_rank),
            instance_rank_info.transfer_engine_info)

    def _get_send_transfer_session(self, ctx_req_id: int) -> TransferSession:
        with self.send_session_cache_lock:
            if ctx_req_id in self.send_session_cache:
                return self.send_session_cache[ctx_req_id]
            else:
                return None

    def _handle_request_data(self, send_id: bytes, message: list[bytes]):

        transfer_gen_side_req_info: TransferGenSideReqInfo = pickle.loads(
            message[1])

        ctx_req_id = transfer_gen_side_req_info.ctx_req_id

        send_transfer_session = self._get_send_transfer_session(ctx_req_id)
        if send_transfer_session is None:
            print(f" _handle_request_data, send_transfer_session is None")
            self._save_peer_transfer_req_info(transfer_gen_side_req_info)
        else:
            print(f" _handle_request_data, send_transfer_session is not None")
            trans_meta = send_transfer_session.extract_trans_meta(
                transfer_gen_side_req_info)
            self.submit_send_task(trans_meta)
            #           # do we need big lock to protect is_active(), remain_count
            if (not send_transfer_session.is_active()):
                with self.send_session_cache_lock:
                    print(" handle_request_data, delete send_session_cache")

                    if ctx_req_id in self.send_session_cache:
                        del self.send_session_cache[ctx_req_id]

    def _get_socket(self, endpoint: str):
        if endpoint not in self.socket_cache:
            self.socket_cache[endpoint] = self._zmq_context.socket(zmq.DEALER)
            self.socket_cache[endpoint].connect(endpoint)
        return self.socket_cache[endpoint]

    def _save_peer_transfer_req_info(
            self, peer_transfer_req_info: TransferGenSideReqInfo):
        with self._peer_transfer_recv_req_info_lock:
            if peer_transfer_req_info.ctx_req_id not in self._peer_transfer_recv_req_info_cache:
                self._peer_transfer_recv_req_info_cache[
                    peer_transfer_req_info.ctx_req_id] = {}
            self._peer_transfer_recv_req_info_cache[
                peer_transfer_req_info.ctx_req_id][
                    peer_transfer_req_info.
                    instance_rank] = peer_transfer_req_info


class Receiver:

    def __init__(self, kv_cache_manager: KVCacheManager,
                 resource_register: ResourceRegister, device_id: int,
                 transfer_agent: BaseTransferAgent):
        self.kv_cache_manager = kv_cache_manager
        self.resource_register = resource_register
        self.device_id = device_id
        self.transfer_agent = transfer_agent
        self.receive_session_cache = {}
        self.receive_session_cache_lock = threading.Lock()

        self._zmq_context = zmq.Context()

        self.context_endpoint_to_instance_info_cache = {}

    def async_receive(self, request: LlmRequest) -> ReceiveSession:
        # for upper layer to create a receive session for a request, and the receiver will async wait the receive finished.

        pass

    def _async_request_data_transfer(self, request: LlmRequest):
        pass

    def _need_register_peer_in_first_request(
            self, disagg_params: DisaggregatedParams) -> bool:
        pass

    def _get_socket(self, endpoint: str):

        pass

    def _get_context_info(self,
                          disagg_params: DisaggregatedParams) -> InstanceInfo:
        pass

    def submit_receive_task(self, trans_recv_meta: TransRecvMeta):

        # async wait the write finished signal from sender
        pass

    def _handle_receiver_loop(self):

        pass


class TransferWorker:

    def __init__(
        self,
        kv_cache_manager: KVCacheManager,
        instance_info: InstanceInfo,
        instance_rank_info: InstanceRankInfo,
        device_id: int,
    ):

        self.kv_cache_manager = kv_cache_manager
        self.resource_register = ResourceRegister(instance_info,
                                                  instance_rank_info)
        self.device_id = device_id
        self.transfer_agent = NixlTransferAgent(
            instance_rank_info.instance_name +
            str(instance_rank_info.instance_rank), True)
        self.sender = Sender(kv_cache_manager, self.resource_register,
                             device_id, self.transfer_agent)
        self.receiver = Receiver(kv_cache_manager, self.resource_register,
                                 device_id, self.transfer_agent)

    def create_send_session(self, request: LlmRequest) -> SendSession:
        return self.sender.create_send_session(request)

    def async_send(self, request: LlmRequest) -> SendSession:
        return self.sender.async_send(request)

    def async_receive(self, request: LlmRequest) -> ReceiveSession:
        return self.receiver.async_receive(request)

    def cancel_request(self, request: LlmRequest):
        pass
