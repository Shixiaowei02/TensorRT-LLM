import concurrent
import pickle
import threading
import uuid
from dataclasses import dataclass
from typing import List, Optional, Tuple, override

import zmq

import tensorrt_llm.bindings
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.disaggregated_params import DisaggregatedParams

from ..base import BaseTransferAgent, ReceiverBase, Request, SenderBase, SessionState, TransResult
from .kv_registry import ExecutorDesc, KVRegistry, WorkerDesc

AttentionTypeCpp = tensorrt_llm.bindings.internal.batch_manager.AttentionType
RequestType = tensorrt_llm.bindings.internal.batch_manager.RequestType


class Session:
    def __init__(self, kv_cache_manager: KVCacheManager, request_id: int, kv_registry: KVRegistry):
        self.kv_cache_manager = kv_cache_manager
        self.request_id = request_id
        self.kv_registry = kv_registry
        self.future = concurrent.futures.Future()
        self.first_extracted = False
        self.encountered_count = 0
        self.expect_count = 0
        self.session_id = str(uuid.uuid4())

    def get_future_for_session(self):
        return self.future

    def extra_trams_meta(self):
        pass


class GenSideReqInfo:
    ctx_req_id: int
    instance_name: str
    instance_rank: int
    block_ids: List[int]
    gen_req_id: int
    session_id: str


@dataclass
class SendMeta:
    session_id: str
    future_for_session: concurrent.futures.Future
    src_kv_ptrs: List[int]
    dst_kv_ptrs: List[int]
    kv_sizes: List[int]
    expect_count: int
    remote_name: str
    src_aux_ptrs: List[int] = None
    dst_aux_ptrs: List[int] = None
    aux_sizes: List[int] = None
    peer_endpoint: Optional[str] = None  # used for send state
    peer_session_id: Optional[str] = None


@dataclass
class RecvMeta:
    session_id: str
    future_for_session: concurrent.futures.Future
    expect_count: int
    remote_name: str


class MessageType:
    TERMINATION = "TERMINATION"
    TASK_STATE = "TASK_STATE"
    PEER_INFO = "PEER_INFO"
    REQUEST_DATA = "REQUEST_DATA"
    REQUEST_INSTANCE_INFO = "REQUEST_INSTANCE_INFO"
    REGISTER_RANK_INFO = "REQUEST_RANK_INFO"


class SendSession(Session):
    def __init__(self, kv_cache_manager: KVCacheManager, request_id: int, kv_registry: KVRegistry):
        super().__init__(kv_cache_manager, request_id, kv_registry)

    def extract_trans_meta(self, dst_info: GenSideReqInfo) -> SendMeta:
        pass

    def get_state(self) -> SessionState:
        pass

    def trigger_send_chunk(self, chunk_id, chunk_num):
        # call sender_submit_send_task
        pass


class ReceiveSession(Session):
    def __init__(self, kv_cache_manager: KVCacheManager, request_id: int, kv_registry: KVRegistry):
        super().__init__(kv_cache_manager, request_id, kv_registry)

    def extract_trans_meta(
        self, peer_instance_info: ExecutorDesc, peer_dp_rank: int
    ) -> Tuple[RecvMeta, List[int]]:
        peer_domain_ranks = self.kv_registry.get_peer_ranks(peer_instance_info, peer_dp_rank)
        expect_count = len(peer_domain_ranks.ranks)
        if not self.first_extracted:
            self.first_extracted = True
            self.expect_count = len(peer_domain_ranks.ranks)
        self.encountered_count = self.encountered_count + 1
        return RecvMeta(
            session_id=self.session_id,
            future_for_session=self.future,
            expect_count=expect_count,
            remote_name=None,
        ), peer_domain_ranks.ranks

    def get_state(self) -> SessionState:
        pass


class Sender(SenderBase):
    def __init__(
        self,
        worker_desc: WorkerDesc,
        exec_desc: ExecutorDesc,
        kv_registry: KVRegistry,
        device_id: int,
        transfer_agent: BaseTransferAgent,
    ):
        self.worker_desc = worker_desc
        self.exec_desc = exec_desc
        self.kv_registry = kv_registry
        self.device_id = device_id
        self.transfer_agent = transfer_agent
        self.send_session_cache = {}
        self.send_session_cache_lock = threading.Lock()

        self._peer_transfer_recv_req_info_cache = {}
        self._peer_transfer_recv_req_info_lock = threading.Lock()

        self._zmq_context = zmq.Context()
        self.server_socket = self._zmq_context.socket(zmq.ROUTER)
        self.server_socket.bind("tcp://*:5555")
        self.server_endpoint = self.server_socket.getsockopt(zmq.LAST_ENDPOINT).decode()

        self.socket_cache = {}

    #  return a session for a request.
    #  upper layer can check the state of the session to know the progress of the request.
    #
    @override
    def async_send(self, request: Request) -> TransResult:
        request_id = request.py_request_id

        if request_id in self.send_session_cache:
            return self.send_session_cache[request_id]
        send_session = SendSession(self.kv_cache_manager, request_id, self.kv_registry)
        self.send_session_cache[request_id] = send_session

        self._handle_send_session(send_session)

        return

    #  for upper layer to create a send session for a request ,only used for pre-allocate flow
    def create_session(self, request: Request) -> SendSession:
        #  for upper layer to create a send session for a request ,only used for pre-allocate flow.
        # the kvcache will not send until session.trigger_send_chunk() is called.

        pass

    def cancel_request(self, request: Request):
        pass

    def submit_send_task(self, trans_send_meta: SendMeta):
        pass

    def _handle_send_session(self, send_session: SendSession):
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

    def _message_is_termination(self, message: List[bytes]):
        return message[0] == str(MessageType.TERMINATION).encode("ascii")

    def _message_is_request_data(self, message: List[bytes]):
        # mapping_info = self._convert_message_to_mapping_info(message)
        return message[0] == str(MessageType.REQUEST_DATA).encode("ascii")

    def _message_is_request_instance_info(self, message: List[bytes]):
        return message[0] == str(MessageType.REQUEST_INSTANCE_INFO).encode("ascii")

    def _message_is_register_rank_info(self, message: List[bytes]):
        return message[0] == str(MessageType.REGISTER_RANK_INFO).encode("ascii")

    def _handle_request_instance_info(self, send_id: bytes, message: List[bytes]):
        assert len(message) == 1
        send_message = [send_id, pickle.dumps(self.instance_info)]
        self.server_socket.send_multipart(send_message)

    def _handle_register_rank_info(self, send_id: bytes, message: List[bytes]):
        instance_rank_info: WorkerDesc = pickle.loads(message[1])
        self.cache_transfer_manager.register_peer(
            instance_rank_info.instance_name, instance_rank_info.instance_rank, instance_rank_info
        )
        self.transfer_agent.load_remote_agent(
            instance_rank_info.instance_name + str(instance_rank_info.instance_rank),
            instance_rank_info.transfer_engine_info,
        )

    def _get_send_transfer_session(self, ctx_req_id: int) -> Session:
        with self.send_session_cache_lock:
            if ctx_req_id in self.send_session_cache:
                return self.send_session_cache[ctx_req_id]
            else:
                return None

    def _handle_request_data(self, send_id: bytes, message: List[bytes]):
        transfer_gen_side_req_info: GenSideReqInfo = pickle.loads(message[1])

        ctx_req_id = transfer_gen_side_req_info.ctx_req_id

        send_transfer_session = self._get_send_transfer_session(ctx_req_id)
        if send_transfer_session is None:
            print(" _handle_request_data, send_transfer_session is None")
            self._save_peer_transfer_req_info(transfer_gen_side_req_info)
        else:
            print(" _handle_request_data, send_transfer_session is not None")
            trans_meta = send_transfer_session.extract_trans_meta(transfer_gen_side_req_info)
            self.submit_send_task(trans_meta)
            #           # do we need big lock to protect is_active(), remain_count
            if not send_transfer_session.is_active():
                with self.send_session_cache_lock:
                    print(" handle_request_data, delete send_session_cache")

                    if ctx_req_id in self.send_session_cache:
                        del self.send_session_cache[ctx_req_id]

    def _get_socket(self, endpoint: str):
        if endpoint not in self.socket_cache:
            self.socket_cache[endpoint] = self._zmq_context.socket(zmq.DEALER)
            self.socket_cache[endpoint].connect(endpoint)
        return self.socket_cache[endpoint]

    def _save_peer_transfer_req_info(self, peer_transfer_req_info: GenSideReqInfo):
        with self._peer_transfer_recv_req_info_lock:
            if peer_transfer_req_info.ctx_req_id not in self._peer_transfer_recv_req_info_cache:
                self._peer_transfer_recv_req_info_cache[peer_transfer_req_info.ctx_req_id] = {}
            self._peer_transfer_recv_req_info_cache[peer_transfer_req_info.ctx_req_id][
                peer_transfer_req_info.instance_rank
            ] = peer_transfer_req_info


class Receiver(ReceiverBase):
    def __init__(
        self,
        worker_desc: WorkerDesc,
        exec_desc: ExecutorDesc,
        kv_registry: KVRegistry,
        device_id: int,
        transfer_agent: BaseTransferAgent,
    ):
        self.worker_desc = worker_desc
        self.exec_desc = exec_desc
        self.kv_registry = kv_registry
        self.device_id = device_id
        self.transfer_agent = transfer_agent
        self.receive_session_cache = {}
        self.receive_session_cache_lock = threading.Lock()

        self._zmq_context = zmq.Context()

        self.context_endpoint_to_instance_info_cache = {}

    @override
    def async_send(self, request: Request) -> TransResult:
        # for upper layer to create a receive session for a request,
        # and the receiver will async wait the receive finished.

        pass

    def _async_request_data_transfer(self, request: Request):
        pass

    def _need_register_peer_in_first_request(self, disagg_params: DisaggregatedParams) -> bool:
        pass

    def _get_socket(self, endpoint: str):
        pass

    def _get_context_info(self, disagg_params: DisaggregatedParams) -> ExecutorDesc:
        pass

    def submit_receive_task(self, trans_recv_meta: RecvMeta):
        # async wait the write finished signal from sender
        pass

    def _handle_receiver_loop(self):
        pass
