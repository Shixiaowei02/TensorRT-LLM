import concurrent
import pickle
import sys
import threading
import time
import uuid
from abc import ABC, abstractmethod
from builtins import dict
from dataclasses import dataclass
from typing import Optional

import zmq

import tensorrt_llm.bindings
import tensorrt_llm.bindings.executor as trtllm
from tensorrt_llm._torch.pyexecutor.kv_cache_transceiver import \
    KvCacheTransceiver
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings import DataType, LlmRequestState
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.sampling_params import SamplingParams

AttentionTypeCpp = tensorrt_llm.bindings.internal.batch_manager.AttentionType
LlmRequestType = tensorrt_llm.bindings.internal.batch_manager.LlmRequestType
DataType = tensorrt_llm.bindings.DataType

nixl_path = "/opt/nvidia/nvda_nixl/lib/python3/dist-packages"
if nixl_path not in sys.path:
    sys.path.insert(0, nixl_path)

import torch
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

    # def close(self):
    # self.agen


@dataclass
class TransReqMeta:
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
class TransReqRecvMeta:
    session_id: str
    future_for_session: concurrent.futures.Future
    expect_count: int
    remote_name: str


class BaseDataSender(ABC):
    #  be called by PyKvCacheTransceiver
    # def __init__(self, kv_cache_manager: KVCacheManager,config: CacheTransceiverConfig):

    @abstractmethod
    def submit_transfer_task(self, transfer_meta_data: TransReqMeta):
        raise NotImplementedError


class BaseDataReceiver(ABC):
    #  be called by PyKvCacheTransceiver
    # def __init__(self, kv_cache_manager: KVCacheManager,config: CacheTransceiverConfig):
    @abstractmethod
    def submit_transfer_task(self, transfer_meta_data: TransReqMeta
                             | TransReqRecvMeta):
        raise NotImplementedError

    @abstractmethod
    def get_endpoint(self):
        raise NotImplementedError


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
    except:
        pass

    try:
        import netifaces
        for interface in netifaces.interfaces():
            addrs = netifaces.ifaddresses(interface)
            if netifaces.AF_INET in addrs:
                for addr in addrs[netifaces.AF_INET]:
                    ip = addr['addr']
                    if not ip.startswith("127.") and not ip.startswith(
                            "169.254"):
                        return ip
    except ImportError:
        pass

    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        if not local_ip.startswith("127."):
            return local_ip
    except:
        pass
    return "127.0.0.1"


class MessageType:
    TERMINATION = "TERMINATION"
    TASK_STATE = "TASK_STATE"
    PEER_INFO = "PEER_INFO"
    REQUEST_DATA = "REQUEST_DATA"
    REQUEST_INSTANCE_INFO = "REQUEST_INSTANCE_INFO"
    REGISTER_RANK_INFO = "REQUEST_RANK_INFO"


class DataSender(BaseDataSender):

    def __init__(self, agent: NixlTransferAgent, device_id: int):
        self.agent = agent
        self.device_id = device_id

        self.session_id_to_count = {}
        self.zmq_context = zmq.Context()
        self.socket_cache = {}

    def submit_transfer_task(self, transfer_meta_data: TransReqMeta):
        print(
            f" data sender submit_transfer_task, transfer_meta_data:{transfer_meta_data}"
        )
        # 将 transfer_meta_data 放入队列，由后台线程/Executor 处理
        if not hasattr(self, '_transfer_queue'):
            import concurrent.futures
            import queue
            import threading
            self._transfer_queue = queue.Queue()
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1)

            def _consumer():
                while True:
                    meta = self._transfer_queue.get()
                    if meta is None:
                        break
                    try:
                        print(f"start handle_transfer_task")
                        self._handle_transfer_task(meta)
                        print(" handle_transfer_task success")
                    except Exception as e:
                        if hasattr(meta, "future_for_session"):
                            meta.future_for_session.set_exception(e)
                    finally:
                        self._transfer_queue.task_done()

            self._background_thread = threading.Thread(target=_consumer,
                                                       daemon=True)
            self._background_thread.start()
        self._transfer_queue.put(transfer_meta_data)

    def _handle_transfer_task(self, transfer_meta_data: TransReqMeta):
        print(
            f" enter _handle_transfer_task, transfer_meta_data:{transfer_meta_data}"
        )
        assert len(transfer_meta_data.src_kv_ptrs) == len(
            transfer_meta_data.dst_kv_ptrs)
        assert len(transfer_meta_data.kv_sizes) == len(
            transfer_meta_data.src_kv_ptrs)
        if transfer_meta_data.session_id not in self.session_id_to_count:
            self.session_id_to_count[transfer_meta_data.session_id] = 0
        src_kv_list = [(src_ptr, size, self.device_id) for src_ptr, size in zip(
            transfer_meta_data.src_kv_ptrs, transfer_meta_data.kv_sizes)]
        dst_kv_list = [(dst_ptr, size, self.device_id) for dst_ptr, size in zip(
            transfer_meta_data.dst_kv_ptrs, transfer_meta_data.kv_sizes)]
        print(f"src_kv_list: {src_kv_list}")
        print(f"dst_kv_list: {dst_kv_list}")
        src_memory_descs = MemoryDescs("VRAM", src_kv_list)
        dst_memory_descs = MemoryDescs("VRAM", dst_kv_list)
        request = TransferRequest(TransferOp.WRITE, src_memory_descs,
                                  dst_memory_descs,
                                  transfer_meta_data.remote_name, '')
        status = self.agent.submit_transfer_requests(request)
        sync_status = "SUCCESS"
        if not status.wait():
            sync_status = "FAILED"
            transfer_meta_data.future_for_session.set_exception(
                RuntimeError("Transfer failed"))
        socket = self._get_socket(transfer_meta_data.peer_endpoint)
        socket.send_multipart([
            str(MessageType.TASK_STATE).encode("ascii"),
            transfer_meta_data.peer_session_id.encode("ascii"),
            sync_status.encode("ascii")
        ])
        # TODO: socket_cache
        print(f"habdle_transfer_task, transfer_meta_data:{transfer_meta_data}")
        self.session_id_to_count[transfer_meta_data.session_id] += 1
        if (self.session_id_to_count[transfer_meta_data.session_id]
                > transfer_meta_data.expect_count):
            raise RuntimeError(
                f"Session {transfer_meta_data.session_id} has more than {transfer_meta_data.expect_count} transfers"
            )
        if (self.session_id_to_count[transfer_meta_data.session_id] ==
                transfer_meta_data.expect_count):
            transfer_meta_data.future_for_session.set_result(sync_status)
            del self.session_id_to_count[transfer_meta_data.session_id]

    def _get_socket(self, peer_endpoint: str):
        if peer_endpoint not in self.socket_cache:
            self.socket_cache[peer_endpoint] = self.zmq_context.socket(
                zmq.DEALER)
            print(f"DataSender get_socket connect to: {peer_endpoint}")
            self.socket_cache[peer_endpoint].connect(peer_endpoint)
        return self.socket_cache[peer_endpoint]


class DataReceiver(BaseDataReceiver):

    def __init__(self, agent: NixlTransferAgent, device_id: int):
        self.agent = agent
        self.device_id = device_id
        self.zmq_context = zmq.Context()
        self.server_socket = self.zmq_context.socket(zmq.ROUTER)
        self.server_socket.bind(f"tcp://{get_local_ip()}:*")
        self.server_endpoint = self.server_socket.getsockopt(
            zmq.LAST_ENDPOINT).decode()
        print(f"DataReceiver server_endpoint: {self.server_endpoint}")
        self.session_id_to_count = {}
        # TODO: add a lock for session_id_to_count and session_id_to_future
        self.session_id_to_future = {}
        self._background_thread = threading.Thread(
            target=self._loop_for_receive_state, daemon=True)
        self._background_thread.start()

    def submit_transfer_task(self, transfer_meta_data: TransReqMeta
                             | TransReqRecvMeta):
        if transfer_meta_data.session_id not in self.session_id_to_count:
            self.session_id_to_count[
                transfer_meta_data.session_id] = transfer_meta_data.expect_count
            self.session_id_to_future[
                transfer_meta_data.
                session_id] = transfer_meta_data.future_for_session
        else:
            assert self.session_id_to_future[
                transfer_meta_data.
                session_id] == transfer_meta_data.future_for_session

    def get_endpoint(self):
        return self.server_endpoint

    def _loop_for_receive_state(self):
        while True:
            message = self.server_socket.recv_multipart()
            send_id = message[0]
            recv_message = message[1:]
            if self._message_is_termination(recv_message):
                break
            elif self._message_is_task_state(recv_message):
                self._handle_task_state(send_id, recv_message)
            else:
                raise ValueError(
                    f" data receiver received unknown message type: {recv_message[0]}"
                )

    def _message_is_termination(self, message: list[bytes]):
        return message[0] == str(MessageType.TERMINATION).encode("ascii")

    def _message_is_task_state(self, message: list[bytes]):
        return message[0] == str(MessageType.TASK_STATE).encode("ascii")

    def _handle_task_state(self, send_id: bytes, message: list[bytes]):
        assert len(message) == 3
        assert message[0].decode("ascii") == str(MessageType.TASK_STATE)
        session_id = message[1].decode("ascii")
        task_state = message[2].decode("ascii")
        if task_state == "SUCCESS":
            print(
                f"task state is success session_id_to_count: {self.session_id_to_count}"
            )
            self.session_id_to_count[session_id] -= 1
            if self.session_id_to_count[session_id] == 0:
                # print(f"session {session_id} is completed , send_id: {send_id}")
                self.session_id_to_future[session_id].set_result("SUCCESS")
                self.session_id_to_future.pop(session_id)
        elif task_state == "FAILED":
            self.session_id_to_future[session_id].set_exception(
                RuntimeError(f"Task state: {task_state}"))
        else:
            raise ValueError(
                f" session {session_id} received unknown task state: {task_state}"
            )


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

    @classmethod
    def from_zmq(cls, msg: list[bytes]):
        return cls(instance_name=msg[0].decode("ascii"),
                   tp_size=int(msg[1].decode("ascii")),
                   pp_size=int(msg[2].decode("ascii")),
                   dp_size=int(msg[3].decode("ascii")),
                   cp_size=int(msg[4].decode("ascii")),
                   kv_head_num_per_rank=int(msg[5].decode("ascii")),
                   tokens_per_block=int(msg[6].decode("ascii")),
                   dims_per_head=int(msg[7].decode("ascii")),
                   element_size=int(msg[8].decode("ascii")),
                   enable_attention_dp=bool(msg[9].decode("ascii")),
                   is_mla=bool(msg[10].decode("ascii")),
                   layer_num_per_pp=pickle.loads(msg[11]),
                   ctx_server_endpoints=pickle.loads(msg[12]))


@dataclass
class TransferReqInfo:
    req_id: int
    start_token_idx: Optional[int] = None
    end_token_idx: Optional[int] = None
    start_layer_idx: Optional[int] = None
    end_layer_idx: Optional[int] = None
    block_ids: Optional[list[int]] = None
    instance_name: Optional[str] = None
    instance_rank: Optional[int] = None


@dataclass
class TransferGenSideReqInfo:
    ctx_req_id: int
    instance_name: str
    instance_rank: int
    block_ids: list[int]
    gen_req_id: int
    session_id: str


@dataclass
class peerDomainRanks:
    domain_pp_size: int
    domain_tp_size: int
    domain_cp_size: int
    duplicate_head_factor: int
    peer_duplicate_head_factor: int
    target_peer_pp_layer_num: list[int]
    ranks: list[int]


class TransferSession:
    _peer_domain_ranks_cache: dict[str, peerDomainRanks] = {}

    def __init__(self, transferReqInfo: TransferReqInfo, transfer_manager,
                 instance_rank_info: InstanceRankInfo):
        self.first_extracted = False
        self.future = concurrent.futures.Future()
        self.instance_rank_info = instance_rank_info
        self.transfer_req_info = transferReqInfo
        self.session_id = str(uuid.uuid4())
        self.transfer_manager = transfer_manager
        self.request_id = transferReqInfo.req_id
# kv_layout: Literal["NHD", "HND"] = "NHD"
# Extract the transmission metadata for the current transfer session. Each rank needs to
# retrieve meta separately. Once all ranks have completed the extraction, the session is
# considered complete.

### compute the ptrs and size , big logic
### sender and receiver side should have differ transferReqInfo class name

    def extract_trans_meta(self,
                           dst_info: TransferGenSideReqInfo) -> TransReqMeta:
        # TODO: compute the ptrs and size

        # TODO: get peer instance rank info and peer instance info
        peer_instance_rank_info: InstanceRankInfo = self.transfer_manager.get_peer_instance_rank_info(
            dst_info.instance_name, dst_info.instance_rank)
        peer_domain_ranks = self._get_peer_peer_target_ranks(
            peer_instance_rank_info, peer_instance_rank_info.dp_rank)
        expect_count = len(peer_domain_ranks.ranks)
        if not self.first_extracted:
            self.first_extracted = True
            self.remain_count = len(peer_domain_ranks.ranks)
        need_send = self._need_to_transfer(peer_domain_ranks,
                                           peer_instance_rank_info)
        kv_factor = 1 if self.instance_rank_info.is_mla else 2
        self_kv_block_size = self.instance_rank_info.layer_num_per_pp[
            self.instance_rank_info.
            pp_rank] * kv_factor * self.instance_rank_info.kv_head_num_per_rank * self.instance_rank_info.tokens_per_block * self.instance_rank_info.dims_per_head * self.instance_rank_info.element_size
        # save
        peer_kv_block_size = peer_instance_rank_info.layer_num_per_pp[
            peer_instance_rank_info.
            pp_rank] * kv_factor * peer_instance_rank_info.kv_head_num_per_rank * peer_instance_rank_info.tokens_per_block * peer_instance_rank_info.dims_per_head * peer_instance_rank_info.element_size

        src_block_ids, dst_block_ids = self.filter_kv_block_ptrs(
            self.transfer_req_info.block_ids, dst_info.block_ids)

        if need_send == False:
            return TransReqMeta(
                session_id=self.session_id,
                future_for_session=self.future,
                src_kv_ptrs=[],
                dst_kv_ptrs=[],
                kv_sizes=[],
                expect_count=expect_count,
                remote_name=peer_instance_rank_info.instance_name +
                str(peer_instance_rank_info.instance_rank),
                peer_endpoint=peer_instance_rank_info.recv_endpoint,
                peer_session_id=dst_info.session_id)
        # self.remain_count = self.remain_count - 1
        src_kv_ptr = self.instance_rank_info.kvcache_ptrs[0]
        dst_kv_ptr = peer_instance_rank_info.kvcache_ptrs[0]
        src_kv_blocks_ptrs = [
            src_kv_ptr + self_kv_block_size * block_id
            for block_id in src_block_ids
        ]
        dst_kv_blocks_ptrs = [
            dst_kv_ptr + peer_kv_block_size * block_id
            for block_id in dst_block_ids
        ]
        print(
            f" src_kv_blocks_ptrs: {src_kv_blocks_ptrs}, src_kv_blocks_ptrs.shape: {len(src_kv_blocks_ptrs)} self_kv_block_size: {self_kv_block_size}, dst_kv_blocks_ptrs: {dst_kv_blocks_ptrs}, dst_kv_blocks_ptrs.shape: {len(dst_kv_blocks_ptrs)} peer_kv_block_size: {peer_kv_block_size}"
        )

        src_kv_blocks_transfer_ptrs, src_kv_blocks_size, dst_kv_blocks_transfer_ptrs, dst_kv_blocks_size = self._gen_kv_block_ptrs_for_all_layers(
            peer_instance_rank_info, src_kv_blocks_ptrs, self_kv_block_size,
            dst_kv_blocks_ptrs, peer_kv_block_size)

        self.remain_count = self.remain_count - 1
        return TransReqMeta(session_id=self.session_id,
                            future_for_session=self.future,
                            src_kv_ptrs=src_kv_blocks_transfer_ptrs,
                            dst_kv_ptrs=dst_kv_blocks_transfer_ptrs,
                            kv_sizes=[src_kv_blocks_size] *
                            len(src_kv_blocks_transfer_ptrs),
                            expect_count=expect_count,
                            remote_name=peer_instance_rank_info.instance_name +
                            str(peer_instance_rank_info.instance_rank),
                            peer_endpoint=peer_instance_rank_info.recv_endpoint,
                            peer_session_id=dst_info.session_id)
        # then compute the ptrs and size , big logic

        # called by receiver side to extract the transfer metadata and get peer target ranks
    def extract_recv_trans_meta(
            self, peer_instance_info: InstanceInfo,
            peer_dp_rank: int) -> tuple[TransReqRecvMeta, list[int]]:

        peer_domain_ranks = self._get_peer_peer_target_ranks(
            peer_instance_info, peer_dp_rank)
        expect_count = len(peer_domain_ranks.ranks)
        if not self.first_extracted:
            self.first_extracted = True
            self.remain_count = len(peer_domain_ranks.ranks)
        self.remain_count = 0
        return TransReqRecvMeta(session_id=self.session_id,
                                future_for_session=self.future,
                                expect_count=expect_count,
                                remote_name=None), peer_domain_ranks.ranks

    def create_gen_side_transfer_req_info(
            self, transfer_req_info: TransferReqInfo,
            disagg_params: DisaggregatedParams) -> TransferGenSideReqInfo:

        return TransferGenSideReqInfo(
            ctx_req_id=disagg_params.ctx_request_id,
            instance_name=transfer_req_info.instance_name,
            instance_rank=transfer_req_info.instance_rank,
            block_ids=transfer_req_info.block_ids,
            gen_req_id=transfer_req_info.req_id,
            session_id=self.session_id)

    def get_future_for_session(self):
        return self.future

    # // Block shape: [numHeads, numTokens, dimsPerHead]

# // CacheBlock shape: [numLayers, 2, mBlockSize]
# [numLayers,kv_factor,heads,tokens,dimsPerHead]

#  select kv blocks according cp , or none if duplicate or mla

    def _need_to_transfer(self, peer_domain_ranks: peerDomainRanks,
                          peer_instance_rank_info: InstanceRankInfo):
        if (peer_domain_ranks.duplicate_head_factor <= 1):
            return True
        peer_dp_rank = peer_instance_rank_info.dp_rank if peer_instance_rank_info.enable_attention_dp else 0
        self_tp_size_per_dp_group = self.instance_rank_info.tp_size // self.instance_rank_info.dp_size if self.instance_rank_info.enable_attention_dp else self.instance_rank_info.tp_size
        self_tprank_in_dp_group = self.instance_rank_info.tp_rank % self_tp_size_per_dp_group
        return (peer_dp_rank % peer_domain_ranks.duplicate_head_factor) == (
            self_tprank_in_dp_group % peer_domain_ranks.duplicate_head_factor)

    def filter_kv_block_ptrs(
            self, src_block_ids: list[int],
            dst_block_ids: list[int]) -> tuple[list[int], list[int]]:
        # pass
        return src_block_ids, dst_block_ids

        #  for cp


#  TODO: we need a dict to store function and params for use to avoid recompute
# return kv_block_ptrs

    def _gen_kv_block_ptrs_for_all_layers(
        self,
        peer_instance_rank_info: InstanceRankInfo,
        src_kv_block_ptrs: list[int],
        src_kv_block_size: int,
        dst_kv_block_ptrs: list[int],
        dst_kv_block_size: int,
    ) -> tuple[list[int], int, list[int],
               int]:  # (kv_block_ptr, kv_block_size), #kv format ,nhd or hnd
        # we may store some info such as offset and stride for use to avoid recompute
        assert len(src_kv_block_ptrs) == len(dst_kv_block_ptrs)
        self_layer_num = self.instance_rank_info.layer_num_per_pp[
            self.instance_rank_info.pp_rank]
        peer_layer_num = peer_instance_rank_info.layer_num_per_pp[
            peer_instance_rank_info.pp_rank]
        kv_factor = 1 if self.instance_rank_info.is_mla else 2
        assert src_kv_block_size == self_layer_num * kv_factor * self.instance_rank_info.kv_head_num_per_rank * self.instance_rank_info.tokens_per_block * self.instance_rank_info.dims_per_head * self.instance_rank_info.element_size
        assert dst_kv_block_size == peer_layer_num * kv_factor * peer_instance_rank_info.kv_head_num_per_rank * peer_instance_rank_info.tokens_per_block * peer_instance_rank_info.dims_per_head * peer_instance_rank_info.element_size
        # just ruturn the src_kv_block_ptrs and dst_kv_block_ptrs if tp and pp are the same
        self_tp_size_per_dp_group = self.instance_rank_info.tp_size // self.instance_rank_info.dp_size if self.instance_rank_info.enable_attention_dp else self.instance_rank_info.tp_size
        peer_tp_size_per_dp_group = peer_instance_rank_info.tp_size // peer_instance_rank_info.dp_size if peer_instance_rank_info.enable_attention_dp else peer_instance_rank_info.tp_size
        self_tprank_in_dp_group = self.instance_rank_info.tp_rank % self_tp_size_per_dp_group
        peer_tprank_in_dp_group = peer_instance_rank_info.tp_rank % peer_tp_size_per_dp_group
        is_duplicate_head = self.instance_rank_info.kv_head_num_per_rank * self_tp_size_per_dp_group != peer_instance_rank_info.kv_head_num_per_rank * peer_tp_size_per_dp_group
        write_all_head = is_duplicate_head or self.instance_rank_info.is_mla or self_tp_size_per_dp_group == peer_tp_size_per_dp_group
        if write_all_head and self.instance_rank_info.pp_size == peer_instance_rank_info.pp_size:
            return src_kv_block_ptrs, src_kv_block_size, dst_kv_block_ptrs, dst_kv_block_size

        # assume
        # selfStartLayer, selfEndLayer,  , peerStartLayer, peerEndLayer
        src_start_layer_id = sum(
            self.instance_rank_info.layer_num_per_pp[:self.instance_rank_info.
                                                     pp_rank])
        src_end_layer_id = src_start_layer_id + self.instance_rank_info.layer_num_per_pp[
            self.instance_rank_info.pp_rank]
        peer_start_layer_id = sum(
            peer_instance_rank_info.layer_num_per_pp[:peer_instance_rank_info.
                                                     pp_rank])
        peer_end_layer_id = peer_start_layer_id + peer_instance_rank_info.layer_num_per_pp[
            peer_instance_rank_info.pp_rank]
        start_layer_id = max(src_start_layer_id, peer_start_layer_id)
        end_layer_id = min(src_end_layer_id, peer_end_layer_id)
        transfer_layer_num = end_layer_id - start_layer_id
        src_start_layer_offset = start_layer_id - src_start_layer_id
        peer_start_layer_offset = start_layer_id - peer_start_layer_id
        if write_all_head:  # write all heads but pp size are different
            fragment_size = transfer_layer_num * kv_factor * self.instance_rank_info.kv_head_num_per_rank * self.instance_rank_info.tokens_per_block * self.instance_rank_info.dims_per_head * self.instance_rank_info.element_size
            src_block_offset = src_start_layer_offset * kv_factor * self.instance_rank_info.kv_head_num_per_rank * self.instance_rank_info.tokens_per_block * self.instance_rank_info.dims_per_head * self.instance_rank_info.element_size
            dst_block_offset = peer_start_layer_offset * kv_factor * peer_instance_rank_info.kv_head_num_per_rank * peer_instance_rank_info.tokens_per_block * peer_instance_rank_info.dims_per_head * peer_instance_rank_info.element_size
            src_kv_blocks_transfer_ptrs = [
                src_kv_block_ptr + src_block_offset
                for src_kv_block_ptr in src_kv_block_ptrs
            ]
            dst_kv_blocks_transfer_ptrs = [
                dst_kv_block_ptr + dst_block_offset
                for dst_kv_block_ptr in dst_kv_block_ptrs
            ]
            return src_kv_blocks_transfer_ptrs, fragment_size, dst_kv_blocks_transfer_ptrs, fragment_size

        # head num are different
        head_fragment_size = self.instance_rank_info.tokens_per_block * self.instance_rank_info.dims_per_head * self.instance_rank_info.element_size
        continue_heads_fragment_size = min(
            self.instance_rank_info.kv_head_num_per_rank,
            peer_instance_rank_info.kv_head_num_per_rank) * head_fragment_size
        self.instance_rank_info.kv_head_num_per_rank * head_fragment_size
        peer_instance_rank_info.kv_head_num_per_rank * head_fragment_size
        src_head_offset = 0
        dst_head_offset = 0
        if (self_tp_size_per_dp_group < peer_tp_size_per_dp_group):
            src_head_offset = peer_tprank_in_dp_group % (
                peer_tp_size_per_dp_group //
                self_tp_size_per_dp_group) * continue_heads_fragment_size
            dst_head_offset = 0
        if (self_tp_size_per_dp_group > peer_tp_size_per_dp_group):
            dst_head_offset = self_tprank_in_dp_group % (
                self_tp_size_per_dp_group //
                peer_tp_size_per_dp_group) * continue_heads_fragment_size
            src_head_offset = 0
        block_num = len(src_kv_block_ptrs)
        src_kv_blocks_transfer_ptrs = []
        dst_kv_blocks_transfer_ptrs = []
        src_layer_kv_ele_size = self.instance_rank_info.kv_head_num_per_rank * self.instance_rank_info.tokens_per_block * self.instance_rank_info.dims_per_head * self.instance_rank_info.element_size
        dst_layer_kv_ele_size = peer_instance_rank_info.kv_head_num_per_rank * peer_instance_rank_info.tokens_per_block * peer_instance_rank_info.dims_per_head * peer_instance_rank_info.element_size
        src_layer_ele_size = src_layer_kv_ele_size * kv_factor
        dst_layer_ele_size = dst_layer_kv_ele_size * kv_factor
        print(
            f" src_layer_ele_size: {src_layer_ele_size}, dst_layer_ele_size: {dst_layer_ele_size}"
        )
        for block_id in range(block_num):
            src_kv_block_ptr = src_kv_block_ptrs[block_id]
            dst_kv_block_ptr = dst_kv_block_ptrs[block_id]
            for layer_id in range(transfer_layer_num):
                src_layer_id = start_layer_id + layer_id
                dst_layer_id = start_layer_id + layer_id
                for kv in range(kv_factor):
                    src_kv_block_transfer_ptr = src_kv_block_ptr + src_layer_ele_size * src_layer_id + src_layer_kv_ele_size * kv + src_head_offset
                    dst_kv_block_transfer_ptr = dst_kv_block_ptr + dst_layer_ele_size * dst_layer_id + dst_layer_kv_ele_size * kv + dst_head_offset
                    src_kv_blocks_transfer_ptrs.append(
                        src_kv_block_transfer_ptr)
                    dst_kv_blocks_transfer_ptrs.append(
                        dst_kv_block_transfer_ptr)
        return src_kv_blocks_transfer_ptrs, continue_heads_fragment_size, dst_kv_blocks_transfer_ptrs, continue_heads_fragment_size

    def _get_peer_peer_target_ranks(self, peer_instance_info: InstanceInfo,
                                    peer_dp_rank: int) -> peerDomainRanks:
        return self.transfer_manager.get_peer_domain_ranks(
            peer_instance_info, peer_dp_rank)

    def is_active(self) -> bool:
        return self.remain_count != 0


class CacheTransferManager:

    def __init__(self, instance_rank_info: InstanceRankInfo,
                 kv_cache_manager: KVCacheManager):
        self.instance_rank_info = instance_rank_info
        self.active_transfer_sessions = []
        self.peer_instance_instance_info_cache = {}
        self.kv_cache_manager = kv_cache_manager
        self._peer_domain_ranks_cache = {}
        self._peer_transfer_req_info_cache = {}

    # This interface is called when connecting the generation instance at the upper-level context
    # sender or receiver. It is invoked only once for each generation rank.
    def register_peer(self, peer_instance_name: str, peer_rank: int,
                      peer_instance_info: InstanceRankInfo):
        self.peer_instance_instance_info_cache[
            peer_instance_name + str(peer_rank)] = peer_instance_info

    # Get the cache info of the peer.
    def get_peer_instance_rank_info(self, peer_instance_name: str,
                                    peer_rank: int) -> InstanceRankInfo:
        return self.peer_instance_instance_info_cache[peer_instance_name +
                                                      str(peer_rank)]

    # This interface is called when disconnecting the generation instance at the upper-level context.
    def unregister_peer(self, name: str, peer_rank: int):
        del self.peer_instance_instance_info_cache[name + str(peer_rank)]

    # Each transmission unit is a transfer session, which typically corresponds
    def create_transfer_session(
        self,
        src: TransferReqInfo,
    ) -> TransferSession:
        session = TransferSession(transferReqInfo=src,
                                  transfer_manager=self,
                                  instance_rank_info=self.instance_rank_info)
        self.active_transfer_sessions.append(session)
        return session

    # create transfer request info from request,which is used to create transfer
    def create_trans_req_info(self, request: LlmRequest) -> TransferReqInfo:

        block_ids = self.kv_cache_manager.get_batch_cache_indices(
            [request.py_request_id])[0]
        return TransferReqInfo(
            req_id=request.py_request_id,
            block_ids=block_ids,
            instance_name=self.instance_rank_info.instance_name,
            instance_rank=self.instance_rank_info.instance_rank,
        )

    # use self_instance info and peer instance info to get target_rank and some other info
    def get_peer_domain_ranks(self, peer_instance_info: InstanceInfo,
                              peer_dp_rank: int) -> peerDomainRanks:
        if str(peer_instance_info.instance_name) + str(
                peer_dp_rank) in self._peer_domain_ranks_cache:
            return self._peer_domain_ranks_cache[
                str(peer_instance_info.instance_name) + str(peer_dp_rank)]
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
        self._peer_domain_ranks_cache[str(peer_instance_info.instance_name) +
                                      str(peer_dp_rank)] = peer_domain_ranks
        return peer_domain_ranks

    def _clean_resource_for_request(self, request_id: int):
        if request_id in self._peer_transfer_req_info_cache:
            del self._peer_transfer_req_info_cache[request_id]

    def process(self):
        for session in self.active_transfer_sessions:
            if not session.is_active():
                # remove session
                # self.remove_transfer_session(session)
                self.active_transfer_sessions.remove(session)


class Transceiver:

    def __init__(self, kv_cache_manager: KVCacheManager, device_id: int,
                 instance_info: InstanceInfo,
                 instance_rank_info: InstanceRankInfo):

        self.kv_cache_manager = kv_cache_manager

        self.instance_rank_info = instance_rank_info

        self.cache_transfer_manager = CacheTransferManager(
            self.instance_rank_info, self.kv_cache_manager)
        self.instance_info = instance_info

        self._zmq_context = zmq.Context()

        self.transfer_agent = NixlTransferAgent(
            self.instance_rank_info.instance_name +
            str(self.instance_rank_info.instance_rank), True)

        kv_cache_memory = self.kv_cache_manager.get_unique_primary_pool()
        memory_desc = (kv_cache_memory.data_ptr(),
                       kv_cache_memory.numel() * kv_cache_memory.element_size(),
                       device_id, "kv_cache_memory")
        reg_memory_desc = RegMemoryDescs("VRAM", [memory_desc])
        # print(f"reg_memory_desc: {reg_memory_desc}")
        self.transfer_agent.register_memory(reg_memory_desc)
        self.data_sender = DataSender(self.transfer_agent, device_id)
        self.data_receiver = DataReceiver(self.transfer_agent, device_id)
        self.server_socket = self._zmq_context.socket(zmq.ROUTER)
        self.server_socket.bind(f"tcp://{get_local_ip()}:*")
        self.server_endpoint = self.server_socket.getsockopt(
            zmq.LAST_ENDPOINT).decode()
        print(f"Transceiver server_endpoint: {self.server_endpoint}")
        self.instance_rank_info.server_endpoint = self.server_endpoint
        self.instance_rank_info.recv_endpoint = self.data_receiver.get_endpoint(
        )
        self.instance_rank_info.transfer_engine_info = bytes(
            self.transfer_agent.get_local_agent_desc())

        # should get from all transceiver

        self.instance_info.ctx_server_endpoints = [self.server_endpoint]

        self.context_endpoint_to_instance_info_cache = {}
        self.socket_cache = {}
        self.request_id_to_send_transfer_session_cache = {}
        self.request_id_to_send_transfer_session_lock = threading.Lock()
        self._peer_transfer_recv_req_info_cache = {}
        self._peer_transfer_recv_req_info_lock = threading.Lock()
        self._sender_background_thread = threading.Thread(
            target=self._handle_sender_loop, daemon=True)
        self._sender_background_thread.start()

    #  upper layer should update the instance info ctx_server_endpoints when it is updated
    def update_instance_info_ctx_server_endpoints(
            self, ctx_server_endpoints: list[str]):
        self.instance_info.ctx_server_endpoints = ctx_server_endpoints

    def async_send(self, request: LlmRequest) -> TransferSession:
        # return future object ?
        transfer_req_info = self.cache_transfer_manager.create_trans_req_info(
            request)
        transfer_session = self.cache_transfer_manager.create_transfer_session(
            transfer_req_info)
        with self.request_id_to_send_transfer_session_lock:
            self.request_id_to_send_transfer_session_cache[
                transfer_req_info.req_id] = transfer_session
        self._handle_send_transfer_session(transfer_session)

        return transfer_session

    def async_receive(self, request: LlmRequest) -> TransferSession:
        return self._async_request_data(request)

    def cancel_request(self, request: LlmRequest):
        pass

    def _handle_send_transfer_session(self, transfer_session: TransferSession):
        transfer_recv_req_info_dict = {}
        with self._peer_transfer_recv_req_info_lock:
            if transfer_session.request_id in self._peer_transfer_recv_req_info_cache:
                transfer_recv_req_info_dict = self._peer_transfer_recv_req_info_cache[
                    transfer_session.request_id]
        for transfer_recv_req_info in transfer_recv_req_info_dict.values():
            trans_meta = transfer_session.extract_trans_meta(
                transfer_recv_req_info)
            self.data_sender.submit_transfer_task(trans_meta)
            if (not transfer_session.is_active()):
                with self.request_id_to_send_transfer_session_lock:
                    print(
                        f" handle_send_transfer_session, delete request_id_to_send_transfer_session_cache: {transfer_session.request_id}"
                    )
                    if transfer_session.request_id in self.request_id_to_send_transfer_session_cache:
                        del self.request_id_to_send_transfer_session_cache[
                            transfer_session.request_id]
                ## TODO: manager clean

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
        with self.request_id_to_send_transfer_session_lock:
            if ctx_req_id in self.request_id_to_send_transfer_session_cache:
                return self.request_id_to_send_transfer_session_cache[
                    ctx_req_id]
            else:
                return None

    def _handle_request_data(self, send_id: bytes, message: list[bytes]):

        transfer_gen_side_req_info: TransferGenSideReqInfo = pickle.loads(
            message[1])

        print(
            f" _handle_request_data, transfer_gen_side_req_info:{transfer_gen_side_req_info}"
        )
        ctx_req_id = transfer_gen_side_req_info.ctx_req_id

        send_transfer_session = self._get_send_transfer_session(ctx_req_id)
        if send_transfer_session is None:
            print(f" _handle_request_data, send_transfer_session is None")
            self._save_peer_transfer_req_info(transfer_gen_side_req_info)
        else:
            print(f" _handle_request_data, send_transfer_session is not None")
            trans_meta = send_transfer_session.extract_trans_meta(
                transfer_gen_side_req_info)
            self.data_sender.submit_transfer_task(trans_meta)
            #           # do we need big lock to protect is_active(), remain_count
            if (not send_transfer_session.is_active()):
                with self.request_id_to_send_transfer_session_lock:
                    print(
                        " handle_request_data, delete request_id_to_send_transfer_session_cache"
                    )

                    if ctx_req_id in self.request_id_to_send_transfer_session_cache:
                        del self.request_id_to_send_transfer_session_cache[
                            ctx_req_id]
                ## TODO: manager clean

    def _async_request_data(self, request: LlmRequest):
        disagg_params: DisaggregatedParams = request.py_disaggregated_params

        context_peer_infos: InstanceInfo = self._get_context_info(disagg_params)

        transfer_req_info = self.cache_transfer_manager.create_trans_req_info(
            request)
        transfer_session = self.cache_transfer_manager.create_transfer_session(
            transfer_req_info)
        transfer_meta, target_ranks = transfer_session.extract_recv_trans_meta(
            context_peer_infos, disagg_params.ctx_dp_rank)

        # map_info = self.gen_map_info(request, disagg_params)
        transfer_recv_req_info = transfer_session.create_gen_side_transfer_req_info(
            transfer_req_info, disagg_params)
        print(f" async_request_data target_ranks: {target_ranks}")
        for rank in target_ranks:
            self._send_data_request(
                context_peer_infos.ctx_server_endpoints[rank],
                transfer_recv_req_info)
        print(f" async_request_data submit_transfer_task: {transfer_meta}")
        self.data_receiver.submit_transfer_task(transfer_meta)

        return transfer_session

    def _need_register_peer_in_first_request(
            self, disagg_params: DisaggregatedParams) -> bool:
        return disagg_params.ctx_leader_endpoint not in self.context_endpoint_to_instance_info_cache

    # self.leader_endpint

    def _get_context_info(self,
                          disagg_params: DisaggregatedParams) -> InstanceInfo:
        if self._need_register_peer_in_first_request(disagg_params):
            socket = self._zmq_context.socket(zmq.DEALER)
            socket.connect(disagg_params.ctx_leader_endpoint)
            print(
                f"get_context_info connect to ctx_leader_endpoint: {disagg_params.ctx_leader_endpoint}"
            )
            message = [str(MessageType.REQUEST_INSTANCE_INFO).encode("ascii")]
            print(f"get_context_info send message: {message}")
            socket.send_multipart(message)
            message = socket.recv_multipart()
            # context_info = InstanceInfo.from_zmq(message)
            context_info = pickle.loads(message[0])
            socket.close()
            for endpoint in context_info.ctx_server_endpoints:
                socket = self._zmq_context.socket(zmq.DEALER)
                socket.connect(endpoint)
                print(f"get_context_info connect to: {endpoint}")
                send_message = []
                send_message.append(
                    str(MessageType.REGISTER_RANK_INFO).encode("ascii"))
                send_message.append(pickle.dumps(self.instance_rank_info))
                socket.send_multipart(send_message)
                socket.close()

            self.context_endpoint_to_instance_info_cache[
                disagg_params.ctx_leader_endpoint] = context_info
            return context_info

        else:  # get context info from cache
            return self.context_endpoint_to_instance_info_cache[
                disagg_params.ctx_leader_endpoint]

    def _send_data_request(self, endpoint: str,
                           transfer_recv_req_info: TransferGenSideReqInfo):
        socket = self._get_socket(endpoint)
        send_message = []
        send_message.append(str(MessageType.REQUEST_DATA).encode("ascii"))
        send_message.append(pickle.dumps(transfer_recv_req_info))
        socket.send_multipart(send_message)

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


class PyNativeKvCacheTransceiver(KvCacheTransceiver):

    def __init__(self, instance_rank_info: InstanceRankInfo,
                 kv_cache_manager: KVCacheManager):
        self.instance_rank_info = instance_rank_info
        self.kv_cache_manager = kv_cache_manager

        self.device_id = 0

        self.transceiver = Transceiver(kv_cache_manager, self.device_id)
        self.send_request_id_to_session = {}
        self.send_request_id_to_request = {}

        self.recv_request_id_to_session = {}
        self.recv_request_id_to_request = {}

    def respond_and_send_async(self, request: LlmRequest):
        session = self.transceiver.async_send(request)
        self.send_request_id_to_session[request.request_id] = session
        self.send_request_id_to_request[request.request_id] = request
        request.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS

    def request_and_receive_sync(self, request: LlmRequest):
        session = self.transceiver.request_and_receive_sync(request)
        # self.recv_request_id_to_session[request.request_id] = session
        session.future_for_session.result()

    def request_and_receive_async(self, request: LlmRequest):
        session = self.transceiver.request_and_receive_async(request)
        self.recv_request_id_to_session[request.request_id] = session
        self.recv_request_id_to_request[request.request_id] = request
        request.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS

    def check_context_transfer_status(self, at_least_request_num: int):
        if (at_least_request_num > 0):

            futures = [
                session.future_for_session
                for session in self.send_request_id_to_session.values()
                if session.is_active()
            ]

            concurrent.futures.wait(
                futures,
                timeout=None,
                return_when=concurrent.futures.FIRST_COMPLETED)
        for session in self.send_request_id_to_session.values():
            if (session.future_for_session.done()):

                request = self.send_request_id_to_request[session.request_id]
                request.state = LlmRequestState.DISAGG_CONTEXT_TRANS_COMPLETE
                del self.send_request_id_to_session[session.request_id]
                del self.send_request_id_to_request[session.request_id]

    def check_gen_transfer_status(self, at_least_request_num: int):
        if (at_least_request_num > 0):
            futures = [
                session.future_for_session
                for session in self.recv_request_id_to_session.values()
                if session.is_active()
            ]
            concurrent.futures.wait(
                futures,
                timeout=None,
                return_when=concurrent.futures.FIRST_COMPLETED)
        for session in self.recv_request_id_to_session.values():
            if (session.future_for_session.done()):
                request = self.recv_request_id_to_request[session.request_id]
                request.state = LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
                del self.recv_request_id_to_session[session.request_id]
                del self.recv_request_id_to_request[session.request_id]


def test_nixl_transfer_agent():
    device = torch.device("cuda:0")
    src_tensor = torch.randn(1024, device=device)
    src_tensor2 = torch.randn(512, device=device)
    dst_tensor = torch.randn(1024, device=device)
    dst_tensor2 = torch.randn(512, device=device)
    agent1 = NixlTransferAgent("agent1", True)
    agent2 = NixlTransferAgent("agent2", True)
    print("Registering memory")
    src_memory_desc = [(src_tensor.data_ptr(),
                        src_tensor.numel() * src_tensor.element_size(), 0),
                       (src_tensor2.data_ptr(),
                        src_tensor2.numel() * src_tensor2.element_size(), 0)]
    dst_memory_desc = [(dst_tensor.data_ptr(),
                        dst_tensor.numel() * dst_tensor.element_size(), 0),
                       (dst_tensor2.data_ptr(),
                        dst_tensor2.numel() * dst_tensor2.element_size(), 0)]
    reg_scr_memory_desc = [
        (src_tensor.data_ptr(), src_tensor.numel() * src_tensor.element_size(),
         0, "src1"),
        (src_tensor2.data_ptr(),
         src_tensor2.numel() * src_tensor2.element_size(), 0, "src2")
    ]
    reg_dst_memory_desc = [
        (dst_tensor.data_ptr(), dst_tensor.numel() * dst_tensor.element_size(),
         0, "dst1"),
        (dst_tensor2.data_ptr(),
         dst_tensor2.numel() * dst_tensor2.element_size(), 0, "dst2")
    ]

    src_memory_descs = MemoryDescs("VRAM", src_memory_desc)
    dst_memory_descs = MemoryDescs("VRAM", dst_memory_desc)
    reg_src_memory_descs = RegMemoryDescs("VRAM", reg_scr_memory_desc)
    reg_dst_memory_descs = RegMemoryDescs("VRAM", reg_dst_memory_desc)
    agent1.register_memory(reg_src_memory_descs)
    agent2.register_memory(reg_dst_memory_descs)
    print("Loading remote agent")
    agent1.load_remote_agent("agent2", agent2.get_local_agent_desc())

    request = TransferRequest(TransferOp.WRITE, src_memory_descs,
                              dst_memory_descs, "agent2", '')

    print("Submitting transfer requests")
    status = agent1.submit_transfer_requests(request)
    print("Waiting for transfer to complete")
    if not status.wait():
        print("Transfer failed")
        return
    print("Transfer completed")
    print(f"src_tensor: {src_tensor}")
    print(f"dst_tensor: {dst_tensor}")

    print(f"src_tensor2: {src_tensor2}")
    print(f"dst_tensor2: {dst_tensor2}")


def test_data_sender_and_receiver():
    device = torch.device("cuda:0")
    src_tensor = torch.randn(1024, device=device)
    src_tensor2 = torch.randn(512, device=device)
    dst_tensor = torch.randn(1024, device=device)
    dst_tensor2 = torch.randn(512, device=device)
    sender_name = "sender"
    receiver_name = "receiver"
    agent1 = NixlTransferAgent(sender_name, True)
    agent2 = NixlTransferAgent(receiver_name, True)

    # session_id_for_receiver = str(uuid.uuid4())
    # register memory for receiver
    reg_dst_memory_desc = [
        (dst_tensor.data_ptr(), dst_tensor.numel() * dst_tensor.element_size(),
         0, "dst1"),
        (dst_tensor2.data_ptr(),
         dst_tensor2.numel() * dst_tensor2.element_size(), 0, "dst2")
    ]
    reg_dst_memory_descs = RegMemoryDescs("VRAM", reg_dst_memory_desc)
    agent2.register_memory(reg_dst_memory_descs)
    # register memory for sender
    reg_src_memory_desc = [
        (src_tensor.data_ptr(), src_tensor.numel() * src_tensor.element_size(),
         0, "src1"),
        (src_tensor2.data_ptr(),
         src_tensor2.numel() * src_tensor2.element_size(), 0, "src2")
    ]
    reg_src_memory_descs = RegMemoryDescs("VRAM", reg_src_memory_desc)
    agent1.register_memory(reg_src_memory_descs)
    # load remote agent for sender
    print("Loading remote agent for sender")
    agent1.load_remote_agent(receiver_name, agent2.get_local_agent_desc())
    # load remote agent for receiver
    # agent2.load_remote_agent(sender_name, agent1.get_local_agent_desc())
    data_sender = DataSender(agent1, 0)
    data_receiver = DataReceiver(agent2, 0)
    recv_endpoint = data_receiver.get_endpoint()
    print(f"recv_endpoint: {recv_endpoint}")
    transfer_meta_data_for_receiver = TransReqRecvMeta(
        session_id=str(uuid.uuid4()),
        future_for_session=concurrent.futures.Future(),
        expect_count=1,
        remote_name=sender_name)
    print(
        f"transfer_meta_data_for_receiver session_id: {transfer_meta_data_for_receiver.session_id}"
    )
    transfer_meta_data_for_sender = TransReqMeta(
        session_id=str(uuid.uuid4()),
        future_for_session=concurrent.futures.Future(),
        src_kv_ptrs=[src_tensor.data_ptr()],
        dst_kv_ptrs=[dst_tensor.data_ptr()],
        kv_sizes=[
            src_tensor.numel() * src_tensor.element_size(),
        ],
        expect_count=1,
        peer_endpoint=recv_endpoint,
        peer_session_id=transfer_meta_data_for_receiver.session_id,
        remote_name=receiver_name,
    )

    transfer_meta_data_for_receiver_2 = TransReqRecvMeta(
        session_id=str(uuid.uuid4()),
        future_for_session=concurrent.futures.Future(),
        expect_count=1,
        remote_name=sender_name,
    )
    print(
        f"transfer_meta_data_for_receiver_2 session_id: {transfer_meta_data_for_receiver_2.session_id}"
    )
    transfer_meta_data_for_sender_2 = TransReqMeta(
        session_id=str(uuid.uuid4()),
        future_for_session=concurrent.futures.Future(),
        src_kv_ptrs=[src_tensor2.data_ptr()],
        dst_kv_ptrs=[dst_tensor2.data_ptr()],
        kv_sizes=[src_tensor2.numel() * src_tensor2.element_size()],
        expect_count=1,
        peer_endpoint=recv_endpoint,
        peer_session_id=transfer_meta_data_for_receiver_2.session_id,
        remote_name=receiver_name,
    )
    print("Submitting transfer tasks")
    data_sender.submit_transfer_task(transfer_meta_data_for_sender)
    data_sender.submit_transfer_task(transfer_meta_data_for_sender_2)
    print("Submitting transfer tasks for receiver")
    data_receiver.submit_transfer_task(transfer_meta_data_for_receiver)
    data_receiver.submit_transfer_task(transfer_meta_data_for_receiver_2)
    print("Waiting for transfer tasks to complete")
    result = transfer_meta_data_for_sender.future_for_session.result()
    print(f"sender result: {result}")
    result = transfer_meta_data_for_receiver.future_for_session.result()
    print(f"receiver result: {result}")
    result = transfer_meta_data_for_sender_2.future_for_session.result()
    print(f"sender_2 result: {result}")
    result = transfer_meta_data_for_receiver_2.future_for_session.result()
    print(f"receiver_2 result: {result}")
    # print(f"src_tensor: {src_tensor}")
    # print(f"dst_tensor: {dst_tensor}")
    # print(f"src_tensor2: {src_tensor2}")
    # print(f"dst_tensor2: {dst_tensor2}")
    assert src_tensor.equal(dst_tensor)
    assert src_tensor2.equal(dst_tensor2)


def test_cache_transceiver():
    mapping = Mapping(world_size=1, rank=0)

    num_layers = 2
    head_dim = 128
    num_kv_heads = 4
    tokens_per_block = 8
    max_seq_len = 256
    max_batch_size = 1
    dtype = DataType.FLOAT
    element_size = 4
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
        dtype=dtype)

    ctx_instance_info = InstanceInfo(instance_name="ctx_instance",
                                     tp_size=1,
                                     pp_size=1,
                                     dp_size=1,
                                     cp_size=1,
                                     kv_head_num_per_rank=num_kv_heads,
                                     tokens_per_block=tokens_per_block,
                                     dims_per_head=head_dim,
                                     element_size=element_size,
                                     enable_attention_dp=False,
                                     is_mla=False,
                                     layer_num_per_pp=[num_layers],
                                     ctx_server_endpoints=None)

    ctx_instance_rank_info = InstanceRankInfo(
        instance_name="ctx_instance",
        instance_rank=0,
        tp_size=1,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        dp_size=1,
        dp_rank=0,
        cp_size=1,
        cp_rank=0,
        kv_head_num_per_rank=num_kv_heads,
        tokens_per_block=tokens_per_block,
        dims_per_head=head_dim,
        element_size=element_size,
        enable_attention_dp=False,
        is_mla=False,
        layer_num_per_pp=[num_layers],
        kvcache_ptrs=[
            ctx_kv_cache_manager.get_unique_primary_pool().data_ptr()
        ],
        aux_ptrs=[],
        server_endpoint="",
        recv_endpoint="",
        transfer_engine_info=bytes())

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
        dtype=dtype)

    gen_instance_info = InstanceInfo(instance_name="gen_instance",
                                     tp_size=1,
                                     pp_size=1,
                                     dp_size=1,
                                     cp_size=1,
                                     kv_head_num_per_rank=num_kv_heads,
                                     tokens_per_block=tokens_per_block,
                                     dims_per_head=head_dim,
                                     element_size=element_size,
                                     enable_attention_dp=False,
                                     is_mla=False,
                                     layer_num_per_pp=[num_layers],
                                     ctx_server_endpoints=None)

    gen_instance_rank_info = InstanceRankInfo(
        instance_name="gen_instance",
        instance_rank=0,
        tp_size=1,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        dp_size=1,
        dp_rank=0,
        cp_size=1,
        cp_rank=0,
        kv_head_num_per_rank=num_kv_heads,
        tokens_per_block=tokens_per_block,
        dims_per_head=head_dim,
        element_size=element_size,
        enable_attention_dp=False,
        is_mla=False,
        layer_num_per_pp=[num_layers],
        kvcache_ptrs=[
            gen_kv_cache_manager.get_unique_primary_pool().data_ptr()
        ],
        aux_ptrs=[],
        server_endpoint="",
        recv_endpoint="",
        transfer_engine_info=bytes())

    print(
        f" ctx instance info kv cache ptrs :{ ctx_instance_rank_info.kvcache_ptrs}"
    )
    print(
        f" gen instance info kv cache ptrs :{ gen_instance_rank_info.kvcache_ptrs}"
    )

    device_id = 0

    ctx_transceiver = Transceiver(ctx_kv_cache_manager, device_id,
                                  ctx_instance_info, ctx_instance_rank_info)
    gen_transceiver = Transceiver(gen_kv_cache_manager, device_id,
                                  gen_instance_info, gen_instance_rank_info)

    sampling_params = SamplingParams()

    ctx_request = LlmRequest(
        request_id=0,
        max_new_tokens=1,
        input_tokens=list(range(256)),
        sampling_config=tensorrt_llm.bindings.SamplingConfig(
            sampling_params._get_sampling_config()),
        is_streaming=False,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY)

    ctx_kv_cache_manager.impl.add_sequence(ctx_request.py_request_id,
                                           ctx_request.prompt_len, 1,
                                           ctx_request)

    ctx_block_ids = ctx_kv_cache_manager.get_batch_cache_indices(
        [ctx_request.py_request_id])[0]

    ctx_block_data_pools = ctx_kv_cache_manager.get_unique_primary_pool()

    random_values = torch.rand(ctx_block_data_pools.shape,
                               dtype=torch.float32,
                               device=ctx_block_data_pools.device)
    ctx_block_data_pools.copy_(random_values)

    gen_request = LlmRequest(
        request_id=1,
        max_new_tokens=1,
        input_tokens=list(range(256)),
        sampling_config=tensorrt_llm.bindings.SamplingConfig(
            sampling_params._get_sampling_config()),
        is_streaming=False,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY)

    gen_request.py_disaggregated_params = DisaggregatedParams(
        ctx_request_id=ctx_request.py_request_id,
        ctx_dp_rank=0,
        ctx_leader_endpoint=ctx_instance_info.ctx_server_endpoints[0])
    gen_kv_cache_manager.impl.add_sequence(gen_request.py_request_id,
                                           gen_request.prompt_len, 1,
                                           gen_request)
    recv_session = gen_transceiver.async_receive(gen_request)

    time.sleep(0.1)
    send_session = ctx_transceiver.async_send(ctx_request)

    print(
        f"send_session.get_future_for_session().result(): {send_session.get_future_for_session().result()}"
    )
    print(
        f"recv_session.get_future_for_session().result(): {recv_session.get_future_for_session().result()}"
    )
    gen_block_ids = gen_kv_cache_manager.get_batch_cache_indices(
        [gen_request.py_request_id])[0]
    print(f"gen_block_ids: {gen_block_ids}")
    gen_block_datas = gen_kv_cache_manager.get_unique_primary_pool(
    )[gen_block_ids]

    ctx_block_datas = ctx_kv_cache_manager.get_unique_primary_pool(
    )[ctx_block_ids]

    print(
        f"ctx_block_datas: {ctx_block_datas}, ctx_block_datas.shape: {ctx_block_datas.shape}, ctx_block_datas.data_ptr: {ctx_block_datas.data_ptr()}"
    )
    print(
        f"gen_block_datas: {gen_block_datas}, gen_block_datas.shape: {gen_block_datas.shape}, gen_block_datas.data_ptr: {gen_block_datas.data_ptr()}"
    )
    assert ctx_block_datas.equal(gen_block_datas)


def test_cache_transceiver_with_tp(ctx_tp, gen_tp):

    ctx_tp_size = ctx_tp
    gen_tp_size = gen_tp

    # mapping = Mapping(world_size=1, rank=0)

    num_layers = 2
    head_dim = 128
    num_kv_heads = 4
    ctx_kv_head_num_per_rank = num_kv_heads // ctx_tp_size
    gen_kv_head_num_per_rank = num_kv_heads // gen_tp_size
    tokens_per_block = 8
    max_seq_len = 256
    max_batch_size = 1
    dtype = DataType.FLOAT
    element_size = 4

    ctx_kv_cache_managers = []
    ctx_instance_info = InstanceInfo(
        instance_name="ctx_instance",
        tp_size=ctx_tp_size,
        pp_size=1,
        dp_size=1,
        cp_size=1,
        kv_head_num_per_rank=ctx_kv_head_num_per_rank,
        tokens_per_block=tokens_per_block,
        dims_per_head=head_dim,
        element_size=element_size,
        enable_attention_dp=False,
        is_mla=False,
        layer_num_per_pp=[num_layers],
        ctx_server_endpoints=None)
    ctx_instance_rank_infos = []

    for ctx in range(ctx_tp_size):
        print(f"create ctx_kv_cache_manager for ctx {ctx}")
        mapping = Mapping(world_size=ctx_tp_size, rank=ctx, tp_size=ctx_tp_size)
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
            dtype=dtype)

        print(
            f" ctx cache pool ptr: {ctx_kv_cache_manager.get_unique_primary_pool().data_ptr()}, ctx cache pool shape: {ctx_kv_cache_manager.get_unique_primary_pool().shape}"
        )
        # random fill the ctx_kv_cache_manager
        ctx_block_data_pool = ctx_kv_cache_manager.get_unique_primary_pool()
        random_values = torch.rand(ctx_block_data_pool.shape,
                                   dtype=torch.float32,
                                   device=ctx_block_data_pool.device)
        ctx_block_data_pool.copy_(random_values)

        ctx_kv_cache_managers.append(ctx_kv_cache_manager)

        ctx_instance_rank_info = InstanceRankInfo(
            instance_name="ctx_instance",
            instance_rank=ctx,
            tp_size=ctx_tp_size,
            tp_rank=ctx,
            pp_size=1,
            pp_rank=0,
            dp_size=1,
            dp_rank=0,
            cp_size=1,
            cp_rank=0,
            kv_head_num_per_rank=ctx_kv_head_num_per_rank,
            tokens_per_block=tokens_per_block,
            dims_per_head=head_dim,
            element_size=element_size,
            enable_attention_dp=False,
            is_mla=False,
            layer_num_per_pp=[num_layers],
            kvcache_ptrs=[
                ctx_kv_cache_manager.get_unique_primary_pool().data_ptr()
            ],
            aux_ptrs=[],
            server_endpoint="",
            recv_endpoint="",
            transfer_engine_info=bytes())
        ctx_instance_rank_infos.append(ctx_instance_rank_info)

    gen_kv_cache_managers = []
    gen_instance_info = InstanceInfo(
        instance_name="gen_instance",
        tp_size=gen_tp_size,
        pp_size=1,
        dp_size=1,
        cp_size=1,
        kv_head_num_per_rank=gen_kv_head_num_per_rank,
        tokens_per_block=tokens_per_block,
        dims_per_head=head_dim,
        element_size=element_size,
        enable_attention_dp=False,
        is_mla=False,
        layer_num_per_pp=[num_layers],
        ctx_server_endpoints=None)
    gen_instance_rank_infos = []
    for gen in range(gen_tp_size):
        mapping = Mapping(world_size=gen_tp_size, rank=gen, tp_size=gen_tp_size)
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
            dtype=dtype)

        print(
            f" gen cache pool ptr: {gen_kv_cache_manager.get_unique_primary_pool().data_ptr()}, gen cache pool shape: {gen_kv_cache_manager.get_unique_primary_pool().shape}"
        )
        gen_kv_cache_managers.append(gen_kv_cache_manager)
        gen_instance_rank_info = InstanceRankInfo(
            instance_name="gen_instance",
            instance_rank=gen,
            tp_size=gen_tp_size,
            tp_rank=gen,
            pp_size=1,
            pp_rank=0,
            dp_size=1,
            dp_rank=0,
            cp_size=1,
            cp_rank=0,
            kv_head_num_per_rank=gen_kv_head_num_per_rank,
            tokens_per_block=tokens_per_block,
            dims_per_head=head_dim,
            element_size=element_size,
            enable_attention_dp=False,
            is_mla=False,
            layer_num_per_pp=[num_layers],
            kvcache_ptrs=[
                gen_kv_cache_manager.get_unique_primary_pool().data_ptr()
            ],
            aux_ptrs=[],
            server_endpoint="",
            recv_endpoint="",
            transfer_engine_info=bytes())
        gen_instance_rank_infos.append(gen_instance_rank_info)

    device_id = 0
    ctx_transceivers = []
    for ctx in range(ctx_tp_size):
        ctx_transceiver = Transceiver(ctx_kv_cache_managers[ctx], device_id,
                                      ctx_instance_info,
                                      ctx_instance_rank_infos[ctx])
        ctx_transceivers.append(ctx_transceiver)

    ctx_server_endpoints = []
    for ctx in range(ctx_tp_size):
        ctx_server_endpoints.append(
            ctx_transceivers[ctx].instance_rank_info.server_endpoint)

    for ctx in range(ctx_tp_size):
        ctx_transceivers[ctx].update_instance_info_ctx_server_endpoints(
            ctx_server_endpoints)

    gen_transceivers = []
    for gen in range(gen_tp_size):
        gen_transceiver = Transceiver(gen_kv_cache_managers[gen], device_id,
                                      gen_instance_info,
                                      gen_instance_rank_infos[gen])
        gen_transceivers.append(gen_transceiver)

    sampling_params = SamplingParams()

    request_len = 16
    ctx_request = LlmRequest(
        request_id=0,
        max_new_tokens=1,
        input_tokens=list(range(request_len)),
        sampling_config=tensorrt_llm.bindings.SamplingConfig(
            sampling_params._get_sampling_config()),
        is_streaming=False,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY)

    for ctx in range(ctx_tp_size):
        ctx_kv_cache_managers[ctx].impl.add_sequence(ctx_request.py_request_id,
                                                     ctx_request.prompt_len, 1,
                                                     ctx_request)

    gen_request = LlmRequest(
        request_id=1,
        max_new_tokens=1,
        input_tokens=list(range(request_len)),
        sampling_config=tensorrt_llm.bindings.SamplingConfig(
            sampling_params._get_sampling_config()),
        is_streaming=False,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY)

    gen_request.py_disaggregated_params = DisaggregatedParams(
        ctx_request_id=ctx_request.py_request_id,
        ctx_dp_rank=0,
        ctx_leader_endpoint=ctx_instance_rank_infos[0].server_endpoint)

    for gen in range(gen_tp_size):
        gen_kv_cache_managers[gen].impl.add_sequence(gen_request.py_request_id,
                                                     gen_request.prompt_len, 1,
                                                     gen_request)

    recv_sessions = []
    for gen in range(gen_tp_size):
        recv_sessions.append(gen_transceivers[gen].async_receive(gen_request))

    send_sessions = []
    print(f" send_sessions before async_send: {send_sessions}")
    for ctx in range(ctx_tp_size):
        send_sessions.append(ctx_transceivers[ctx].async_send(ctx_request))

    for ctx in range(ctx_tp_size):
        print(
            f"send_sessions[{ctx}].get_future_for_session().result(): {send_sessions[ctx].get_future_for_session().result()}"
        )

    for gen in range(gen_tp_size):
        print(
            f"recv_sessions[{gen}].get_future_for_session().result(): {recv_sessions[gen].get_future_for_session().result()}"
        )

    ctx_data_tensors = []
    for ctx in range(ctx_tp_size):
        ctx_data_pool_tensor = ctx_kv_cache_managers[
            ctx].get_unique_primary_pool()
        ctx_block_ids = ctx_kv_cache_managers[ctx].get_batch_cache_indices(
            [ctx_request.py_request_id])[0]
        ctx_block_data_tensors = ctx_data_pool_tensor[ctx_block_ids]
        ctx_data_tensors.append(ctx_block_data_tensors)

    gen_data_tensors = []
    for gen in range(gen_tp_size):
        gen_data_pool_tensor = gen_kv_cache_managers[
            gen].get_unique_primary_pool()
        gen_block_ids = gen_kv_cache_managers[gen].get_batch_cache_indices(
            [gen_request.py_request_id])[0]
        gen_block_data_tensors = gen_data_pool_tensor[gen_block_ids]
        gen_data_tensors.append(gen_block_data_tensors)

    print(
        f"ctx_data_tensors: {ctx_data_tensors}, ctx_data_tensors.shape: {ctx_data_tensors[0].shape}"
    )
    print(
        f"gen_data_tensors: {gen_data_tensors} , gen_data_tensors.shape: {gen_data_tensors[0].shape}"
    )


if __name__ == "__main__":
    # test_data_sender_and_receiver()
    test_cache_transceiver_with_tp(2, 1)
