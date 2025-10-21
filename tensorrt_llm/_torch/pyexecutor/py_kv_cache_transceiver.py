import concurrent
import sys
from abc import ABC
from typing import Optional

import zmq

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

    def load_remote_agent(self, name: str, agent_desc: str):
        self.agent.add_remote_agent(bytes(agent_desc))

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


class TransReqMeta:
    session_id: str
    future_for_session: concurrent.futures.Future
    src_kv_ptrs: list[int]
    dst_kv_ptrs: list[int]
    kv_sizes: list[int]
    src_aux_ptrs: list[int]
    dst_aux_ptrs: list[int]
    aux_sizes: list[int]
    expect_count: int
    peer_endpoint: Optional[str]  # used for send state
    peer_session_id: Optional[int]


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
    def submit_transfer_task(self, transfer_meta_data: TransReqMeta):
        raise NotImplementedError

    @abstractmethod
    def get_endpoint(self):
        raise NotImplementedError


class DataSender(BaseDataSender):

    def __init__(self, agent: NixlTransferAgent, device_id: int):
        self.agent = agent
        self.device_id = device_id

        self.session_id_to_count = {}
        self.zmq_context = zmq.Context()

    def submit_transfer_task(self, transfer_meta_data: TransReqMeta):
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
                        self._handle_transfer_task(meta)
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
        assert len(transfer_meta_data.src_kv_ptrs) == len(
            transfer_meta_data.dst_kv_ptrs)
        assert len(transfer_meta_data.kv_sizes) == len(
            transfer_meta_data.src_kv_ptrs)
        if transfer_meta_data.peer_session_id not in self.session_id_to_count:
            self.session_id_to_count[transfer_meta_data.peer_session_id] = 0
        src_kv_list = [(src_ptr, size, self.device_id) for src_ptr, size in zip(
            transfer_meta_data.src_kv_ptrs, transfer_meta_data.kv_sizes)]
        dst_kv_list = [(dst_ptr, size, self.device_id) for dst_ptr, size in zip(
            transfer_meta_data.dst_kv_ptrs, transfer_meta_data.kv_sizes)]
        src_memory_descs = MemoryDescs("VRAM", src_kv_list)
        dst_memory_descs = MemoryDescs("VRAM", dst_kv_list)
        request = TransferRequest(TransferOp.WRITE, src_memory_descs,
                                  dst_memory_descs, "agent2", '')
        status = self.agent.submit_transfer_requests(request)
        sync_status = "SUCCESS"
        if not status.wait():
            sync_status = "FAILED"
            transfer_meta_data.future_for_session.set_exception(
                RuntimeError("Transfer failed"))
        socket = self.zmq_context.socket(zmq.DEALER)
        socket.connect(transfer_meta_data.peer_endpoint)
        socket.send_multipart([
            str(transfer_meta_data.peer_session_id).encode("ascii"),
            sync_status.encode("ascii")
        ])
        # TODO: socket_cache
        self.session_id_to_count[transfer_meta_data.peer_session_id] += 1
        if (self.session_id_to_count[transfer_meta_data.peer_session_id] ==
                transfer_meta_data.expect_count):
            transfer_meta_data.future_for_session.set_result(sync_status)
            self.session_id_to_count.pop(transfer_meta_data.peer_session_id)
        if (self.session_id_to_count[transfer_meta_data.peer_session_id]
                > transfer_meta_data.expect_count):
            raise RuntimeError(
                f"Session {transfer_meta_data.peer_session_id} has more than {transfer_meta_data.expect_count} transfers"
            )


class DataReceiver(BaseDataReceiver):

    def __init__(self, agent: NixlTransferAgent, device_id: int):
        self.agent = agent
        self.device_id = device_id

    def submit_transfer_task(self, transfer_meta_data: TransReqMeta):
        raise NotImplementedError

    def get_endpoint(self):
        raise NotImplementedError

    def _loop_for_receive_state(self):
        pass


def main():

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


if __name__ == "__main__":
    main()
