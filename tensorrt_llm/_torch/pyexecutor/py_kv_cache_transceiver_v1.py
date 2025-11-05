from tensorrt_llm._torch.disaggregation.base import ExecutorDesc, Receiver, Sender, WorkerDesc
from tensorrt_llm._torch.disaggregation.native import ResourceRegister
from tensorrt_llm._torch.disaggregation.nixl import NixlTransferAgent
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager


class TransferWorker:
    def __init__(
        self,
        kv_cache_manager: KVCacheManager,
        worker_desc: WorkerDesc,
        exec_desc: ExecutorDesc,
        device_id: int,
    ):
        self.kv_cache_manager = kv_cache_manager
        self.resource_register = ResourceRegister(exec_desc, worker_desc)
        self.transfer_agent = NixlTransferAgent(
            worker_desc.instance_name + str(worker_desc.instance_rank), True
        )
        self.sender = Sender(
            kv_cache_manager, self.resource_register, device_id, self.transfer_agent
        )
        self.receiver = Receiver(
            kv_cache_manager, self.resource_register, device_id, self.transfer_agent
        )
