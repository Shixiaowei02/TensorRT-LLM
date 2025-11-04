from tensorrt_llm._torch.disaggregation.base import Receiver, ReceiveSession, Sender, SendSession
from tensorrt_llm._torch.disaggregation.native import (
    InstanceInfo,
    InstanceRankInfo,
    ResourceRegister,
)
from tensorrt_llm._torch.disaggregation.nixl import NixlTransferAgent
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager


class TransferWorker:
    def __init__(
        self,
        kv_cache_manager: KVCacheManager,
        instance_info: InstanceInfo,
        instance_rank_info: InstanceRankInfo,
        device_id: int,
    ):
        self.kv_cache_manager = kv_cache_manager
        self.resource_register = ResourceRegister(instance_info, instance_rank_info)
        self.device_id = device_id
        self.transfer_agent = NixlTransferAgent(
            instance_rank_info.instance_name + str(instance_rank_info.instance_rank), True
        )
        self.sender = Sender(
            kv_cache_manager, self.resource_register, device_id, self.transfer_agent
        )
        self.receiver = Receiver(
            kv_cache_manager, self.resource_register, device_id, self.transfer_agent
        )

    def create_send_session(self, request: LlmRequest) -> SendSession:
        return self.sender.create_send_session(request)

    def async_send(self, request: LlmRequest) -> SendSession:
        return self.sender.async_send(request)

    def async_receive(self, request: LlmRequest) -> ReceiveSession:
        return self.receiver.async_receive(request)

    def cancel_request(self, request: LlmRequest):
        pass
