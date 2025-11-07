import sys

from ..base.agent import BaseTransferAgent, RegMemoryDescs, TransferRequest, TransferStatus

nixl_path = "/opt/nvidia/nvda_nixl/lib/python3/dist-packages"
if nixl_path not in sys.path:
    sys.path.insert(0, nixl_path)

from nixl._api import nixl_agent, nixl_agent_config, nixl_xfer_handle  # noqa: E402


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
        agent_config = nixl_agent_config(enable_prog_thread=use_prog_thread, backends=["UCX"])
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

    def check_remote_descs(self, name: str, memory_descs: list[int]) -> bool:
        raise NotImplementedError

    def notify_sync_message(self, name: str, sync_message: str):
        raise NotImplementedError

    def submit_transfer_requests(self, request: TransferRequest) -> TransferStatus:
        src_xfer_descs = self.agent.get_xfer_descs(request.src_descs.descs, request.src_descs.type)
        dst_xfer_descs = self.agent.get_xfer_descs(request.dst_descs.descs, request.dst_descs.type)
        handle = self.agent.initialize_xfer(
            request.op, src_xfer_descs, dst_xfer_descs, request.remote_name, request.sync_message
        )
        status = self.agent.transfer(handle)
        assert status != "ERR"
        return NixlTransferStatus(self.agent, handle)
