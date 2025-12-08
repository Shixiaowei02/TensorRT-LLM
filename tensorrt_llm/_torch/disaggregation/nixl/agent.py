"""NIXL Transfer Agent implementations.

This module provides two implementations:
1. BindingsNixlTransferAgent - Uses the standalone nixl_bindings C++ module with GIL release support
2. NixlTransferAgent - Uses the Python nixl library directly (fallback)

The standalone nixl_bindings module is separate from the main trtllm bindings,
so trtllm can still function normally even without NIXL dependencies.
"""

from tensorrt_llm._utils import nvtx_range

# Import base classes for type compatibility
from ..base.agent import BaseTransferAgent, RegMemoryDescs, TransferRequest, TransferStatus

# Try to import the standalone tensorrt_llm_transfer_agent_binding module
# Located at tensorrt_llm/ (same level as bindings.so)
_AGENT_BINDING_AVAILABLE = False
try:
    import tensorrt_llm.tensorrt_llm_transfer_agent_binding as _agent_binding  # noqa: E402

    _AGENT_BINDING_AVAILABLE = True

    # Import from standalone module
    BaseAgentConfig = _agent_binding.BaseAgentConfig
    CppNixlTransferAgent = _agent_binding.NixlTransferAgent
    CppNixlTransferStatus = _agent_binding.NixlTransferStatus
    CppTransferStatus = _agent_binding.TransferStatus
    MemoryType = _agent_binding.MemoryType
    MemoryDesc = _agent_binding.MemoryDesc
    MemoryDescs = _agent_binding.MemoryDescs
    TransferOp = _agent_binding.TransferOp
    CppTransferRequest = _agent_binding.TransferRequest
    AgentDesc = _agent_binding.AgentDesc
    poll_until_complete = _agent_binding.poll_until_complete

except ImportError:
    # tensorrt_llm_transfer_agent_binding not available, will fall back to Python nixl or raise error
    pass


def is_transfer_agent_binding_available() -> bool:
    """Check if the standalone tensorrt_llm_transfer_agent_binding module is available."""
    return _AGENT_BINDING_AVAILABLE


class BindingsNixlTransferStatus(TransferStatus):
    """TransferStatus wrapper using C++ bindings with GIL release."""

    def __init__(self, cpp_status):
        self._cpp_status = cpp_status

    def is_completed(self) -> bool:
        """Check if transfer is completed (releases GIL)."""
        return self._cpp_status.is_completed()

    @nvtx_range("BindingsNixlTransferStatus.wait")
    def wait(self) -> bool:
        """Wait for transfer to complete (releases GIL)."""
        return self._cpp_status.wait_with_status() != 0

    def wait_with_status(self) -> int:
        """Wait and return status code (releases GIL)."""
        return self._cpp_status.wait_with_status()


class BindingsNixlTransferAgent(BaseTransferAgent):
    """NixlTransferAgent using C++ bindings with GIL release support.

    This implementation uses the standalone nixl_bindings C++ module which releases
    the GIL during blocking operations like wait().

    The nixl_bindings module is independent from the main trtllm bindings,
    so trtllm can function normally even without NIXL.
    """

    def __init__(self, name: str, use_prog_thread: bool = True, num_workers: int = 1):
        if not _AGENT_BINDING_AVAILABLE:
            raise ImportError(
                "tensorrt_llm_transfer_agent_binding module is not available. "
                "Please build with NIXL_ROOT set to enable NIXL support."
            )
        config = BaseAgentConfig(
            name,
            use_prog_thread,
            multi_thread=False,
            use_listen_thread=False,
            num_workers=num_workers,
        )
        self._cpp_agent = CppNixlTransferAgent(config)
        self.name = name

    def register_memory(self, descs: RegMemoryDescs):
        """Register memory regions."""
        cpp_descs = self._convert_to_cpp_descs(descs)
        self._cpp_agent.register_memory(cpp_descs)

    def deregister_memory(self, descs: RegMemoryDescs):
        """Deregister memory regions."""
        cpp_descs = self._convert_to_cpp_descs(descs)
        self._cpp_agent.deregister_memory(cpp_descs)

    def load_remote_agent(self, name: str, agent_desc: bytes):
        """Load a remote agent by its descriptor (bytes)."""
        # AgentDesc expects std::string which can hold binary data
        desc_str = agent_desc if isinstance(agent_desc, bytes) else agent_desc.encode()
        cpp_desc = AgentDesc(desc_str)
        self._cpp_agent.load_remote_agent(name, cpp_desc)

    def load_remote_agent_by_connection(self, name: str, connection_info: str):
        """Load a remote agent by connection info."""
        self._cpp_agent.load_remote_agent_by_connection(name, connection_info)

    def get_local_agent_desc(self) -> bytes:
        """Get the local agent descriptor as bytes."""
        agent_desc = self._cpp_agent.get_local_agent_desc()
        return agent_desc.backend_agent_desc  # Returns bytes

    def get_local_connection_info(self) -> str:
        """Get the local connection info."""
        return self._cpp_agent.get_local_connection_info()

    def invalidate_remote_agent(self, name: str):
        """Invalidate a remote agent."""
        self._cpp_agent.invalidate_remote_agent(name)

    def check_remote_descs(self, name: str, memory_descs) -> bool:
        """Check if remote descriptors are available."""
        cpp_descs = self._convert_to_cpp_descs(memory_descs)
        return self._cpp_agent.check_remote_descs(name, cpp_descs)

    def notify_sync_message(self, name: str, sync_message: str):
        """Send a sync message to a remote agent."""
        self._cpp_agent.notify_sync_message(name, sync_message)

    def get_notified_sync_messages(self):
        """Get notified sync messages."""
        return self._cpp_agent.get_notified_sync_messages()

    @nvtx_range("BindingsNixlTransferAgent.submit_transfer_requests")
    def submit_transfer_requests(self, request: TransferRequest) -> TransferStatus:
        """Submit transfer requests and return status."""
        cpp_request = self._convert_to_cpp_request(request)
        cpp_status = self._cpp_agent.submit_transfer_requests(cpp_request)
        return BindingsNixlTransferStatus(cpp_status)

    def _convert_to_cpp_descs(self, descs: RegMemoryDescs) -> "MemoryDescs":
        """Convert Python RegMemoryDescs to C++ MemoryDescs.

        RegMemoryDescs.descs is List[Tuple[int, int, int, str]] = (ptr, size, device_id, name)
        """
        mem_type = self._convert_memory_type(descs.type)
        cpp_desc_list = []
        for d in descs.descs:
            if isinstance(d, tuple):
                # (ptr, size, device_id, name) format
                ptr, size, device_id = d[0], d[1], d[2]
                cpp_desc_list.append(MemoryDesc(ptr, size, device_id))
            else:
                # Object with attributes
                cpp_desc_list.append(MemoryDesc(d.ptr, d.size, d.device_id))
        return MemoryDescs(mem_type, cpp_desc_list)

    def _convert_memory_type(self, py_type: str) -> "MemoryType":
        """Convert Python memory type string to C++ MemoryType."""
        type_map = {
            "DRAM": MemoryType.DRAM,
            "VRAM": MemoryType.VRAM,
            "GPU": MemoryType.VRAM,
            "BLK": MemoryType.BLK,
            "OBJ": MemoryType.OBJ,
            "FILE": MemoryType.FILE,
        }
        return type_map.get(py_type.upper(), MemoryType.VRAM)

    def _convert_transfer_op(self, py_op: str) -> "TransferOp":
        """Convert Python transfer op string to C++ TransferOp."""
        if py_op.upper() == "READ":
            return TransferOp.READ
        return TransferOp.WRITE

    def _convert_to_cpp_request(self, request: TransferRequest) -> "CppTransferRequest":
        """Convert Python TransferRequest to C++ TransferRequest."""
        op = self._convert_transfer_op(request.op)
        src_descs = self._convert_to_cpp_descs(request.src_descs)
        dst_descs = self._convert_to_cpp_descs(request.dst_descs)
        return CppTransferRequest(
            op, src_descs, dst_descs, request.remote_name, request.sync_message
        )


# For backward compatibility, also keep the Python nixl-based implementation
NixlTransferAgent = None
NixlTransferStatus = None

try:
    from nixl import nixl_agent, nixl_agent_config, nixl_xfer_handle  # noqa: E402

    # For Python nixl, we need poll_until_complete
    if _AGENT_BINDING_AVAILABLE:
        _poll_until_complete = poll_until_complete
    else:
        # Fallback pure Python implementation
        import time

        def _poll_until_complete(
            check_fn, initial_sleep_sec=0.0001, max_sleep_sec=0.01, timeout_sec=0.0
        ):
            """Poll a check function until it returns 'DONE'."""
            start_time = time.monotonic()
            sleep_sec = initial_sleep_sec

            while True:
                status = check_fn()
                if status == "DONE":
                    elapsed = time.monotonic() - start_time
                    return (True, elapsed)

                elapsed = time.monotonic() - start_time
                if timeout_sec > 0.0 and elapsed >= timeout_sec:
                    return (False, elapsed)

                time.sleep(sleep_sec)
                sleep_sec = min(sleep_sec * 2.0, max_sleep_sec)

    class NixlTransferStatus(TransferStatus):
        """TransferStatus using Python nixl library."""

        def __init__(self, agent: nixl_agent, handle: nixl_xfer_handle):
            self.agent = agent
            self.handle = handle

        def is_completed(self):
            status = self.agent.check_xfer_state(self.handle)
            return status == "DONE"

        def wait(self, timeout_sec: float = 0.0) -> bool:
            """Wait for transfer to complete, releasing GIL during sleep."""
            success, _ = _poll_until_complete(
                lambda: self.agent.check_xfer_state(self.handle),
                initial_sleep_sec=0.0001,
                max_sleep_sec=0.01,
                timeout_sec=timeout_sec,
            )
            return success

    class NixlTransferAgent(BaseTransferAgent):
        """NixlTransferAgent using Python nixl library."""

        def __init__(self, name: str, use_prog_thread: bool, num_workers: int = 1):
            self.name = name
            agent_config = nixl_agent_config(
                enable_prog_thread=use_prog_thread, backends=["UCX"], num_threads=num_workers
            )
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

        @nvtx_range("NixlTransferAgent.submit_transfer_requests")
        def submit_transfer_requests(self, request: TransferRequest) -> TransferStatus:
            src_xfer_descs = self.agent.get_xfer_descs(
                request.src_descs.descs, request.src_descs.type
            )
            dst_xfer_descs = self.agent.get_xfer_descs(
                request.dst_descs.descs, request.dst_descs.type
            )
            handle = self.agent.initialize_xfer(
                request.op,
                src_xfer_descs,
                dst_xfer_descs,
                request.remote_name,
                request.sync_message,
            )
            status = self.agent.transfer(handle)
            assert status != "ERR"
            return NixlTransferStatus(self.agent, handle)

except ImportError:
    # nixl library not available
    pass
