from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Union


class TransferOp:
    READ = "READ"
    WRITE = "WRITE"


@dataclass
class MemoryDesc:
    ptr: int
    size: int
    device_id: int


@dataclass
class MemoryDescs:
    type: str
    descs: List[Union[Tuple[int, int, int], MemoryDesc]]


@dataclass
class RegMemoryDescs:
    type: str
    descs: List[Tuple[int, int, int, str]]


@dataclass
class TransferRequest:
    op: TransferOp
    src_descs: MemoryDescs
    dst_descs: MemoryDescs
    remote_name: str
    sync_message: str


class TransferStatus(ABC):
    @abstractmethod
    def is_completed(self) -> bool: ...

    @abstractmethod
    def wait(self, timeout: float | None = None) -> None: ...


class BaseTransferAgent(ABC):
    @abstractmethod
    def register_memory(self, descs: MemoryDescs) -> None: ...

    @abstractmethod
    def deregister_memory(self, descs: MemoryDescs) -> None: ...

    @abstractmethod
    def load_remote_agent(self, name: str, agent_desc: str) -> None: ...

    @abstractmethod
    def get_local_agent_desc(self) -> str: ...

    @abstractmethod
    def invalidate_remote_agent(self, name: str) -> None: ...

    @abstractmethod
    def submit_transfer_requests(self, request: TransferRequest) -> TransferStatus: ...

    @abstractmethod
    def notify_sync_message(self, name: str, sync_message: str) -> None: ...

    @abstractmethod
    def check_remote_descs(self, name: str, memory_descs: List[int]) -> bool: ...
