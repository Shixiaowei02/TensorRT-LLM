from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from tensorrt_llm import DisaggregatedParams


@dataclass
class KVSlice:
    """Supports transmitting only part of the request cache, e.g, chunks or layers."""

    start_token_idx: Optional[int] = None
    end_token_idx: Optional[int] = None
    start_layer_idx: Optional[int] = None
    end_layer_idx: Optional[int] = None
    block_ids: List[int] = field(default_factory=list)

    is_last_slice: bool = False

    # def __post_init__(self) -> None:
    #     if self.start_token_idx < 0 or self.end_token_idx < 0:
    #         raise ValueError("token indices must be non-negative")
    #     if self.start_layer_idx < 0 or self.end_layer_idx < 0:
    #         raise ValueError("layer indices must be non-negative")
    #     if self.start_token_idx > self.end_token_idx:
    #         raise ValueError("start_token_idx cannot be greater than end_token_idx")
    #     if self.start_layer_idx > self.end_layer_idx:
    #         raise ValueError("start_layer_idx cannot be greater than end_layer_idx")
    #     if any(b < 0 for b in self.block_ids):
    #         raise ValueError("block_ids must contain non-negative integers")


class State(Enum):
    """States of a transfer session."""

    INIT = "Init"  # Session contains only the required members for construction.
    READY = "Ready"  # Resources are ready for processing.
    TRANSFERRING = "Transferring"  # Data is being transffered.
    FINISHED = "Finished"  # Processing is finished.
    ERR = "Err"  # An error has occurred.


TaskIdType = int


@dataclass
class SessionState:
    state: State
    finished_tasks: List[TaskIdType]


@dataclass
class SessionArgsBase:
    request_id: int
    disagg_params: DisaggregatedParams


class SenderBase(ABC): ...


class ReceiverBase(ABC): ...


class TxSessionBase(ABC):
    def __init__(self, sender: SenderBase, args: SessionArgsBase):
        self.session_args = args

    @abstractmethod
    def get_state(self) -> SessionState: ...

    @abstractmethod
    def poll_task(self, id: TaskIdType) -> State: ...

    @abstractmethod
    def send(self, slice: KVSlice) -> TaskIdType: ...

    @abstractmethod
    def get_exception(self) -> Optional[Exception]: ...


class RxSessionBase(ABC):
    def __init__(self, receiver: ReceiverBase, args: SessionArgsBase):
        self.session_args = args

    @abstractmethod
    def get_state(self) -> SessionState: ...

    @abstractmethod
    def poll_task(self, id: TaskIdType) -> State: ...

    @abstractmethod
    def receive(self, slice: KVSlice) -> TaskIdType: ...

    @abstractmethod
    def get_exception(self) -> Optional[Exception]: ...
