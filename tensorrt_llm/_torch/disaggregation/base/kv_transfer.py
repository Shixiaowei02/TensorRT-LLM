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
    start_layer: Optional[int] = None
    end_layer: Optional[int] = None
    blocks: List[int] = field(default_factory=list)

    is_last_slice: bool = False

    # def __post_init__(self) -> None:
    #     if self.start_token_idx < 0 or self.end_token_idx < 0:
    #         raise ValueError("token indices must be non-negative")
    #     if self.start_layer < 0 or self.end_layer < 0:
    #         raise ValueError("layer indices must be non-negative")
    #     if self.start_token_idx > self.end_token_idx:
    #         raise ValueError("start_token_idx cannot be greater than end_token_idx")
    #     if self.start_layer > self.end_layer:
    #         raise ValueError("start_layer cannot be greater than end_layer")
    #     if any(b < 0 for b in self.blocks):
    #         raise ValueError("block_ids must contain non-negative integers")


class State(Enum):
    """States of a transfer session."""

    INIT = "Init"  # Session contains only the required members for construction.
    READY = "Ready"  # Resources are ready for processing.
    TRANSFERRING = "Transferring"  # Data is being transffered.
    FINISHED = "Finished"  # Processing is finished.
    META_DATA_SENT = "MetaDataSent"  # Meta data has been sent.
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

    @property
    @abstractmethod
    def state(self) -> SessionState: ...

    @abstractmethod
    def poll_task(self, id: TaskIdType) -> State: ...

    @abstractmethod
    def send(self, slice: KVSlice) -> TaskIdType: ...

    """
    Async send slice to the peer. return the task id. Task state can be polled by poll_task().
    """

    @property
    @abstractmethod
    def exception(self) -> Optional[Exception]: ...


class RxSessionBase(ABC):
    def __init__(self, receiver: ReceiverBase, args: SessionArgsBase):
        self.session_args = args

    @property
    @abstractmethod
    def state(self) -> SessionState: ...

    @abstractmethod
    def poll_task(self, id: TaskIdType) -> State: ...

    @abstractmethod
    def receive(self, slice: KVSlice) -> TaskIdType: ...

    """
    Async receive slice from the peer. return the task id. Task state can be polled by poll_task().
    """

    @property
    @abstractmethod
    def exception(self) -> Optional[Exception]: ...
