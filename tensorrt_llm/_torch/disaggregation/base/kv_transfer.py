from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


@dataclass
class KVSlice:
    """Supports transmitting only part of the request cache, e.g, chunks or layers."""

    start_token_idx: int
    end_token_idx: int
    start_layer_idx: int
    end_layer_idx: int
    block_ids: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.start_token_idx < 0 or self.end_token_idx < 0:
            raise ValueError("token indices must be non-negative")
        if self.start_layer_idx < 0 or self.end_layer_idx < 0:
            raise ValueError("layer indices must be non-negative")
        if self.start_token_idx > self.end_token_idx:
            raise ValueError("start_token_idx cannot be greater than end_token_idx")
        if self.start_layer_idx > self.end_layer_idx:
            raise ValueError("start_layer_idx cannot be greater than end_layer_idx")
        if any(b < 0 for b in self.block_ids):
            raise ValueError("block_ids must contain non-negative integers")


class SessionState(Enum):
    """States of a transfer session."""

    INIT = "Init"  # Session contains only the required members for construction.
    READY = "Ready"  # Resources are ready for processing.
    TRANSFERRING = "Transferring"  # Data is being transffered.
    FINISHED = "Finished"  # Processing is finished.
    ERR = "Err"  # An error has occurred.


class TxSessionBase(ABC):
    """Sending session for each complete LLM request."""

    """
    Sends a KVSlice to the receiver.
    """

    @abstractmethod
    def send(self, req: KVSlice) -> None: ...

    """
    Gets the current state of the session.
    """

    @abstractmethod
    def get_state() -> SessionState: ...

    """
    Gets any exception that occurred during the session.
    """

    @abstractmethod
    def get_exception(self) -> Optional[Exception]: ...


class RxSessionBase(ABC):
    """Receiving session for each complete LLM request."""

    """
    Gets the current state of the session.
    """

    @abstractmethod
    def get_state() -> SessionState: ...

    """
    Gets any exception that occurred during the session.
    """

    @abstractmethod
    def get_exception(self) -> Optional[Exception]: ...


class SenderBase(ABC):
    """Handles cache control signals within a single rank."""

    """
    Creates a sending session for a given LLM request ID.
    """

    @abstractmethod
    def create_session(self, llm_rid: int) -> TxSessionBase: ...


class ReceiverBase(ABC):
    """Handles cache control signals within a single rank."""

    """
    Creates a receiving session for a given LLM request ID.
    """

    @abstractmethod
    def create_session(self, llm_rid: int) -> RxSessionBase: ...
