from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SessionState(Enum):
    INIT = "Init"
    PENDING = "Pending"
    RUNNING = "Running"
    SUCCESS = "Success"
    FAILED = "Failed"
    CANCELLED = "Cancelled"


@dataclass
class TransResult:
    state: SessionState
    error_message: Optional[str] = None


@dataclass
class Request:
    llm_rid: int
    session_id: int
    start_token_idx: int
    end_token_idx: int
    start_layer_idx: int
    end_layer_idx: int


class Sender:
    def async_send(self, request: Request) -> TransResult: ...


class Receiver:
    def async_receive(self, request: Request) -> TransResult: ...
