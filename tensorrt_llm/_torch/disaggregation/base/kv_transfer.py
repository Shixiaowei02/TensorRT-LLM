import concurrent
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from tensorrt_llm import DisaggregatedParams


@dataclass
class TransferGenSideReqInfo:
    ctx_req_id: int
    instance_name: str
    instance_rank: int
    block_ids: list[int]
    disagg_id: str
    start_token_id: Optional[int] = None


@dataclass
class AgentSendArgs:
    future_for_task: concurrent.futures.Future
    src_kv_ptrs: list[int]
    dst_kv_ptrs: list[int]
    kv_sizes: list[int]
    expect_count: int
    remote_name: str
    src_aux_ptrs: list[int] = None
    dst_aux_ptrs: list[int] = None
    aux_sizes: list[int] = None
    peer_endpoint: Optional[str] = None  # used for send state
    disagg_id: Optional[str] = None
    slice_id: Optional[int] = None
    expect_slice_num: Optional[int] = None


@dataclass
class AgentRecvArgs:
    disagg_id: str
    futrure_for_task: concurrent.futures.Future
    expect_count: int
    remote_name: str
    slice_id: int


@dataclass
class KVSlice:
    """Supports transmitting only part of the request cache, e.g, chunks or layers."""

    start_token_idx: int
    end_token_idx: int
    start_layer_idx: int
    end_layer_idx: int

    expect_slice_num: int

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


class SliceTaskState(Enum):
    INIT = "Init"
    READY = "Ready"
    TRANSFERRING = "Transferring"
    FINISHED = "Finished"
    ERR = "Err"


class SliceSenderTaskBase(ABC):
    def __init__(self, slice: KVSlice, disagg_params: DisaggregatedParams, slice_id: int): ...

    @abstractmethod
    def get_state(self) -> SliceTaskState: ...

    ### the send task wrap the send slice

    @abstractmethod
    def extract_trans_meta(self, dst_info: TransferGenSideReqInfo) -> AgentSendArgs: ...

    ### extract agent send args from send slice and transfer reqinfo for sender submission


class SliceReceiverTaskBase(ABC):
    def __init__(self, slice: KVSlice, disagg_params: DisaggregatedParams, slice_id: int): ...

    @abstractmethod
    def get_state(self) -> SliceTaskState: ...

    @abstractmethod
    def extract_trans_meta(self) -> AgentRecvArgs: ...

    @abstractmethod
    def create_gen_side_transfer_req_info(self) -> TransferGenSideReqInfo: ...


class SenderSessionBase(ABC):
    ### the sender session wrap the sender and all sender slice tasks for a request
    def __init__(self, request_id: int, disagg_params: DisaggregatedParams, sender):
        self.request_id = request_id
        self.disagg_params = disagg_params
        self.sender = sender
        self.slice_tasks = []

    @abstractmethod
    def get_state(self) -> SessionState: ...

    @abstractmethod
    def async_send(self, slice: KVSlice) -> None:
        ...
        ## call sender.async_send_slice to create a sender slice task and add to the sender session


class ReceiverSessionBase(ABC):
    def __init__(self, request_id: int, disagg_params: DisaggregatedParams, receiver):
        self.request_id = request_id
        self.disagg_params = disagg_params
        self.receiver = receiver
        self.slice_tasks = []

    ### the receiver session wraps the receiver and all receiver slice tasks for a request
    @abstractmethod
    def get_state(self) -> SessionState: ...

    @abstractmethod
    def async_receive(self, slice: KVSlice) -> None:
        ...
        ## call receiver.async_receive_slice to create a receiver slice task and add to the receiver session


class SenderBase(ABC):
    @abstractmethod
    def get_sender_session_state(self, disagg_id: str) -> SessionState: ...

    @abstractmethod
    def init_init_session_resource(self, disagg_id: str) -> None: ...

    @abstractmethod
    def clear_sender_session_resource(self, disagg_id: str) -> None: ...

    @abstractmethod
    def async_send_slice(
        self, disagg_params: DisaggregatedParams, slice: KVSlice
    ) -> SliceSenderTaskBase: ...

    # create a send slice task and may submit the send task
    # the upper layer can check the state of the send slice task to know the progress of the async send

    @abstractmethod
    def submit_send_task(self, agent_send_args: AgentSendArgs) -> None: ...

    ### submit the send task to the agent


class ReceiverBase(ABC):
    @abstractmethod
    def get_receiver_session_state(self, disagg_id: str) -> SessionState: ...

    @abstractmethod
    def init_init_session_resource(self, disagg_id: str) -> None: ...

    @abstractmethod
    def clear_receiver_session_resource(self, disagg_id: str) -> None: ...

    @abstractmethod
    def async_receive_slice(
        self, disagg_params: DisaggregatedParams, slice: KVSlice
    ) -> SliceReceiverTaskBase:
        ### create a receive slice task and may submit the receive task
        ### the upper layer can check the state of the receive slice task to know the progress of the async receive
        ...

    @abstractmethod
    def submit_receive_task(self, agent_recv_args: AgentRecvArgs) -> None:
        ### submit the receive task to the agent
        ...
