from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest


@dataclass
class AuxBufferMeta:
    ptrs: list[int]
    size: list[int]
    item_sizes: list[int] = field(default_factory=list)
    device: str = "cpu"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ptrs": self.ptrs,
            "size": self.size,
            "item_sizes": self.item_sizes,
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuxBufferMeta":
        return cls(
            ptrs=data["ptrs"],
            size=data["size"],
            item_sizes=data.get("item_sizes", []),
            device=data.get("device", "cpu"),
        )


class AuxBufferBase(ABC):
    """
    Abstract base class defining the interface for auxiliary buffer management.
    """

    @abstractmethod
    def alloc_slot(self) -> int:
        """
        Allocate a free slot and return its index.

        Raises:
            ValueError: If no slot is available.
        """
        ...

    @abstractmethod
    def free_slot(self, slot: int) -> None:
        """
        Release the specified slot.
        """
        ...

    @abstractmethod
    def get_meta(self) -> AuxBufferMeta:
        """
        Retrieve meta-information about the underlying buffer(s).
        Returns buffer info (e.g., pointers, sizes, device).
        """
        ...

    @abstractmethod
    def fill_slot(self, slot: int, request: LlmRequest) -> None:
        """
        Fill/overwrite the contents of the given slot with data from the request.
        """
        ...

    @abstractmethod
    def get_slot_tokens(self, slot: int) -> tuple[list[int], list[int]]:
        """
        Get the token data (e.g., first/draft tokens) from the specified slot.
        """
        ...


class AuxBuffer(AuxBufferBase):
    def __init__(self, max_slot_num: int, beam_width: int, max_draft_len: int, device: str = "cpu"):
        self.max_slot_num = max_slot_num
        self.beam_width = beam_width
        self.max_draft_len = max_draft_len
        self.device = device

        self.free_slots = deque(list(range(max_slot_num)))
        self.occupied_slots: set = set()

        data_type = torch.int32
        self.first_tokens_buffer = torch.empty(
            max_slot_num, beam_width, dtype=data_type, device=device
        )

        self.draft_tokens_buffer = torch.empty(
            max_slot_num, max_draft_len, dtype=data_type, device=device
        )

        self.meta = AuxBufferMeta(
            ptrs=[self.first_tokens_buffer.data_ptr(), self.draft_tokens_buffer.data_ptr()],
            size=[
                self.first_tokens_buffer.numel() * self.first_tokens_buffer.element_size(),
                self.draft_tokens_buffer.numel() * self.draft_tokens_buffer.element_size(),
            ],
            item_sizes=[
                self.first_tokens_buffer[0].numel() * self.first_tokens_buffer.element_size(),
                self.draft_tokens_buffer[0].numel() * self.draft_tokens_buffer.element_size(),
            ],
            device=device,
        )

    def alloc_slot(self) -> int:
        if not self.free_slots:
            raise ValueError("No free slot available")
        slot_id = self.free_slots.popleft()
        if slot_id in self.occupied_slots:
            raise RuntimeError(f"Slot {slot_id} is already in use")
        self.occupied_slots.add(slot_id)
        return slot_id

    def free_slot(self, slot: int) -> None:
        if slot not in self.occupied_slots:
            raise ValueError(f"Attempting to free unused slot {slot}")
        self.occupied_slots.remove(slot)
        if slot < 0 or slot >= self.max_slots:
            raise ValueError(f"Invalid slot_id {slot}")
        self.free_slots.append(slot)

    def get_meta(self) -> AuxBufferMeta:
        return self.meta

    def fill_slot(self, slot: int, request: LlmRequest) -> None:
        first_gen_tokens = request.get_last_tokens()
        draft_tokens = request.py_draft_tokens

        if len(first_gen_tokens) > self.beam_width:
            raise ValueError(
                f"first_gen_tokens length {len(first_gen_tokens)} exceeds beam_width {self.beam_width}"
            )
        if len(draft_tokens) > self.max_draft_len:
            raise ValueError(
                f"draft_tokens length {len(draft_tokens)} exceeds max_draft_len {self.max_draft_len}"
            )

        self.first_tokens_buffer[slot][: len(first_gen_tokens)].copy_(
            torch.tensor(first_gen_tokens, dtype=torch.int32, device="cpu")
        )
        self.draft_tokens_buffer[slot][: len(draft_tokens)].copy_(
            torch.tensor(draft_tokens, dtype=torch.int32, device="cpu")
        )

    def get_slot_tokens(self, slot) -> tuple[list[int], list[int]]:
        first_gen_tokens = self.first_tokens_buffer[slot].tolist()
        draft_tokens = self.draft_tokens_buffer[slot].tolist()

        return first_gen_tokens, draft_tokens
