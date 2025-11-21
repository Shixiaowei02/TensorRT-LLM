from collections import deque
from dataclasses import dataclass, field
from typing import Tuple

import torch

from tensorrt_llm._torch.disaggregation.base.kv_transfer import MetaBufferBase
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest


@dataclass
class MetaBufferInfo:
    ptrs: list[int]
    size: list[int]
    item_sizes: list[int] = field(default_factory=list)
    device: str = "cpu"


class MetaBuffer(MetaBufferBase):
    def __init__(self, max_slot_num: int, beam_width: int, max_draft_len: int):
        self.max_slot_num = max_slot_num
        self.beam_width = beam_width
        self.max_draft_len = max_draft_len

        self.free_slots = deque(list(range(max_slot_num)))

        data_type = torch.int32
        device = "cpu"
        self.first_tokens_buffer = torch.empty(
            max_slot_num, beam_width, dtype=data_type, device=device
        )

        self.draft_tokens_buffer = torch.empty(
            max_slot_num, max_draft_len, dtype=data_type, device=device
        )

        self.buffer_info = MetaBufferInfo(
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
        if len(self.free_slots) == 0:
            raise ValueError("No free slot available")
        return self.free_slots.popleft()

    def free_slot(self, slot_id: int) -> None:
        self.free_slots.append(slot_id)

    def get_buffer_info(self) -> MetaBufferInfo:
        return self.buffer_info

    def set_buffer(self, slot_id: int, request: LlmRequest) -> None:
        first_gen_tokens = request.get_last_tokens()
        draft_tokens = request.py_draft_tokens

        self.first_tokens_buffer[slot_id][: len(first_gen_tokens)].copy_(
            torch.tensor(first_gen_tokens, dtype=torch.int32, device="cpu")
        )
        self.draft_tokens_buffer[slot_id][: len(draft_tokens)].copy_(
            torch.tensor(draft_tokens, dtype=torch.int32, device="cpu")
        )

    def extra_buffer(self, slot_id) -> Tuple[list[int], list[int]]:
        first_gen_tokens = self.first_tokens_buffer[slot_id].tolist()
        draft_tokens = self.draft_tokens_buffer[slot_id].tolist()

        return first_gen_tokens, draft_tokens
