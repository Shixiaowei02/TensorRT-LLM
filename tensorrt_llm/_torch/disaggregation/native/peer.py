from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import msgpack

from tensorrt_llm._torch.disaggregation.native.aux_buffer import AuxBufferMeta
from tensorrt_llm._torch.disaggregation.native.region.registrar import PeerRegistrar


@dataclass
class RankInfo:
    instance_name: str
    instance_rank: int
    tp_size: int
    tp_rank: int
    pp_size: int
    pp_rank: int
    dp_size: int
    dp_rank: int
    cp_size: int
    cp_rank: int
    device_id: int
    kv_heads_per_rank: int
    # [numLayers, kv_factor, heads, tokens, dims_per_head]
    tokens_per_block: int
    dims_per_head: int
    element_bytes: int
    enable_attention_dp: bool
    is_mla: bool
    layer_num_per_pp: List[int]
    kv_ptrs: List[int]
    aux_ptrs: List[int]
    server_endpoint: str
    self_endpoint: str
    transfer_engine_info: bytes
    aux_meta: Optional[AuxBufferMeta]

    def to_bytes(self) -> bytes:
        data = asdict(self)
        data["aux_meta"] = self.aux_meta.to_dict() if self.aux_meta is not None else None
        return msgpack.packb(data)

    @classmethod
    def from_bytes(cls, data: bytes) -> "RankInfo":
        unpacked = msgpack.unpackb(data)
        if unpacked.get("aux_meta") is not None:
            unpacked["aux_meta"] = AuxBufferMeta.from_dict(unpacked["aux_meta"])
        return cls(**unpacked)


@dataclass
class PeerOverlap:
    overlap_pp_size: int = 0
    overlap_tp_size: int = 0
    overlap_cp_size: int = 0
    duplicate_head_factor: int = 1
    peer_duplicate_head_factor: int = 1
    target_peer_pp_layer_num: List[int] = field(default_factory=list)
    ranks: List[int] = field(default_factory=list)


class PeerMapper:
    def __init__(
        self,
        registrar: PeerRegistrar,
    ):
        self._registrar = registrar
        self._overlap_cache: Dict[str, PeerOverlap] = {}

        # cache self info
        self._ri = self._registrar.rank_info

    @staticmethod
    def _find_overlap(self_val, peer_val, self_rank, peer_rank=None):
        if self_val <= peer_val:
            overlap = peer_val // self_val
            start = self_rank * overlap + (peer_rank * peer_val if peer_rank is not None else 0)
            end = start + overlap
        else:
            ratio = self_val // peer_val
            start = (self_rank // ratio) + (peer_rank * peer_val if peer_rank is not None else 0)
            overlap = 1
            end = start + overlap

        return overlap, start, end

    # ---------------- peer overlap targets ----------------
    def get_peer_overlap_targets(self, peer_rank_info: RankInfo, peer_dp_rank: int) -> PeerOverlap:
        peer_ri = peer_rank_info
        key = self._unique_key(peer_ri.instance_name, peer_dp_rank)
        if key in self._overlap_cache:
            return self._overlap_cache[key]

        # compute pp overlap and target layers
        self_start_layer = sum(self._ri.layer_num_per_pp[: self._ri.pp_rank])
        self_end_layer = self_start_layer + self._ri.layer_num_per_pp[self._ri.pp_rank]

        pre = 0
        tgt_pp_ranks: List[int] = []
        tgt_pp_layer_num: List[int] = []
        for p in range(peer_ri.pp_size):
            peer_start_layer = pre
            peer_end_layer = peer_start_layer + peer_ri.layer_num_per_pp[p]
            if self_start_layer < peer_end_layer and self_end_layer > peer_start_layer:
                tgt_pp_ranks.append(p)
                tgt_pp_layer_num.append(
                    min(peer_end_layer, self_end_layer) - max(peer_start_layer, self_start_layer)
                )
            pre += peer_ri.layer_num_per_pp[p]

        if tgt_pp_ranks == []:
            # no overlap found
            targets = PeerOverlap()
            self._overlap_cache[key] = targets
            return targets

        peer_start_pp = tgt_pp_ranks[0]
        overlap_pp_size = len(tgt_pp_ranks)
        peer_end_pp = peer_start_pp + overlap_pp_size

        # tp per dp-group
        self_tp_per_dp = self._tp_per_dp(self._ri)
        peer_tp_per_dp = self._tp_per_dp(peer_ri)
        self_tp_rank_in_dp = self._ri.tp_rank % self_tp_per_dp

        overlap_tp_size, peer_start_tp, peer_end_tp = self._find_overlap(
            self_tp_per_dp, peer_tp_per_dp, self_tp_rank_in_dp, peer_dp_rank
        )
        overlap_cp_size, peer_start_cp, peer_end_cp = self._find_overlap(
            self._ri.cp_size, peer_ri.cp_size, self._ri.cp_rank
        )

        ranks: List[int] = []
        for pp in range(peer_start_pp, peer_end_pp):
            for cp in range(peer_start_cp, peer_end_cp):
                for tp in range(peer_start_tp, peer_end_tp):
                    ranks.append(pp * peer_ri.tp_size * peer_ri.cp_size + cp * peer_ri.tp_size + tp)

        factor_self = self._ri.kv_heads_per_rank * self_tp_per_dp
        factor_peer = peer_ri.kv_heads_per_rank * peer_tp_per_dp
        dup_head = max(1, factor_self // factor_peer)
        peer_dup_head = max(1, factor_peer // factor_self)

        targets = PeerOverlap(
            overlap_pp_size=overlap_pp_size,
            overlap_tp_size=overlap_tp_size,
            overlap_cp_size=overlap_cp_size,
            duplicate_head_factor=dup_head,
            peer_duplicate_head_factor=peer_dup_head,
            target_peer_pp_layer_num=tgt_pp_layer_num,
            ranks=ranks,
        )
        self._overlap_cache[key] = targets
        return targets
