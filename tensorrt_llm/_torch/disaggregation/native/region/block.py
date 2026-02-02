from typing import List

import numpy as np
from peer_mapper import RankInfo

from tensorrt_llm._torch.disaggregation.base.kv_extractor import (
    MemoryRegion,
    Region,
    RegionMapperBase,
    RegionPair,
)


class IdentityMapper(RegionMapperBase):
    """
    ---- mapper_identity ----

    Pass-through mapping. Do not change pointers or sizes.

    src_ptrs: [ S0 ] [ S1 ] [ S2 ] ...
                |      |      |
                v      v      v
    dst_ptrs: [ D0 ] [ D1 ] [ D2 ] ...
    """

    def map(self, src_regions: List[Region], dst_regions: List[Region]) -> List[RegionPair]:
        return [
            RegionPair(src=src_region, dst=dst_region)
            for src_region, dst_region in zip(src_regions, dst_regions)
        ]


class HeadMatchMapper(RegionMapperBase):
    """
    ---- mapper_head_match ----

    Move/copy entire contiguous block(s) (multi-layer fragment) as a single chunk.
    Align by whole fragment size (frag_size) and apply a constant source/destination block offset.

    src_ptrs:  [ S0 ]         [ S1 ]          ...
                 |              |
              + src_off      + src_off
                 |              |
          [ S0 + src_off ] [ S1 + src_off ]   ->  (each points to a frag of size frag_size)
                   copy whole frag
                 |              |
                 v              v
          [ D0 + dst_off ] [ D1 + dst_off ]   ->  (destination frags)
    """

    def __init__(
        self,
        transfer_layers: int,
        kv_factor: int,
        src_layer_off: int,
        dst_layer_off: int,
        self_ri: RankInfo,
        peer_ri: RankInfo,
    ):
        self._frag_size = self._block_size(transfer_layers, kv_factor, self_ri)
        self._src_block_off = self._block_size(src_layer_off, kv_factor, self_ri)
        self._dst_block_off = self._block_size(dst_layer_off, kv_factor, peer_ri)

    def map(self, src_regions: List[Region], dst_regions: List[Region]) -> List[RegionPair]:
        mapped_regions = []
        for src_region, dst_region in zip(src_regions, dst_regions):
            src_memory = MemoryRegion(
                ptr=src_region.memory.ptr + self._src_block_off,
                size=self._frag_size,
            )
            dst_memory = MemoryRegion(
                ptr=dst_region.memory.ptr + self._dst_block_off,
                size=self._frag_size,
            )
            mapped_regions.append(
                RegionPair(
                    src=Region(memory=src_memory, region=None),
                    dst=Region(memory=dst_memory, region=None),
                )
            )
        return mapped_regions

    def _block_size(self, layer_num: int, kv_factor: int, ri: RankInfo) -> int:
        return (
            layer_num
            * kv_factor
            * ri.kv_heads_per_rank
            * ri.tokens_per_block
            * ri.dims_per_head
            * ri.element_bytes
        )


class HeadMismatchMapper(RegionMapperBase):
    """
    ---- mapper_head_mismatch ----

    Fine-grained mapping when head counts or TP/DP partitioning differ.
    Split layers into per-head (or contiguous-heads) fragments and map them individually.
    Handles kv_factor (e.g., key+value duplication) and TP/DP head offsets.

    Source (layers x heads):
    L0: [S00 S01] [S02 S03] ...
    L1: [S10 S11] [S12 S13] ...

    Destination (layers x heads, different layout possible):
    L0': [D00] [D01] [D02] ...
    L1': [D10] [D11] ...

    Mapping (each arrow = copy cont_heads_frag):
    [S00 S01] -> [D00]
    [S02 S03] -> [D01]
    [S10 S11] -> [D02]
    """

    def __init__(
        self,
        transfer_layers: int,
        kv_factor: int,
        self_tp_per_dp: int,
        peer_tp_per_dp: int,
        self_tp_rank: int,
        peer_tp_rank: int,
        src_layer_off: int,
        peer_layer_off: int,
        self_ri: RankInfo,
        peer_ri: RankInfo,
    ):
        self._ri = self_ri
        self._peer_ri = peer_ri

        # Each pool has shape [max_blocks, num_layers, kv_factor, num_kv_heads, tokens_per_block,
        # dims_per_head, element_bytes].
        bytes_per_head = self._ri.tokens_per_block * self._ri.dims_per_head * self._ri.element_bytes
        self._bytes_cont_heads = (
            min(self._ri.kv_heads_per_rank, peer_ri.kv_heads_per_rank) * bytes_per_head
        )

        self._src_head_off, self._dst_head_off = self._compute_head_offsets(
            self_tp_per_dp,
            peer_tp_per_dp,
            self_tp_rank,
            peer_tp_rank,
            self._bytes_cont_heads,
        )

        self._layer_indices = np.arange(transfer_layers, dtype=np.int64)
        self._kv_indices = np.arange(kv_factor, dtype=np.int64)
        self._peer_layer_off = peer_layer_off

    def map(self, src_regions: List[Region], dst_regions: List[Region]) -> List[RegionPair]:
        mapped_regions = []
        for src_region, dst_region in zip(src_regions, dst_regions):
            # Extract necessary memory information from Region and calculate source fragments
            src_bases = np.array([src_region.memory.ptr], dtype=np.int64)
            src_frags = self._get_frags(
                bases=src_bases,
                layer_indices=self._layer_indices,
                layer_kv_num=self._get_layer_kv_num(self._ri),
                kv_indices=self._kv_indices,
                head_off=self._src_head_off,
                kv_factor=self._kv_indices.size,
            )

            # Extract necessary memory information from Region and calculate destination fragments
            dst_bases = np.array([dst_region.memory.ptr], dtype=np.int64)
            dst_frags = self._get_frags(
                bases=dst_bases,
                layer_indices=self._peer_layer_off + self._layer_indices,
                layer_kv_num=self._get_layer_kv_num(self._peer_ri),
                kv_indices=self._kv_indices,
                head_off=self._dst_head_off,
                kv_factor=self._kv_indices.size,
            )

            # Map source fragments to destination fragments
            for src_frag, dst_frag in zip(src_frags.flatten(), dst_frags.flatten()):
                src_memory = MemoryRegion(ptr=src_frag, bytes=self._bytes_cont_heads)
                dst_memory = MemoryRegion(ptr=dst_frag, bytes=self._bytes_cont_heads)

                # Create new RegionPair
                mapped_regions.append(
                    RegionPair(
                        src=Region(memory=src_memory, region=None),
                        dst=Region(memory=dst_memory, region=None),
                    )
                )

        return mapped_regions

    @staticmethod
    def _compute_head_offsets(
        self_tp_per_dp: int,
        peer_tp_per_dp: int,
        self_tp_rank: int,
        peer_tp_rank: int,
        bytes_cont_heads: int,
    ) -> tuple[int, int]:
        if self_tp_per_dp == peer_tp_per_dp:
            return 0, 0
        ratio = max(self_tp_per_dp, peer_tp_per_dp) // min(self_tp_per_dp, peer_tp_per_dp)
        if self_tp_per_dp < peer_tp_per_dp:
            return (peer_tp_rank % ratio) * bytes_cont_heads, 0
        else:
            return 0, (self_tp_rank % ratio) * bytes_cont_heads

    @staticmethod
    def _get_layer_kv_num(ri: RankInfo) -> int:
        return ri.kv_heads_per_rank * ri.tokens_per_block * ri.dims_per_head * ri.element_bytes

    @staticmethod
    def _get_frags(bases, layer_indices, layer_kv_num, kv_indices, head_off, kv_factor):
        layer_num = layer_kv_num * kv_factor
        return (
            bases[:, None, None]
            + layer_num * layer_indices[None, :, None]
            + layer_kv_num * kv_indices[None, None, :]
            + head_off
        )
