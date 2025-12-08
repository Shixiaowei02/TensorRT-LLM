from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Union

import msgpack
import numpy as np

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.native.kv_meta_buffer import MetaBufferInfo
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import get_size_in_bytes, nvtx_range


@dataclass
class PeerOverlapTargets:
    overlap_pp_size: int
    overlap_tp_size: int
    overlap_cp_size: int
    duplicate_head_factor: int
    peer_duplicate_head_factor: int
    target_peer_pp_layer_num: List[int]
    ranks: List[int]


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
    kv_head_num_per_rank: int
    #  [numLayers,kv_factor,heads,tokens,dimsPerHead]
    tokens_per_block: int
    dims_per_head: int
    element_size: int
    enable_attention_dp: bool
    is_mla: bool
    layer_num_per_pp: List[int]
    kvcache_ptrs: List[int]
    aux_ptrs: List[int]
    server_endpoint: str
    recv_endpoint: str
    transfer_engine_info: bytes
    meta_buffer_info: Optional[MetaBufferInfo]

    def to_bytes(self) -> bytes:
        data = asdict(self)
        data["meta_buffer_info"] = (
            self.meta_buffer_info.to_dict() if self.meta_buffer_info is not None else None
        )
        return msgpack.packb(data)

    @classmethod
    def from_bytes(cls, data: bytes) -> "RankInfo":
        unpacked = msgpack.unpackb(data)
        if unpacked["meta_buffer_info"] is not None:
            unpacked["meta_buffer_info"] = MetaBufferInfo.from_dict(unpacked["meta_buffer_info"])
        return cls(**unpacked)


@dataclass
class InstanceInfo:
    instance_name: str
    tp_size: int
    pp_size: int
    dp_size: int
    cp_size: int
    kv_head_num_per_rank: int
    tokens_per_block: int
    dims_per_head: int
    element_size: int
    enable_attention_dp: bool
    is_mla: bool
    layer_num_per_pp: List[int]
    ctx_server_endpoints: List[str]

    def to_bytes(self) -> bytes:
        return msgpack.packb(asdict(self))

    @classmethod
    def from_bytes(cls, data: bytes) -> "InstanceInfo":
        return cls(**msgpack.unpackb(data))


@dataclass
class KVPoolAttributes:
    kv_cache_ptrs: List[int]
    kv_cache_block_sizes: List[int]


class KVPtrExtractor:
    def __init__(self, kv_pool_attributes: Union[KVPoolAttributes, KVCacheManager]):
        if self._is_kv_cache_manager(kv_pool_attributes):
            self.kv_pool_attributes = self._attrs_from_manager(kv_pool_attributes)
        else:
            self.kv_pool_attributes = kv_pool_attributes

    def extract_kv_block_ptrs(self, kv_block_ids: List[int], pool_idx: int = 0) -> List[int]:
        """
        Given KV block ids, return the base pointers for each block in the chosen pool.

        Args:
            kv_block_ids: list of integer block ids (0-based).
            pool_idx: index of the pool to use (default 0). Must be valid for configured pools.

        Returns:
            List[int]: list of pointers (one pointer per block id).
        """
        ptrs = self.kv_pool_attributes.kv_cache_ptrs
        sizes = self.kv_pool_attributes.kv_cache_block_sizes

        if pool_idx < 0 or pool_idx >= len(ptrs):
            raise IndexError(f"pool_idx {pool_idx} out of range, available pools: {len(ptrs)}")

        base_ptr = ptrs[pool_idx]
        block_size = sizes[pool_idx]
        return [base_ptr + block_size * bid for bid in kv_block_ids]

    # ---------------- internal helpers ----------------
    @staticmethod
    def _is_kv_cache_manager(obj) -> bool:
        # A light-weight duck-typing check to decide whether obj behaves like KVCacheManager.
        # We expect KVCacheManager to provide get_unique_primary_pool() and dtype attribute.
        return hasattr(obj, "get_unique_primary_pool") and hasattr(obj, "dtype")

    @staticmethod
    def _attrs_from_manager(manager: KVCacheManager) -> KVPoolAttributes:
        try:
            pools = manager.get_unique_primary_pool()
        except Exception as ex:
            raise ValueError("Failed to obtain pool(s) from KVCacheManager: " + str(ex))

        # Normalize to list
        if isinstance(pools, (list, tuple)):
            pool_list = list(pools)
        else:
            pool_list = [pools]

        element_size = get_size_in_bytes(1, manager.dtype)

        ptrs: List[int] = []
        block_sizes: List[int] = []
        for p in pool_list:
            # data_ptr
            if hasattr(p, "data_ptr") and callable(getattr(p, "data_ptr")):
                ptr = p.data_ptr()
            elif isinstance(p, int):
                ptr = int(p)
            else:
                raise ValueError("Pool object does not expose data_ptr(): " + repr(p))
            ptrs.append(ptr)

            # numel -> block size
            # use p[0] , p is a tensor shape (num_blocks,block_size)
            if hasattr(p, "numel") and callable(getattr(p, "numel")):
                n = p[0].numel()
            elif hasattr(p, "num_elements"):
                n = getattr(p[0], "num_elements")
            else:
                raise ValueError("Pool object does not expose numel()/num_elements: " + repr(p))

            block_sizes.append(int(n) * int(element_size))

        return KVPoolAttributes(kv_cache_ptrs=ptrs, kv_cache_block_sizes=block_sizes)


@dataclass
class CopyArgs:
    src_ptrs: List[int]
    dst_ptrs: List[int]
    size: int


class PeerRegistrar:
    def __init__(
        self,
        rank_info: RankInfo,
        instance_info: InstanceInfo,
    ):
        self.ri = rank_info
        self.ii = instance_info

        self._peer_ri_cache: Dict[str, RankInfo] = {}

    # ---------------- public simple APIs ----------------
    def register(self, peer_name: str, peer_rank: int, peer_ri: RankInfo):
        # TODO: check  if peer is valid for registration
        self._peer_ri_cache[self._key(peer_name, peer_rank)] = peer_ri

    def unregister(self, peer_name: str, peer_rank: int):
        del self._peer_ri_cache[self._key(peer_name, peer_rank)]

    def get_peer_rank_info(self, peer_name: str, peer_rank: int):
        return self._peer_ri_cache[self._key(peer_name, peer_rank)]

    def get_self_instance_info(self) -> InstanceInfo:
        return self.ii

    def get_self_rank_info(self) -> RankInfo:
        return self.ri

    def _key(self, name: str, rank: int) -> str:
        return name + str(rank)


class PeerMapperBase(ABC):
    PtrMapper = Callable[[List[int], int, List[int], int], CopyArgs]

    def __init__(
        self,
        registrar: PeerRegistrar,
        kv_cache_manager: KVCacheManager,
    ): ...

    @abstractmethod
    def get_peer_overlap_targets(
        self, peer_ii: InstanceInfo, peer_dp_rank: int
    ) -> PeerOverlapTargets: ...

    @abstractmethod
    def get_kv_ptrs_mapper(self, peer_ri: RankInfo) -> PtrMapper: ...


class PeerMapper(PeerMapperBase):
    def __init__(
        self,
        registrar: PeerRegistrar,
        kv_cache_manager: KVCacheManager,
    ):
        super().__init__(registrar, kv_cache_manager)
        self._registrar = registrar
        self._overlap_cache: Dict[str, PeerOverlapTargets] = {}
        self._kv_mapper_cache: Dict[str, callable] = {}
        self._kv_pool_cache: Dict[str, KVPtrExtractor] = {}

        self.kv_block_ptr_extractor = KVPtrExtractor(kv_cache_manager)

        self.ri = self._registrar.get_self_rank_info()
        self.ii = self._registrar.get_self_instance_info()

    # ---------------- kv pool extractor ----------------
    def get_kv_extractor(self, peer_name: str, peer_rank: int) -> KVPtrExtractor:
        k = self._key(peer_name, peer_rank)
        if k not in self._kv_pool_cache:
            peer_ri = self._registrar.get_peer_rank_info(peer_name, peer_rank)
            kv_f = 1 if peer_ri.is_mla else 2
            block_sz = self._block_size(peer_ri, kv_f)
            extractor = KVPtrExtractor(
                kv_pool_attributes=KVPoolAttributes(
                    kv_cache_ptrs=peer_ri.kvcache_ptrs, kv_cache_block_sizes=[block_sz]
                )
            )
            self._kv_pool_cache[k] = extractor
        return self._kv_pool_cache[k]

    # ---------------- peer overlap targets ----------------
    def get_peer_overlap_targets(
        self, peer_ii: InstanceInfo, peer_dp_rank: int
    ) -> PeerOverlapTargets:
        k = self._key(peer_ii.instance_name, peer_dp_rank)
        if k in self._overlap_cache:
            return self._overlap_cache[k]

        # compute pp overlap and target layers
        self_start_layer = sum(self.ri.layer_num_per_pp[: self.ri.pp_rank])
        self_end_layer = self_start_layer + self.ri.layer_num_per_pp[self.ri.pp_rank]

        pre = 0
        tgt_pp_ranks: List[int] = []
        tgt_pp_layer_num: List[int] = []
        for p in range(peer_ii.pp_size):
            peer_start = pre
            peer_end = peer_start + peer_ii.layer_num_per_pp[p]
            if self_start_layer < peer_end and self_end_layer > peer_start:
                tgt_pp_ranks.append(p)
                tgt_pp_layer_num.append(
                    min(peer_end, self_end_layer) - max(peer_start, self_start_layer)
                )
            pre += peer_ii.layer_num_per_pp[p]

        peer_pp_start = tgt_pp_ranks[0]
        overlap_pp_size = len(tgt_pp_ranks)
        peer_pp_end = peer_pp_start + overlap_pp_size

        # tp per dp-group
        self_tp_per_dp = self._tp_per_dp(self.ri)
        peer_tp_per_dp = self._tp_per_dp(peer_ii)
        self_tp_rank_in_dp = self.ri.tp_rank % self_tp_per_dp

        # compute tp overlap
        if self_tp_per_dp <= peer_tp_per_dp:
            overlap_tp = peer_tp_per_dp // self_tp_per_dp
            peer_tp_start = self_tp_rank_in_dp * overlap_tp + peer_dp_rank * peer_tp_per_dp
            peer_tp_end = peer_tp_start + overlap_tp
        else:
            ratio = self_tp_per_dp // peer_tp_per_dp
            peer_tp_start = self_tp_rank_in_dp // ratio + peer_dp_rank * peer_tp_per_dp
            overlap_tp = 1
            peer_tp_end = peer_tp_start + overlap_tp

        # cp overlap
        if self.ri.cp_size <= peer_ii.cp_size:
            overlap_cp = peer_ii.cp_size // self.ri.cp_size
            peer_cp_start = self.ri.cp_rank * overlap_cp
            peer_cp_end = peer_cp_start + overlap_cp
        else:
            ratio = self.ri.cp_size // peer_ii.cp_size
            peer_cp_start = self.ri.cp_rank // ratio
            overlap_cp = 1
            peer_cp_end = peer_cp_start + overlap_cp

        ranks: List[int] = []
        for pp in range(peer_pp_start, peer_pp_end):
            for cp in range(peer_cp_start, peer_cp_end):
                for tp in range(peer_tp_start, peer_tp_end):
                    ranks.append(pp * peer_ii.tp_size * peer_ii.cp_size + cp * peer_ii.tp_size + tp)

        dup_head = max(
            1,
            self.ri.kv_head_num_per_rank
            * self_tp_per_dp
            // (peer_ii.kv_head_num_per_rank * peer_tp_per_dp),
        )
        peer_dup_head = max(
            1,
            peer_ii.kv_head_num_per_rank
            * peer_tp_per_dp
            // (self.ri.kv_head_num_per_rank * self_tp_per_dp),
        )

        pd = PeerOverlapTargets(
            overlap_pp_size=overlap_pp_size,
            overlap_tp_size=(peer_tp_per_dp if "overlap_tp" in locals() else overlap_tp),
            overlap_cp_size=overlap_cp,
            duplicate_head_factor=dup_head,
            peer_duplicate_head_factor=peer_dup_head,
            target_peer_pp_layer_num=tgt_pp_layer_num,
            ranks=ranks,
        )
        self._overlap_cache[k] = pd
        return pd

    def get_peer_registrar(self) -> PeerRegistrar:
        return self._registrar

    # ---------------- kv block ptrs mapper ----------------
    def get_kv_ptrs_mapper(self, peer_ri: RankInfo) -> PeerMapperBase.PtrMapper:
        k = self._key(peer_ri.instance_name, peer_ri.instance_rank)
        if k in self._kv_mapper_cache:
            return self._kv_mapper_cache[k]

        kv_factor = 1 if self.ri.is_mla else 2
        self_tp_per_dp = self._tp_per_dp(self.ri)
        peer_tp_per_dp = self._tp_per_dp(peer_ri)
        self_tp_rank = self.ri.tp_rank % self_tp_per_dp
        peer_tp_rank = peer_ri.tp_rank % peer_tp_per_dp

        # head_num_per_rank =1 when is_dup_head
        is_dup_head = (
            self.ri.kv_head_num_per_rank * self_tp_per_dp
            != peer_ri.kv_head_num_per_rank * peer_tp_per_dp
        )
        head_match = is_dup_head or self.ri.is_mla or self_tp_per_dp == peer_tp_per_dp
        logger.debug(
            f"head_match: {head_match}, is_dup_head: {is_dup_head}, self.ri.is_mla: {self.ri.is_mla}, "
            f"self_tp_per_dp: {self_tp_per_dp}, peer_tp_per_dp: {peer_tp_per_dp}"
        )
        # fast identity when write_all and same pp_size
        if head_match and self.ri.pp_size == peer_ri.pp_size:
            mapper = self._identity_kv_mapper()
            self._kv_mapper_cache[k] = mapper
            return mapper

        # compute overlapping layers
        self_start_layer = sum(self.ri.layer_num_per_pp[: self.ri.pp_rank])
        self_end_layer = self_start_layer + self.ri.layer_num_per_pp[self.ri.pp_rank]
        peer_start_layer = sum(peer_ri.layer_num_per_pp[: peer_ri.pp_rank])
        peer_end_layer = peer_start_layer + peer_ri.layer_num_per_pp[peer_ri.pp_rank]
        start = max(self_start_layer, peer_start_layer)
        end = min(self_end_layer, peer_end_layer)
        transfer_layers = end - start
        src_offset_layers = start - self_start_layer
        peer_offset_layers = start - peer_start_layer

        if head_match:
            mapper = self._build_head_match_kv_mapper(
                transfer_layers, kv_factor, src_offset_layers, peer_offset_layers, peer_ri
            )
            self._kv_mapper_cache[k] = mapper
            return mapper

        # head mismatch case
        mapper = self._build_head_mismatch_kv_mapper(
            transfer_layers,
            kv_factor,
            self_tp_per_dp,
            peer_tp_per_dp,
            self_tp_rank,
            peer_tp_rank,
            src_offset_layers,
            peer_offset_layers,
            peer_ri,
        )
        self._kv_mapper_cache[k] = mapper
        return mapper

    # ---------------- private helpers ----------------
    def _key(self, name: str, rank: int) -> str:
        return name + str(rank)

    def _tp_per_dp(self, info: RankInfo) -> int:
        return (
            info.tp_size // info.dp_size
            if getattr(info, "enable_attention_dp", False)
            else info.tp_size
        )

    def _block_size(self, ri: RankInfo, kv_factor: int) -> int:
        return (
            ri.layer_num_per_pp[ri.pp_rank]
            * kv_factor
            * ri.kv_head_num_per_rank
            * ri.tokens_per_block
            * ri.dims_per_head
            * ri.element_size
        )

    """
    ---- identity_kv_mapper ----

    Pass-through mapping. Do not change pointers or sizes.

    src_ptrs: [ S0 ] [ S1 ] [ S2 ] ...
                |      |      |
                v      v      v
    dst_ptrs: [ D0 ] [ D1 ] [ D2 ] ...
    """

    def _identity_kv_mapper(self) -> PeerMapperBase.PtrMapper:
        def kv_mapper(
            src_ptrs: List[int], src_size: int, dst_ptrs: List[int], dst_size: int
        ) -> CopyArgs:
            return src_ptrs, dst_ptrs, dst_size

        return kv_mapper

    """
    ---- write_head_match_kv_mapper ----

    Move/copy entire contiguous block(s) (multi-layer fragment) as a single chunk.
    Align by whole fragment size (frag_sz) and apply a constant source/destination block offset.

    src_ptrs:  [ S0 ]         [ S1 ]          ...
                 |              |
              + src_off      + src_off
                 |              |
          [ S0 + src_off ] [ S1 + src_off ]   ->  (each points to a frag of size frag_sz)
                   copy whole frag
                 |              |
                 v              v
          [ D0 + dst_off ] [ D1 + dst_off ]   ->  (destination frags)
    """

    def _build_head_match_kv_mapper(
        self,
        transfer_layers: int,
        kv_factor: int,
        src_layer_off: int,
        dst_layer_off: int,
        peer_ri: RankInfo,
    ) -> PeerMapperBase.PtrMapper:
        frag_sz = (
            transfer_layers
            * kv_factor
            * self.ri.kv_head_num_per_rank
            * self.ri.tokens_per_block
            * self.ri.dims_per_head
            * self.ri.element_size
        )
        src_block_off = (
            src_layer_off
            * kv_factor
            * self.ri.kv_head_num_per_rank
            * self.ri.tokens_per_block
            * self.ri.dims_per_head
            * self.ri.element_size
        )
        dst_block_off = (
            dst_layer_off
            * kv_factor
            * peer_ri.kv_head_num_per_rank
            * peer_ri.tokens_per_block
            * peer_ri.dims_per_head
            * peer_ri.element_size
        )

        @nvtx_range("head_match_kv_mapper")
        def kv_mapper(
            src_ptrs: List[int],
            src_size: int,
            dst_ptrs: List[int],
            dst_size: int,
            s_frag: int = frag_sz,
            s_off: int = src_block_off,
            d_off: int = dst_block_off,
        ) -> CopyArgs:
            s_trans = [p + s_off for p in src_ptrs]
            d_trans = [p + d_off for p in dst_ptrs]
            return s_trans, d_trans, s_frag

        return kv_mapper

    """
    ---- head_mismatch_kv_mapper ----

    Fine-grained mapping when head counts or TP/DP partitioning differ.
    Split layers into per-head (or contiguous-heads) fragments and map them individually.
    Handles kv_factor (e.g., key+value duplication) and TP/DP head offsets.

    Source (layers x heads):
    L0: [S00 S01] [S02 S03] ...
    L1: [S10 S11] [S12 S13] ...

    Destination (layers x heads, different layout possible):
    L0': [D00] [D01] [D02] ...
    L1': [D10] [D11] ...

    Mapping (each arrow = copy cont_head_frag):
    [S00 S01] -> [D00]
    [S02 S03] -> [D01]
    [S10 S11] -> [D02]
    """

    def _build_head_mismatch_kv_mapper(
        self,
        transfer_layers: int,
        kv_factor: int,
        self_tp_per_dp: int,
        peer_tp_per_dp: int,
        self_tp_rank: int,
        peer_tp_rank: int,
        src_layer_off: int,
        peer_layer_off: int,
        peer_ri: RankInfo,
    ) -> PeerMapperBase.PtrMapper:
        head_frag = self.ri.tokens_per_block * self.ri.dims_per_head * self.ri.element_size
        cont_head_frag = min(self.ri.kv_head_num_per_rank, peer_ri.kv_head_num_per_rank) * head_frag

        src_head_off = 0
        dst_head_off = 0
        if self_tp_per_dp < peer_tp_per_dp:
            ratio = peer_tp_per_dp // self_tp_per_dp
            src_head_off = (peer_tp_rank % ratio) * cont_head_frag
        elif self_tp_per_dp > peer_tp_per_dp:
            ratio = self_tp_per_dp // peer_tp_per_dp
            dst_head_off = (self_tp_rank % ratio) * cont_head_frag

        src_layer_kv_num = (
            self.ri.kv_head_num_per_rank
            * self.ri.tokens_per_block
            * self.ri.dims_per_head
            * self.ri.element_size
        )
        dst_layer_kv_num = (
            peer_ri.kv_head_num_per_rank
            * peer_ri.tokens_per_block
            * peer_ri.dims_per_head
            * peer_ri.element_size
        )
        src_layer_num = src_layer_kv_num * kv_factor
        dst_layer_num = dst_layer_kv_num * kv_factor

        @nvtx_range("head_mismatch_kv_mapper")
        def kv_mapper(
            src_ptrs: List[int],
            src_size: int,
            dst_ptrs: List[int],
            dst_size: int,
            transfer_layers: int = transfer_layers,
            src_head_off: int = src_head_off,
            src_layer_num: int = src_layer_num,
            src_layer_kv_num: int = src_layer_kv_num,
            dst_head_off: int = dst_head_off,
            dst_layer_num: int = dst_layer_num,
            dst_layer_kv_num: int = dst_layer_kv_num,
            cont_frag: int = cont_head_frag,
        ) -> CopyArgs:
            # Use numpy for vectorized computation
            src_bases = np.array(src_ptrs, dtype=np.int64)  # shape: (n_ptrs,)
            dst_bases = np.array(dst_ptrs, dtype=np.int64)  # shape: (n_ptrs,)

            # Layer indices
            layer_indices = np.arange(transfer_layers, dtype=np.int64)  # shape: (transfer_layers,)
            src_layer_indices = src_layer_off + layer_indices
            dst_layer_indices = peer_layer_off + layer_indices

            # KV indices
            kv_indices = np.arange(kv_factor, dtype=np.int64)  # shape: (kv_factor,)

            # Compute all combinations using broadcasting
            # Shape: (n_ptrs, transfer_layers, kv_factor)
            src_out = (
                src_bases[:, None, None]
                + src_layer_num * src_layer_indices[None, :, None]
                + src_layer_kv_num * kv_indices[None, None, :]
                + src_head_off
            )
            dst_out = (
                dst_bases[:, None, None]
                + dst_layer_num * dst_layer_indices[None, :, None]
                + dst_layer_kv_num * kv_indices[None, None, :]
                + dst_head_off
            )

            return src_out.ravel().tolist(), dst_out.ravel().tolist(), cont_frag

        return kv_mapper
