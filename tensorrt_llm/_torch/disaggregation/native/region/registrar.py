from typing import Dict

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.native.peer import RankInfo
from tensorrt_llm._torch.disaggregation.resource.extractor import RegionExtractorBase
from tensorrt_llm._torch.disaggregation.resource.mapper import RegionMapperBase


class PeerRegistrar:
    def __init__(
        self,
        self_rank_info: RankInfo,
    ):
        self._ri = self_rank_info
        self._peer_ri_cache: Dict[str, RankInfo] = {}
        self._peer_region_map_cache: Dict[str, RegionMapperBase] = {}
        self._peer_ext_cache: Dict[str, RegionExtractorBase] = {}

    # ---------------- public simple APIs ----------------
    def register(self, peer_name: str, peer_rank: int, peer_ri: RankInfo):
        # TODO: check if peer is valid for registration
        if not self._check_peer_compatible(peer_ri):
            raise ValueError(
                f"PeerRegistrar.register: peer {peer_name} (rank={peer_rank}) is incompatible with local rank."
            )
        self._peer_ri_cache[self._unique_key(peer_name, peer_rank)] = peer_ri

    def unregister(self, peer_name: str, peer_rank: int):
        key = self._unique_key(peer_name, peer_rank)
        if key in self._peer_ri_cache:
            del self._peer_ri_cache[key]

    def get_peer_rank_info(self, peer_name: str, peer_rank: int):
        return self._peer_ri_cache[self._unique_key(peer_name, peer_rank)]

    @property
    def self_rank_info(self) -> RankInfo:
        return self._ri

    def _unique_key(self, name: str, rank: int) -> str:
        return name + str(rank)

    def _check_peer_compatible(self, peer_ri: RankInfo) -> bool:
        if self._ri.is_mla != peer_ri.is_mla:
            logger.warning(
                "PeerRegistrar: compatibility check failed: 'is_mla' differs "
                f"(local={self._ri.is_mla}, peer={peer_ri.is_mla})."
            )
            return False
        if self._ri.cp_size != 1 or peer_ri.cp_size != 1:
            logger.warning(
                "PeerRegistrar: unsupported configuration: context parallelism (cp_size) "
                f"must be 1 for both local and peer ranks (local={self._ri.cp_size}, peer={peer_ri.cp_size})."
            )
            return False
        if self._ri.element_bytes != peer_ri.element_bytes:
            logger.warning(
                "PeerRegistrar: element size mismatch "
                f"(local={self._ri.element_bytes} bytes, peer={peer_ri.element_bytes} bytes)."
            )
            return False
        if self._ri.tokens_per_block != peer_ri.tokens_per_block:
            logger.warning(
                "PeerRegistrar: tokens_per_block mismatch "
                f"(local={self._ri.tokens_per_block}, peer={peer_ri.tokens_per_block})."
            )
            return False
        if self._ri.dims_per_head != peer_ri.dims_per_head:
            logger.warning(
                "PeerRegistrar: dims_per_head mismatch "
                f"(local={self._ri.dims_per_head}, peer={peer_ri.dims_per_head})."
            )
            return False

        self_layers = sum(self._ri.layer_num_per_pp)
        peer_layers = sum(peer_ri.layer_num_per_pp)
        if self_layers != peer_layers:
            logger.warning(
                "PeerRegistrar: total layer count mismatch "
                f"(local={self_layers}, peer={peer_layers})."
            )
            return False

        if self._ri.is_mla:
            if peer_ri.kv_heads_per_rank != 1 or self._ri.kv_heads_per_rank != 1:
                logger.warning(
                    "PeerRegistrar: MLA mode requires exactly 1 KV head per rank for both local and peer."
                    f" (local={self._ri.kv_heads_per_rank}, peer={peer_ri.kv_heads_per_rank})"
                )
                return False
        return True

    def get_kv_map(self, peer_ri: RankInfo):
        key = self._unique_key(peer_ri.instance_name, peer_ri.instance_rank)
        if key in self._kv_map_cache:
            return self._kv_map_cache[key]

        kv_factor = 1 if self._ri.is_mla else 2
        self_tp_per_dp = self._tp_per_dp(self._ri)
        peer_tp_per_dp = self._tp_per_dp(peer_ri)
        self_tp_rank = self._ri.tp_rank % self_tp_per_dp
        peer_tp_rank = peer_ri.tp_rank % peer_tp_per_dp

        # head_num_per_rank = 1 when is_dup_head
        is_dup_head = (
            self._ri.kv_heads_per_rank * self_tp_per_dp
            != peer_ri.kv_heads_per_rank * peer_tp_per_dp
        )
        head_match = is_dup_head or self._ri.is_mla or self_tp_per_dp == peer_tp_per_dp
        logger.debug(
            "KVMapperFactory.get_kv_map: "
            f"head_match={head_match}, is_dup_head={is_dup_head}, self_is_mla={self._ri.is_mla}, "
            f"self_tp_per_dp={self_tp_per_dp}, peer_tp_per_dp={peer_tp_per_dp}"
        )
        # fast identity when write_all and same pp_size
        if head_match and self._ri.pp_size == peer_ri.pp_size:
            mapper = self._kv_map_identity()
            self._kv_map_cache[key] = mapper
            return mapper

        # compute overlapping layers
        self_start_layer = sum(self._ri.layer_num_per_pp[: self._ri.pp_rank])
        self_end_layer = self_start_layer + self._ri.layer_num_per_pp[self._ri.pp_rank]
        peer_start_layer = sum(peer_ri.layer_num_per_pp[: peer_ri.pp_rank])
        peer_end_layer = peer_start_layer + peer_ri.layer_num_per_pp[peer_ri.pp_rank]
        start = max(self_start_layer, peer_start_layer)
        end = min(self_end_layer, peer_end_layer)
        transfer_layers = end - start
        self_layer_offset = start - self_start_layer
        peer_layer_offset = start - peer_start_layer

        if head_match:
            mapper = self._kv_map_head_match(
                transfer_layers, kv_factor, self_layer_offset, peer_layer_offset, peer_ri
            )
            self._kv_map_cache[key] = mapper
            return mapper

        # head mismatch case
        mapper = self._kv_map_head_mismatch(
            transfer_layers,
            kv_factor,
            self_tp_per_dp,
            peer_tp_per_dp,
            self_tp_rank,
            peer_tp_rank,
            self_layer_offset,
            peer_layer_offset,
            peer_ri,
        )
        self._kv_map_cache[key] = mapper
        return mapper

    # ---------------- private helpers ----------------
    def _unique_key(self, name: str, rank: int) -> str:
        return name + str(rank)

    def _tp_per_dp(self, info: RankInfo) -> int:
        return (
            info.tp_size // info.dp_size
            if getattr(info, "enable_attention_dp", False)
            else info.tp_size
        )
