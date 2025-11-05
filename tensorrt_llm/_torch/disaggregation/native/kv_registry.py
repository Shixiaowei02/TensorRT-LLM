from dataclasses import dataclass
from typing import List


@dataclass
class WorkerDesc:
    instance_rank: int
    tp_rank: int
    pp_rank: int
    dp_rank: int
    cp_rank: int
    #  [numLayers,kv_factor,heads,tokens,dimsPerHead]
    dims_per_head: int
    enable_attention_dp: bool
    kvcache_ptrs: List[int]
    aux_ptrs: List[int]
    server_endpoint: str
    recv_endpoint: str
    transfer_engine_info: bytes


@dataclass
class ExecutorDesc:
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


@dataclass
class DomainPeerRanks:
    domain_pp_size: int
    domain_tp_size: int
    domain_cp_size: int
    duplicate_head_factor: int
    peer_duplicate_head_factor: int
    target_peer_pp_layer_num: List[int]
    ranks: List[int]


class KVRegistry:
    def __init__(self, exec_desc: WorkerDesc, instance_info: ExecutorDesc):
        self.exec_desc = exec_desc
        self.instance_info = instance_info
        self.peer_exec_cache = {}
        self.peer_domain_ranks_cache = {}

    def register_peer(self, peer_instance_name: str, peer_rank: int, peer_exec: WorkerDesc):
        self.peer_exec_cache[peer_instance_name + str(peer_rank)] = peer_exec

    def unregister_peer(self, peer_instance_name: str, peer_rank: int):
        del self.peer_exec_cache[peer_instance_name + str(peer_rank)]

    def get_peer_exec(self, peer_instance_name: str, peer_rank: int):
        return self.peer_exec_cache[peer_instance_name + str(peer_rank)]

    def get_instance_info(self):
        return self.instance_info

    def get_exec_desc(self):
        return self.exec_desc

    def get_peer_ranks(self, peer_instance_info: ExecutorDesc, peer_dp_rank: int):
        # return the mapping rank relationship between self and peer in asymmetric parallelism case

        if peer_instance_info.instance_name + str(peer_dp_rank) in self.peer_domain_ranks_cache:
            return self.peer_domain_ranks_cache[
                peer_instance_info.instance_name + str(peer_dp_rank)
            ]
        peer_pp_rank_start = 0
        peer_pp_rank_end = 0
        self_start_layer_id = sum(self.exec_desc.layer_num_per_pp[: self.exec_desc.pp_rank])
        self_end_layer_id = (
            self_start_layer_id + self.exec_desc.layer_num_per_pp[self.exec_desc.pp_rank]
        )
        pre_peer_pp_layer_id = 0
        target_peer_pp_ranks = []
        target_peer_pp_layer_num = []
        for pp_rank in range(peer_instance_info.pp_size):
            peer_pp_start_layer_id = pre_peer_pp_layer_id
            peer_pp_end_layer_id = (
                peer_pp_start_layer_id + peer_instance_info.layer_num_per_pp[pp_rank]
            )
            if (
                self_start_layer_id < peer_pp_end_layer_id
                and self_end_layer_id > peer_pp_start_layer_id
            ):
                target_peer_pp_ranks.append(pp_rank)
                target_peer_pp_layer_num.append(
                    min(peer_pp_end_layer_id, self_end_layer_id)
                    - max(peer_pp_start_layer_id, self_start_layer_id)
                )
            pre_peer_pp_layer_id += peer_instance_info.layer_num_per_pp[pp_rank]

        peer_pp_rank_start = target_peer_pp_ranks[0]
        domain_pp_size = len(target_peer_pp_ranks)
        peer_pp_rank_end = peer_pp_rank_start + len(target_peer_pp_ranks)

        # dp
        self_tp_size_per_dp_group = (
            self.exec_desc.tp_size / self.exec_desc.dp_size
            if self.exec_desc.enable_attention_dp
            else self.exec_desc.tp_size
        )
        peer_tp_size_per_dp_group = (
            peer_instance_info.tp_size / peer_instance_info.dp_size
            if peer_instance_info.enable_attention_dp
            else peer_instance_info.tp_size
        )
        self_tprank_in_dp_group = self.exec_desc.tp_rank % self_tp_size_per_dp_group
        peer_tp_rank_start = 0
        peer_tp_rank_end = 0
        domain_tp_size = 1
        if self_tp_size_per_dp_group <= peer_tp_size_per_dp_group:
            domain_tp_size = peer_tp_size_per_dp_group // self_tp_size_per_dp_group
            peer_tp_rank_start = (
                self_tprank_in_dp_group * domain_tp_size + peer_dp_rank * peer_tp_size_per_dp_group
            )
            peer_tp_rank_end = peer_tp_rank_start + domain_tp_size
        else:
            peer_tp_rank_start = (
                self_tprank_in_dp_group // (self_tp_size_per_dp_group // peer_tp_size_per_dp_group)
                + peer_dp_rank * peer_tp_size_per_dp_group
            )
            peer_tp_rank_end = peer_tp_rank_start + domain_tp_size
        # peer_tp_rank_start = self_tprank_in_dp_group * domain_tp_size + peer_dp_rank * peer_tp_size_per_dp_group
        # peer_tp_rank_end = peer_tp_rank_start + domain_tp_size
        peer_cp_rank_start = 0
        peer_cp_rank_end = 0
        domain_cp_size = 1
        if self.exec_desc.cp_size <= peer_instance_info.cp_size:
            domain_cp_size = peer_instance_info.cp_size // self.exec_desc.cp_size
            peer_cp_rank_start = self.exec_desc.cp_rank * domain_cp_size
            peer_cp_rank_end = peer_cp_rank_start + domain_cp_size
        else:
            peer_cp_rank_start = self.exec_desc.cp_rank // (
                self.exec_desc.cp_size // peer_instance_info.cp_size
            )
            peer_cp_rank_end = peer_cp_rank_start + domain_cp_size
        ranks = []
        for pp_rank in range(peer_pp_rank_start, peer_pp_rank_end):
            for cp_rank in range(peer_cp_rank_start, peer_cp_rank_end):
                for tp_rank in range(peer_tp_rank_start, peer_tp_rank_end):
                    ranks.append(
                        pp_rank * peer_instance_info.tp_size * peer_instance_info.cp_size
                        + cp_rank * peer_instance_info.tp_size
                        + tp_rank
                    )
        duplicate_head_factor = max(
            1,
            self.exec_desc.kv_head_num_per_rank
            * self_tp_size_per_dp_group
            // (peer_instance_info.kv_head_num_per_rank * peer_tp_size_per_dp_group),
        )
        peer_duplicate_head_factor = max(
            1,
            peer_instance_info.kv_head_num_per_rank
            * peer_tp_size_per_dp_group
            // (self.exec_desc.kv_head_num_per_rank * self_tp_size_per_dp_group),
        )

        peer_domain_ranks = DomainPeerRanks(
            domain_pp_size=domain_pp_size,
            domain_tp_size=domain_tp_size,
            domain_cp_size=domain_cp_size,
            duplicate_head_factor=duplicate_head_factor,
            peer_duplicate_head_factor=peer_duplicate_head_factor,
            target_peer_pp_layer_num=target_peer_pp_layer_num,
            ranks=ranks,
        )
        self.peer_domain_ranks_cache[peer_instance_info.instance_name + str(peer_dp_rank)] = (
            peer_domain_ranks
        )
        return peer_domain_ranks

    def get_kv_block_ptrs_extractor(self, peer_exec: WorkerDesc):
        # return the kv block ptrs extractor for the peer. the extractor will be used
        # to extract the kv block ptrs and submit to transfer agent.
        pass

        return
