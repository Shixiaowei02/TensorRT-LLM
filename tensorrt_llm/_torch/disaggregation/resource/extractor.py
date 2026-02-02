from dataclasses import dataclass
from typing import List

from tensorrt_llm._torch.disaggregation.base.kv_extractor import (
    DataLayout,
    DataRole,
    KVRegionSpec,
    MemoryRegion,
    NonNegRange,
    Region,
    RegionExtractorBase,
)
from tensorrt_llm._torch.disaggregation.native.kv_mapper_ import RankInfo
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import get_size_in_bytes


@dataclass
class KVPoolAttrs:
    """Attributes for a single (primary) KV memory pool."""

    pool_ptrs: List[int]
    block_bytes: List[int]


class KVRegionExtractorV1(RegionExtractorBase):
    """
    Descriptor and region extractor for KV cache pool managed by KVCacheManager.
    Provides region descriptors for adapting block-wise view.
    """

    def __init__(self, kv_cache_manager: KVCacheManager, rank_info: RankInfo):
        self._kv_pool_attrs = self._attrs_from_manager(kv_cache_manager)
        self._rank_info = rank_info
        self._data_layout = DataLayout.HND

    @staticmethod
    def _attrs_from_manager(manager: KVCacheManager) -> KVPoolAttrs:
        try:
            pools = manager.get_unique_primary_pool()
        except Exception as ex:
            raise ValueError(f"Failed to get pool(s): {ex}")

        pool_list = list(pools) if isinstance(pools, (list, tuple)) else [pools]
        elem_bytes = get_size_in_bytes(1, manager.dtype)
        ptrs, block_sizes = [], []

        for p in pool_list:
            if hasattr(p, "data_ptr") and callable(p.data_ptr):
                try:
                    ptr = int(p.data_ptr())
                except Exception as ex:
                    raise ValueError(f"Fail to call data_ptr(): {ex}")
            elif isinstance(p, int):
                ptr = int(p)
            else:
                raise ValueError(f"Pool object lacks 'data_ptr' and is not int: {p!r}")
            ptrs.append(ptr)

            try:
                if hasattr(p, "__getitem__") and hasattr(p[0], "numel"):
                    n = int(p[0].numel())
                elif hasattr(p, "numel") and callable(p.numel):
                    n = int(p.numel())
                else:
                    raise RuntimeError("Cannot determine element count")
            except Exception as ex:
                raise ValueError(f"Failed to get block size from {p!r}: {ex}")

            block_sizes.append(n * elem_bytes)

        return KVPoolAttrs(pool_ptrs=ptrs, block_bytes=block_sizes)

    def extract(self, region_ids: List[int]) -> List[Region]:
        """
        Given a list of region_ids, returns a list of Region,
        each describing the memory region and corresponding cache specification
        for that block.
        """
        assert len(self._kv_pool_attrs.pool_ptrs) == 1
        pool_idx = 0
        attrs = self._kv_pool_attrs
        rank = self._rank_info

        layer_start = rank.pp_rank * rank.layer_num_per_pp
        layer_range = NonNegRange(layer_start, layer_start + rank.layer_num_per_pp - 1)
        head_start = rank.tp_rank * rank.kv_heads_per_rank
        head_range = NonNegRange(head_start, head_start + rank.kv_heads_per_rank - 1)

        descriptors = []
        for k, block_id in enumerate(region_ids):
            token_start = k * rank.tokens_per_block
            token_range = NonNegRange(token_start, token_start + rank.tokens_per_block - 1)
            region = KVRegionSpec(
                layers=layer_range,
                heads=head_range,
                tokens=token_range,
                role=DataRole.KEY | DataRole.VALUE,
            )
            data_desc = Region(
                memory=MemoryRegion(
                    ptr=attrs.pool_ptrs[pool_idx] + block_id * attrs.block_bytes[0],
                    size=attrs.block_bytes[0],
                ),
                region=region,
            )
            descriptors.append(data_desc)

        return descriptors
