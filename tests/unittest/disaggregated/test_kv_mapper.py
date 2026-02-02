import pytest


class DummyMemoryRegion:
    def __init__(self, ptr, bytes):
        self.ptr = ptr
        self.bytes = bytes


class DummyRegion:
    def __init__(self, memory, region):
        self.memory = memory
        self.region = region


class DummyKVRegionSpec:
    def __init__(self, layers, heads, tokens, role=None):
        self.layers = layers
        self.heads = heads
        self.tokens = tokens
        self.role = role


class DummyNonNegRange:
    def __init__(self, start, end):
        self.start = start
        self.end = end


def calculate_region_overlap(src_region, dst_region) -> bool:
    """Determine if a source region and a destination region overlap.

    Args:
        src_region: Source region.
        dst_region: Destination region.

    Returns:
        True if regions overlap, False otherwise.
    """
    src_layers = src_region.region.layers
    src_heads = src_region.region.heads
    src_tokens = src_region.region.tokens

    dst_layers = dst_region.region.layers
    dst_heads = dst_region.region.heads
    dst_tokens = dst_region.region.tokens

    layer_overlap = not (src_layers.end < dst_layers.start or dst_layers.end < src_layers.start)
    head_overlap = not (src_heads.end < dst_heads.start or dst_heads.end < src_heads.start)
    token_overlap = not (src_tokens.end < dst_tokens.start or dst_tokens.end < src_tokens.start)

    return layer_overlap and head_overlap and token_overlap


@pytest.mark.parametrize("total_layers", [2])  # Total number of layers
@pytest.mark.parametrize("total_heads", [2])  # Number of total attention heads
@pytest.mark.parametrize("src_pp_size", [1])  # Pipeline parallel size for source
@pytest.mark.parametrize("src_tp_size", [2])  # Tensor parallel size for source
@pytest.mark.parametrize("dst_pp_size", [2])  # Pipeline parallel size for destination
@pytest.mark.parametrize("dst_tp_size", [1])  # Tensor parallel size for destination
@pytest.mark.parametrize("tokens_per_block", [32])  # Tokens per block
@pytest.mark.parametrize("total_tokens", [32])  # Total tokens per sequence
@pytest.mark.parametrize("print_log", [False])  # Whether to print logs
def test_kv_mapper_with_dynamic_dst_mapping(
    total_layers,
    total_heads,
    src_pp_size,
    src_tp_size,
    dst_pp_size,
    dst_tp_size,
    tokens_per_block,
    total_tokens,
    print_log,
):
    """Test KVMapper by dynamically generating filtered dst regions per src rank."""
    # Validate divisibility
    if total_layers % src_pp_size != 0:
        pytest.skip("total_layers not divisible by src_pp_size")
    if total_layers % dst_pp_size != 0:
        pytest.skip("total_layers not divisible by dst_pp_size")
    if total_heads % src_tp_size != 0:
        pytest.skip("total_heads not divisible by src_tp_size")
    if total_heads % dst_tp_size != 0:
        pytest.skip("total_heads not divisible by dst_tp_size")
    if total_tokens % tokens_per_block != 0:
        pytest.skip("total_tokens not divisible by tokens_per_block")

    num_token_blocks = total_tokens // tokens_per_block  # Number of token blocks
    region_ptr_base = 0
    element_bytes = 2
    kv_factor = 2

    # Generate src_regions and test for each src rank
    for pp_rank in range(src_pp_size):
        for tp_rank in range(src_tp_size):
            src_regions = []
            region_ptr = region_ptr_base

            # Generate src_regions based on pp_rank and tp_rank
            l0 = pp_rank * (total_layers // src_pp_size)
            l1 = l0 + (total_layers // src_pp_size) - 1
            h0 = tp_rank * (total_heads // src_tp_size)
            h1 = h0 + (total_heads // src_tp_size) - 1

            for block in range(num_token_blocks):
                t0 = block * tokens_per_block
                t1 = t0 + tokens_per_block - 1

                spec = DummyKVRegionSpec(
                    DummyNonNegRange(l0, l1),
                    DummyNonNegRange(h0, h1),
                    DummyNonNegRange(t0, t1),
                    role="key|value",
                )

                region_bytes = (
                    (l1 - l0 + 1) * kv_factor * (h1 - h0 + 1) * (t1 - t0 + 1) * element_bytes
                )
                mem = DummyMemoryRegion(region_ptr, region_bytes)
                src_regions.append(DummyRegion(memory=mem, region=spec))
                region_ptr += region_bytes

            # Validate src regions count
            assert len(src_regions) == num_token_blocks, (
                f"Error in src_regions count for pp_rank={pp_rank}, tp_rank={tp_rank}: "
                f"Got {len(src_regions)}, Expected: {num_token_blocks}"
            )

            # Dynamically generate filtered dst_regions for the current src_regions
            dst_regions_dynamic = []
            region_ptr = 1000000
            for dst_pp_rank in range(dst_pp_size):
                for dst_tp_rank in range(dst_tp_size):
                    dst_l0 = dst_pp_rank * (total_layers // dst_pp_size)
                    dst_l1 = dst_l0 + (total_layers // dst_pp_size) - 1
                    dst_h0 = dst_tp_rank * (total_heads // dst_tp_size)
                    dst_h1 = dst_h0 + (total_heads // dst_tp_size) - 1

                    for block in range(num_token_blocks):
                        dt0 = block * tokens_per_block
                        dt1 = dt0 + tokens_per_block - 1

                        spec = DummyKVRegionSpec(
                            DummyNonNegRange(dst_l0, dst_l1),
                            DummyNonNegRange(dst_h0, dst_h1),
                            DummyNonNegRange(dt0, dt1),
                            role="key|value",
                        )

                        mem = DummyMemoryRegion(
                            region_ptr,
                            (dst_l1 - dst_l0 + 1)
                            * kv_factor
                            * (dst_h1 - dst_h0 + 1)
                            * (dt1 - dt0 + 1)
                            * element_bytes,
                        )
                        candidate_region = DummyRegion(memory=mem, region=spec)

                        # Only include dst_regions that overlap with src_regions
                        if any(
                            calculate_region_overlap(src, candidate_region) for src in src_regions
                        ):
                            dst_regions_dynamic.append(candidate_region)

                        region_ptr += mem.bytes

            if not dst_regions_dynamic:
                if print_log:
                    print(
                        f"Skipping src rank pp_rank={pp_rank}, tp_rank={tp_rank} as no overlapping dst regions found."
                    )
                continue

            # Initialize KVMapper and map regions
            mapper = None  # Replace with actual KVMapper initialization
            out_pairs = mapper.map(src_regions, dst_regions_dynamic)

            # Calculate expected mapping pairs
            expected_pairs = sum(
                calculate_region_overlap(src, dst)
                for src in src_regions
                for dst in dst_regions_dynamic
            )

            if print_log:
                print("\n" + "=" * 60)
                print(f"Testing src pp_rank={pp_rank}, tp_rank={tp_rank}")
                print(
                    f"Src Regions: {len(src_regions)}, Filtered Dst Regions: {len(dst_regions_dynamic)}"
                )
                print(f"Output Mapped RegionPairs: {len(out_pairs)} (Expected: {expected_pairs})")
                for i, pair in enumerate(out_pairs):
                    print(f"Pair {i + 1}: Src {pair.src} -> Dst {pair.dst}")

            # Validate output pair count
            assert len(out_pairs) == expected_pairs, (
                f"Error in mapped pairs count for pp_rank={pp_rank}, tp_rank={tp_rank}: "
                f"Got {len(out_pairs)}, Expected: {expected_pairs}."
            )
