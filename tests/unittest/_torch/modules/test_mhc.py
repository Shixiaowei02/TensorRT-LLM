# Copied and modified from https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_mhc
import time
from collections import defaultdict

import pytest
import torch

from tensorrt_llm._torch.modules.mhc.hyper_connection import HCHead, mHC

# Global dictionary to store timing statistics
timing_stats = defaultdict(lambda: {"total_time": 0.0, "count": 0, "times": []})


def generate_pre_data(
    n: int,
    hc_mult: int,
    hidden_size: int,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 20,
) -> dict[str, torch.Tensor | float]:
    """Generate test data for big fuse operator."""
    torch.random.manual_seed(42)

    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    device = "cuda"

    residual = (
        (torch.randn((n, hc_mult, hidden_size), dtype=torch.float, device=device) / hidden_size)
        .mul(1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))
        .bfloat16()
    )

    fn = (
        torch.randn((hc_mult3, hc_mult, hidden_size), dtype=torch.float, device=device)
        * 1e-4
        * (1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))
    ).flatten(1, 2)

    hc_scale = torch.randn((3,), dtype=torch.float, device=device) * 0.1

    hc_base = torch.randn((hc_mult3,), dtype=torch.float, device=device) * 0.1

    return {
        "residual": residual,
        "fn": fn,
        "hc_scale": hc_scale,
        "hc_base": hc_base,
        "rms_eps": rms_eps,
        "hc_pre_eps": hc_pre_eps,
        "hc_sinkhorn_eps": hc_sinkhorn_eps,
        "hc_post_mult_value": hc_post_mult_value,
        "sinkhorn_repeat": sinkhorn_repeat,
    }


def generate_post_data(
    n: int,
    hidden_size: int,
    hc_mult: int,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """Generate test data for post operator."""
    torch.random.manual_seed(42)

    x = torch.randn((n, hidden_size), dtype=torch.bfloat16, device=device) / hidden_size
    residual = torch.randn((n, hc_mult, hidden_size), dtype=torch.bfloat16, device=device)
    post_layer_mix = torch.randn((n, hc_mult, 1), dtype=torch.float32, device=device)
    comb_res_mix = torch.randn((n, hc_mult, hc_mult), dtype=torch.float32, device=device)

    return {
        "x": x,
        "residual": residual,
        "post_layer_mix": post_layer_mix,
        "comb_res_mix": comb_res_mix,
    }


def generate_head_data(
    m: int,
    hidden_size: int,
    hc_mult: int,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """Generate test data for post operator."""
    torch.random.manual_seed(42)

    x = torch.randn((m, hc_mult, hidden_size), dtype=torch.bfloat16, device=device) / hidden_size
    hc_fn = torch.randn((hc_mult, hc_mult * hidden_size), dtype=torch.float32, device=device)
    hc_base = torch.randn((hc_mult,), dtype=torch.float32, device=device)
    hc_scale = torch.randn((1,), dtype=torch.float32, device=device)

    return {
        "x": x,
        "hc_fn": hc_fn,
        "hc_scale": hc_scale,
        "hc_base": hc_base,
    }


@pytest.mark.parametrize("n", [1, 32, 64, 128, 256, 512, 4096, 8192])
@pytest.mark.parametrize("hidden_size", [4096])
@pytest.mark.parametrize("hc_mult", [4])
@pytest.mark.parametrize("backend", ["cuda", "cutile", "tilelang"])
def test_mhc_pre_mapping(n: int, hidden_size: int, hc_mult: int, backend: str):
    test_data = generate_pre_data(
        n=n,
        hc_mult=hc_mult,
        hidden_size=hidden_size,
    )

    # Create vanilla reference for comparison
    ref_module = mHC(
        mult=hc_mult,
        hidden_size=hidden_size,
        sinkhorn_iters=test_data["sinkhorn_repeat"],
        dtype=None,
        eps=test_data["hc_pre_eps"],
        norm_eps=test_data["rms_eps"],
        post_mult_value=test_data["hc_post_mult_value"],
        backend="vanilla",
    ).cuda()
    ref_module.fn.copy_(test_data["fn"])
    ref_module.scale.copy_(test_data["hc_scale"])
    ref_module.base.copy_(test_data["hc_base"])

    test_module = mHC(
        mult=hc_mult,
        hidden_size=hidden_size,
        sinkhorn_iters=test_data["sinkhorn_repeat"],
        dtype=None,
        eps=test_data["hc_pre_eps"],
        norm_eps=test_data["rms_eps"],
        post_mult_value=test_data["hc_post_mult_value"],
        backend=backend,
    ).cuda()
    test_module.fn.copy_(test_data["fn"])
    test_module.scale.copy_(test_data["hc_scale"])
    test_module.base.copy_(test_data["hc_base"])

    # Warm up both vanilla and test modules
    for _ in range(50):
        ref_module.pre_mapping(test_data["residual"])
        test_module.pre_mapping(test_data["residual"])
    torch.cuda.synchronize()

    # Timing with 500 iterations
    for _ in range(500):
        start_time = time.perf_counter()
        post_mix_ref, comb_mix_ref, layer_input_ref = test_module.pre_mapping(test_data["residual"])
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        # Record timing with shape information
        test_key = f"pre_mapping_{backend}_{n}_{hidden_size}"
        timing_stats[test_key]["total_time"] += elapsed
        timing_stats[test_key]["count"] += 1
        timing_stats[test_key]["times"].append(elapsed)

    # Compare outputs with vanilla backend
    if backend != "vanilla":
        post_mix_vanilla, comb_mix_vanilla, layer_input_vanilla = ref_module.pre_mapping(
            test_data["residual"]
        )
        # cutile/deepgemm backends use tf32 MMA which has larger rounding diffs
        torch.testing.assert_close(post_mix_vanilla, post_mix_ref, rtol=1e-4, atol=1e-3)
        # comb_mix goes through Sinkhorn normalization (exp → iterated row/col
        # normalize) which amplifies small tf32 rounding differences.
        torch.testing.assert_close(comb_mix_vanilla, comb_mix_ref, rtol=1e-3, atol=5e-3)
        torch.testing.assert_close(layer_input_vanilla, layer_input_ref, rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("n", [64, 128, 4096, 8192])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize("hc_mult", [4])
@pytest.mark.parametrize("backend", ["cuda", "cutile", "tilelang"])
def test_mhc_post_mapping(n: int, hidden_size: int, hc_mult: int, backend: str):
    test_data = generate_post_data(
        n=n,
        hc_mult=hc_mult,
        hidden_size=hidden_size,
    )

    # Create vanilla reference for comparison
    ref_module = mHC(mult=hc_mult, hidden_size=hidden_size, sinkhorn_iters=10, backend="vanilla")

    test_module = mHC(mult=hc_mult, hidden_size=hidden_size, sinkhorn_iters=10, backend=backend)

    # Warm up both vanilla and test modules
    for _ in range(50):
        ref_module.post_mapping(**test_data)
        test_module.post_mapping(**test_data)
    torch.cuda.synchronize()

    # Timing with 500 iterations
    for _ in range(500):
        start_time = time.perf_counter()
        output = test_module.post_mapping(**test_data)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        # Record timing with shape information
        test_key = f"post_mapping_{backend}_{n}_{hidden_size}"
        timing_stats[test_key]["total_time"] += elapsed
        timing_stats[test_key]["count"] += 1
        timing_stats[test_key]["times"].append(elapsed)

    # Compare outputs with vanilla backend
    if backend != "vanilla":
        output_ref = ref_module.post_mapping(**test_data)
        # bf16 I/O kernels have FMA ordering differences vs vanilla fp32 path
        if backend in ("tilelang", "cuda"):
            torch.testing.assert_close(output_ref, output, rtol=1e-2, atol=0.1)
        else:
            torch.testing.assert_close(output_ref, output, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("m", [64, 128, 4096, 8192])
@pytest.mark.parametrize("hidden_size", [4096])
@pytest.mark.parametrize("hc_mult", [4])
@pytest.mark.parametrize("backend", ["cuda", "cutile", "tilelang"])
def test_hc_head(m: int, hidden_size: int, hc_mult: int, backend: str):
    test_data = generate_head_data(
        m=m,
        hc_mult=hc_mult,
        hidden_size=hidden_size,
    )

    ref_module = HCHead(mult=hc_mult, hidden_size=hidden_size, backend="vanilla").cuda()
    ref_module.fn.copy_(test_data["hc_fn"])
    ref_module.scale.copy_(test_data["hc_scale"])
    ref_module.base.copy_(test_data["hc_base"])

    test_module = HCHead(mult=hc_mult, hidden_size=hidden_size, backend=backend).cuda()
    test_module.fn.copy_(test_data["hc_fn"])
    test_module.scale.copy_(test_data["hc_scale"])
    test_module.base.copy_(test_data["hc_base"])

    # Warm up both vanilla and test modules
    for _ in range(50):
        ref_module(test_data["x"])
        test_module(test_data["x"])
    torch.cuda.synchronize()

    # Timing with 500 iterations
    for _ in range(500):
        start_time = time.perf_counter()
        output = test_module(test_data["x"])
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        # Record timing with shape information
        test_key = f"hc_head_{backend}_{m}_{hidden_size}"
        timing_stats[test_key]["total_time"] += elapsed
        timing_stats[test_key]["count"] += 1
        timing_stats[test_key]["times"].append(elapsed)

    # Compare outputs with vanilla backend
    if backend != "vanilla":
        output_ref = ref_module(test_data["x"])
        if backend in ("tilelang", "cuda"):
            torch.testing.assert_close(output_ref, output, rtol=1e-2, atol=0.1)
        else:
            torch.testing.assert_close(output_ref, output, rtol=1e-4, atol=2e-3)


@pytest.fixture(scope="session", autouse=True)
def print_timing_stats():
    """Fixture to print timing statistics at the end of test session."""
    yield

    if timing_stats:
        print("\n" + "=" * 100)
        print("Backend Performance Statistics (Time in microseconds)")
        print("=" * 100)

        # Group by test type and shape
        test_shape_groups = {}
        for key in timing_stats.keys():
            parts = key.split("_")
            # Extract test type (pre_mapping, post_mapping, hc_head) and shape params
            if "pre_mapping" in key:
                test_type = "pre_mapping"
                shape_str = f"{parts[-2]}_{parts[-1]}"
                shape_label = f"(n={parts[-2]}, hidden={parts[-1]})"
            elif "post_mapping" in key:
                test_type = "post_mapping"
                shape_str = f"{parts[-2]}_{parts[-1]}"
                shape_label = f"(n={parts[-2]}, hidden={parts[-1]})"
            elif "hc_head" in key:
                test_type = "hc_head"
                shape_str = f"{parts[-2]}_{parts[-1]}"
                shape_label = f"(m={parts[-2]}, hidden={parts[-1]})"
            else:
                continue

            backend = parts[-3]

            group_key = (test_type, shape_str, shape_label)
            if group_key not in test_shape_groups:
                test_shape_groups[group_key] = {}
            test_shape_groups[group_key][backend] = timing_stats[key]

        # Sort by test type, then by shape
        for test_type, shape_str, shape_label in sorted(test_shape_groups.keys()):
            print(f"\n{test_type.upper()} {shape_label}:")
            print("-" * 100)

            backends = test_shape_groups[(test_type, shape_str, shape_label)]
            for backend in sorted(backends.keys()):
                stats = backends[backend]
                times = sorted(stats["times"])
                n = len(times)
                lo, hi = n // 10, n - n // 10
                trimmed = times[lo:hi] if hi > lo else times

                avg_time = sum(trimmed) / len(trimmed)
                med_time = trimmed[len(trimmed) // 2]
                min_time = trimmed[0]
                max_time = trimmed[-1]

                # Convert to microseconds
                print(
                    f"  {backend:12s}: "
                    f"median={med_time * 1e6:8.1f}us  "
                    f"avg={avg_time * 1e6:8.1f}us  "
                    f"min={min_time * 1e6:8.1f}us  "
                    f"max={max_time * 1e6:8.1f}us  "
                    f"runs={n} (trimmed {len(trimmed)})"
                )

            # Speedup using median of trimmed times
            med_times = {}
            for backend in backends.keys():
                times = sorted(backends[backend]["times"])
                n = len(times)
                lo, hi = n // 10, n - n // 10
                trimmed = times[lo:hi] if hi > lo else times
                med_times[backend] = trimmed[len(trimmed) // 2]

            if len(backends) > 1:
                slowest = max(med_times.values())
                print("\n  Speedup (median, vs slowest):")
                for backend in sorted(med_times.keys()):
                    speedup = slowest / med_times[backend] if med_times[backend] > 0 else float("inf")
                    print(f"    {backend:12s}: {speedup:.2f}x")

        print("\n" + "=" * 100)


@pytest.mark.parametrize("n", [8])
@pytest.mark.parametrize("hidden_size", [4096])
@pytest.mark.parametrize("hc_mult", [4])
@pytest.mark.parametrize("backend", ["cuda", "cutile", "tilelang"])
def test_ncu_pre_mapping(n: int, hidden_size: int, hc_mult: int, backend: str):
    """Profiling-only test: warmup outside profiler range, single call inside.

    Usage:
      ncu --set full --nvtx --nvtx-include "mhc_profile/" -o ncu_pre_<backend> \
        pytest test_mhc.py -k "ncu_pre_mapping and <backend>" --no-header -rN
    """
    test_data = generate_pre_data(n=n, hc_mult=hc_mult, hidden_size=hidden_size)
    mod = mHC(
        mult=hc_mult, hidden_size=hidden_size, sinkhorn_iters=test_data["sinkhorn_repeat"],
        dtype=None, eps=test_data["hc_pre_eps"], norm_eps=test_data["rms_eps"],
        post_mult_value=test_data["hc_post_mult_value"], backend=backend,
    ).cuda()
    mod.fn.copy_(test_data["fn"])
    mod.scale.copy_(test_data["hc_scale"])
    mod.base.copy_(test_data["hc_base"])

    for _ in range(50):
        mod.pre_mapping(test_data["residual"])
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push("mhc_profile")
    mod.pre_mapping(test_data["residual"])
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    torch.manual_seed(42)
    pytest.main()
