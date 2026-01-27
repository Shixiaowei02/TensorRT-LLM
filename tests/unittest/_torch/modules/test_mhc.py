# Copied and modified from https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_mhc
import pytest
import torch

from tensorrt_llm._torch.modules.hyper_connection import HCHead, mHC


def generate_pre_data(
    n: int,
    hc_mult: int,
    hidden_size: int,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 10,
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


@pytest.mark.parametrize("n", [512, 1024, 2048, 8192])
@pytest.mark.parametrize("hidden_size", [1280, 2560, 4096])
@pytest.mark.parametrize("hc_mult", [4])
@pytest.mark.parametrize("backend", ["deepgemm", "tilelang"])
def test_mhc_pre_mapping(n: int, hidden_size: int, hc_mult: int, backend: str):
    test_data = generate_pre_data(
        n=n,
        hc_mult=hc_mult,
        hidden_size=hidden_size,
    )

    ref_module = mHC(
        mult=hc_mult,
        hidden_size=hidden_size,
        sinkhorn_iters=test_data["sinkhorn_repeat"],
        dtype=None,
        eps=test_data["hc_pre_eps"],
        norm_eps=test_data["rms_eps"],
        post_mult_value=test_data["hc_post_mult_value"],
        backend="vanilla",
    )
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
    )
    test_module.fn.copy_(test_data["fn"])
    test_module.scale.copy_(test_data["hc_scale"])
    test_module.base.copy_(test_data["hc_base"])

    # Test on pre-mapping
    post_mix_fused, comb_mix_fused, layer_input_fused = ref_module.pre_mapping(
        test_data["residual"]
    )
    post_mix_ref, comb_mix_ref, layer_input_ref = test_module.pre_mapping(test_data["residual"])

    # Compare outputs
    torch.testing.assert_close(post_mix_fused, post_mix_ref)
    torch.testing.assert_close(comb_mix_fused, comb_mix_ref)
    torch.testing.assert_close(layer_input_fused, layer_input_ref)


@pytest.mark.parametrize("n", [4096])
@pytest.mark.parametrize("hidden_size", [1280, 2560, 7168])
@pytest.mark.parametrize("hc_mult", [4])
@pytest.mark.parametrize("backend", ["deepgemm", "tilelang"])
def test_mhc_post_mapping(n: int, hidden_size: int, hc_mult: int, backend: str):
    test_data = generate_post_data(
        n=n,
        hc_mult=hc_mult,
        hidden_size=hidden_size,
    )

    ref_module = mHC(mult=hc_mult, hidden_size=hidden_size, sinkhorn_iters=10, backend="vanilla")

    test_module = mHC(mult=hc_mult, hidden_size=hidden_size, sinkhorn_iters=10, backend=backend)

    # Test on pre-mapping
    output_ref = ref_module.post_mapping(**test_data)
    output = test_module.post_mapping(**test_data)

    # Compare outputs
    torch.testing.assert_close(output_ref, output)


@pytest.mark.parametrize("m", [1024, 4096])
@pytest.mark.parametrize("hidden_size", [2560, 4096])
@pytest.mark.parametrize("hc_mult", [4])
@pytest.mark.parametrize("backend", ["tilelang"])
def test_hc_head(m: int, hidden_size: int, hc_mult: int, backend: str):
    test_data = generate_head_data(
        m=m,
        hc_mult=hc_mult,
        hidden_size=hidden_size,
    )

    ref_module = HCHead(mult=hc_mult, hidden_size=hidden_size, backend="vanilla")
    ref_module.fn.copy_(test_data["hc_fn"])
    ref_module.scale.copy_(test_data["hc_scale"])
    ref_module.base.copy_(test_data["hc_base"])

    test_module = HCHead(mult=hc_mult, hidden_size=hidden_size, backend=backend)
    test_module.fn.copy_(test_data["hc_fn"])
    test_module.scale.copy_(test_data["hc_scale"])
    test_module.base.copy_(test_data["hc_base"])

    # Test on pre-mapping
    output_ref = ref_module(test_data["x"])
    output = test_module(test_data["x"])

    # TileLang backend may have larger numerical differences due to bf16 GEMM
    torch.testing.assert_close(output_ref, output, rtol=1e-4, atol=1e-3)


if __name__ == "__main__":
    torch.manual_seed(42)
    pytest.main()
