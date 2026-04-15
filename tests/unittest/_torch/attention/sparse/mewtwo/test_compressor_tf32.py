#!/usr/bin/env python3
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for TF32 vs FP32 computation paths in the Mewtwo Compressor wkv_gate."""

import os
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.attention_backend.sparse.mewtwo.compressor import _to_float
from tensorrt_llm._torch.modules.linear import Linear

DEVICE = "cuda"
# Typical Mewtwo dimensions
DIM = 4096
HEAD_DIM = 512
STATE_DIM = 2 * HEAD_DIM  # overlap=True for compress_ratio=4
OUT_DIM = STATE_DIM * 2  # wkv + gate


def _create_wkv_gate():
    """Create an fp32 Linear layer matching the compressor's wkv_gate."""
    gate = Linear(
        DIM,
        OUT_DIM,
        bias=False,
        dtype=torch.float32,
        quant_config=None,
        skip_create_weights_in_init=False,
        use_custom_cublas_mm=True,
    ).to(DEVICE)
    gate.weight.data.normal_(0, 0.02)
    return gate


def _fp32_path(x: torch.Tensor, wkv_gate: Linear) -> torch.Tensor:
    """FP32 path: through nn.Linear (the old default)."""
    return wkv_gate(x.float())


def _tf32_path(x: torch.Tensor, wkv_gate: Linear) -> torch.Tensor:
    """TF32 path: via cublas_mm (the new default, matching DSA)."""
    return torch.ops.trtllm.cublas_mm(_to_float(x), wkv_gate.weight.t(), None, out_dtype=None)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    yield
    torch.cuda.empty_cache()


@pytest.mark.parametrize("num_tokens", [1, 16, 128, 512])
def test_tf32_path_matches_fp32(num_tokens):
    """TF32 and FP32 paths produce numerically close results.

    TF32 uses 10-bit mantissa (vs 23-bit FP32), so we allow a slightly
    relaxed tolerance while still requiring high cosine similarity.
    """
    wkv_gate = _create_wkv_gate()
    x = torch.randn(num_tokens, DIM, device=DEVICE, dtype=torch.bfloat16)

    with torch.no_grad():
        out_fp32 = _fp32_path(x, wkv_gate)
        out_tf32 = _tf32_path(x, wkv_gate)

    assert out_fp32.shape == out_tf32.shape

    # TF32 has ~10 bits of mantissa precision vs FP32's 23 bits, but the
    # inputs are bf16 (8-bit mantissa) so the practical gap is small.
    a, b = out_fp32.float().flatten(), out_tf32.float().flatten()
    cos_sim = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    assert cos_sim >= 0.999, f"cos_sim={cos_sim:.6f}"

    max_diff = (a - b).abs().max().item()
    scale = max(a.abs().max().item(), b.abs().max().item(), 1e-3)
    rel_err = max_diff / scale
    assert rel_err <= 5e-2, f"rel_err={rel_err:.6f}, max_diff={max_diff:.6f}"


@pytest.mark.parametrize("num_tokens", [1, 128])
def test_fp32_path_produces_correct_results(num_tokens):
    """FP32 path (via custom cublas Linear) is close to plain F.linear reference."""
    wkv_gate = _create_wkv_gate()
    x = torch.randn(num_tokens, DIM, device=DEVICE, dtype=torch.bfloat16)

    with torch.no_grad():
        out_fp32 = _fp32_path(x, wkv_gate)
        out_ref = F.linear(x.float(), wkv_gate.weight)

    # The custom cublas_mm-backed Linear may use a different GEMM algorithm
    # than F.linear, so we check cosine similarity rather than exact match.
    a, b = out_fp32.float().flatten(), out_ref.float().flatten()
    cos_sim = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    assert cos_sim >= 0.9999, f"cos_sim={cos_sim:.6f}"


@pytest.mark.parametrize("num_tokens", [1, 128])
def test_tf32_path_produces_correct_results(num_tokens):
    """TF32 path (via cublas_mm) is close to a plain F.linear reference."""
    wkv_gate = _create_wkv_gate()
    x = torch.randn(num_tokens, DIM, device=DEVICE, dtype=torch.bfloat16)

    with torch.no_grad():
        out_tf32 = _tf32_path(x, wkv_gate)
        out_ref = F.linear(x.float(), wkv_gate.weight)

    a, b = out_tf32.float().flatten(), out_ref.float().flatten()
    cos_sim = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    assert cos_sim >= 0.999, f"cos_sim={cos_sim:.6f}"


def test_env_var_selects_fp32_path():
    """MEWTWO_COMPRESSOR_FP32=1 env var causes module-level flag to be True."""
    # Re-import the module with the env var set to verify behavior.
    import importlib

    import tensorrt_llm._torch.attention_backend.sparse.mewtwo.compressor as mod

    with mock.patch.dict(os.environ, {"MEWTWO_COMPRESSOR_FP32": "1"}):
        importlib.reload(mod)
        assert mod._USE_FP32_COMPRESSOR is True

    # Restore default
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("MEWTWO_COMPRESSOR_FP32", None)
        importlib.reload(mod)
        assert mod._USE_FP32_COMPRESSOR is False


def test_to_float_output_dtype():
    """_to_float helper casts to float32."""
    x_bf16 = torch.randn(4, 16, device=DEVICE, dtype=torch.bfloat16)
    x_fp32 = _to_float(x_bf16)
    assert x_fp32.dtype == torch.float32
    # Values should match after cast
    torch.testing.assert_close(x_fp32, x_bf16.float())
