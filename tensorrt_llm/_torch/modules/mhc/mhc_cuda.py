"""MHC CUDA kernels — torch.ops.trtllm interface.

Kernels (cpp/tensorrt_llm/kernels/mhcKernels/):
  - mhc_big_fuse:        fused RMS + sigmoid + Sinkhorn + pre_apply_mix
                         (NUM_SPLITS=1: normal, =16: with split-K reduction)
  - mhc_gemm_sqrsum_fma: FP32 FMA GEMM + sqrsum (fused, inline PTX)
  - mhc_hc_head_apply:   RMS norm + sigmoid + weighted sum
  - mhc_post_mapping:    out = post * x + comb.T @ residual
DeepGEMM wrapper:
  - gemm_rms_dg:         TF32 GEMM + sqrsum via DeepGEMM (optional split-K)

Backend auto-selection for pre_mapping (profiled on B200, 148 SMs):
  M <= 64:          FMA              + big_fuse<1>
  64 < M <= 1024:   DG split-K 16   + big_fuse<16>
  M > 1024:         DG (no split)   + big_fuse<1>
  Falls back to FMA when DeepGEMM is unavailable.
"""

import torch

_DG_NUM_SPLITS = 16
_DG_SPLITK_M_THRESHOLD = 64
_DG_NOSPLIT_M_THRESHOLD = 1024


# ---------------------------------------------------------------------------
# Python API — low-level (kernel-level interfaces)
# ---------------------------------------------------------------------------


def mhc_big_fuse_cuda(
    y_acc: torch.Tensor,
    r_acc: torch.Tensor,
    residual: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    layer_input: torch.Tensor,
    M: int,
    K: int,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    num_splits: int = 1,
):
    torch.ops.trtllm.mhc_big_fuse(
        y_acc, r_acc, residual, hc_scale, hc_base,
        post_mix, comb_mix, layer_input,
        M, K, hidden_size,
        rms_eps, hc_pre_eps, hc_sinkhorn_eps,
        hc_post_mult_value, sinkhorn_repeat,
        num_splits,
    )


_dg_fn_cache = None
_dg_available = None


def _get_dg_fn():
    """Import tf32_hc_prenorm_gemm (standalone deep_gemm or bundled).

    Returns the function, or None if DeepGEMM is unavailable.
    """
    global _dg_fn_cache, _dg_available
    if _dg_available is not None:
        return _dg_fn_cache
    try:
        from deep_gemm import tf32_hc_prenorm_gemm
        _dg_fn_cache = tf32_hc_prenorm_gemm
    except ImportError:
        try:
            from tensorrt_llm.deep_gemm import tf32_hc_prenorm_gemm
            _dg_fn_cache = tf32_hc_prenorm_gemm
        except ImportError:
            _dg_fn_cache = None
    _dg_available = _dg_fn_cache is not None
    return _dg_fn_cache


def mhc_gemm_rms_dg_cuda(
    x: torch.Tensor,
    w_nk: torch.Tensor,
    M: int,
    N: int,
    K: int,
    num_splits: int = 16,
):
    """DeepGEMM TF32 GEMM + sqrsum with optional split-K.

    Args:
        x:     [M, K] bfloat16 input
        w_nk:  [N, K] float32 weight (K-major, as required by DeepGEMM)
        num_splits: 1 for no split-K, >1 for split-K

    Returns (y_acc, r_acc, num_splits):
        num_splits == 1: y_acc [M, N] fp32, r_acc [M] fp32
        num_splits > 1:  y_acc [num_splits, M, N] fp32, r_acc [num_splits, M] fp32
    """
    dg_fn = _get_dg_fn()
    assert dg_fn is not None, "DeepGEMM is not available"
    x = x.contiguous()
    w_nk = w_nk.contiguous()

    if num_splits <= 1:
        y_acc = torch.empty((M, N), dtype=torch.float32, device=x.device)
        r_acc = torch.empty((M,), dtype=torch.float32, device=x.device)
        dg_fn(x, w_nk, y_acc, r_acc)
        return y_acc, r_acc, 1
    else:
        y_acc = torch.empty((num_splits, M, N), dtype=torch.float32, device=x.device)
        r_acc = torch.empty((num_splits, M), dtype=torch.float32, device=x.device)
        dg_fn(x, w_nk, y_acc, r_acc, num_splits=num_splits)
        return y_acc, r_acc, num_splits


def mhc_gemm_rms_fma_cuda(
    x: torch.Tensor,
    w: torch.Tensor | None,
    M: int,
    N: int,
    K: int,
    w_t: torch.Tensor | None = None,
):
    """Split-N FP32 FMA fused GEMM + sqrsum on CUDA cores (no tensor cores).

    Returns (y_acc [M, N] fp32, r_acc [M] fp32).
    """
    x = x.contiguous()
    if w_t is None:
        w_t = w.t().contiguous()

    num_k_blocks = 1
    k_chunk = K

    y_acc = torch.empty((M, N), dtype=torch.float32, device=x.device)
    r_acc = torch.empty((M,), dtype=torch.float32, device=x.device)

    torch.ops.trtllm.mhc_gemm_sqrsum_fma(
        x, w_t, y_acc, r_acc,
        M, N, K, num_k_blocks, k_chunk,
    )

    return y_acc, r_acc


# ---------------------------------------------------------------------------
# Python API — high-level (drop-in for mhc_pre_mapping_fused)
# ---------------------------------------------------------------------------


def mhc_pre_mapping_fused(
    x: torch.Tensor,
    w_t: torch.Tensor,
    residual: torch.Tensor,
    n: int,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
):
    """Full pre-mapping pipeline: GEMM+sqrsum -> big_fuse.

    Auto-selects the fastest GEMM backend based on M:
      M <= 64:          FMA CUDA-core GEMM  + big_fuse<1>
      64 < M <= 1024:   DeepGEMM split-K 16 + big_fuse<16>  (fused reduction)
      M > 1024:         DeepGEMM no-split   + big_fuse<1>
    Falls back to FMA when DeepGEMM is unavailable.

    Args:
        w_t: [N, K] float32 weight (row-major, pre-transposed).
    """
    residual = residual.contiguous()
    hc_scale = hc_scale.to(torch.float32).contiguous()
    hc_base = hc_base.to(torch.float32).contiguous()

    M, K = x.shape
    N = w_t.shape[0]
    n2 = n * n

    num_splits = 1
    if M <= _DG_SPLITK_M_THRESHOLD or _get_dg_fn() is None:
        y_acc, r_acc = mhc_gemm_rms_fma_cuda(x, None, M, N, K, w_t=w_t)
    elif M <= _DG_NOSPLIT_M_THRESHOLD:
        y_acc, r_acc, num_splits = mhc_gemm_rms_dg_cuda(
            x, w_t, M, N, K, num_splits=_DG_NUM_SPLITS,
        )
    else:
        y_acc, r_acc, num_splits = mhc_gemm_rms_dg_cuda(
            x, w_t, M, N, K, num_splits=1,
        )

    residual_3d = residual.view(M, n, hidden_size)

    post_mix = torch.empty((M, n), dtype=torch.float32, device=x.device)
    comb_mix = torch.empty((M, n2), dtype=torch.float32, device=x.device)
    layer_input = torch.empty(
        (M, hidden_size), dtype=torch.bfloat16, device=x.device
    )

    mhc_big_fuse_cuda(
        y_acc.contiguous(),
        r_acc.contiguous(),
        residual_3d.contiguous(),
        hc_scale,
        hc_base,
        post_mix,
        comb_mix,
        layer_input,
        M, K, hidden_size,
        rms_eps, hc_pre_eps, hc_sinkhorn_eps,
        hc_post_mult_value, sinkhorn_repeat,
        num_splits=num_splits,
    )

    return post_mix, comb_mix, layer_input


# ---------------------------------------------------------------------------
# Python API — post_mapping
# ---------------------------------------------------------------------------


def mhc_post_mapping_cuda(
    residual: torch.Tensor,
    x: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """Post-mapping: out = post * x + comb.T @ residual.

    Args:
        residual: [B, n, hidden_size] bf16
        x:        [B, hidden_size]    bf16
        post_mix: [B, n]             fp32
        comb_mix: [B, n, n]          fp32
        n:        number of hyper-connection heads

    Returns: [B, n, hidden_size] bf16.
    """
    residual = residual.contiguous()
    x = x.contiguous()
    post_mix = post_mix.to(torch.float32).contiguous()
    comb_mix = comb_mix.to(torch.float32).contiguous()

    B = residual.shape[0]
    hidden_size = residual.shape[2]

    out = torch.empty((B, n, hidden_size), dtype=torch.bfloat16, device=x.device)

    torch.ops.trtllm.mhc_post_mapping(
        residual, x, post_mix, comb_mix, out,
        B, hidden_size,
    )

    return out


# ---------------------------------------------------------------------------
# Python API — HCHead
# ---------------------------------------------------------------------------


def mhc_hc_head_cuda(
    x: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    mult: int,
    hidden_size: int,
    norm_eps: float = 1e-5,
    eps: float = 1e-5,
) -> torch.Tensor:
    """HCHead forward: RMS-normed GEMM -> sigmoid -> weighted sum.

    Args:
        x:           [M, mult, hidden_size] bf16 input
        fn:          [mult, K] fp32 weight (K = mult * hidden_size)
        scale:       [1] fp32
        base:        [mult] fp32
        mult:        number of hyper-connection heads (typically 4)
        hidden_size: per-head dimension
        norm_eps:    RMS norm epsilon
        eps:         sigmoid offset epsilon

    Returns: [M, hidden_size] bf16
    """
    M = x.shape[0]
    K = mult * hidden_size

    x_flat = x.reshape(M, K).contiguous()
    fn_t = fn.to(torch.float32).contiguous()
    scale = scale.to(torch.float32).contiguous()
    base = base.to(torch.float32).contiguous()

    mixes, sqrsum = mhc_gemm_rms_fma_cuda(
        x_flat, None, M, mult, K, w_t=fn_t,
    )

    out = torch.empty((M, hidden_size), dtype=torch.bfloat16, device=x.device)

    torch.ops.trtllm.mhc_hc_head_apply(
        mixes, sqrsum, x.reshape(M, mult, hidden_size).contiguous(), out,
        scale, base,
        M, mult, hidden_size, K,
        float(norm_eps), float(eps),
    )

    return out
