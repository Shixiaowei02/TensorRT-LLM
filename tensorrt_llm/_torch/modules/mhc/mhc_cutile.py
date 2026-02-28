"""cuTile MHC (multi-head Hyper-Connection) kernels.

Moved from tilegym/ops/cutile/mhc.py so that trtllm no longer depends on the
TileGym package for MHC ops.  All kernel semantics are identical.
"""

import os
from math import ceil
from types import SimpleNamespace
from typing import Dict, Tuple

import cuda.tile as ct
import torch

try:
    import cuda.tile_experimental as ct_experimental

    _HAS_TILE_EXPERIMENTAL = True
except (ModuleNotFoundError, ImportError, OSError):
    ct_experimental = None  # type: ignore[misc, assignment]
    _HAS_TILE_EXPERIMENTAL = False

# Type aliases for constants
ConstInt = ct.Constant[int]
LOG2E = 1.4426950408889634

# Cache tuned configs so steady-state benchmark timing excludes autotune overhead.
_SPLIT_GEMM_CFG_CACHE: Dict[Tuple, SimpleNamespace] = {}
_BIG_FUSE_CFG_CACHE: Dict[Tuple, SimpleNamespace] = {}
_POST_MAPPING_CFG_CACHE: Dict[Tuple, SimpleNamespace] = {}

# Autotune only when cuda.tile_experimental is installed; otherwise use default configs.
# Set DISABLE_CUTILE_TUNE=1 to force default configs even when tile_experimental is available.
_AUTOTUNE_DISABLED = os.environ.get("DISABLE_CUTILE_TUNE", "0") == "1"


def _autotune_enabled() -> bool:
    """True only when tile_experimental is installed and autotune is not disabled."""
    return _HAS_TILE_EXPERIMENTAL and not _AUTOTUNE_DISABLED


def _default_split_gemm_cfg(N, max_split_k=None):
    """Conservative defaults when autotuning is disabled."""
    tile_n = 8 if N <= 8 else (16 if N <= 16 else 32)
    split_k = min(4, max_split_k) if (max_split_k is not None and max_split_k >= 1) else 4
    return SimpleNamespace(
        TILE_SIZE_M=128,
        TILE_SIZE_N=tile_n,
        TILE_SIZE_K=128,
        SPLIT_K=split_k,
        GROUP_SIZE_M=8,
    )


def _default_big_fuse_cfg():
    """Conservative defaults when autotuning is disabled."""
    return SimpleNamespace(TILE_SIZE_H=512, occupancy=2)


def _default_post_mapping_cfg():
    """Conservative defaults when autotuning is disabled."""
    return SimpleNamespace(TILE_SIZE_C=1024, occupancy=4)


def _device_sm(device: torch.device) -> Tuple[int, int]:
    props = torch.cuda.get_device_properties(device)
    return props.major, props.minor


def _to_cfg_namespace(cfg) -> SimpleNamespace:
    if isinstance(cfg, dict):
        cfg = SimpleNamespace(**cfg)
    if not hasattr(cfg, "TILE_SIZE_M") and hasattr(cfg, "m"):
        cfg.TILE_SIZE_M = cfg.m
    if not hasattr(cfg, "TILE_SIZE_N") and hasattr(cfg, "n"):
        cfg.TILE_SIZE_N = cfg.n
    if not hasattr(cfg, "TILE_SIZE_K") and hasattr(cfg, "k"):
        cfg.TILE_SIZE_K = cfg.k
    if not hasattr(cfg, "SPLIT_K") and hasattr(cfg, "split_k"):
        cfg.SPLIT_K = cfg.split_k
    if not hasattr(cfg, "GROUP_SIZE_M") and hasattr(cfg, "group_size_m"):
        cfg.GROUP_SIZE_M = cfg.group_size_m
    return cfg


def _split_gemm_cache_key(x, w, M, N, K, max_split_k):
    return (
        x.device.type,
        x.device.index,
        _device_sm(x.device),
        str(x.dtype),
        str(w.dtype),
        M,
        N,
        K,
        max_split_k,
    )


def _big_fuse_cache_key(x, w, n, hidden_size, M, K, sinkhorn_repeat):
    return (
        x.device.type,
        x.device.index,
        _device_sm(x.device),
        str(x.dtype),
        str(w.dtype),
        n,
        hidden_size,
        M,
        K,
        sinkhorn_repeat,
    )


def _post_mapping_cache_key(residual, x, n, C):
    return (
        residual.device.type,
        residual.device.index,
        _device_sm(residual.device),
        str(residual.dtype),
        str(x.dtype),
        n,
        C,
        residual.shape[0],  # B
    )


def _compute_bid(tile_id, num_bid_in_group, num_bid_m, GROUP_SIZE_M):
    group_id = tile_id // num_bid_in_group
    first_bid_m = group_id * GROUP_SIZE_M
    group_size_m = ct.minimum(num_bid_m - first_bid_m, GROUP_SIZE_M)
    bid_m = first_bid_m + (tile_id % group_size_m)
    bid_n = (tile_id % num_bid_in_group) // group_size_m
    return bid_m, bid_n


def _sigmoid(x):
    return 1.0 / (1.0 + ct.exp(-x))


# ============================================================================
# Kernel 1: Split-K fused GEMM + RMS
# ============================================================================


@ct.kernel(occupancy=2)
def mhc_split_gemm_rms_kernel(
    X,
    W,
    Y_acc,
    R_acc,
    M: int,
    N: int,
    K: int,
    TILE_SIZE_M: ConstInt,
    TILE_SIZE_N: ConstInt,
    TILE_SIZE_K: ConstInt,
    SPLIT_K: ConstInt,
    GROUP_SIZE_M: ConstInt,
):
    """Split-K fused GEMM + RMS compute kernel for mHC.

    Key optimization: All blocks compute RMS to avoid wasting registers.
    Each block computes partial RMS for its K-tile range, which are later
    summed in the finalize kernel.
    """
    tile_id = ct.bid(0)
    bid_k = ct.bid(1)
    zero_pad = ct.PaddingMode.ZERO

    num_bid_m = ct.cdiv(M, TILE_SIZE_M)
    num_bid_n = ct.cdiv(N, TILE_SIZE_N)
    num_bid_in_group = GROUP_SIZE_M * num_bid_n
    bid_m, bid_n = _compute_bid(tile_id, num_bid_in_group, num_bid_m, GROUP_SIZE_M)
    k_tiles = ct.cdiv(K, TILE_SIZE_K)
    k_tiles_per_split = ct.cdiv(k_tiles, SPLIT_K)
    k_tile_start = bid_k * k_tiles_per_split
    k_tile_end = ct.minimum(k_tile_start + k_tiles_per_split, k_tiles)

    # Keep rms_acc 2D from the start to avoid a 1D->2D ct.reshape before store
    # (ct.reshape can cause tileiras compilation failures on SM 100).
    rms_acc = ct.full((TILE_SIZE_M, 1), 0.0, dtype=ct.float32)
    accumulator = ct.full((TILE_SIZE_M, TILE_SIZE_N), 0.0, dtype=ct.float32)
    mma_dtype = ct.tfloat32 if (X.dtype == ct.float32 or W.dtype == ct.float32) else X.dtype

    for k_tile in range(k_tile_start, k_tile_end):
        a = ct.load(
            X,
            index=(bid_m, k_tile),
            shape=(TILE_SIZE_M, TILE_SIZE_K),
            padding_mode=zero_pad,
            allow_tma=True,
            latency=2,
        )
        # Do NOT use allow_tma for W: when N is small (e.g. HCHead N=4),
        # the row stride (N * sizeof(dtype)) may violate TMA's 16-byte
        # alignment requirement, silently producing garbage.
        b = ct.load(
            W,
            index=(k_tile, bid_n),
            shape=(TILE_SIZE_K, TILE_SIZE_N),
            padding_mode=zero_pad,
            latency=2,
        )

        # Compute RMS from a BEFORE MMA conversion so a_fp32 dies before
        # MMA accumulator registers are live — reduces peak register pressure.
        a_fp32 = ct.astype(a, ct.float32)
        rms_acc = rms_acc + ct.sum(a_fp32 * a_fp32, axis=1, keepdims=True)

        a_mma = ct.astype(a_fp32, mma_dtype)
        b_mma = ct.astype(b, mma_dtype)
        accumulator = ct.mma(a_mma, b_mma, acc=accumulator)

    bid_m_k = bid_m + bid_k * num_bid_m
    ct.store(Y_acc, index=(bid_m_k, bid_n), tile=accumulator)

    # Store RMS partial results - will be summed across bid_n in finalize kernel
    # Using bid_n as additional dimension for partial sums
    ct.store(R_acc, index=(bid_m_k, bid_n), tile=rms_acc)


# ============================================================================
# Kernel 2: Finalize split-K + scale/bias/sigmoid
# ============================================================================


@ct.kernel
def mhc_finalize_scale_bias_sigmoid_kernel(
    Y_acc,
    R_acc,
    Y,
    R,
    n: int,
    alpha_pre: float,
    alpha_post: float,
    alpha_res: float,
    norm_eps: float,
    Bias,
    M: int,
    N: int,
    K: int,
    TILE_SIZE_M: ConstInt,
    TILE_SIZE_N: ConstInt,
    SPLIT_K: ConstInt,
):
    """Finalize split-K + fused scale/bias/sigmoid kernel for mHC."""
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    num_bid_m = ct.cdiv(M, TILE_SIZE_M)

    y_accum = ct.full((TILE_SIZE_M, TILE_SIZE_N), 0.0, dtype=ct.float32)
    r_accum = ct.full((TILE_SIZE_M, 1), 0.0, dtype=ct.float32)

    # Sum across split_k dimension
    for split_idx in range(SPLIT_K):
        bid_m_k = bid_m + split_idx * num_bid_m
        y_tile = ct.load(
            Y_acc,
            index=(bid_m_k, bid_n),
            shape=(TILE_SIZE_M, TILE_SIZE_N),
            padding_mode=ct.PaddingMode.ZERO,
        )
        y_accum = y_accum + y_tile

        # RMS is independent of bid_n; each bid_n block stores the same partial RMS.
        # Loading the current bid_n avoids over-counting when num_bid_n > 1.
        r_tile = ct.load(
            R_acc,
            index=(bid_m_k, bid_n),
            shape=(TILE_SIZE_M, 1),
            padding_mode=ct.PaddingMode.ZERO,
        )
        r_tile = ct.astype(r_tile, ct.float32)
        r_accum = r_accum + r_tile

    denom = ct.full((TILE_SIZE_M, 1), K * 1.0, dtype=ct.float32)
    mean = ct.truediv(r_accum, denom)
    eps_tile = ct.full((TILE_SIZE_M, 1), norm_eps, dtype=ct.float32)
    rstd = ct.rsqrt(mean + eps_tile)
    ones = ct.full((TILE_SIZE_M, 1), 1.0, dtype=ct.float32)
    r = ct.truediv(ones, rstd)
    if bid_n == 0:
        r_out = ct.astype(r, R.dtype)
        ct.store(R, index=(bid_m, 0), tile=r_out)

    offsets = ct.arange(TILE_SIZE_N, dtype=ct.int32)
    col_ids = bid_n * TILE_SIZE_N + offsets
    bias = ct.load(Bias, index=(bid_n,), shape=(TILE_SIZE_N,), padding_mode=ct.PaddingMode.ZERO)
    bias = ct.reshape(bias, (1, TILE_SIZE_N))

    one = ct.full((TILE_SIZE_N,), 1.0, dtype=ct.float32)
    zero = ct.full((TILE_SIZE_N,), 0.0, dtype=ct.float32)
    mask_pre = ct.where(ct.less(col_ids, n), one, zero)
    mask_post = ct.where(ct.less(col_ids, 2 * n), one, zero)
    mask_post = mask_post - mask_pre
    mask_res = one - mask_pre - mask_post

    scale = alpha_pre * mask_pre + alpha_post * mask_post + alpha_res * mask_res
    scale = ct.reshape(scale, (1, TILE_SIZE_N))

    linear = ct.truediv(y_accum * scale, r) + ct.astype(bias, ct.float32)
    sigmoid_linear = _sigmoid(linear)
    two_sigmoid = sigmoid_linear * 2.0

    mask_pre = ct.reshape(mask_pre, (1, TILE_SIZE_N))
    mask_post = ct.reshape(mask_post, (1, TILE_SIZE_N))
    mask_res = ct.reshape(mask_res, (1, TILE_SIZE_N))

    out = linear * mask_res + sigmoid_linear * mask_pre + two_sigmoid * mask_post
    out = ct.astype(out, Y.dtype)
    ct.store(Y, index=(bid_m, bid_n), tile=out)


# ============================================================================
# Kernel 3: Apply residual (post-mapping)
# ============================================================================


@ct.kernel
def mhc_apply_residual_kernel(
    X,
    F_out,
    Y_post,
    Y_res,
    Out,
    C: int,
    n: ct.Constant[int],
    TILE_SIZE_C: ConstInt,
):
    """Apply H_res and H_post to residual stream (in-place on Out)."""
    # Shapes:
    # - X: [B, n, C] view of residual stream
    # - F_out: [B, C]
    # - Y_post: [B, n]
    # - Y_res: [B, n, n]
    # - Out: [B, n, C]
    row = ct.bid(0)
    c_tile = ct.bid(1)
    compute_dtype = (
        ct.float32
        if (X.dtype == ct.float32 or F_out.dtype == ct.float32 or Y_post.dtype == ct.float32)
        else X.dtype
    )

    f_tile = ct.load(
        F_out,
        index=(row, c_tile),
        shape=(1, TILE_SIZE_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    f_tile = ct.astype(f_tile, compute_dtype)

    h_post = ct.load(
        Y_post,
        index=(row, 0),
        shape=(1, n),
        padding_mode=ct.PaddingMode.ZERO,
    )
    h_post = ct.reshape(h_post, (n, 1))
    h_post = ct.astype(h_post, compute_dtype)

    h_res = ct.load(
        Y_res,
        index=(row, 0, 0),
        shape=(1, n, n),
        padding_mode=ct.PaddingMode.ZERO,
    )
    h_res = ct.reshape(h_res, (n, n))
    h_res = ct.astype(h_res, compute_dtype)

    acc = ct.full((n, TILE_SIZE_C), 0.0, dtype=compute_dtype)
    for j in range(n):
        x_row = ct.load(
            X,
            index=(row, j, c_tile),
            shape=(1, 1, TILE_SIZE_C),
            padding_mode=ct.PaddingMode.ZERO,
        )
        x_row = ct.reshape(x_row, (1, TILE_SIZE_C))
        x_row = ct.astype(x_row, compute_dtype)
        h_col = ct.extract(h_res, (0, j), shape=(n, 1))
        x_row = ct.broadcast_to(x_row, (n, TILE_SIZE_C))
        h_col = ct.broadcast_to(h_col, (n, TILE_SIZE_C))
        prod = h_col * x_row
        acc = acc + prod
    h_post = ct.broadcast_to(h_post, (n, TILE_SIZE_C))
    f_tile = ct.broadcast_to(f_tile, (n, TILE_SIZE_C))
    x_post = h_post * f_tile
    out_tile = acc + x_post
    out_tile = ct.astype(out_tile, Out.dtype)
    out_tile = ct.reshape(out_tile, (1, n, TILE_SIZE_C))
    ct.store(Out, index=(row, 0, c_tile), tile=out_tile)


# ============================================================================
# Kernel 3b: Optimized post-mapping (replaces mhc_apply_residual for post_mapping)
# ============================================================================


@ct.kernel(occupancy=4)
def mhc_post_mapping_kernel(
    Residual,  # [B, n, C] bfloat16
    X,  # [B, C] bfloat16
    PostMix,  # [B, n] float32
    CombMix,  # [B, n, n] float32  (NOT transposed)
    Out,  # [B, n, C] bfloat16
    C: int,
    n: ConstInt,
    TILE_SIZE_C: ConstInt,
):
    """Optimized post-mapping: out = post * x + comb_res_mix.T @ residual.

    1-CTA-per-token: grid (B,).  Each CTA streams through the hidden dimension.
    Bulk residual loads (1 TMA descriptor per tile) + latency=4 for pipelining
    across loop iterations.  comb/post loaded once per token.
    """
    row = ct.bid(0)

    # ---- Load small tensors ONCE per token (L2-cached, ~80 bytes) ----
    post_vec = ct.load(
        PostMix,
        index=(row, 0),
        shape=(1, n),
        padding_mode=ct.PaddingMode.ZERO,
    )
    comb_mat = ct.load(
        CombMix,
        index=(row, 0, 0),
        shape=(1, n, n),
        padding_mode=ct.PaddingMode.ZERO,
    )
    post_col = ct.reshape(post_vec, (n, 1))
    comb_2d = ct.reshape(comb_mat, (n, n))

    # ---- Stream through hidden dimension (like big_fuse section 9) ----
    num_tiles = ct.cdiv(C, TILE_SIZE_C)
    for c_tile in range(num_tiles):
        # BULK load: all n residual rows in one transaction (no TMA —
        # matching TileLang's TL_DISABLE_TMA_LOWER for strided access)
        all_res = ct.load(
            Residual,
            index=(row, 0, c_tile),
            shape=(1, n, TILE_SIZE_C),
            padding_mode=ct.PaddingMode.ZERO,
            latency=4,
        )
        x_tile = ct.load(
            X,
            index=(row, c_tile),
            shape=(1, TILE_SIZE_C),
            padding_mode=ct.PaddingMode.ZERO,
            latency=4,
        )

        all_res_2d = ct.reshape(all_res, (n, TILE_SIZE_C))
        x_fp32 = ct.astype(ct.reshape(x_tile, (1, TILE_SIZE_C)), ct.float32)

        # comb.T @ residual: out[i,h] = sum_j comb[j,i] * res[j,h]
        acc = ct.full((n, TILE_SIZE_C), 0.0, dtype=ct.float32)
        for j in range(n):
            res_row = ct.extract(all_res_2d, (j, 0), shape=(1, TILE_SIZE_C))
            res_fp32 = ct.astype(res_row, ct.float32)
            comb_row = ct.extract(comb_2d, (j, 0), shape=(1, n))
            comb_col = ct.reshape(comb_row, (n, 1))
            acc = acc + ct.broadcast_to(comb_col, (n, TILE_SIZE_C)) * ct.broadcast_to(
                res_fp32, (n, TILE_SIZE_C)
            )

        # post * x
        out_tile = acc + ct.broadcast_to(post_col, (n, TILE_SIZE_C)) * ct.broadcast_to(
            x_fp32, (n, TILE_SIZE_C)
        )

        out_tile = ct.astype(out_tile, Out.dtype)
        out_tile = ct.reshape(out_tile, (1, n, TILE_SIZE_C))
        ct.store(Out, index=(row, 0, c_tile), tile=out_tile, latency=4)


# ============================================================================
# Kernel 4: Sinkhorn-Knopp normalization
# ============================================================================


@ct.kernel
def mhc_sinkhorn_kernel(
    Y,
    n: ct.Constant[int],
):
    """Sinkhorn-Knopp normalization for residual block (in-place on Y)."""
    row = ct.bid(0)
    total = n * n
    mat = ct.load(Y, index=(row, 0), shape=(1, total))
    mat = ct.reshape(mat, (n, n))
    mat = ct.astype(mat, ct.float32)
    mat = ct.exp2(mat * LOG2E)

    for _ in range(20):
        row_sum = ct.sum(mat, axis=1, keepdims=True)
        mat = ct.truediv(mat, row_sum)
        col_sum = ct.sum(mat, axis=0, keepdims=True)
        mat = ct.truediv(mat, col_sum)

    mat = ct.reshape(mat, (1, total))
    mat = ct.astype(mat, Y.dtype)
    ct.store(Y, index=(row, 0), tile=mat)


# ============================================================================
# Kernel 5: Big-fuse (replaces finalize + Sinkhorn + pre_apply_mix)
# ============================================================================


@ct.kernel(occupancy=2)
def mhc_big_fuse_kernel(
    Y_pp,  # [M, 2n] float32  – pre+post columns of GEMM output
    Y_res,  # [M, n²] float32  – res columns of GEMM output
    R_acc,  # [M, num_bid_n] float32  – partial sqrsum
    Residual,  # [M, n, hidden_size] bfloat16
    HcScale,  # [4] float32  (padded from 3)
    BiasPP,  # [2n] float32  – pre+post bias
    BiasRes,  # [n²] float32  – res bias
    PostMix,  # [M, n] float32  (output)
    CombMix,  # [M, n*n] float32  (output)
    LayerInput,  # [M, hidden_size] bfloat16  (output)
    M: int,
    K: int,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    n: ConstInt,  # = mult  (compile-time)
    sinkhorn_repeat: ConstInt,  # number of Sinkhorn iterations
    TILE_SIZE_H: ConstInt,  # hidden-dim tile width
):
    """Fused: split-K reduce + RMS + sigmoid + Sinkhorn + pre_apply_mix.

    One CTA per token.  Replaces the finalize kernel, PyTorch Sinkhorn,
    and PyTorch pre_apply_mix with a single kernel launch.

    The Python wrapper splits Y_acc and bias into pre+post (2n cols) and
    res (n² cols) so that every load shape is a power of two.
    ct.extract offsets are tile-based (offset * shape), so we extract
    pre / post from the 2n-wide pp row at tile offsets 0 and 1.
    """
    token = ct.bid(0)
    n2 = n * n
    nn2 = 2 * n  # pre+post width (compile-time)

    # ---- 1. Load GEMM partials & sqrsum for this token ----------------------
    pp_row = ct.load(
        Y_pp,
        index=(token, 0),
        shape=(1, nn2),
        padding_mode=ct.PaddingMode.ZERO,
    )
    pp_row = ct.astype(pp_row, ct.float32)

    res_row = ct.load(
        Y_res,
        index=(token, 0),
        shape=(1, n2),
        padding_mode=ct.PaddingMode.ZERO,
    )
    res_row = ct.astype(res_row, ct.float32)

    # sqrsum: all bid_n copies are identical for SPLIT_K=1; read column 0.
    sqrsum = ct.load(
        R_acc,
        index=(token, 0),
        shape=(1, 1),
        padding_mode=ct.PaddingMode.ZERO,
    )
    sqrsum = ct.astype(sqrsum, ct.float32)

    # ---- 2. RMS normalize ---------------------------------------------------
    K_tile = ct.full((1, 1), K * 1.0, dtype=ct.float32)
    eps_rms = ct.full((1, 1), rms_eps, dtype=ct.float32)
    rstd = ct.rsqrt(ct.truediv(sqrsum, K_tile) + eps_rms)  # (1, 1)

    rstd_pp = ct.broadcast_to(rstd, (1, nn2))
    pp_row = pp_row * rstd_pp

    rstd_res = ct.broadcast_to(rstd, (1, n2))
    res_row = res_row * rstd_res

    # ---- 3. Load scale [4] and bias sub-vectors ----------------------------
    scale_vec = ct.load(
        HcScale,
        index=(0,),
        shape=(4,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    pp_bias = ct.load(
        BiasPP,
        index=(0,),
        shape=(nn2,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    res_bias = ct.load(
        BiasRes,
        index=(0,),
        shape=(n2,),
        padding_mode=ct.PaddingMode.ZERO,
    )

    # ---- 4. Pre columns [0, n): sigmoid(x * scale[0] + base) + pre_eps ----
    # ct.extract uses tile-based offset: physical = offset * shape
    pre_vals = ct.extract(pp_row, (0, 0), shape=(1, n))  # cols 0..n-1
    pre_bias_v = ct.extract(pp_bias, (0,), shape=(n,))  # bias[0..n-1]
    pre_bias_v = ct.reshape(pre_bias_v, (1, n))
    s0 = ct.extract(scale_vec, (0,), shape=(1,))
    s0 = ct.reshape(s0, (1, 1))
    s0 = ct.broadcast_to(s0, (1, n))
    pre_eps_tile = ct.full((1, n), hc_pre_eps, dtype=ct.float32)
    pre_mix = _sigmoid(pre_vals * s0 + pre_bias_v) + pre_eps_tile  # (1, n)

    # ---- 5. Post columns [n, 2n): sigmoid(x * scale[1] + base) * mult_val -
    post_vals = ct.extract(pp_row, (0, 1), shape=(1, n))  # cols n..2n-1
    post_bias_v = ct.extract(pp_bias, (1,), shape=(n,))  # bias[n..2n-1]
    post_bias_v = ct.reshape(post_bias_v, (1, n))
    s1 = ct.extract(scale_vec, (1,), shape=(1,))
    s1 = ct.reshape(s1, (1, 1))
    s1 = ct.broadcast_to(s1, (1, n))
    post_mix = _sigmoid(post_vals * s1 + post_bias_v) * hc_post_mult_value  # (1, n)

    # ---- 6. Res columns: scale + bias → Sinkhorn ---------------------------
    res_bias = ct.reshape(res_bias, (1, n2))
    s2 = ct.extract(scale_vec, (2,), shape=(1,))
    s2 = ct.reshape(s2, (1, 1))
    s2 = ct.broadcast_to(s2, (1, n2))
    res_linear = res_row * s2 + res_bias  # (1, n²)

    # ---- 7. Sinkhorn normalization (matches sinkhorn_normalize_ref) ---------
    mat = ct.reshape(res_linear, (n, n))
    # Step 1: softmax(dim=-1) + eps  →  exp / row_sum + eps
    mat = ct.exp(mat)
    row_sum = ct.sum(mat, axis=1, keepdims=True)  # (n, 1)
    mat = ct.truediv(mat, row_sum)
    eps_nn = ct.full((n, n), hc_sinkhorn_eps, dtype=ct.float32)
    mat = mat + eps_nn
    # Step 2: col normalize  →  x / (col_sum + eps)
    col_sum = ct.sum(mat, axis=0, keepdims=True)  # (1, n)
    eps_1n = ct.full((1, n), hc_sinkhorn_eps, dtype=ct.float32)
    mat = ct.truediv(mat, col_sum + eps_1n)
    # Steps 3+:  (sinkhorn_repeat − 1) iterations of row/col normalize
    eps_n1 = ct.full((n, 1), hc_sinkhorn_eps, dtype=ct.float32)
    for _ in range(sinkhorn_repeat - 1):
        row_sum = ct.sum(mat, axis=1, keepdims=True) + eps_n1
        mat = ct.truediv(mat, row_sum)
        col_sum = ct.sum(mat, axis=0, keepdims=True) + eps_1n
        mat = ct.truediv(mat, col_sum)

    # ---- 8. Store post_mix and comb_mix ------------------------------------
    ct.store(PostMix, index=(token, 0), tile=post_mix)
    comb_flat = ct.reshape(mat, (1, n2))
    ct.store(CombMix, index=(token, 0), tile=comb_flat)

    # ---- 9. pre_apply_mix: stream through hidden dimension -----------------
    # layer_input[h] = Σ_j  pre_mix[j] * residual[token, j, h]
    # Load all n rows at once (contiguous along [n, H] dims), extract per-row.
    hidden_tiles = ct.cdiv(hidden_size, TILE_SIZE_H)
    for h_tile in range(hidden_tiles):
        # Single TMA load for all n rows — avoids n separate loads
        all_x = ct.load(
            Residual,
            index=(token, 0, h_tile),
            shape=(1, n, TILE_SIZE_H),
            padding_mode=ct.PaddingMode.ZERO,
            allow_tma=True,
            latency=4,
        )
        all_x_2d = ct.reshape(all_x, (n, TILE_SIZE_H))

        # Weighted sum: Σ_j pre_mix[j] * residual[token, j, h_tile]
        # Per-row fp32 cast reduces peak live registers by (n-1)*TILE_SIZE_H.
        acc = ct.full((1, TILE_SIZE_H), 0.0, dtype=ct.float32)
        for j in range(n):
            x_row = ct.astype(ct.extract(all_x_2d, (j, 0), shape=(1, TILE_SIZE_H)), ct.float32)
            pre_j = ct.extract(pre_mix, (0, j), shape=(1, 1))
            pre_j = ct.broadcast_to(pre_j, (1, TILE_SIZE_H))
            acc = acc + pre_j * x_row

        acc_bf16 = ct.astype(acc, ct.bfloat16)
        ct.store(LayerInput, index=(token, h_tile), tile=acc_bf16, latency=3)


# ============================================================================
# Autotune search space
# ============================================================================


def _mhc_split_gemm_rms_autotune_configs():
    tile_ms = (64, 128)
    # Include smaller tile_n values so the autotuner finds valid configs
    # when N < 32 (e.g. N=24 for hc_mult=4).  MMA requires N >= 8.
    tile_ns = (8, 16, 32)
    tile_ks = (64, 128, 256)
    split_ks = (1, 2, 4, 8, 16)
    group_size_ms = (8, 16)
    for tile_m in tile_ms:
        for tile_n in tile_ns:
            for tile_k in tile_ks:
                for split_k in split_ks:
                    for group_size_m in group_size_ms:
                        yield SimpleNamespace(
                            TILE_SIZE_M=tile_m,
                            TILE_SIZE_N=tile_n,
                            TILE_SIZE_K=tile_k,
                            SPLIT_K=split_k,
                            GROUP_SIZE_M=group_size_m,
                        )


def _mhc_big_fuse_autotune_configs():
    for tile_h in (128, 256, 512, 1024):
        for occ in (1, 2, 4):
            yield SimpleNamespace(TILE_SIZE_H=tile_h, occupancy=occ)


def _mhc_post_mapping_autotune_configs():
    for tile_c in (256, 512, 1024):
        for occ in (1, 2, 4):
            yield SimpleNamespace(TILE_SIZE_C=tile_c, occupancy=occ)


# ============================================================================
# Python launch wrappers
# ============================================================================


def cutile_autotune_mhc_split_gemm_rms(stream, x, w, M, N, K, cfg=None, max_split_k=None):
    split_key = None
    if cfg is None:
        split_key = _split_gemm_cache_key(x, w, M, N, K, max_split_k)
        cached_cfg = _SPLIT_GEMM_CFG_CACHE.get(split_key)
        if cached_cfg is not None:
            cfg = SimpleNamespace(**vars(cached_cfg))
        elif not _autotune_enabled():
            cfg = _default_split_gemm_cfg(N, max_split_k)
    if cfg is not None:
        cfg = _to_cfg_namespace(cfg)

        num_bid_n = ceil(N / cfg.TILE_SIZE_N)
        y_acc = torch.empty((M * cfg.SPLIT_K, N), device=x.device, dtype=torch.float32)
        # R_acc now stores partial RMS for all N blocks
        r_acc = torch.empty((M * cfg.SPLIT_K, num_bid_n), device=x.device, dtype=torch.float32)
        grid = (
            ceil(M / cfg.TILE_SIZE_M) * ceil(N / cfg.TILE_SIZE_N),
            cfg.SPLIT_K,
            1,
        )
        ct.launch(
            stream,
            grid,
            mhc_split_gemm_rms_kernel,
            (
                x,
                w,
                y_acc,
                r_acc,
                M,
                N,
                K,
                cfg.TILE_SIZE_M,
                cfg.TILE_SIZE_N,
                cfg.TILE_SIZE_K,
                cfg.SPLIT_K,
                cfg.GROUP_SIZE_M,
            ),
        )
        return y_acc, r_acc, cfg

    # ----- Autotune: search over tile sizes, split-K, and group size -----
    configs = list(_mhc_split_gemm_rms_autotune_configs())
    if max_split_k is not None:
        configs = [c for c in configs if c.SPLIT_K <= max_split_k]
    max_split_k_val = max(cfg.SPLIT_K for cfg in configs)
    max_num_bid_n = max(ceil(N / cfg.TILE_SIZE_N) for cfg in configs)
    y_acc = torch.empty((M * max_split_k_val, N), device=x.device, dtype=torch.float32)
    r_acc = torch.empty((M * max_split_k_val, max_num_bid_n), device=x.device, dtype=torch.float32)
    tuned = ct_experimental.autotune_launch(
        stream,
        grid_fn=lambda cfg: (
            ceil(M / cfg.TILE_SIZE_M) * ceil(N / cfg.TILE_SIZE_N),
            cfg.SPLIT_K,
            1,
        ),
        kernel=mhc_split_gemm_rms_kernel,
        args_fn=lambda cfg: (
            x,
            w,
            y_acc,
            r_acc,
            M,
            N,
            K,
            cfg.TILE_SIZE_M,
            cfg.TILE_SIZE_N,
            cfg.TILE_SIZE_K,
            cfg.SPLIT_K,
            cfg.GROUP_SIZE_M,
        ),
        search_space=configs,
    )
    best_cfg = tuned.tuned_config
    if split_key is not None:
        _SPLIT_GEMM_CFG_CACHE[split_key] = SimpleNamespace(
            TILE_SIZE_M=best_cfg.TILE_SIZE_M,
            TILE_SIZE_N=best_cfg.TILE_SIZE_N,
            TILE_SIZE_K=best_cfg.TILE_SIZE_K,
            SPLIT_K=best_cfg.SPLIT_K,
            GROUP_SIZE_M=best_cfg.GROUP_SIZE_M,
        )
    return y_acc, r_acc, best_cfg


def mhc_finalize_scale_bias_sigmoid(
    y_acc: torch.Tensor,
    r_acc: torch.Tensor,
    n: int,
    alpha_pre: float,
    alpha_post: float,
    alpha_res: float,
    bias: torch.Tensor,
    M: int,
    K: int,
    norm_eps: float = 0.0,
    **kwargs,
):
    cfg = kwargs.pop("cfg", None)
    split_k = kwargs.pop("split_k", None)
    tile_m = kwargs.pop("tile_m", None)
    tile_n = kwargs.pop("tile_n", None)
    if cfg is not None:
        tile_m = cfg.TILE_SIZE_M
        tile_n = cfg.TILE_SIZE_N
        split_k = cfg.SPLIT_K

    y_acc = y_acc.contiguous()
    r_acc = r_acc.contiguous()
    bias = bias.contiguous()
    N = y_acc.shape[1]

    y = torch.empty((M, N), device=y_acc.device, dtype=bias.dtype)
    r = torch.empty((M, 1), device=y_acc.device, dtype=torch.float32)

    grid = (ceil(M / tile_m), ceil(N / tile_n), 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        mhc_finalize_scale_bias_sigmoid_kernel,
        (
            y_acc,
            r_acc,
            y,
            r,
            n,
            float(alpha_pre),
            float(alpha_post),
            float(alpha_res),
            float(norm_eps),
            bias,
            M,
            N,
            K,
            tile_m,
            tile_n,
            split_k,
        ),
    )
    return y, r


# ============================================================================
# Public API  (drop-in replacements for tilegym.ops.cutile.mhc)
# ============================================================================


_MIN_TILE_N = 8  # Minimum MMA tile width; also the smallest tile_n in the autotune space.


def mhc_gemm_rms_scale(
    x: torch.Tensor,
    w: torch.Tensor,
    n: int,
    alpha_pre: float,
    alpha_post: float,
    alpha_res: float,
    bias: torch.Tensor,
    norm_eps: float = 0.0,
    **kwargs,
):
    """Fused GEMM + RMS-norm + scale/bias/sigmoid for mHC pre-mapping."""
    cfg = kwargs.pop("cfg", None)
    kwargs.pop("w_nt", None)
    w = w.contiguous()

    M, K = x.shape
    _, N = w.shape

    # When N < _MIN_TILE_N, pad W and bias so the kernel never needs to
    # clip a tile store to fewer columns than TILE_SIZE_N.  CuTile's
    # boundary-clipped stores can corrupt the valid columns on some GPUs.
    N_orig = N
    if N < _MIN_TILE_N:
        pad = _MIN_TILE_N - N
        w = torch.nn.functional.pad(w, (0, pad))  # [K, _MIN_TILE_N]
        bias = torch.nn.functional.pad(bias, (0, pad))  # [_MIN_TILE_N]
        N = _MIN_TILE_N

    y_acc, r_acc, cfg = cutile_autotune_mhc_split_gemm_rms(
        torch.cuda.current_stream(),
        x,
        w,
        M,
        N,
        K,
        cfg=cfg,
    )
    y, r = mhc_finalize_scale_bias_sigmoid(
        y_acc,
        r_acc,
        n,
        alpha_pre,
        alpha_post,
        alpha_res,
        bias,
        M,
        K,
        norm_eps=norm_eps,
        cfg=cfg,
    )

    # Slice back to the original N columns if we padded.
    if N_orig < _MIN_TILE_N:
        y = y[:, :N_orig].contiguous()

    return y, r


def mhc_apply_residual(
    x: torch.Tensor,
    f_out: torch.Tensor,
    y: torch.Tensor,
    n: int,
    **kwargs,
):
    """Apply H_res and H_post to residual stream via cuTile kernel."""
    x = x.contiguous()
    f_out = f_out.contiguous()
    y = y.contiguous()
    B, nC = x.shape
    C = f_out.shape[1]
    # Use view for [B, n, C] without changing external layout.
    x_view = x.view(B, n, C)
    y_post = y.narrow(1, n, n)
    y_res = y.narrow(1, 2 * n, n * n).view(B, n, n)
    out = torch.empty_like(x)
    out_view = out.view(B, n, C)

    TILE_SIZE_C = 1024
    grid = (B, ceil(C / TILE_SIZE_C), 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        mhc_apply_residual_kernel,
        (
            x_view,
            f_out,
            y_post,
            y_res,
            out_view,
            C,
            n,
            TILE_SIZE_C,
        ),
    )
    return out


def mhc_post_mapping(
    residual: torch.Tensor,
    x: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    n: int,
    **kwargs,
):
    """Optimized post-mapping: takes tensors directly, no y_packed indirection.

    Args:
        residual: [B, n, C] bfloat16
        x: [B, C] bfloat16
        post_mix: [B, n] float32
        comb_mix: [B, n, n] float32 (NOT pre-transposed)
        n: number of hyper-connection heads
    """
    cfg = kwargs.pop("cfg", None)

    residual = residual.contiguous()
    x = x.contiguous()
    post_mix = post_mix.contiguous()
    comb_mix = comb_mix.contiguous()

    B = residual.shape[0]
    C = residual.shape[2]

    residual_3d = residual.view(B, n, C)
    x_2d = x.view(B, C)
    post_2d = post_mix.view(B, n)
    comb_3d = comb_mix.view(B, n, n)
    out = torch.empty((B, n, C), dtype=residual.dtype, device=residual.device)

    # --- Resolve config ---
    cache_key = None
    if cfg is None:
        cache_key = _post_mapping_cache_key(residual, x, n, C)
        cached_cfg = _POST_MAPPING_CFG_CACHE.get(cache_key)
        if cached_cfg is not None:
            cfg = SimpleNamespace(**vars(cached_cfg))
        elif not _autotune_enabled():
            cfg = _default_post_mapping_cfg()

    if cfg is not None:
        grid = (B,)
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            mhc_post_mapping_kernel,
            (
                residual_3d,
                x_2d,
                post_2d,
                comb_3d,
                out,
                C,
                n,
                cfg.TILE_SIZE_C,
            ),
        )
    else:
        if cache_key is None:
            cache_key = _post_mapping_cache_key(residual, x, n, C)
        tuned = ct_experimental.autotune_launch(
            torch.cuda.current_stream(),
            grid_fn=lambda cfg: (B,),
            kernel=mhc_post_mapping_kernel,
            args_fn=lambda cfg: (
                residual_3d,
                x_2d,
                post_2d,
                comb_3d,
                out,
                C,
                n,
                cfg.TILE_SIZE_C,
            ),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
            search_space=_mhc_post_mapping_autotune_configs,
        )
        tuned_cfg = tuned.tuned_config
        _POST_MAPPING_CFG_CACHE[cache_key] = SimpleNamespace(
            TILE_SIZE_C=tuned_cfg.TILE_SIZE_C,
            occupancy=tuned_cfg.occupancy,
        )

    return out


def mhc_sinkhorn(
    y: torch.Tensor,
    n: int,
    **kwargs,
):
    """Sinkhorn-Knopp normalization for the residual mix block (in-place)."""
    y = y.contiguous()
    M, _ = y.shape
    y_view = y.narrow(1, 2 * n, n * n)
    grid = (M,)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        mhc_sinkhorn_kernel,
        (
            y_view,
            n,
        ),
    )
    return y


def mhc_pre_mapping_fused(
    x: torch.Tensor,
    w: torch.Tensor,
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
    **kwargs,
):
    """Fused pre-mapping: GEMM+sqrsum kernel → big-fuse kernel.

    Returns (post_mix, comb_mix, layer_input) directly — no PyTorch
    Sinkhorn or apply_mix needed.
    """
    cfg = kwargs.pop("cfg", None)
    cfg_big_fuse = kwargs.pop("cfg_big_fuse", None)
    w = w.contiguous()
    residual = residual.contiguous()
    hc_scale = hc_scale.to(torch.float32)
    hc_base = hc_base.to(torch.float32)

    M, K = x.shape
    _, N = w.shape

    # --- Stage 1: GEMM + sqrsum (reuse existing optimised MMA kernel) -------
    # big_fuse kernel assumes SPLIT_K=1 (indexes y_acc by token directly).
    y_acc, r_acc, cfg = cutile_autotune_mhc_split_gemm_rms(
        torch.cuda.current_stream(),
        x,
        w,
        M,
        N,
        K,
        cfg=cfg,
        max_split_k=1,
    )

    # --- Stage 2: big-fuse (replaces finalize + Sinkhorn + apply_mix) -------
    n2 = n * n
    residual_3d = residual.view(M, n, hidden_size)

    # Split Y_acc and bias into pre+post (2n cols) and res (n² cols).
    # This ensures every ct.load/ct.extract shape dim is a power of two.
    # 2n=8 and n²=16 are both powers of two for n=4.
    y_pp = y_acc[:, : 2 * n].contiguous()  # [M, 2n]
    y_res = y_acc[:, 2 * n : 2 * n + n2].contiguous()  # [M, n²]
    bias_pp = hc_base[: 2 * n]  # [2n] (1D slice, already contiguous)
    bias_res = hc_base[2 * n : 2 * n + n2]  # [n²] (1D slice, already contiguous)
    hc_scale_pad = torch.nn.functional.pad(
        hc_scale,
        (0, 4 - hc_scale.numel()),
    )  # [4]

    post_mix = torch.empty((M, n), dtype=torch.float32, device=x.device)
    comb_mix = torch.empty((M, n2), dtype=torch.float32, device=x.device)
    layer_input = torch.empty((M, hidden_size), dtype=torch.bfloat16, device=x.device)

    if cfg_big_fuse is None:
        cached_big_cfg = _BIG_FUSE_CFG_CACHE.get(
            _big_fuse_cache_key(x, w, n, hidden_size, M, K, sinkhorn_repeat)
        )
        if cached_big_cfg is not None:
            cfg_big_fuse = SimpleNamespace(**vars(cached_big_cfg))
        elif not _autotune_enabled():
            cfg_big_fuse = _default_big_fuse_cfg()

    if cfg_big_fuse is not None:
        ct.launch(
            torch.cuda.current_stream(),
            (M,),
            mhc_big_fuse_kernel,
            (
                y_pp,
                y_res,
                r_acc,
                residual_3d,
                hc_scale_pad,
                bias_pp,
                bias_res,
                post_mix,
                comb_mix,
                layer_input,
                M,
                K,
                hidden_size,
                float(rms_eps),
                float(hc_pre_eps),
                float(hc_sinkhorn_eps),
                float(hc_post_mult_value),
                n,
                sinkhorn_repeat,
                cfg_big_fuse.TILE_SIZE_H,
            ),
        )
    else:
        tuned = ct_experimental.autotune_launch(
            torch.cuda.current_stream(),
            grid_fn=lambda cfg: (M,),
            kernel=mhc_big_fuse_kernel,
            args_fn=lambda cfg: (
                y_pp,
                y_res,
                r_acc,
                residual_3d,
                hc_scale_pad,
                bias_pp,
                bias_res,
                post_mix,
                comb_mix,
                layer_input,
                M,
                K,
                hidden_size,
                float(rms_eps),
                float(hc_pre_eps),
                float(hc_sinkhorn_eps),
                float(hc_post_mult_value),
                n,
                sinkhorn_repeat,
                cfg.TILE_SIZE_H,
            ),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
            search_space=_mhc_big_fuse_autotune_configs,
        )
        tuned_cfg = tuned.tuned_config
        _BIG_FUSE_CFG_CACHE[_big_fuse_cache_key(x, w, n, hidden_size, M, K, sinkhorn_repeat)] = (
            SimpleNamespace(TILE_SIZE_H=tuned_cfg.TILE_SIZE_H, occupancy=tuned_cfg.occupancy)
        )

    return post_mix, comb_mix, layer_input
