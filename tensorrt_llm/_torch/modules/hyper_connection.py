# Copied and modified from https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_mhc
import math
from typing import Optional

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn
from torch.nn.parameter import Parameter

from tensorrt_llm.deep_gemm import tf32_hc_prenorm_gemm


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    },
)
def mhc_pre_big_fuse_tilelang(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    residual,
    post_mix,
    comb_mix,
    layer_input,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 16,
    hc_mult: int = 4,
):
    """Deeply fused kernels, everything other than gemm & sqrsum in mHC pre block."""
    num_tokens = T.dynamic("num_tokens")
    hc_mult3 = hc_mult * (2 + hc_mult)
    hidden_block = math.gcd(512, hidden_size)

    gemm_out_mul: T.Tensor[[n_splits, num_tokens, hc_mult3], T.float32]
    gemm_out_sqrsum: T.Tensor[[n_splits, num_tokens], T.float32]
    hc_scale: T.Tensor[[3], T.float32]
    hc_base: T.Tensor[[hc_mult3], T.float32]
    residual: T.Tensor[[num_tokens, hc_mult, hidden_size], T.bfloat16]
    # outputs
    post_mix: T.Tensor[[num_tokens, hc_mult], T.float32]
    comb_mix: T.Tensor[[num_tokens, hc_mult * hc_mult], T.float32]
    layer_input: T.Tensor[[num_tokens, hidden_size], T.bfloat16]

    with T.Kernel(num_tokens, threads=96) as i:
        ##################################################################
        # _pre_norm_fn_fwd_norm
        rms = T.alloc_fragment(1, T.float32)
        mixes = T.alloc_fragment(hc_mult3, T.float32)
        T.clear(mixes)
        rms[0] = 0
        for i_split in T.serial(n_splits):
            rms[0] += gemm_out_sqrsum[i_split, i]
        rms[0] = T.rsqrt(rms[0] / (hc_mult * hidden_size) + rms_eps)
        for j in T.Parallel(hc_mult3):
            mixes[j] = 0
            for i_split in T.serial(n_splits):
                mixes[j] += gemm_out_mul[i_split, i, j]
            mixes[j] *= rms[0]
        mixes_shared = T.alloc_shared(hc_mult3, T.float32)
        T.copy(mixes, mixes_shared)

        if T.get_thread_binding() < 32:
            ##################################################################
            # _pre_split_mixes_fwd (post & comb)
            cm = T.alloc_fragment((hc_mult, hc_mult), T.float32)
            for j in T.Parallel(hc_mult):
                post_mix[i, j] = (
                    T.sigmoid(mixes_shared[j + hc_mult] * hc_scale[1] + hc_base[j + hc_mult])
                    * hc_post_mult_value
                )
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = (
                    mixes_shared[j * hc_mult + k + hc_mult * 2] * hc_scale[2]
                    + hc_base[j * hc_mult + k + hc_mult * 2]
                )

            ##################################################################
            # _sinkhorn_fwd
            row_sum = T.alloc_fragment(hc_mult, T.float32)
            col_sum = T.alloc_fragment(hc_mult, T.float32)

            # comb = comb.softmax(-1) + eps
            row_max = T.alloc_fragment(hc_mult, T.float32)
            T.reduce_max(cm, row_max, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = T.exp(cm[j, k] - row_max[j])
            T.reduce_sum(cm, row_sum, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / row_sum[j] + hc_sinkhorn_eps

            # comb = comb / (comb.sum(-2) + eps)
            T.reduce_sum(cm, col_sum, dim=0)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            for _ in T.serial(sinkhorn_repeat - 1):
                # comb = comb / (comb.sum(-1) + eps)
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (row_sum[j] + hc_sinkhorn_eps)

                # comb = comb / (comb.sum(-2) + eps)
                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            # save comb_mix to global memory
            for j, k in T.Parallel(hc_mult, hc_mult):
                comb_mix[i, j * hc_mult + k] = cm[j, k]
        else:
            ##################################################################
            # _pre_split_mixes_fwd (pre)
            pre_mix_shared = T.alloc_shared(hc_mult, T.float32)
            for j in T.Parallel(hc_mult):
                pre_mix_shared[j] = (
                    T.sigmoid(
                        mixes_shared[j] * hc_scale[0] + hc_base[j],
                    )
                    + hc_pre_eps
                )
            ###################################################################
            # _pre_apply_mix_fwd
            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                xs = T.alloc_shared((hc_mult, hidden_block), T.float32)
                xl = T.alloc_fragment((hc_mult, hidden_block), T.float32)
                T.copy(residual[i, 0, i0_h * hidden_block], xs)
                T.copy(xs, xl)

                ol = T.alloc_fragment(hidden_block, T.float32)
                T.clear(ol)

                for i_hc in T.serial(hc_mult):
                    pre = pre_mix_shared[i_hc]
                    for i1_h in T.Parallel(hidden_block):
                        ol[i1_h] += pre * xl[i_hc, i1_h]

                T.copy(ol, layer_input[i, i0_h * hidden_block])


# Copied from https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_mhc/example_mhc_pre.py
@tilelang.jit
def mhc_pre_gemm_sqrsum_tilelang(
    x,
    fn,
    out,
    sqrsum,
    hc_mult3: int,
    hc_hidden_size: int,
    token_block: int = 32,
    hidden_block: int = 256,
) -> tilelang.JITKernel:
    """Not highly optimized TileLang implementation of fused gemm and sqrsum in mHC pre block."""
    assert hc_mult3 <= 32  # should be 24 usually
    num_tokens = T.dynamic("num_tokens")
    assert hc_hidden_size % hidden_block == 0

    x: T.Tensor((num_tokens, hc_hidden_size), T.bfloat16)
    fn: T.Tensor((hc_mult3, hc_hidden_size), T.float32)
    out: T.Tensor((num_tokens, hc_mult3), T.float32)
    sqrsum: T.Tensor((num_tokens), T.float32)

    with T.Kernel(T.ceildiv(num_tokens, token_block)) as px:
        out_frag = T.alloc_fragment((token_block, 32), T.float32)
        sqrsum_part = T.alloc_fragment((token_block, 4), T.float32)
        T.clear(out_frag)
        T.clear(sqrsum_part)
        for pz in T.Pipelined(hc_hidden_size // hidden_block, num_stages=2):
            x_smem_16 = T.alloc_shared((token_block, hidden_block), T.bfloat16)
            fn_smem = T.alloc_shared((32, hidden_block), T.float32)

            T.annotate_layout({x_smem_16: tilelang.layout.make_swizzled_layout(x_smem_16)})

            T.copy(x[px * token_block, pz * hidden_block], x_smem_16)
            T.copy(fn[0, pz * hidden_block], fn_smem)

            x_frag_16 = T.alloc_fragment((token_block, hidden_block), T.bfloat16)
            T.copy(x_smem_16, x_frag_16)
            x_frag = T.alloc_fragment((token_block, hidden_block), T.float32)
            T.copy(x_frag_16, x_frag)

            for jj in T.serial(hidden_block // 4):
                for i, j in T.Parallel(token_block, 4):
                    sqrsum_part[i, j] += x_frag[i, jj * 4 + j] * x_frag[i, jj * 4 + j]

            # should be TF32 gemm
            T.gemm(
                x_frag,
                fn_smem,
                out_frag,
                transpose_A=False,
                transpose_B=True,
                wg_wait=0,
                clear_accum=False,
            )
        sqrsum_l = T.alloc_fragment(token_block, T.float32)
        T.reduce_sum(sqrsum_part, sqrsum_l)
        for i in T.Parallel(token_block):
            sqrsum[px * token_block + i] = sqrsum_l[i]
        for i, j in T.Parallel(token_block, 32):
            if j < hc_mult3:
                out[px * token_block + i, j] = out_frag[i, j]


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    },
)
def mhc_post_tilelang(
    a, b, c, d, x, hc: int, hidden: int, n_thr: int = 128, h_blk: int = 1024
) -> tilelang.JITKernel:
    # rename for shorter code
    n = T.dynamic("num_tokens")
    h = hidden

    h_blk = math.gcd(hidden, h_blk)
    a: T.Tensor((n, hc, hc), T.float32)
    b: T.Tensor((n, hc, h), T.bfloat16)
    c: T.Tensor((n, hc), T.float32)
    d: T.Tensor((n, h), T.bfloat16)
    x: T.Tensor((n, hc, h), T.bfloat16)
    with T.Kernel(n, threads=n_thr) as i_n:
        x_shared = T.alloc_shared((hc, h_blk), T.bfloat16)
        b_shared = T.alloc_shared((hc, h_blk), T.bfloat16)
        d_shared = T.alloc_shared(h_blk, T.bfloat16)

        x_local = T.alloc_fragment((hc, h_blk), T.float32)
        b_local = T.alloc_fragment((hc, h_blk), T.float32)
        d_local = T.alloc_fragment(h_blk, T.float32)

        a_local = T.alloc_fragment((hc, hc), T.float32)
        c_local = T.alloc_fragment(hc, T.float32)
        T.copy(a[i_n, 0, 0], a_local)
        T.copy(c[i_n, 0], c_local)

        for i0_h in T.Pipelined(T.ceildiv(h, h_blk), num_stages=2):
            T.copy(b[i_n, 0, i0_h * h_blk], b_shared)
            T.copy(d[i_n, i0_h * h_blk], d_shared)

            T.copy(b_shared, b_local)
            T.copy(d_shared, d_local)
            for i_hco, i1_h in T.Parallel(hc, h_blk):
                x_local[i_hco, i1_h] = c_local[i_hco] * d_local[i1_h]
                for i_hci in T.serial(hc):
                    x_local[i_hco, i1_h] += a_local[i_hci, i_hco] * b_local[i_hci, i1_h]
            T.copy(x_local, x_shared)

            T.copy(x_shared, x[i_n, 0, i0_h * h_blk])


# Triton kernels for mHC
@triton.jit
def mhc_pre_gemm_sqrsum_triton_kernel(
    x_ptr,
    fn_ptr,
    out_ptr,
    sqrsum_ptr,
    num_tokens,
    hc_mult3,
    hc_hidden_size,
    BLOCK_H: tl.constexpr,
):
    """Triton kernel computing one output element per program."""
    # Each program computes one element
    pid = tl.program_id(0)
    token_idx = pid // hc_mult3
    n_idx = pid % hc_mult3

    if token_idx >= num_tokens:
        return

    # Accumulate dot product
    acc = 0.0
    sqr_acc = 0.0

    h_idx = 0
    while h_idx < hc_hidden_size:
        h_offsets = h_idx + tl.arange(0, BLOCK_H)
        h_mask = h_offsets < hc_hidden_size

        # Load x values
        x_offsets = token_idx * hc_hidden_size + h_offsets
        x_vals = tl.load(x_ptr + x_offsets, mask=h_mask, other=0.0).to(tl.float32)

        # Load fn weights
        fn_offsets = n_idx * hc_hidden_size + h_offsets
        fn_vals = tl.load(fn_ptr + fn_offsets, mask=h_mask, other=0.0).to(tl.float32)

        # Accumulate
        acc += tl.sum(x_vals * fn_vals)

        # Compute sqrsum for first channel only
        if n_idx == 0:
            sqr_acc += tl.sum(x_vals * x_vals)

        h_idx += BLOCK_H

    # Store output
    out_offset = token_idx * hc_mult3 + n_idx
    tl.store(out_ptr + out_offset, acc)

    # Store sqrsum for first channel only
    if n_idx == 0:
        tl.store(sqrsum_ptr + token_idx, sqr_acc)


@triton.jit
def mhc_post_triton_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    x_ptr,
    num_tokens,
    hc,
    hidden,
    BLOCK_H: tl.constexpr,
):
    """Optimized Triton kernel for mHC post mapping.
    Computes: result = post * x + comb.mT @ residual
    Each program handles one (token, hc_out) combination.
    """
    pid = tl.program_id(0)

    token_idx = pid // hc
    i_hco = pid % hc

    if token_idx >= num_tokens:
        return

    # Load c value once for this output channel
    c_val = tl.load(c_ptr + token_idx * hc + i_hco)

    # Process hidden dimension in blocks
    for h_block in range(tl.cdiv(hidden, BLOCK_H)):
        h_start = h_block * BLOCK_H
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        h_mask = h_offsets < hidden

        # Load d values for this hidden block
        d_offsets = token_idx * hidden + h_offsets
        d_vals = tl.load(d_ptr + d_offsets, mask=h_mask, other=0.0).to(tl.float32)

        # Compute term1: c * d
        x_val = c_val * d_vals

        # Compute term2: sum over i_hci of a[token, i_hci, i_hco] * b[token, i_hci, h]
        for i_hci in range(hc):
            a_offset = token_idx * hc * hc + i_hci * hc + i_hco
            a_val = tl.load(a_ptr + a_offset)

            b_offsets = token_idx * hc * hidden + i_hci * hidden + h_offsets
            b_vals = tl.load(b_ptr + b_offsets, mask=h_mask, other=0.0).to(tl.float32)

            x_val += a_val * b_vals

        # Store result
        x_offsets = token_idx * hc * hidden + i_hco * hidden + h_offsets
        tl.store(x_ptr + x_offsets, x_val.to(tl.bfloat16), mask=h_mask)


def mhc_pre_gemm_sqrsum_triton(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
    hc_mult3: int,
    hc_hidden_size: int,
):
    """Triton kernel for fused gemm and sqrsum in mHC pre block."""
    # x: [num_tokens, hc_hidden_size]
    # fn: [hc_mult3, hc_hidden_size]
    # Compute out = x @ fn.T and sqrsum = sum(x^2)

    num_tokens = x.shape[0]

    # Define block size for hidden dimension
    BLOCK_H = 512

    # Launch one program per output element
    grid = (num_tokens * hc_mult3,)

    mhc_pre_gemm_sqrsum_triton_kernel[grid](
        x,
        fn,
        out,
        sqrsum,
        num_tokens,
        hc_mult3,
        hc_hidden_size,
        BLOCK_H=BLOCK_H,
    )


def mhc_post_triton(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    x: torch.Tensor,
    hc: int,
    hidden: int,
):
    """Triton kernel for mHC post mapping."""
    # a: comb_res_mix [num_tokens, hc, hc]
    # b: residual [num_tokens, hc, hidden]
    # c: post_layer_mix [num_tokens, hc]
    # d: x (layer output) [num_tokens, hidden]
    # x: output [num_tokens, hc, hidden]

    num_tokens = a.shape[0]

    # Define block size for hidden dimension
    BLOCK_H = 256  # Larger block for better memory coalescing

    # Launch one program per (token, hc_out) combination
    grid = (num_tokens * hc,)

    mhc_post_triton_kernel[grid](
        a,
        b,
        c,
        d,
        x,
        num_tokens,
        hc,
        hidden,
        BLOCK_H=BLOCK_H,
    )


def sinkhorn_normalize_ref(x: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


class mHC(nn.Module):
    def __init__(
        self,
        mult: int,
        hidden_size: int,
        sinkhorn_iters: int,
        dtype: Optional[torch.dtype] = None,
        eps: float = 1e-6,
        norm_eps: float = 1e-6,
        sinkhorn_eps: float = 1e-6,
        post_mult_value: float = 1.0,
        n_splits: int = 1,
        backend: str = "tilelang",
    ):
        super().__init__()
        self.mult = mult
        self.hidden_size = hidden_size
        self.sinkhorn_iters = sinkhorn_iters
        self.dtype = dtype
        self.eps = eps
        self.norm_eps = norm_eps
        self.sinkhorn_eps = sinkhorn_eps
        self.post_mult_value = post_mult_value
        self.n_splits = n_splits
        self.backend = backend
        self.mix_hc = (2 + self.mult) * self.mult
        self.hc_dim = self.mult * self.hidden_size

        # Parameters
        self.fn = Parameter(
            torch.empty((self.mix_hc, self.hc_dim), dtype=torch.float32), requires_grad=False
        ).cuda()
        self.base = Parameter(
            torch.empty((self.mix_hc,), dtype=torch.float32), requires_grad=False
        ).cuda()
        self.scale = Parameter(torch.empty((3,), dtype=torch.float32), requires_grad=False).cuda()

    def pre_mapping(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [b,s,hc,d], hc_fn: [mix_hc,hc*d], hc_scale: [3], hc_base: [mix_hc], y: [b,s,hc,d]
        if self.backend == "vanilla":
            assert self.mult == x.shape[-2]
            residual_flat = x.flatten(-2, -1).float()
            sqrsum = residual_flat.square().sum(-1)
            mixes = (
                residual_flat
                @ self.fn.T
                * (sqrsum.unsqueeze(-1) / self.fn.shape[-1] + self.norm_eps).rsqrt()
            )
            scale = torch.cat(
                [
                    self.scale[0].expand(self.mult),
                    self.scale[1].expand(self.mult),
                    self.scale[2].expand(self.mult * self.mult),
                ],
            )
            mixes = mixes * scale + self.base
            pre_mix = mixes[:, : self.mult].sigmoid().unsqueeze(-1) + self.eps
            post_mix = (
                mixes[:, self.mult : 2 * self.mult].sigmoid() * self.post_mult_value
            ).unsqueeze(-1)
            res_mix = mixes[:, 2 * self.mult :].view(-1, self.mult, self.mult)
            res_mix = sinkhorn_normalize_ref(
                res_mix, repeat=self.sinkhorn_iters, eps=self.sinkhorn_eps
            )
            layer_input = (x * pre_mix).sum(-2).bfloat16()
            return post_mix, res_mix, layer_input
        elif self.backend == "deepgemm":
            assert self.mult == x.shape[-2]
            gemm_m = x.shape[0]
            gemm_n, gemm_k = self.fn.shape
            residual_flat = x.flatten(-2, -1).to(torch.bfloat16)
            d = (
                torch.empty((gemm_m, gemm_n), dtype=torch.float, device="cuda")
                if self.n_splits == 1
                else torch.empty((self.n_splits, gemm_m, gemm_n), dtype=torch.float, device="cuda")
            )
            s = (
                torch.empty((gemm_m,), dtype=torch.float, device="cuda")
                if self.n_splits == 1
                else torch.empty((self.n_splits, gemm_m), dtype=torch.float, device="cuda")
            )
            num_splits = None if self.n_splits == 1 else self.n_splits
            tf32_hc_prenorm_gemm(residual_flat, self.fn, d, s, num_splits=num_splits)
            mixes = d * (s.unsqueeze(-1) / gemm_k + self.norm_eps).rsqrt()
            scale = torch.cat(
                [
                    self.scale[0].expand(self.mult),
                    self.scale[1].expand(self.mult),
                    self.scale[2].expand(self.mult * self.mult),
                ],
            )
            mixes = mixes * scale + self.base
            pre_mix = mixes[:, : self.mult].sigmoid().unsqueeze(-1) + self.eps
            post_mix = (
                mixes[:, self.mult : 2 * self.mult].sigmoid() * self.post_mult_value
            ).unsqueeze(-1)
            res_mix = mixes[:, 2 * self.mult :].view(-1, self.mult, self.mult)
            res_mix = sinkhorn_normalize_ref(
                res_mix, repeat=self.sinkhorn_iters, eps=self.sinkhorn_eps
            )
            layer_input = (x * pre_mix).sum(-2).bfloat16()
            return post_mix, res_mix, layer_input
        elif self.backend == "tilelang":
            # Validate shapes
            assert x.dtype == torch.bfloat16
            assert self.mult == x.shape[-2]
            assert self.hidden_size == x.shape[-1]
            hc_mult2 = self.mult * self.mult
            hc_mult3 = self.mult * 2 + hc_mult2

            hc_hidden_size = self.mult * self.hidden_size
            assert self.fn.shape[0] == hc_mult3
            assert self.fn.shape[1] == hc_hidden_size
            assert self.scale.shape == (3,)
            assert self.base.shape == (hc_mult3,)

            outer_shape = x.shape[:-2]

            residual_flat = x.view(-1, self.mult, self.hidden_size)
            num_tokens = residual_flat.shape[0]
            fn_flat = self.fn

            post_mix = torch.empty(num_tokens, self.mult, dtype=torch.float32, device=x.device)
            comb_mix = torch.empty(num_tokens, hc_mult2, dtype=torch.float32, device=x.device)
            layer_input = torch.empty(
                num_tokens, self.hidden_size, dtype=torch.bfloat16, device=x.device
            )

            gemm_out_mul = torch.empty(
                self.n_splits, num_tokens, hc_mult3, dtype=torch.float32, device=x.device
            )
            gemm_out_sqrsum = torch.empty(
                self.n_splits, num_tokens, dtype=torch.float32, device=x.device
            )
            assert self.n_splits == 1, (
                "The simple TileLang version gemm_sqrsum doesn't support split-k"
            )
            mhc_pre_gemm_sqrsum_tilelang(
                residual_flat.view(num_tokens, self.mult * self.hidden_size),
                fn_flat,
                gemm_out_mul.squeeze(0),
                gemm_out_sqrsum.squeeze(0),
                hc_mult3,
                self.mult * self.hidden_size,
            )
            mhc_pre_big_fuse_tilelang(
                gemm_out_mul,
                gemm_out_sqrsum,
                self.scale,
                self.base,
                residual_flat,
                post_mix,
                comb_mix,
                layer_input,
                self.hidden_size,
                self.norm_eps,
                self.eps,
                self.sinkhorn_eps,
                self.post_mult_value,
                self.sinkhorn_iters,
                self.n_splits,
                self.mult,
            )
            post_mix = post_mix.view(*outer_shape, self.mult, 1)
            comb_mix = comb_mix.view(*outer_shape, self.mult, self.mult)
            layer_input = layer_input.view(*outer_shape, self.hidden_size)
            return post_mix, comb_mix, layer_input
        elif self.backend == "triton":
            assert x.dtype == torch.bfloat16
            assert self.mult == x.shape[-2]
            assert self.hidden_size == x.shape[-1]
            hc_mult2 = self.mult * self.mult
            hc_mult3 = self.mult * 2 + hc_mult2

            hc_hidden_size = self.mult * self.hidden_size
            assert self.fn.shape[0] == hc_mult3
            assert self.fn.shape[1] == hc_hidden_size
            assert self.scale.shape == (3,)
            assert self.base.shape == (hc_mult3,)

            outer_shape = x.shape[:-2]

            residual_flat = x.view(-1, self.mult, self.hidden_size)
            num_tokens = residual_flat.shape[0]
            fn_flat = self.fn

            gemm_out = torch.empty(num_tokens, hc_mult3, dtype=torch.float32, device=x.device)
            gemm_sqrsum = torch.empty(num_tokens, dtype=torch.float32, device=x.device)

            mhc_pre_gemm_sqrsum_triton(
                residual_flat.view(num_tokens, self.mult * self.hidden_size),
                fn_flat,
                gemm_out,
                gemm_sqrsum,
                hc_mult3,
                self.mult * self.hidden_size,
            )

            mixes = gemm_out * (gemm_sqrsum.unsqueeze(-1) / hc_hidden_size + self.norm_eps).rsqrt()

            scale = torch.cat(
                [
                    self.scale[0].expand(self.mult),
                    self.scale[1].expand(self.mult),
                    self.scale[2].expand(self.mult * self.mult),
                ],
            )
            mixes = mixes * scale + self.base
            pre_mix = mixes[:, : self.mult].sigmoid().unsqueeze(-1) + self.eps
            post_mix = (
                mixes[:, self.mult : 2 * self.mult].sigmoid() * self.post_mult_value
            ).unsqueeze(-1)
            res_mix = mixes[:, 2 * self.mult :].view(-1, self.mult, self.mult)
            res_mix = sinkhorn_normalize_ref(
                res_mix, repeat=self.sinkhorn_iters, eps=self.sinkhorn_eps
            )
            layer_input = (residual_flat.float() * pre_mix).sum(-2).bfloat16()

            post_mix = post_mix.view(*outer_shape, self.mult, 1)
            comb_mix = res_mix.view(*outer_shape, self.mult, self.mult)
            layer_input = layer_input.view(*outer_shape, self.hidden_size)
            return post_mix, comb_mix, layer_input

    def post_mapping(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        if self.backend == "vanilla" or self.backend == "deepgemm":
            # x: [b,s,d], residual: [b,s,hc,d], post: [b,s,hc], comb: [b,s,hc,hc], y: [b,s,hc,d]
            # y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
            term2 = torch.bmm(comb_res_mix.mT, residual.float())
            return (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()
        elif self.backend == "tilelang":
            out = torch.empty_like(residual)
            mhc_post_tilelang(
                comb_res_mix,
                residual,
                post_layer_mix.squeeze(-1),
                x,
                out,
                residual.shape[-2],
                residual.shape[-1],
            )
            return out
        elif self.backend == "triton":
            outer_shape = residual.shape[:-2]
            residual_flat = residual.view(-1, residual.shape[-2], residual.shape[-1])
            x_flat = x.view(-1, x.shape[-1])
            comb_flat = comb_res_mix.view(-1, comb_res_mix.shape[-2], comb_res_mix.shape[-1])
            post_flat = post_layer_mix.view(-1, post_layer_mix.shape[-2])

            hc = residual_flat.shape[-2]
            hidden = residual_flat.shape[-1]

            out = torch.empty_like(residual_flat)

            mhc_post_triton(
                comb_flat,
                residual_flat,
                post_flat,
                x_flat,
                out,
                hc,
                hidden,
            )

            return out.view(*outer_shape, hc, hidden)


class HCHead(nn.Module):
    def __init__(
        self,
        mult: int,
        hidden_size: int,
        eps: float = 1e-6,
        norm_eps: float = 1e-6,
        backend: str = "tilelang",
    ):
        super().__init__()
        self.mult = mult
        self.hidden_size = hidden_size
        self.eps = eps
        self.norm_eps = norm_eps
        self.fn = nn.Parameter(
            torch.empty((self.mult, self.mult * self.hidden_size), dtype=torch.float32),
            requires_grad=False,
        ).cuda()
        self.base = nn.Parameter(
            torch.empty((self.mult,), dtype=torch.float32), requires_grad=False
        ).cuda()
        self.scale = nn.Parameter(
            torch.empty((1,), dtype=torch.float32), requires_grad=False
        ).cuda()
        self.backend = backend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backend == "vanilla":
            shape, dtype = x.size(), x.dtype
            x = x.flatten(-2, -1).float()
            rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
            mixes = F.linear(x, self.fn) * rsqrt
            pre = torch.sigmoid(mixes * self.scale + self.base) + self.eps
            y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
            return y.to(dtype)
        elif self.backend == "deepgemm":
            raise NotImplementedError("HC head GEMM n <= 32 is too small for DeepGEMM to support.")
        elif self.backend == "tilelang":
            # x: [b,s,hc,d], fn: [hc,hc*d], y: [b,s,d]
            shape, dtype = x.size(), x.dtype
            x_flat = x.flatten(-2, -1).to(torch.bfloat16)
            gemm_m = x_flat.shape[0]
            gemm_n, gemm_k = self.fn.shape
            d = torch.empty((gemm_m, gemm_n), dtype=torch.float, device=x.device)
            s = torch.empty((gemm_m,), dtype=torch.float, device=x.device)
            mhc_pre_gemm_sqrsum_tilelang(
                x_flat,
                self.fn,
                d,
                s,
                self.mult,
                self.mult * self.hidden_size,
            )
            mixes = d * (s.unsqueeze(-1) / gemm_k + self.norm_eps).rsqrt()
            pre = torch.sigmoid(mixes * self.scale + self.base) + self.eps
            y = torch.sum(pre.unsqueeze(-1) * x_flat.float().view(shape), dim=2)
            return y.to(dtype)
        elif self.backend == "triton":
            shape, dtype = x.size(), x.dtype
            x_flat = x.flatten(-2, -1).to(torch.bfloat16)
            gemm_m = x_flat.shape[0]
            gemm_n, gemm_k = self.fn.shape
            d = torch.empty((gemm_m, gemm_n), dtype=torch.float, device=x.device)
            s = torch.empty((gemm_m,), dtype=torch.float, device=x.device)
            mhc_pre_gemm_sqrsum_triton(
                x_flat,
                self.fn,
                d,
                s,
                self.mult,
                self.mult * self.hidden_size,
            )
            mixes = d * (s.unsqueeze(-1) / gemm_k + self.norm_eps).rsqrt()
            pre = torch.sigmoid(mixes * self.scale + self.base) + self.eps
            y = torch.sum(pre.unsqueeze(-1) * x_flat.float().view(shape), dim=2)
            return y.to(dtype)
