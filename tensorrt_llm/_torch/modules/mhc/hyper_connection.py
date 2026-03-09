# Copied and modified from https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_mhc
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm.deep_gemm import tf32_hc_prenorm_gemm

try:
    from tensorrt_llm._torch.modules.mhc.mhc_cutile import (
        mhc_apply_residual as mhc_apply_residual_cutile,
    )
    from tensorrt_llm._torch.modules.mhc.mhc_cutile import (
        mhc_gemm_rms_scale as mhc_gemm_rms_scale_cutile,
    )
    from tensorrt_llm._torch.modules.mhc.mhc_cutile import (
        mhc_post_mapping as mhc_post_mapping_cutile,
    )
    from tensorrt_llm._torch.modules.mhc.mhc_cutile import (
        mhc_pre_mapping_fused as mhc_pre_mapping_fused_cutile,
    )

    _cutile_available = True
except Exception as _e:
    _cutile_available = False
    mhc_apply_residual_cutile = None
    mhc_gemm_rms_scale_cutile = None
    mhc_post_mapping_cutile = None
    mhc_pre_mapping_fused_cutile = None

try:
    from tensorrt_llm._torch.modules.mhc.mhc_tilelang import mhc_post as mhc_post_tilelang
    from tensorrt_llm._torch.modules.mhc.mhc_tilelang import (
        mhc_pre_big_fuse as mhc_pre_big_fuse_tilelang,
    )
    from tensorrt_llm._torch.modules.mhc.mhc_tilelang import (
        mhc_pre_gemm_sqrsum as mhc_pre_gemm_sqrsum_tilelang,
    )

    _tilelang_available = True
except Exception as _e:
    _tilelang_available = False
    mhc_post_tilelang = None
    mhc_pre_big_fuse_tilelang = None
    mhc_pre_gemm_sqrsum_tilelang = None

try:
    from tensorrt_llm._torch.modules.mhc.mhc_cuda import mhc_hc_head_cuda, mhc_post_mapping_cuda
    from tensorrt_llm._torch.modules.mhc.mhc_cuda import (
        mhc_pre_mapping_fused as mhc_pre_mapping_fused_cuda,
    )

    _cuda_available = True
except Exception as _e:
    _cuda_available = False
    mhc_hc_head_cuda = None
    mhc_post_mapping_cuda = None
    mhc_pre_mapping_fused_cuda = None


def _require_cutile():
    if not _cutile_available:
        raise RuntimeError(
            "CuTile backend is unavailable. Install cuda.tile or use a different backend."
        )


def _require_tilelang():
    if not _tilelang_available:
        raise RuntimeError(
            "TileLang backend is unavailable. Install tilelang or use a different backend."
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
        backend: str = "cuda",
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
        self.fn = nn.Parameter(
            torch.empty((self.mix_hc, self.hc_dim), dtype=torch.float32), requires_grad=False
        )
        self.base = nn.Parameter(
            torch.empty((self.mix_hc,), dtype=torch.float32), requires_grad=False
        )
        self.scale = nn.Parameter(torch.empty((3,), dtype=torch.float32), requires_grad=False)

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
        elif self.backend == "cutile":
            _require_cutile()
            assert x.dtype == torch.bfloat16
            assert self.mult == x.shape[-2]
            assert self.hidden_size == x.shape[-1]
            outer_shape = x.shape[:-2]
            residual_flat = x.view(-1, self.mult, self.hidden_size)
            num_tokens = residual_flat.shape[0]

            # Fully fused: GEMM+sqrsum kernel → big-fuse kernel
            # (replaces old finalize + PyTorch Sinkhorn + PyTorch apply_mix)
            post_mix, comb_mix, layer_input = mhc_pre_mapping_fused_cutile(
                residual_flat.view(num_tokens, self.hc_dim),
                self.fn.T.contiguous(),
                residual_flat,
                self.mult,
                self.scale,
                self.base,
                self.hidden_size,
                self.norm_eps,
                self.eps,
                self.sinkhorn_eps,
                self.post_mult_value,
                self.sinkhorn_iters,
            )

            post_mix = post_mix.view(*outer_shape, self.mult, 1)
            comb_mix = comb_mix.view(*outer_shape, self.mult, self.mult)
            layer_input = layer_input.view(*outer_shape, self.hidden_size)
            return post_mix, comb_mix, layer_input
        elif self.backend == "cuda":
            if not _cuda_available:
                raise RuntimeError(
                    "Raw CUDA backend is unavailable. "
                    "Ensure torch.utils.cpp_extension and CUDA toolkit are installed."
                )
            assert x.dtype == torch.bfloat16
            assert self.mult == x.shape[-2]
            assert self.hidden_size == x.shape[-1]
            outer_shape = x.shape[:-2]
            residual_flat = x.view(-1, self.mult, self.hidden_size)
            num_tokens = residual_flat.shape[0]

            post_mix, comb_mix, layer_input = mhc_pre_mapping_fused_cuda(
                residual_flat.view(num_tokens, self.hc_dim),
                self.fn.contiguous(),
                residual_flat,
                self.mult,
                self.scale,
                self.base,
                self.hidden_size,
                self.norm_eps,
                self.eps,
                self.sinkhorn_eps,
                self.post_mult_value,
                self.sinkhorn_iters,
            )

            post_mix = post_mix.view(*outer_shape, self.mult, 1)
            comb_mix = comb_mix.view(*outer_shape, self.mult, self.mult)
            layer_input = layer_input.view(*outer_shape, self.hidden_size)
            return post_mix, comb_mix, layer_input
        elif self.backend == "tilelang":
            _require_tilelang()
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
                self.fn,
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
        elif self.backend == "cutile":
            _require_cutile()
            outer_shape = residual.shape[:-2]
            n = self.mult
            hidden = residual.shape[-1]
            residual_flat = residual.view(-1, n, hidden)
            B = residual_flat.shape[0]

            out = mhc_post_mapping_cutile(
                residual_flat,
                x.reshape(B, hidden),
                post_layer_mix.view(B, n),
                comb_res_mix.view(B, n, n),
                n,
            )
            return out.view(*outer_shape, n, hidden)
        elif self.backend == "cuda":
            if not _cuda_available:
                raise RuntimeError(
                    "Raw CUDA backend is unavailable. "
                    "Ensure torch.utils.cpp_extension and CUDA toolkit are installed."
                )
            outer_shape = residual.shape[:-2]
            n = self.mult
            hidden = residual.shape[-1]
            residual_flat = residual.view(-1, n, hidden)
            B = residual_flat.shape[0]

            out = mhc_post_mapping_cuda(
                residual_flat,
                x.reshape(B, hidden),
                post_layer_mix.view(B, n),
                comb_res_mix.view(B, n, n),
                n,
            )
            return out.view(*outer_shape, n, hidden)
        elif self.backend == "tilelang":
            _require_tilelang()
            outer_shape = residual.shape[:-2]
            residual_flat = residual.view(-1, residual.shape[-2], residual.shape[-1])
            x_flat = x.view(-1, x.shape[-1])
            comb_flat = comb_res_mix.view(-1, comb_res_mix.shape[-2], comb_res_mix.shape[-1])
            post_flat = post_layer_mix.view(-1, post_layer_mix.shape[-2])

            hc = residual_flat.shape[-2]
            hidden = residual_flat.shape[-1]

            out = torch.empty_like(residual_flat)
            mhc_post_tilelang(
                comb_flat,
                residual_flat,
                post_flat.squeeze(-1),
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
        backend: str = "cuda",
    ):
        super().__init__()
        self.mult = mult
        self.hidden_size = hidden_size
        self.eps = eps
        self.norm_eps = norm_eps
        self.fn = nn.Parameter(
            torch.empty((self.mult, self.mult * self.hidden_size), dtype=torch.float32),
            requires_grad=False,
        )
        self.base = nn.Parameter(
            torch.empty((self.mult,), dtype=torch.float32), requires_grad=False
        )
        self.scale = nn.Parameter(torch.empty((1,), dtype=torch.float32), requires_grad=False)
        self.backend = backend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backend == "vanilla":
            shape, dtype = x.size(), x.dtype
            x = x.flatten(-2, -1).float()
            rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
            mixes = F.linear(x, self.fn) * rsqrt
            pre = torch.sigmoid(mixes * self.scale + self.base) + self.eps
            y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
            return y.to(dtype)
        elif self.backend == "deepgemm":
            raise NotImplementedError("HC head GEMM n <= 32 is too small for DeepGEMM to support.")
        elif self.backend == "cutile":
            _require_cutile()
            shape, dtype = x.size(), x.dtype
            x_flat = x.flatten(-2, -1)  # [batch, mult*hidden_size], bf16
            # mhc_gemm_rms_scale with n=mult: all columns are "pre" (sigmoid)
            y, _r = mhc_gemm_rms_scale_cutile(
                x_flat,
                self.fn.T,  # [mult*hidden_size, mult]
                self.mult,  # n=mult: all columns are pre (sigmoid)
                float(self.scale),  # alpha_pre
                0.0,  # alpha_post (no post columns)
                0.0,  # alpha_res (no res columns)
                self.base,  # bias
                norm_eps=self.norm_eps,
            )
            # y: [batch, mult] — all sigmoid columns
            pre = y + self.eps
            y = torch.sum(pre.unsqueeze(-1) * x_flat.float().view(shape), dim=1)
            return y.to(dtype)
        elif self.backend == "tilelang":
            _require_tilelang()
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
            y = torch.sum(pre.unsqueeze(-1) * x_flat.float().view(shape), dim=1)
            return y.to(dtype)
        elif self.backend == "cuda":
            if not _cuda_available:
                raise RuntimeError("CUDA MHC kernels not available")
            dtype = x.dtype
            x_bf16 = x.to(torch.bfloat16).contiguous()
            y = mhc_hc_head_cuda(
                x_bf16,
                self.fn,
                self.scale,
                self.base,
                self.mult,
                self.hidden_size,
                norm_eps=self.norm_eps,
                eps=self.eps,
            )
            return y.to(dtype)

    def skip_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Skip HCHead computation for pipeline parallelism on non-last ranks."""
        return x
