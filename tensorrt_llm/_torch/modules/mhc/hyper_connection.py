# Multi-Head Hyper-Connection (mHC) module
# Based on: "Hyper-Connections" (https://arxiv.org/abs/2409.19606)
from typing import Optional

import torch
from torch import nn

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

    def post_mapping(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
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


class HCHead(nn.Module):
    def __init__(
        self,
        mult: int,
        hidden_size: int,
        eps: float = 1e-6,
        norm_eps: float = 1e-6,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
