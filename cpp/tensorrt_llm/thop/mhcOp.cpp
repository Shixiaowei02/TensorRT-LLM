/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/kernels/mhcKernels/mhcKernels.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>
#include <torch/extension.h>

namespace tk = tensorrt_llm::kernels::mhc;

namespace
{

void mhcBigFuseOp(torch::Tensor y_acc, torch::Tensor r_acc, torch::Tensor residual, torch::Tensor hc_scale,
    torch::Tensor hc_base, torch::Tensor post_mix, torch::Tensor comb_mix, torch::Tensor layer_input, int64_t M,
    int64_t K, int64_t hidden_size, double rms_eps, double hc_pre_eps, double hc_sinkhorn_eps,
    double hc_post_mult_value, int64_t sinkhorn_repeat)
{
    auto stream = at::cuda::getCurrentCUDAStream();

    tk::mhcBigFuseLaunch(y_acc.data_ptr<float>(), r_acc.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16 const*>(residual.data_ptr<at::BFloat16>()), hc_scale.data_ptr<float>(),
        hc_base.data_ptr<float>(), post_mix.data_ptr<float>(), comb_mix.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(layer_input.data_ptr<at::BFloat16>()), static_cast<int>(M),
        static_cast<int>(K), static_cast<int>(hidden_size), static_cast<float>(rms_eps), static_cast<float>(hc_pre_eps),
        static_cast<float>(hc_sinkhorn_eps), static_cast<float>(hc_post_mult_value), static_cast<int>(sinkhorn_repeat),
        stream);
}

void mhcSqrsumOp(torch::Tensor x, torch::Tensor r, int64_t M, int64_t K)
{
    auto stream = at::cuda::getCurrentCUDAStream();

    tk::mhcSqrsumLaunch(reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr<at::BFloat16>()), r.data_ptr<float>(),
        static_cast<int>(M), static_cast<int>(K), stream);
}

void mhcGemmSqrsumFmaOp(torch::Tensor x, torch::Tensor w, torch::Tensor y, torch::Tensor r, int64_t M, int64_t N,
    int64_t K, int64_t num_k_blocks, int64_t k_chunk)
{
    auto stream = at::cuda::getCurrentCUDAStream();

    tk::mhcGemmSqrsumFmaLaunch(reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr<at::BFloat16>()), w.data_ptr<float>(),
        y.data_ptr<float>(), r.data_ptr<float>(), static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
        static_cast<int>(num_k_blocks), static_cast<int>(k_chunk), num_k_blocks > 1, stream);
}

void mhcGemmOp(torch::Tensor x, torch::Tensor w, torch::Tensor y, int64_t M, int64_t N, int64_t K)
{
    auto handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(handle, at::cuda::getCurrentCUDAStream());

    float alpha = 1.0f, beta = 0.0f;

    auto status
        = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
            &alpha, w.data_ptr(), CUDA_R_16BF, static_cast<int>(N), x.data_ptr(), CUDA_R_16BF, static_cast<int>(K),
            &beta, y.data_ptr(), CUDA_R_32F, static_cast<int>(N), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cublasGemmEx failed with status ", static_cast<int>(status));
}

void mhcHcHeadApplyOp(torch::Tensor mixes, torch::Tensor sqrsum, torch::Tensor x, torch::Tensor out,
    torch::Tensor scale, torch::Tensor base_t, int64_t M, int64_t mult, int64_t hidden_size, int64_t K, double norm_eps,
    double eps)
{
    auto stream = at::cuda::getCurrentCUDAStream();

    tk::mhcHcHeadApplyLaunch(mixes.data_ptr<float>(), sqrsum.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), scale.data_ptr<float>(),
        base_t.data_ptr<float>(), static_cast<int>(M), static_cast<int>(mult), static_cast<int>(hidden_size),
        static_cast<int>(K), static_cast<float>(norm_eps), static_cast<float>(eps), stream);
}

void mhcPostMappingOp(torch::Tensor residual, torch::Tensor x, torch::Tensor post_mix, torch::Tensor comb_mix,
    torch::Tensor out, int64_t B, int64_t hidden_size)
{
    auto stream = at::cuda::getCurrentCUDAStream();

    tk::mhcPostMappingLaunch(reinterpret_cast<__nv_bfloat16 const*>(residual.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr<at::BFloat16>()), post_mix.data_ptr<float>(),
        comb_mix.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), static_cast<int>(B),
        static_cast<int>(hidden_size), stream);
}

} // anonymous namespace

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "mhc_big_fuse("
        "Tensor y_acc, Tensor r_acc, Tensor residual, "
        "Tensor hc_scale, Tensor hc_base, "
        "Tensor(a!) post_mix, Tensor(b!) comb_mix, Tensor(c!) layer_input, "
        "int M, int K, int hidden_size, "
        "float rms_eps, float hc_pre_eps, float hc_sinkhorn_eps, "
        "float hc_post_mult_value, int sinkhorn_repeat) -> ()");

    m.def("mhc_sqrsum(Tensor x, Tensor(a!) r, int M, int K) -> ()");

    m.def(
        "mhc_gemm_sqrsum_fma("
        "Tensor x, Tensor w, Tensor(a!) y, Tensor(b!) r, "
        "int M, int N, int K, int num_k_blocks, int k_chunk) -> ()");

    m.def(
        "mhc_gemm("
        "Tensor x, Tensor w, Tensor(a!) y, "
        "int M, int N, int K) -> ()");

    m.def(
        "mhc_hc_head_apply("
        "Tensor mixes, Tensor sqrsum, Tensor x, Tensor(a!) out, "
        "Tensor scale, Tensor base_t, "
        "int M, int mult, int hidden_size, int K, "
        "float norm_eps, float eps) -> ()");

    m.def(
        "mhc_post_mapping("
        "Tensor residual, Tensor x, "
        "Tensor post_mix, Tensor comb_mix, Tensor(a!) out, "
        "int B, int hidden_size) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("mhc_big_fuse", &mhcBigFuseOp);
    m.impl("mhc_sqrsum", &mhcSqrsumOp);
    m.impl("mhc_gemm_sqrsum_fma", &mhcGemmSqrsumFmaOp);
    m.impl("mhc_gemm", &mhcGemmOp);
    m.impl("mhc_hc_head_apply", &mhcHcHeadApplyOp);
    m.impl("mhc_post_mapping", &mhcPostMappingOp);
}
