# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Native cuBLASLt tensor-scaled FP8 GEMM."""

import torch
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import ScaleFormat, format_signatures
from tokenspeed_kernel.thirdparty.cuda.cublas_fp8 import (
    cublas_fp8_tensor,
    cublas_fp8_tensor_available,
)


def cublaslt_mm_fp8_tensor_scaled(
    A,
    B,
    A_scales,
    B_scales,
    out_dtype,
    *,
    alpha,
    block_size,
    out=None,
):
    if alpha is not None or block_size is not None:
        raise ValueError("cuBLASLt tensor-scaled FP8 takes no alpha or block_size")
    A, B = A.contiguous(), B.T.contiguous().T
    direct = out is not None and out.is_contiguous()
    result = (
        out
        if direct
        else torch.empty((A.shape[0], B.shape[1]), device=A.device, dtype=out_dtype)
    )
    if result.dtype != out_dtype:
        raise ValueError("FP8 output dtype differs from out_dtype")
    cublas_fp8_tensor(A, B, A_scales, B_scales, result)
    return out.copy_(result) if out is not None and not direct else result


if cublas_fp8_tensor_available():
    register_kernel(
        "gemm",
        "mm",
        name="cublaslt_mm_fp8_tensor_scaled",
        solution="cublaslt",
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(12, 1),
        ),
        signatures=format_signatures(
            ("a", "b"),
            "scaled-fp8",
            {torch.float8_e4m3fn},
            scale=ScaleFormat(storage_dtype=torch.float32, granularity="tensor"),
        ),
        traits={
            "n_align": frozenset({8}),
            "k_align": frozenset({16}),
            "a_inner_stride_one": frozenset({True}),
            "b_inner_stride_one": frozenset({False}),
            "out_dtype": frozenset({torch.float16, torch.bfloat16}),
        },
        priority=Priority.PERFORMANT + 3,
    )(cublaslt_mm_fp8_tensor_scaled)
