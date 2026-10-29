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

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import (
    ScaleFormat,
    dense_tensor_format,
    format_signature,
    tensor_format,
)


@triton.jit(do_not_specialize=["M"])
def _nvfp4_quant_kernel(X, G, Q, S, M, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    x = tl.load(X + row * K + offsets, offsets < K, other=0).to(tl.float32)
    blocks = tl.reshape(x, [BLOCK // 16, 16])
    amax = tl.max(tl.abs(blocks), 1)
    global_inverse = tl.load(G)
    scale = (global_inverse * (amax / 6.0)).to(tl.float8e4nv)
    scale_float = scale.to(tl.float32)
    inverse = tl.where(scale_float > 0, global_inverse / scale_float, 0.0)
    normalized = tl.reshape(blocks * inverse[:, None], [BLOCK // 2, 2])
    low, high = tl.split(normalized)
    packed = tl.inline_asm_elementwise(
        "{ .reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, $2, $1; mov.b32 $0, {b, b, b, b}; }",
        "=r,f,f",
        [low, high],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    halves = tl.arange(0, BLOCK // 2)
    groups = tl.arange(0, BLOCK // 16)
    tl.store(Q + row * (K // 2) + halves, packed.to(tl.uint8), halves < K // 2)
    tl.store(S + row * (K // 16) + groups, scale, groups < K // 16)


def quantize_nvfp4_linear(x, global_inverse):
    if x.ndim != 2 or x.shape[1] % 16 or not x.is_contiguous():
        raise ValueError(
            "NVFP4 requires contiguous [M, K] input with K divisible by 16"
        )
    m, k = x.shape
    packed = torch.empty((m, k // 2), dtype=torch.uint8, device=x.device)
    scales = torch.empty((m, k // 16), dtype=torch.float8_e4m3fn, device=x.device)
    if m:
        _nvfp4_quant_kernel[(m,)](
            x, global_inverse, packed, scales, m, k, triton.next_power_of_2(k)
        )
    return packed, scales


@triton.jit(do_not_specialize=["M"])
def _nvfp4_mm_kernel(
    A,
    B,
    A_scales,
    B_scales,
    C,
    Input_global,
    Weight_global,
    # Activation-side geometry follows the batch and stays runtime; the
    # weight-side geometry is fixed for the process lifetime.
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bn: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_asm,
    stride_asg,
    stride_bsn: tl.constexpr,
    stride_bsg: tl.constexpr,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k_start in range(0, K, BLOCK_K):
        packed_k = k_start // 2 + tl.arange(0, BLOCK_K // 2)
        scale_k = k_start // 16 + tl.arange(0, BLOCK_K // 16)
        a = tl.load(
            A + offs_m[:, None] * stride_am + packed_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (packed_k[None, :] < K // 2),
            other=0.0,
        )
        a_scale = tl.load(
            A_scales + offs_m[:, None] * stride_asm + scale_k[None, :] * stride_asg,
            mask=(offs_m[:, None] < M) & (scale_k[None, :] < K // 16),
            other=0.0,
        )

        b = tl.load(
            B + offs_n[:, None] * stride_bn + packed_k[None, :] * stride_bk,
            mask=(offs_n[:, None] < N) & (packed_k[None, :] < K // 2),
            other=0.0,
        )
        b_scale = tl.load(
            B_scales + offs_n[:, None] * stride_bsn + scale_k[None, :] * stride_bsg,
            mask=(offs_n[:, None] < N) & (scale_k[None, :] < K // 16),
            other=0.0,
        )

        acc = tl.dot_scaled(
            a,
            a_scale,
            "e2m1",
            b.trans(),
            b_scale,
            "e2m1",
            acc=acc,
            fast_math=True,
        )

    acc *= 1.0 / (tl.load(Input_global) * tl.load(Weight_global))
    tl.store(
        C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@register_kernel(
    "gemm",
    "nvfp4_linear",
    name="triton_nvfp4_linear",
    solution="triton",
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(12, 0),
        max_arch_version=ArchVersion(12, 1),
    ),
    signatures=frozenset(
        format_signature(
            x=dense_tensor_format(dtype),
            weight=tensor_format(
                "nvfp4",
                torch.uint8,
                scale=ScaleFormat(
                    storage_dtype=torch.float8_e4m3fn,
                    granularity="block",
                    block_shape=(16,),
                ),
            ),
        )
        for dtype in (torch.float16, torch.bfloat16)
    ),
    traits={},
    priority=Priority.PORTABLE,
)
def triton_nvfp4_linear(
    x, weight, weight_scale, input_global_scale, weight_global_scale, *, bias
):
    m, k = x.shape
    n = weight.shape[0]
    if weight.shape[1] * 2 != k or weight_scale.shape != (n, k // 16):
        raise ValueError("NVFP4 weight/scale shapes do not match activation K")
    packed, scales = quantize_nvfp4_linear(x, input_global_scale)
    output = torch.empty((m, n), dtype=x.dtype, device=x.device)
    if m and n:
        _nvfp4_mm_kernel[(triton.cdiv(m, 16), triton.cdiv(n, 32))](
            packed,
            weight,
            scales,
            weight_scale,
            output,
            input_global_scale,
            weight_global_scale,
            m,
            n,
            k,
            packed.stride(0),
            packed.stride(1),
            weight.stride(0),
            weight.stride(1),
            scales.stride(0),
            scales.stride(1),
            weight_scale.stride(0),
            weight_scale.stride(1),
            output.stride(0),
            output.stride(1),
            BLOCK_M=16,
            BLOCK_N=32,
            BLOCK_K=128,
            num_warps=4,
        )
    if bias is not None:
        output.add_(bias)
    return output
