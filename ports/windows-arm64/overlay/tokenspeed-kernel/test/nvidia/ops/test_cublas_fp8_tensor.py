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

"""Real cuBLASLt FP8 execution, output buffers, bias and CUDA streams."""

import pytest
import torch
from tokenspeed_kernel.ops.gemm import mm
from tokenspeed_kernel.thirdparty.cuda.cublas_fp8 import cublas_fp8_tensor_available

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 12
    or not cublas_fp8_tensor_available(),
    reason="requires built SM12x cuBLASLt FP8 library",
)


def operands(m, n, k):
    torch.manual_seed(42)
    a = torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn).T
    return a, b, torch.tensor([0.25], device="cuda"), torch.tensor([0.5], device="cuda")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("m", [1, 7, 65])
@pytest.mark.parametrize("shape", [(8, 16), (64, 128), (128, 256)])
def test_native_cublas_fp8(dtype, m, shape):
    a, b, sa, sb = operands(m, *shape)
    actual = mm(
        a,
        b,
        A_scales=sa,
        B_scales=sb,
        out_dtype=dtype,
        override="cublaslt_mm_fp8_tensor_scaled",
    )
    expected = ((a.double() * sa.double()) @ (b.double() * sb.double())).to(dtype)
    torch.testing.assert_close(
        actual, expected, atol=1e-3, rtol=8e-3 if dtype == torch.bfloat16 else 1e-3
    )


@pytest.mark.parametrize("strided", [False, True])
def test_cublas_fp8_out_and_bias(strided):
    a, b, sa, sb = operands(7, 64, 128)
    storage = torch.empty(14 if strided else 7, 64, dtype=torch.bfloat16, device="cuda")
    out = storage[::2] if strided else storage
    bias = torch.randn(64, dtype=out.dtype, device="cuda")
    actual = mm(
        a,
        b,
        A_scales=sa,
        B_scales=sb,
        bias=bias,
        out=out,
        out_dtype=out.dtype,
        override="cublaslt_mm_fp8_tensor_scaled",
    )
    expected = ((a.double() * sa.double()) @ (b.double() * sb.double())).to(
        out.dtype
    ) + bias
    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=8e-3)


def test_cublas_fp8_cuda_streams():
    a, b, sa, sb = operands(7, 64, 128)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    results = []
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            results.append(
                mm(
                    a,
                    b,
                    A_scales=sa,
                    B_scales=sb,
                    out_dtype=torch.bfloat16,
                    override="cublaslt_mm_fp8_tensor_scaled",
                )
            )
    for stream in streams:
        stream.synchronize()
    expected = ((a.double() * sa.double()) @ (b.double() * sb.double())).to(
        torch.bfloat16
    )
    for actual in results:
        torch.testing.assert_close(actual, expected, atol=1e-3, rtol=8e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("m", [65, 129])
def test_portable_full_batch_fp8_token_quantization(dtype, m):
    from tokenspeed_kernel.ops.quantization.triton import (
        triton_quantize_fp8_per_token_portable,
    )

    torch.manual_seed(42)
    x = torch.randn(m, 128, device="cuda", dtype=dtype)
    x[0] = 0
    x[1] = 1e-10
    actual, scales = triton_quantize_fp8_per_token_portable(
        x,
        granularity="token",
        group_size=None,
        scale_encoding="float32",
        enable_pdl=False,
    )
    # Round independently evaluated IEEE divisions once to FP32. GPU scalar
    # division by a constant can instead multiply by a rounded reciprocal.
    amax = x.double().abs().amax(-1, keepdim=True)
    expected_scales = (amax / 448).clamp(min=1.0 / (448 * 512)).float()
    multiplier = (448.0 / amax).clamp(max=448 * 512).float()
    expected_codes = (x.float() * multiplier).clamp(-448, 448).to(torch.float8_e4m3fn)
    torch.testing.assert_close(scales, expected_scales, atol=0, rtol=0)
    torch.testing.assert_close(actual.float(), expected_codes.float(), atol=0, rtol=0)
