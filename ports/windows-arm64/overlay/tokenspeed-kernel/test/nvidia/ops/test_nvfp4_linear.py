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

"""NVFP4 decode batches reuse the same compiled GPU programs."""

import pytest
import torch
from tokenspeed_kernel.ops.gemm._triton.nvfp4 import (
    _nvfp4_mm_kernel,
    _nvfp4_quant_kernel,
    triton_nvfp4_linear,
)
from utils import assert_no_triton_compile


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM12x",
)
def test_nvfp4_batch_changes_do_not_compile():
    weight = torch.zeros(64, 64, dtype=torch.uint8, device="cuda")
    scales = torch.ones(64, 8, device="cuda").to(torch.float8_e4m3fn)
    gain = torch.ones(1, device="cuda")
    inputs = [
        torch.randn(batch, 128, device="cuda", dtype=torch.bfloat16)
        for batch in (1, 7, 16, 31, 32, 63)
    ]
    triton_nvfp4_linear(inputs[0], weight, scales, gain, gain, bias=None)
    with assert_no_triton_compile(_nvfp4_quant_kernel, _nvfp4_mm_kernel):
        for x in inputs[1:]:
            actual = triton_nvfp4_linear(x, weight, scales, gain, gain, bias=None)
            torch.testing.assert_close(actual, torch.zeros_like(actual), atol=0, rtol=0)
