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

"""Native CUDA/cuBLASLt FP8 tensor scaling through the project FFI library."""

import functools
import os
import sys
from pathlib import Path

import torch

_LIBRARY = Path(__file__).parent / "objs/cublas_fp8_tensor/cublas_fp8_tensor.so"
_DLL_HANDLES: list[object] = []


def cublas_fp8_tensor_available() -> bool:
    return _LIBRARY.is_file()


@functools.cache
def _load_module():
    import tvm_ffi

    if sys.platform == "win32":
        _DLL_HANDLES.append(
            os.add_dll_directory(str(Path(os.environ["CUDA_PATH"]) / "bin"))
        )
    return tvm_ffi.load_module(str(_LIBRARY))


@functools.cache
def _workspace(device: torch.device, stream: int):
    return torch.empty(8 * 1024 * 1024, device=device, dtype=torch.uint8)


def cublas_fp8_tensor(a, b, a_scale, b_scale, out):
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("cuBLASLt FP8 expects compatible two-dimensional matrices")
    if a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn:
        raise TypeError("cuBLASLt FP8 expects E4M3 operands")
    if not a.is_cuda or any(t.device != a.device for t in (b, a_scale, b_scale, out)):
        raise ValueError("cuBLASLt FP8 tensors must share a CUDA device")
    if any(s.numel() != 1 or s.dtype != torch.float32 for s in (a_scale, b_scale)):
        raise ValueError("cuBLASLt FP8 requires one FP32 scale per operand")
    if (
        out.dtype not in (torch.float16, torch.bfloat16)
        or out.shape != (a.shape[0], b.shape[1])
        or not out.is_contiguous()
    ):
        raise ValueError("cuBLASLt FP8 output must be contiguous FP16/BF16 [M,N]")
    if not a.is_contiguous() or not b.t().is_contiguous():
        raise ValueError("cuBLASLt FP8 requires row-major A and column-major B")
    if not a.shape[0] or not b.shape[1]:
        return out
    stream = torch.cuda.current_stream(a.device).cuda_stream
    _load_module().cublas_fp8_tensor(
        a, b, a_scale, b_scale, out, _workspace(a.device, stream), stream
    )
    return out
