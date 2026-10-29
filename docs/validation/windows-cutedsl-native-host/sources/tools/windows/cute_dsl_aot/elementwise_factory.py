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

"""Example factory for the Windows compiler runner; no Torch dependency."""

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream
from cutlass.cute.runtime import make_ptr


@cute.kernel
def elementwise(
    a: cute.Pointer,
    b: cute.Pointer,
    c: cute.Pointer,
    n: cutlass.Int32,
    operation: cutlass.Constexpr,
):
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    threads, _, _ = cute.arch.block_dim()
    i = bx * threads + tx
    if i < n:
        layout = cute.make_layout(n)
        ta = cute.make_tensor(a, layout)
        tb = cute.make_tensor(b, layout)
        tc = cute.make_tensor(c, layout)
        if cutlass.const_expr(operation == "add"):
            tc[i] = ta[i] + tb[i]
        else:
            tc[i] = ta[i] * tb[i]


def compile_elementwise(parameters):
    """Return a compiled add/multiply kernel with three pointers and int32 N."""
    operation = parameters["operation"]
    block_size = parameters["block_size"]
    if operation not in ("add", "multiply") or block_size not in (128, 256):
        raise ValueError("Explicit add/multiply and block size 128/256 are required")

    @cute.jit
    def launch(
        a: cute.Pointer,
        b: cute.Pointer,
        c: cute.Pointer,
        n: cutlass.Int32,
        stream: CUstream,
    ):
        elementwise(a, b, c, n, operation).launch(
            grid=(cute.ceil_div(n, block_size), 1, 1),
            block=(block_size, 1, 1),
            stream=stream,
        )

    pointer = make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=16)
    return cute.compile(
        launch,
        pointer,
        pointer,
        pointer,
        cutlass.Int32(1),
        CUstream(0),
        options="--keep-ptx --keep-cubin",
    )


@cute.kernel
def affine(
    a: cute.Pointer,
    c: cute.Pointer,
    n: cutlass.Int32,
    stride: cutlass.Int64,
    scale: cutlass.Float32,
    offset: cutlass.Float32,
):
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    threads, _, _ = cute.arch.block_dim()
    i = bx * threads + tx
    if i < n:
        ta = cute.make_tensor(a, cute.make_layout(n * stride))
        tc = cute.make_tensor(c, cute.make_layout(n))
        tc[i] = ta[cutlass.Int64(i) * stride] * scale + offset


def compile_affine(parameters):
    """A custom runtime scalar/stride ABI, independent of the router."""
    block_size = parameters["block_size"]
    if block_size not in (128, 256):
        raise ValueError("Explicit block size 128/256 is required")

    @cute.jit
    def launch(
        a: cute.Pointer,
        c: cute.Pointer,
        n: cutlass.Int32,
        stride: cutlass.Int64,
        scale: cutlass.Float32,
        offset: cutlass.Float32,
        stream: CUstream,
    ):
        affine(a, c, n, stride, scale, offset).launch(
            grid=(cute.ceil_div(n, block_size), 1, 1),
            block=(block_size, 1, 1),
            stream=stream,
        )

    pointer = make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=16)
    return cute.compile(
        launch,
        pointer,
        pointer,
        cutlass.Int32(1),
        cutlass.Int64(1),
        cutlass.Float32(1),
        cutlass.Float32(0),
        CUstream(0),
        options="--keep-ptx --keep-cubin",
    )


@cute.kernel
def transpose(a: cute.Pointer, c: cute.Pointer, m: cutlass.Int32, n: cutlass.Int32):
    tx, ty, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    col, row = bx * 16 + tx, by * 16 + ty
    if row < m and col < n:
        ta = cute.make_tensor(a, cute.make_layout((m, n), stride=(n, 1)))
        tc = cute.make_tensor(c, cute.make_layout((n, m), stride=(m, 1)))
        tc[col, row] = ta[row, col]


def compile_transpose(parameters):
    """A custom two-dimensional grid and runtime matrix dimensions."""
    if parameters:
        raise ValueError("Transpose has no compile-time parameters")

    @cute.jit
    def launch(
        a: cute.Pointer,
        c: cute.Pointer,
        m: cutlass.Int32,
        n: cutlass.Int32,
        stream: CUstream,
    ):
        transpose(a, c, m, n).launch(
            grid=(cute.ceil_div(n, 16), cute.ceil_div(m, 16), 1),
            block=(16, 16, 1),
            stream=stream,
        )

    pointer = make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=16)
    return cute.compile(
        launch,
        pointer,
        pointer,
        cutlass.Int32(1),
        cutlass.Int32(1),
        CUstream(0),
        options="--keep-ptx --keep-cubin",
    )
