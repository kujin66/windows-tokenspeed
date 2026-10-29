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

"""Compile the unchanged vendored router and its original host launch program."""

import importlib.util
from pathlib import Path

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream
from cutlass.cute.runtime import make_ptr


def compile_router(parameters):
    m, n, k = (parameters[name] for name in ("m", "n", "k"))
    block = parameters["block_size"]
    if m not in (1, 2, 4) or n != 896 or k != 7168 or block not in (64, 128, 256, 512):
        raise ValueError("Unsupported K3 router host specialization")
    spec = importlib.util.spec_from_file_location(
        "router_kernel", Path(__file__).with_name("_kernel.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    gemm = module.LLBf16Dotprod(
        k=k,
        bs=block,
        main_vec_width=8,
        tail_vec_width=4,
        use_pdl=False,
        out_dtype=cutlass.Float32,
        has_bias=False,
    )

    @cute.jit
    def launch(
        a: cute.Pointer,
        b: cute.Pointer,
        c: cute.Pointer,
        bias: cute.Pointer,
        stream: CUstream,
    ):
        ta = cute.make_tensor(a, cute.make_layout((m, k), stride=(k, 1)))
        tb = cute.make_tensor(b, cute.make_layout((n, k), stride=(k, 1)))
        tc = cute.make_tensor(c, cute.make_layout((m, n), stride=(n, 1)))
        t_bias = cute.make_tensor(bias, cute.make_layout(n))
        gemm(ta, tb, tc, t_bias, m, k, cutlass.Int32(n), stream)

    bf16 = make_ptr(cutlass.BFloat16, 0, cute.AddressSpace.gmem, assumed_align=16)
    fp32 = make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=16)
    return cute.compile(
        launch,
        bf16,
        bf16,
        fp32,
        fp32,
        CUstream(0),
        options="--ptxas-options -maxrregcount=64",
    )
