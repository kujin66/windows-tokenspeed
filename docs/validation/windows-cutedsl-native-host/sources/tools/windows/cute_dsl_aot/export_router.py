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

"""Export the unmodified TokenSpeed CuTe BF16 dot-product router kernel."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import struct
import sys
from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr


def export(m, n, k, bs, output, source_module):
    gemm = source_module.LLBf16Dotprod(
        k=k,
        bs=bs,
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
        m: cutlass.Constexpr,
        n: cutlass.Constexpr,
        k: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        ta = cute.make_tensor(a, cute.make_layout((m, k), stride=(k, 1)))
        tb = cute.make_tensor(b, cute.make_layout((n, k), stride=(k, 1)))
        tc = cute.make_tensor(c, cute.make_layout((m, n), stride=(n, 1)))
        t_bias = cute.make_tensor(bias, cute.make_layout(n))
        gemm(ta, tb, tc, t_bias, m, k, cutlass.Int32(n), stream)

    bf16 = make_ptr(cutlass.BFloat16, 0, cute.AddressSpace.gmem, assumed_align=16)
    fp32 = make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=16)
    compiled = cute.compile(
        launch,
        bf16,
        bf16,
        fp32,
        fp32,
        m,
        n,
        k,
        cuda.CUstream(0),
        options="--keep-ptx --keep-cubin --ptxas-options -maxrregcount=64",
    )
    ptx = compiled.__ptx__
    if isinstance(ptx, bytes):
        ptx = ptx.decode()
    cubin = bytes(compiled.__cubin__)
    entries = re.findall(r"\.entry\s+([^\s(]+)\s*\((.*?)\)", ptx, re.S)
    if len(entries) != 1:
        raise ValueError(f"Expected one device kernel: {entries}")
    params = [x.strip() for x in entries[0][1].split(",")]
    # Static CuTe layouts leave only the four tensor pointers in the GPU ABI.
    if len(params) != 4 or any(
        not re.fullmatch(r"\.param \.align 8 \.b8 \w+\[8\]", p) for p in params
    ):
        raise ValueError(f"Unexpected device ABI: {params}")
    name = f"router-m{m}-n{n}-k{k}-bs{bs}"
    (output / f"{name}.cubin").write_bytes(cubin)
    (output / f"{name}.ptx").write_bytes(ptx.encode("utf-8"))
    return {
        "name": name,
        "shape_mnk": [m, n, k],
        "block": [bs, 1, 1],
        "grid": [n, 1, 1],
        "shared_memory_bytes": m * 4 * (bs // 32),
        "arch": "sm_121a",
        "input_dtype": "bfloat16",
        "output_dtype": "float32",
        "layout": "contiguous A[M,K], B[N,K], C[M,N]",
        "has_bias": False,
        "enable_pdl": False,
        "entry": entries[0][0],
        "abi": ["a:u64", "b:u64", "c:u64", "unused_bias:u64"],
        "ptx_parameters": params,
        "cubin_sha256": hashlib.sha256(cubin).hexdigest(),
        "ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest(),
        "cubin_bytes": len(cubin),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--block-sizes", nargs="+", type=int, required=True)
    args = parser.parse_args()
    if any(bs not in (64, 128, 256, 512) for bs in args.block_sizes):
        parser.error("Block sizes must be 64, 128, 256 or 512")
    args.output = args.output.resolve()
    args.source = args.source.resolve()
    spec = importlib.util.spec_from_file_location(
        "tokenspeed_ll_bf16_source", args.source
    )
    source_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source_module)
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["CUTE_DSL_ARCH"] = "sm_121a"
    os.environ["CUTE_DSL_DUMP_DIR"] = str(args.output)
    old_cwd = Path.cwd()
    os.chdir(args.output)
    try:
        # Model-size cases and vector/scalar tail cases. No split-K or clusters.
        shapes = [(m, 896, 7168, bs) for m in (1, 2, 4) for bs in args.block_sizes]
        shapes += [(1, 17, k, 256) for k in (8, 320, 1032, 2056)]
        cases = [export(*shape, args.output, source_module) for shape in shapes]
    finally:
        os.chdir(old_cwd)
    manifest = {
        "schema": 1,
        "compiler_host": {
            "os": sys.platform,
            "python": sys.version,
            "machine": platform.machine(),
        },
        "source": "tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cute_dsl/ll_bf16/_kernel.py",
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cutedsl": importlib.metadata.version("nvidia-cutlass-dsl"),
        "cuda_bindings": importlib.metadata.version("cuda-bindings"),
        "options": "--keep-ptx --keep-cubin --ptxas-options -maxrregcount=64",
        "kernels": cases,
    }
    if sys.platform == "win32":
        with Path(sys.executable).open("rb") as executable:
            executable.seek(0x3C)
            offset = struct.unpack("<I", executable.read(4))[0]
            executable.seek(offset + 4)
            manifest["compiler_host"]["python_pe_machine"] = hex(
                struct.unpack("<H", executable.read(2))[0]
            )
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f'Exported {len(cases)} kernels to {args.output / "manifest.json"}')


if __name__ == "__main__":
    main()
