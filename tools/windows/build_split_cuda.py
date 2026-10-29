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

import argparse
import importlib.util
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(
    description="CUDA frontend uses 14.44; generated host C++ and link use 14.51"
)
parser.add_argument("group", choices=["rmsnorm_fused_parallel", "sampling_chain"])
args = parser.parse_args()
spec = importlib.util.spec_from_file_location(
    "kernel_setup", root / "tokenspeed-kernel/python/setup.py"
)
m = importlib.util.module_from_spec(spec)
with patch("setuptools.setup"):
    spec.loader.exec_module(m)
b = m.CudaKernelBuilder([], verbose=True)
includes = b._resolve_include_dirs()
old = "C:/Program Files/Microsoft Visual Studio/2022/Enterprise/VC/Tools/MSVC/14.44.35207/bin/Hostarm64/arm64"
flags = [
    "-std=c++17",
    "-O3",
    "-DNDEBUG",
    "-use_fast_math",
    "--expt-relaxed-constexpr",
    "--compiler-options=/MD,/EHsc,/utf-8,/Zc:__cplusplus,/Zc:preprocessor",
    "-DFLASHINFER_ENABLE_BF16",
    "-DFLASHINFER_ENABLE_F16",
    "-DENABLE_BF16",
    "-DENABLE_FP8",
    "-gencode=arch=compute_121a,code=sm_121a",
]
output = root / ".port/split-cuda" / args.group
output.mkdir(parents=True, exist_ok=True)
objects = []
group = next(g for g in m.KERNEL_GROUPS if g[0] == args.group)
for source in group[1]:
    name = source.stem
    generated = output / (name + ".cu.cpp")
    subprocess.run(
        [
            m.NVCC,
            *flags,
            *["-I" + d for d in includes],
            "-ccbin",
            old,
            "--cuda",
            str(source),
            "-o",
            str(generated),
        ],
        check=True,
    )
    obj = output / (name + ".obj")
    subprocess.run(
        [
            "cl",
            "/nologo",
            "/std:c++17",
            "/O2",
            "/MD",
            "/EHsc",
            "/utf-8",
            "/Zc:__cplusplus",
            "/Zc:preprocessor",
            *["/I" + d for d in includes],
            "/c",
            str(generated),
            "/Fo" + str(obj),
        ],
        check=True,
    )
    objects.append(str(obj))
import tvm_ffi

subprocess.run(
    [
        "link.exe",
        "/DLL",
        *objects,
        "/LIBPATH:" + m.CUDA_HOME + "/lib/arm64",
        "cudart.lib",
        "cuda.lib",
        str(Path(tvm_ffi.__file__).parent / "lib/tvm_ffi.lib"),
        "/OUT:" + str(output / (args.group + ".dll")),
    ],
    check=True,
)
