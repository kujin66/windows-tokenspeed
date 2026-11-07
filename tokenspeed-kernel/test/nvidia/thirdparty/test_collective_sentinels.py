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

import importlib.util
import os
import platform
import subprocess
from pathlib import Path

import pytest
import torch
import tvm_ffi

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.cuda is None,
    reason="NVIDIA CUDA required",
)


@pytest.fixture(scope="module")
def sentinel_module(tmp_path_factory):
    library = os.environ.get("TOKENSPEED_COLLECTIVE_SENTINEL_LIBRARY")
    if library:
        return tvm_ffi.load_module(library)
    csrc = (
        Path(__file__).resolve().parents[3]
        / "python/tokenspeed_kernel/thirdparty/cuda/csrc"
    )
    ffi_root = Path(tvm_ffi.__file__).parent
    includes = [csrc, csrc / "include", ffi_root / "include"]
    data_dir = os.environ.get("TOKENSPEED_FLASHINFER_DATA_DIR")
    if data_dir:
        data = Path(data_dir)
    else:
        flashinfer_spec = importlib.util.find_spec("flashinfer")
        if flashinfer_spec is None:
            raise RuntimeError("FlashInfer headers are required for collective tests")
        data = Path(flashinfer_spec.origin).parent / "data"
    includes.extend(
        data / name
        for name in [
            "csrc/nv_internal",
            "csrc/nv_internal/include",
            "include",
            "cutlass/include",
            "spdlog/include",
        ]
    )
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}" + ("a" if major >= 9 else "")
    flags = [
        "-std=c++17",
        "--expt-relaxed-constexpr",
        f"-gencode=arch=compute_{arch},code=sm_{arch}",
        "-DFLASHINFER_ENABLE_BF16",
        "-DFLASHINFER_ENABLE_F16",
    ]
    cuda = Path(
        os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    )
    build = tmp_path_factory.mktemp("collective-sentinel-build")
    output = build / ("sentinels.dll" if os.name == "nt" else "sentinels.so")
    if os.name == "nt":
        flags.extend(
            [
                "--use-local-env",
                "--compiler-options=/MD,/EHsc,/utf-8,/Zc:__cplusplus,/Zc:preprocessor",
            ]
        )
        target = (
            "arm64" if platform.machine().lower() in {"arm64", "aarch64"} else "x64"
        )
        links = [
            str(ffi_root / "lib/tvm_ffi.lib"),
            str(cuda / "lib" / target / "cuda.lib"),
        ]
    else:
        flags.append("--compiler-options=-fPIC")
        links = [f"-L{ffi_root / 'lib'}", "-ltvm_ffi", f"-L{cuda / 'lib64'}", "-lcuda"]
    # Use the caller's prepared compiler environment, including portable MSVC.
    nvcc = cuda / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
    command = [
        str(nvcc),
        "--shared",
        *flags,
        *[f"-I{path}" for path in includes],
        str(Path(__file__).with_name("collective_sentinel_test.cu")),
        *links,
        "-o",
        str(output),
    ]
    subprocess.run(command, cwd=build, check=True)
    return tvm_ffi.load_module(str(output))


def test_collective_sentinel_bits_and_sign_detection(sentinel_module):
    output = torch.empty((3, 3, 3), device="cuda", dtype=torch.int64)
    sentinel_module.collective_sentinels(output)
    expected = torch.tensor(
        [[[bits, 1, 0]] * 3 for bits in [0x8000, 0x8000, 0x80000000]],
        device="cuda",
        dtype=torch.int64,
    )
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_fmt_rounding_thresholds_preserve_integer_values(sentinel_module):
    for index in range(8):
        divisor = 10 ** (index + 1)
        expected = 2**31 + (2**32 + divisor - 1) // divisor
        assert sentinel_module.fmt_rounding_threshold(index) == expected


def test_public_half_math_matches_torch(sentinel_module):
    values = torch.tensor([-2, -1, 0, 1, 2], device="cuda", dtype=torch.float16)
    output = torch.empty((2, values.numel()), device="cuda", dtype=torch.float16)
    sentinel_module.half_math(values, output)
    torch.testing.assert_close(output[0], torch.exp2(values), rtol=0, atol=0)
    torch.testing.assert_close(output[1], torch.tanh(values), rtol=1e-3, atol=5e-4)
