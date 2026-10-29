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

"""Reproduce missing VS layout detection, repair it, and run a real CUDA kernel.

Invoke through tools/windows/run.py using the project Python environment.
Only the copied vcvarsall.bat in this repository's .port directory is moved.
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[2]
port = (root / ".port").resolve()
msvc = Path(os.environ["VCTOOLSINSTALLDIR"]).resolve()
marker = msvc.parents[2] / "Auxiliary/Build/vcvarsall.bat"
if not marker.is_relative_to(port):
    raise RuntimeError(
        f"Regression only permits the portable workspace toolchain: {marker}"
    )
original = marker.read_bytes()
backup = marker.with_suffix(".bat.regression-backup")
if backup.exists():
    raise RuntimeError(f"Existing regression backup must be inspected first: {backup}")


def run(command, log_name):
    result = subprocess.run(
        command,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    (port / log_name).write_text(result.stdout, encoding="utf-8")
    return result


with tempfile.TemporaryDirectory(prefix="nvcc1451-check-", dir=port) as directory:
    work = Path(directory)
    probe = work / "probe.cu"
    probe.write_text("#include <cuda_runtime.h>", encoding="utf-8")
    executable = work / "cuda-smoke.exe"
    marker.rename(backup)
    try:
        negative = run(
            [
                "nvcc",
                "--use-local-env",
                "-c",
                str(probe),
                "-o",
                str(work / "probe.obj"),
            ],
            "nvcc1451-layout-negative.log",
        )
        if (
            negative.returncode == 0
            or "Host compiler targets unsupported OS" not in negative.stdout
        ):
            raise AssertionError(
                f"Missing layout did not reproduce the expected detection error: {negative.stdout}"
            )
        positive = run(
            [
                sys.executable,
                str(root / "tools/windows/run.py"),
                "nvcc",
                "--use-local-env",
                "-std=c++17",
                "-arch=sm_121",
                "-Xcompiler",
                "/MD,/EHsc,/utf-8,/Zc:__cplusplus,/Zc:preprocessor",
                str(root / "tools/windows/cuda_smoke.cu"),
                "-o",
                str(executable),
                "-v",
            ],
            "nvcc1451-cuda-smoke-build.log",
        )
        if positive.returncode:
            raise AssertionError(positive.stdout)
        if marker.read_bytes() != original:
            raise AssertionError(
                "Automatic repair did not restore the official script contents"
            )
        gpu = run([str(executable)], "nvcc1451-cuda-smoke-run.log")
        if gpu.returncode:
            raise AssertionError(gpu.stdout)
        result = json.loads(gpu.stdout)
        if not (
            result["passed"]
            and result["msc_ver"] == 1951
            and result["msc_full_ver"] == 195136260
            and result["cuda_runtime_version"] == 13040
            and result["elements"] == 4097
        ):
            raise AssertionError(result)
        (port / "nvcc1451-cuda-smoke-result.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        report = {
            "passed": True,
            "missing_layout_exit_code": negative.returncode,
            "missing_layout_error": "Host compiler targets unsupported OS",
            "automatic_layout_repair_passed": True,
            "direct_nvcc_exit_code": positive.returncode,
            "vcvarsall_sha256": hashlib.sha256(original).hexdigest(),
            "gpu": result,
        }
        (port / "nvcc1451-layout-regression.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(
            "PASS: missing-layout failure reproduced; automatic layout repair, direct MSVC 14.51 build and GPU vector-add passed"
        )
    finally:
        # Restore only this workspace copy, including when a compiler check fails.
        if marker.exists():
            if marker.read_bytes() != original:
                raise RuntimeError(
                    f"Layout file changed unexpectedly; original retained at {backup}"
                )
            backup.unlink()
        else:
            backup.rename(marker)
