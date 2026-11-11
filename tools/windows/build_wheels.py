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

"""Build the four project wheels in a prepared native ARM64 venv."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import shutil
import struct
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run_build_command(
    command: tuple[str, ...], *, cwd: Path, env: dict[str, str]
) -> None:
    """Retry only Windows Rust temporary-archive sharing violations, at most twice."""
    for attempt in range(3):
        result = subprocess.run(
            command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        sys.stdout.buffer.write(result.stdout)
        sys.stdout.buffer.flush()
        if result.returncode == 0:
            return
        archive_lock = os.name == "nt" and all(
            message in result.stdout
            for message in (
                b"failed to remove temporary directory",
                b"temp-archive",
                b"(os error 32)",
            )
        )
        if not archive_lock or b"error[E" in result.stdout or attempt == 2:
            raise subprocess.CalledProcessError(
                result.returncode, command, output=result.stdout
            )
        print(
            f"Windows Rust temporary archive is locked; retrying ({attempt + 2}/3).",
            flush=True,
        )
        time.sleep(attempt + 1)


def pe_machine(contents: bytes) -> int:
    """Read the COFF machine field; reject truncated or non-PE inputs."""
    if len(contents) < 64 or contents[:2] != b"MZ":
        raise ValueError("Missing DOS header")
    offset = struct.unpack_from("<I", contents, 60)[0]
    if offset + 6 > len(contents) or contents[offset : offset + 4] != b"PE\0\0":
        raise ValueError("Missing PE signature")
    return struct.unpack_from("<H", contents, offset + 4)[0]


def audit_wheel(path: Path) -> dict:
    """Verify every native payload, including PE libraries named .so."""
    with zipfile.ZipFile(path) as archive:
        native = [
            n for n in archive.namelist() if n.endswith((".pyd", ".dll", ".exe", ".so"))
        ]
        for name in native:
            if pe_machine(archive.read(name)) != 0xAA64:
                raise ValueError(f"Not ARM64: {path.name}:{name}")
        if native and not path.name.endswith("-win_arm64.whl"):
            raise ValueError(
                f"Native payload in an incorrectly tagged wheel: {path.name}"
            )
    return {
        "wheel": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "native_files": native,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cuda-arch", required=True, help="nvcc architecture, e.g. 121a"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="New output directory"
    )
    args = parser.parse_args()
    if sys.platform != "win32" or platform.machine().lower() not in {
        "arm64",
        "aarch64",
    }:
        parser.error("Run with native Windows ARM64 Python")
    if sys.prefix == sys.base_prefix:
        parser.error("Run inside a dedicated venv; wheels will be installed there")
    for command in (
        "cl.exe",
        "clang.exe",
        "link.exe",
        "cargo.exe",
        "cmake.exe",
        "ninja.exe",
        "git.exe",
    ):
        if not shutil.which(command):
            parser.error(
                f"Missing {command}; activate an ARM64 MSVC developer environment"
            )
    cuda = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not cuda or not (Path(cuda) / "bin/nvcc.exe").is_file():
        parser.error("Set CUDA_HOME or CUDA_PATH to the Windows CUDA SDK")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    wheels = output / "wheels"
    wheels.mkdir()
    env = dict(os.environ)
    git_command = ["git", "-c", f"safe.directory={ROOT.as_posix()}"]
    revision = subprocess.check_output(
        [*git_command, "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    env.setdefault("TOKENSPEED_KERNEL_GIT_SHA", revision)
    env.update(
        PYTHONPATH="",
        PYO3_PYTHON=sys.executable,
        CMAKE_GENERATOR="Ninja",
        CUDA_HOME=cuda,
        TOKENSPEED_KERNEL_BACKEND="cuda",
        TOKENSPEED_CUDA_ARCH=args.cuda_arch,
        FLASHINFER_CUDA_ARCH_LIST=args.cuda_arch,
        DISTUTILS_USE_SDK="1",
        MSSdk="1",
    )
    env.setdefault("CARGO_TARGET_DIR", str(output / "cargo"))

    def run(*command: str) -> None:
        run_build_command(command, cwd=ROOT, env=env)

    for component in (
        "tokenspeed-scheduler",
        "tokenspeed-engine",
        "tokenspeed-kernel/python",
        "python",
    ):
        command = [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(ROOT / component),
            "--no-build-isolation",
            "--no-deps",
            "--no-index",
            "--wheel-dir",
            str(wheels),
        ]
        if component == "tokenspeed-scheduler":
            command.append(f"--config-settings=build-dir={output / 'scheduler'}")
            command.append(
                "--config-settings=cmake.args="
                "-DTOKENSPEED_SCHEDULER_BUILD_TESTS=ON;"
                "-DTOKENSPEED_SCHEDULER_BUILD_PYTHON=ON;-DINSTALL_GTEST=OFF"
            )
            command.append(
                "--config-settings=cmake.define.TOKENSPEED_SCHEDULER_BUILD_TESTS=ON"
            )
            command.append("--config-settings=cmake.define.INSTALL_GTEST=OFF")
        else:
            # Windows indexers can keep staging directories open after packaging.
            stage = output / f"stage-{component.replace('/', '-')}"
            command.extend(
                [
                    "--config-settings=--build-option=--keep-temp",
                    "--config-settings=--build-option="
                    + shlex.quote(f"--bdist-dir={stage.as_posix()}"),
                ]
            )
        run(*command)
    built = sorted(wheels.glob("*.whl"))
    if len(built) != 4:
        raise RuntimeError(f"Expected four wheels, found {len(built)}")
    audits = [audit_wheel(path) for path in built]
    native_groups = sorted(
        Path(name).parent.name
        for item in audits
        for name in item["native_files"]
        if "/thirdparty/cuda/objs/" in name
    )
    if len(native_groups) != 24 or len(set(native_groups)) != 24:
        raise RuntimeError(f"Expected all 24 native CUDA groups, found {native_groups}")
    diff = subprocess.check_output([*git_command, "diff", "HEAD", "--binary"], cwd=ROOT)
    report = {
        "revision": revision,
        "build_dependencies": {
            name: importlib.metadata.version(name)
            for name in ("torch", "apache-tvm-ffi", "tokenspeed-triton")
        },
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "python": platform.python_version(),
        "cuda_arch": args.cuda_arch,
        "kernel_groups": native_groups,
        "wheels": audits,
    }
    (output / "build.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--no-index",
        "--force-reinstall",
        *map(str, built),
    )


if __name__ == "__main__":
    main()
