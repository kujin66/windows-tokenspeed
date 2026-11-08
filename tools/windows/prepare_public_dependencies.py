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

"""Prepare public ARM64 Torch/TVM/Triton wheels and FlashInfer build headers."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import zipfile
from pathlib import Path

PACKAGES = (
    {
        "name": "torch",
        "version": "2.16.0.dev20261101+cu134",
        "index": "https://pypi.nvidia.com/nvtorch_oot_nightly/",
        "filename": "torch-2.16.0.dev20261101+cu134-cp313-cp313-win_arm64.whl",
        "sha256": "318703ffdddbc71781b2585b75d168bdd561077a065abff078bc44df0f8234f7",
    },
    {
        "name": "apache-tvm-ffi",
        "version": "0.1.15.post2",
        "index": "https://pypi.org/simple",
        "filename": "apache_tvm_ffi-0.1.15.post2-cp313-cp313-win_arm64.whl",
        "sha256": "b7a95723af4332888bca44ea12d436d579b87b939b344c54b057818089429b40",
    },
    {
        "name": "flashinfer-python",
        "version": "0.6.17",
        "index": "https://pypi.org/simple",
        "filename": "flashinfer_python-0.6.17-py3-none-any.whl",
        "sha256": "ab7ce3aec42faf2f6079aa840c2407ac00ecd277a0524cdb708a8ff4a64dad90",
    },
    {
        "name": "triton-windows",
        "version": "3.9.0.post30",
        "index": "https://pypi.org/simple",
        "filename": "triton_windows-3.9.0.post30-cp312-abi3-win_arm64.whl",
        "sha256": "14efee65f0ae116996696fd3b22f8319002894d0772f8c6055f879fac814f0e1",
    },
)


def verify(directory: Path) -> list[dict]:
    verified = []
    for package in PACKAGES:
        path = directory / package["filename"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != package["sha256"]:
            raise ValueError(f"SHA256 mismatch: {path}")
        verified.append(dict(package, verified=True))
    return verified


def adapt_windows_fmt(headers: Path) -> dict:
    """Keep fmt's eight uint32 constants without out-of-range UTF-32 escapes."""
    path = headers / "spdlog/include/spdlog/fmt/bundled/format.h"
    original = path.read_bytes()
    expected = "63bd5a34d139bfca1727bfbaf941afbb5751c871adaaa91b77dce58a70723073"
    if hashlib.sha256(original).hexdigest() != expected:
        raise ValueError("Unexpected fmt header in pinned FlashInfer wheel")
    text = original.decode("utf-8")
    values = [
        0x9999999A,
        0x828F5C29,
        0x80418938,
        0x80068DB9,
        0x8000A7C6,
        0x800010C7,
        0x800001AE,
        0x8000002B,
    ]
    slash = chr(92)
    start = text.index('  return U"' + slash + "x9999999a")
    end = text.index(";", start) + 1
    high = (
        'u"' + "".join(slash + f"x{value >> 16:04x}" for value in values) + '"[index]'
    )
    low = (
        'u"' + "".join(slash + f"x{value & 65535:04x}" for value in values) + '"[index]'
    )
    replacement = (
        f"  return (static_cast<uint32_t>({high}) << 16u) |"
        + chr(10)
        + f"         static_cast<uint32_t>({low});"
    )
    path.write_text(
        text[:start] + replacement + text[end:], encoding="utf-8", newline=""
    )
    return {
        "path": path.relative_to(headers).as_posix(),
        "original_sha256": expected,
        "patched_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def adapt_windows_headers(headers: Path, *, patch: Path) -> list[dict]:
    """Apply the checked-in Windows syntax/ABI fixes to pinned build headers."""
    # Read as text so a Windows checkout's CRLF cannot alter hunk matching.
    patch_text = patch.read_text(encoding="utf-8")
    patch_data = patch_text.encode("utf-8")
    names = [
        line.split(" b/", 1)[1]
        for line in patch_text.splitlines()
        if line.startswith("diff --git ")
    ]
    original = {
        name: hashlib.sha256((headers / name).read_bytes()).hexdigest()
        for name in names
    }
    # Zero context avoids whitespace-only context lines being trimmed by hooks.
    apply_command = ["git", "apply", "--no-index", "--unidiff-zero"]
    # Even --no-index discovers enclosing repositories and silently skips paths
    # outside the current subdirectory. Treat extracted headers as standalone.
    env = dict(os.environ, GIT_CEILING_DIRECTORIES=str(headers.resolve().parent))
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE"):
        env.pop(name, None)
    subprocess.run(
        [*apply_command, "--check"],
        input=patch_data,
        cwd=headers,
        env=env,
        check=True,
    )
    subprocess.run(apply_command, input=patch_data, cwd=headers, env=env, check=True)
    records = [
        {
            "path": name,
            "original_sha256": original[name],
            "patched_sha256": hashlib.sha256((headers / name).read_bytes()).hexdigest(),
        }
        for name in names
    ]
    for record in records:
        if record["original_sha256"] == record["patched_sha256"]:
            raise ValueError(f"Compatibility patch was not applied: {record['path']}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if (
        sys.platform != "win32"
        or platform.machine().lower() not in {"arm64", "aarch64"}
        or sys.version_info[:2] != (3, 13)
    ):
        parser.error("The pinned wheels require native Windows ARM64 Python 3.13")
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.verify_only:
        for package in PACKAGES:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "download",
                    f"{package['name']}=={package['version']}",
                    "--no-deps",
                    "--only-binary=:all:",
                    "--index-url",
                    package["index"],
                    "--dest",
                    str(args.output),
                ],
                check=True,
            )
    records = verify(args.output)
    # Only headers/data are needed for native compilation. Installing FlashInfer's
    # runtime dependency graph would request Linux-only communication packages.
    headers = args.output.resolve() / "flashinfer-headers"
    with zipfile.ZipFile(args.output / PACKAGES[2]["filename"]) as archive:
        for name in archive.namelist():
            if not name.startswith("flashinfer/data/") or name.endswith("/"):
                continue
            target = headers / name.removeprefix("flashinfer/data/")
            if not target.resolve().is_relative_to(headers):
                raise ValueError(f"Invalid header archive entry: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    records[2]["build_data_dir"] = str(headers)
    records[2]["build_header_patches"] = [adapt_windows_fmt(headers)]
    patch = Path(__file__).with_name("flashinfer-build-compat.patch").resolve()
    records[2]["build_header_patches"].extend(
        adapt_windows_headers(headers, patch=patch)
    )
    records[2]["compatibility_patch_sha256"] = hashlib.sha256(
        patch.read_text(encoding="utf-8").encode("utf-8")
    ).hexdigest()
    (args.output / "public-dependencies.json").write_text(
        json.dumps(records, indent=2) + chr(10), encoding="utf-8"
    )
    print(
        "Verified NVIDIA Torch, PyPI TVM FFI, Triton Windows and FlashInfer build headers"
    )


if __name__ == "__main__":
    main()
