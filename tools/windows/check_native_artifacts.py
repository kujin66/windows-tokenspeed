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

"""Verify native ARM64 wheel binaries, installed files and compiler provenance.

Run through tools/windows/run.py after installing the wheels in dist/.
"""

import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
import struct
import sys
import zipfile
from ctypes import wintypes
from pathlib import Path

root = Path(__file__).resolve().parents[2]


def inspect_pe(data):
    if data[:2] != b"MZ":
        raise AssertionError("Expected Windows PE binary")
    offset = struct.unpack_from("<I", data, 0x3C)[0]
    if data[offset : offset + 4] != b"PE" + bytes(2):
        raise AssertionError("Invalid PE signature")
    machine = struct.unpack_from("<H", data, offset + 4)[0]
    if machine != 0xAA64:
        raise AssertionError(f"Expected native ARM64 (0xaa64), found {machine:#x}")
    return {
        "machine": hex(machine),
        "linker_version": list(data[offset + 26 : offset + 28]),
    }


if sys.platform != "win32" or struct.calcsize("P") != 8:
    raise AssertionError("Requires 64-bit Windows Python")
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
kernel32.GetCurrentProcess.restype = wintypes.HANDLE
kernel32.IsWow64Process2.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(wintypes.USHORT),
    ctypes.POINTER(wintypes.USHORT),
]
kernel32.IsWow64Process2.restype = wintypes.BOOL
process_machine, native_machine = wintypes.USHORT(), wintypes.USHORT()
if not kernel32.IsWow64Process2(
    kernel32.GetCurrentProcess(),
    ctypes.byref(process_machine),
    ctypes.byref(native_machine),
):
    raise ctypes.WinError(ctypes.get_last_error())
if process_machine.value != 0 or native_machine.value != 0xAA64:
    raise AssertionError("Python must run natively on ARM64 without WOW64 emulation")
python_pe = inspect_pe(Path(sys.executable).read_bytes())

records = []
for wheel_path in sorted((root / "dist").glob("*.whl")):
    with zipfile.ZipFile(wheel_path) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        package = next(
            line[6:]
            for line in archive.read(metadata_name).decode().splitlines()
            if line.startswith("Name: ")
        )
        distribution = importlib.metadata.distribution(package)
        wheel_info = archive.read(metadata_name.replace("METADATA", "WHEEL")).decode()
        tags = [
            line[5:] for line in wheel_info.splitlines() if line.startswith("Tag: ")
        ]
        natives, identities, checked_files = [], [], 0
        for name in archive.namelist():
            if not name.endswith(
                (".py", ".pyd", ".exe", ".dll", ".so", "build-identity.json")
            ):
                continue
            data = archive.read(name)
            installed = Path(distribution.locate_file(name))
            if not installed.is_file() or installed.read_bytes() != data:
                raise AssertionError(f"Installed file does not match wheel: {name}")
            checked_files += 1
            if name.endswith((".pyd", ".exe", ".dll", ".so")):
                pe = inspect_pe(data)
                if pe["linker_version"] != [14, 51]:
                    raise AssertionError(f"Expected MSVC linker 14.51 in {name}: {pe}")
                natives.append(
                    {
                        "filename": name,
                        **pe,
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "installed_matches_wheel": True,
                    }
                )
            if name.endswith("build-identity.json"):
                identity = json.loads(data)
                if (
                    identity["cuda_frontend_ccbin"]
                    or "19.51.36260 for ARM64" not in identity["host_compiler_version"]
                ):
                    raise AssertionError(f"Unexpected CUDA compiler provenance: {name}")
                if (
                    identity["cuda_version"] != [13, 4]
                    or "--use-local-env" not in identity["compile_flags"]
                ):
                    raise AssertionError(f"Unexpected CUDA flags/version: {name}")
                identities.append(identity)
        if natives and not all(tag.endswith("win_arm64") for tag in tags):
            raise AssertionError(
                f"Native wheel lacks ARM64 platform tag: {wheel_path.name}"
            )
        records.append(
            {
                "wheel": wheel_path.name,
                "sha256": hashlib.sha256(wheel_path.read_bytes()).hexdigest(),
                "tags": tags,
                "checked_installed_files": checked_files,
                "native_binaries": natives,
                "cuda_build_identities": identities,
            }
        )
if len(records) != 4 or sum(len(record["native_binaries"]) for record in records) != 6:
    raise AssertionError("Expected four project wheels with six native binaries")
cuda_manifest = json.loads((Path(os.environ["CUDA_PATH"]) / "version.json").read_text())
if cuda_manifest["cuda"]["version"] != "13.4.0":
    raise AssertionError("Expected CUDA SDK 13.4.0")
report = {
    "passed": True,
    "python": platform.python_version(),
    "python_pe": python_pe,
    "process_machine": hex(process_machine.value),
    "native_machine": hex(native_machine.value),
    "emulated": False,
    "cuda_sdk": cuda_manifest["cuda"]["version"],
    "cuda_nvcc": cuda_manifest["cuda_nvcc"]["version"],
    "artifacts": records,
}
(root / ".port/native-artifacts-verification.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
print(
    "PASS: native ARM64 Python without emulation; 4 installed wheels match; all 6 native binaries are ARM64 and linked by MSVC 14.51"
)
