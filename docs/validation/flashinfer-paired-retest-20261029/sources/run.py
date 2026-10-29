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

import json
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", message="_get_vc_env is private;.*", category=UserWarning
    )
    from setuptools._distutils._msvccompiler import _get_vc_env

root = Path(__file__).resolve().parents[2]
env = dict(os.environ)
for key, value in _get_vc_env("arm64").items():
    env[key.upper()] = value
msvc = root / ".port/msvc/Contents/VC/Tools/MSVC/14.51.36231"
msvc = Path(env.get("TOKENSPEED_MSVC_ROOT", str(msvc)))
# nvcc recognizes MSVC by walking up from cl.exe to this VS layout file,
# even with --use-local-env. Raw compiler VSIX archives do not include it.
vcvarsall = msvc.parents[2] / "Auxiliary/Build/vcvarsall.bat"
if not vcvarsall.is_file():
    source = Path(env["VCINSTALLDIR"]) / "Auxiliary/Build/vcvarsall.bat"
    if not source.is_file():
        raise RuntimeError(
            f"Missing official Visual Studio environment script: {source}"
        )
    vcvarsall.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, vcvarsall)
old = Path(env["VCTOOLSINSTALLDIR"])
for key in ("PATH", "INCLUDE", "LIB", "LIBPATH"):
    env[key] = env.get(key, "").replace(str(old), str(msvc))
env["VCTOOLSINSTALLDIR"] = str(msvc) + os.sep
env["VCTOOLSVERSION"] = msvc.name

env["PATH"] = (
    str(msvc / "bin/Hostarm64/arm64")
    + os.pathsep
    + str(Path(sys.executable).parent)
    + os.pathsep
    + env["PATH"]
)
env["CUDA_HOME"] = env["CUDA_PATH"] = (
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4"
)
env["CUDA_VISIBLE_DEVICES"] = "0"
env["MAX_JOBS"] = "1"
env["PYTHONUTF8"] = "1"
env["CMAKE_BUILD_PARALLEL_LEVEL"] = "2"

# The Visual Studio generator can select its installed 14.44 toolset instead
# of the portable compiler on PATH. Ninja honors the prepared CC/CXX values.
env.setdefault("CMAKE_GENERATOR", "Ninja")
env["VIRTUAL_ENV"] = str(root / ".venv")
env["DISTUTILS_USE_SDK"] = "1"
env["MSSdk"] = "1"
env["CC"] = env["CXX"] = "cl"
env["LIB"] += os.pathsep + str(Path(sys.base_prefix) / "libs")
env["PATH"] = str(Path(sys.base_prefix)) + os.pathsep + env["PATH"]
for name, suffix in {
    "TRITON_CACHE_DIR": "triton",
    "FLASHINFER_WORKSPACE_BASE": "flashinfer",
    "FLASHINFER_CACHE_DIR": "flashinfer",
    "TOKENSPEED_FLASHINFER_AUTOTUNE_CACHE_DIR": "autotune",
}.items():
    env.setdefault(name, str(root / ".port" / suffix))
args = sys.argv[1:]
if args == ["--print-env"]:
    # Export only prepared values, never unrelated inherited credentials.
    json.dump(
        {key: value for key, value in env.items() if os.environ.get(key) != value},
        sys.stdout,
    )
    raise SystemExit(0)
if not args:
    raise SystemExit("Usage: run.py <command> [arguments] | --print-env")
args[0] = shutil.which(args[0], path=env["PATH"]) or args[0]
raise SystemExit(subprocess.call(args, env=env))
