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

"""Compile and execute CuTe's own host launch code as a native ARM64 DLL."""

import ctypes
import hashlib
import inspect
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import torch
from tokenspeed_kernel.cute_dsl_compile import (
    _require_preparation,
    compile_cached_host_artifact,
    pe_machine,
    validate_artifact,
)
from tokenspeed_kernel.cute_dsl_runtime import _SCALARS

_C_TYPES = {
    "int8_t": ctypes.c_int8,
    "uint8_t": ctypes.c_uint8,
    "int16_t": ctypes.c_int16,
    "uint16_t": ctypes.c_uint16,
    "int32_t": ctypes.c_int32,
    "uint32_t": ctypes.c_uint32,
    "int64_t": ctypes.c_int64,
    "uint64_t": ctypes.c_uint64,
    "void *": ctypes.c_void_p,
    "void*": ctypes.c_void_p,
}
_DTYPES = {
    "Float16": torch.float16,
    "BFloat16": torch.bfloat16,
    "Float32": torch.float32,
    "Float64": torch.float64,
    "Int8": torch.int8,
    "Uint8": torch.uint8,
    "Int16": torch.int16,
    "Int32": torch.int32,
    "Int64": torch.int64,
    "Boolean": torch.bool,
}
_modules = {}
_module_lock = threading.Lock()


def build_host_artifact(
    *,
    compiler_python: Path,
    source: Path,
    factory: str,
    parameters: dict,
    arch: str,
    cache: Path,
    dependencies: tuple[Path, ...],
    cuda_home: Path,
    cxx: Path,
) -> Path:
    """Compile device code and its ARM64 host launcher, then link a cached DLL.

    Args:
        compiler_python, source, factory, parameters, arch, dependencies: Same
            contracts as compile_cached_host_artifact. No device ABI or launch
            geometry is supplied by the caller; CuTe exports its host program.
        cache: Root for the exported compiler artifacts and native DLL cache.
        cuda_home: Explicit CUDA SDK with native ARM64 libraries.
        cxx: MSVC cl.exe in a prepared ARM64 compiler environment (INCLUDE/LIB).

    Returns a new or validated existing artifact directory. Compilation and
    native linking run only during preparation. Inputs/toolchains must remain
    immutable for the build; declare all imported source files as dependencies.
    """
    _require_preparation()
    cache, cuda_home, cxx = (Path(p).resolve() for p in (cache, cuda_home, cxx))
    exported = compile_cached_host_artifact(
        compiler_python=compiler_python,
        source=source,
        factory=factory,
        parameters=parameters,
        arch=arch,
        cache=cache / "export",
        dependencies=dependencies,
    )
    runtime = Path(__file__).with_name("_cute_dsl_native.cpp")
    tool_files = [
        cxx,
        cxx.with_name("link.exe"),
        runtime,
        Path(__file__),
        cuda_home / "lib/arm64/cudart.lib",
        cuda_home / "lib/arm64/cuda.lib",
        *(
            cuda_home / "include" / name
            for name in (
                "cuda.h",
                "cuda_runtime_api.h",
                "cuda_runtime.h",
                "driver_types.h",
            )
        ),
    ]
    identity = {
        "export": hashlib.sha256((exported / "artifact.json").read_bytes()).hexdigest(),
        "tool_files": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tool_files
        },
        "environment": {
            name: os.environ.get(name, "")
            for name in ("PATH", "INCLUDE", "LIB", "CL", "_CL_", "LINK")
        },
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    destination = cache / "native" / digest
    if destination.exists():
        manifest = validate_artifact(destination)
        if manifest["native_build"] != identity:
            raise ValueError("Native CuTe cache identity mismatch")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="cute-native-", dir=destination.parent
    ) as temp:
        staging = Path(temp)
        manifest = validate_artifact(exported)
        for name in (
            "artifact.json",
            "kernel.cubin",
            "kernel.ptx",
            *manifest["host_files"],
        ):
            shutil.copyfile(exported / name, staging / name)
        if int.from_bytes((staging / "host.o").read_bytes()[:2], "little") != 0xAA64:
            raise ValueError("CuTe did not export an ARM64 COFF host object")
        command = [
            str(cxx),
            "/nologo",
            "/LD",
            "/MD",
            "/EHsc",
            "/std:c++17",
            "/O2",
            f"/I{cuda_home / 'include'}",
            str(staging / "adapter.cpp"),
            str(runtime),
            "/link",
            str(staging / "host.o"),
            f"/LIBPATH:{cuda_home / 'lib/arm64'}",
            "cudart.lib",
            "cuda.lib",
            "/MACHINE:ARM64",
            f"/OUT:{staging / 'host.dll'}",
            f"/IMPLIB:{staging / 'host.lib'}",
        ]
        run = subprocess.run(
            command,
            cwd=staging,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        (destination.parent / f"{digest}-build.log").write_text(
            run.stdout + run.stderr, encoding="utf-8"
        )
        if run.returncode:
            raise RuntimeError(
                f"Native CuTe host link failed: {(run.stdout + run.stderr)[-8000:]}"
            )
        if pe_machine(staging / "host.dll") != 0xAA64:
            raise ValueError("Native CuTe DLL is not Windows ARM64")
        if any(
            hashlib.sha256(Path(p).read_bytes()).hexdigest() != h
            for p, h in identity["tool_files"].items()
        ):
            raise RuntimeError("Native toolchain inputs changed during compilation")
        manifest["native_build"] = identity
        manifest["compile_inputs"] = json.loads(
            (exported / "cache-inputs.json").read_text(encoding="utf-8")
        )["inputs"]
        manifest["host_files"]["host.dll"] = hashlib.sha256(
            (staging / "host.dll").read_bytes()
        ).hexdigest()
        (staging / "artifact.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        validate_artifact(staging)
        try:
            staging.rename(destination)
        except FileExistsError:
            existing = validate_artifact(destination)
            if existing["native_build"] != identity:
                raise ValueError("Native CuTe cache identity mismatch")
    return destination


class _LoadedHost:
    def __init__(self, directory: Path, *, device: torch.device, cuda_home: Path):
        self.device = device
        self.dll_directory = os.add_dll_directory(str(cuda_home / "bin"))
        self.library = ctypes.CDLL(str(directory / "host.dll"))
        ptr = ctypes.c_void_p
        self.library.ts_prepare.argtypes = [ctypes.c_int]
        self.library.ts_prepare.restype = ctypes.c_int
        self.library.ts_release.argtypes = []
        self.library.ts_release.restype = ctypes.c_int
        self.library.ts_parameter_size.argtypes = [ctypes.c_int]
        self.library.ts_parameter_size.restype = ctypes.c_size_t
        self.library.ts_run.argtypes = [ctypes.POINTER(ptr)]
        self.library.ts_run.restype = ctypes.c_int
        torch.empty(0, device=device)
        code = self.library.ts_prepare(device.index)
        if code:
            raise RuntimeError(f"CUDA host module initialization failed: {code}")


class CuTeHostProgram:
    """Call the original compiled host program from native ARM64 Python.

    Args:
        directory: A build_host_artifact result with verified DLL/object hashes.
        device: Explicit current CUDA device. A loaded DLL is bound to this
            device; multiple-device reuse of the same DLL is rejected.
        cuda_home: SDK directory used to resolve the native CUDA runtime DLL.

    Pointers, scalars and flat fake-tensor descriptors adapt automatically;
    current Torch streams fill cudaStream_t arguments. Other by-value aggregates
    require exact-size bytes. No files are read and no compiler/load operations
    run during warmed calls or graph capture. Loaded DLLs and CUDA libraries
    remain alive for the process lifetime, including all captured graph replays.
    """

    def __init__(self, directory: Path, *, device: torch.device, cuda_home: Path):
        _require_preparation()
        if pe_machine(Path(sys.executable)) != 0xAA64:
            raise RuntimeError("Native CuTe host programs require Windows ARM64 Python")
        directory, cuda_home = Path(directory).resolve(), Path(cuda_home).resolve()
        self.device = torch.device(device)
        if (
            self.device.type != "cuda"
            or self.device.index is None
            or torch.cuda.current_device() != self.device.index
        ):
            raise ValueError(
                "Select an explicit CUDA device before loading a host program"
            )
        self.metadata = validate_artifact(directory)
        major, minor = torch.cuda.get_device_capability(self.device)
        if not re.fullmatch(rf"sm_{major}{minor}[af]?", self.metadata["arch"]):
            raise ValueError("Host program architecture differs from selected GPU")
        if (
            self.metadata["host"]["target"] != "aarch64-pc-windows-msvc"
            or pe_machine(directory / "host.dll") != 0xAA64
        ):
            raise ValueError("Host program is not a native Windows ARM64 DLL")
        self.parameters = self.metadata["host"]["parameters"]
        self.signature = inspect.Signature(
            [
                inspect.Parameter(p["name"], inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for p in self.parameters
                if p["kind"] != "stream"
            ]
        )
        self.parameter_types = []
        with _module_lock:
            loaded = _modules.get(directory)
            if loaded is None:
                loaded = _LoadedHost(directory, device=self.device, cuda_home=cuda_home)
                _modules[directory] = loaded
            elif loaded.device != self.device:
                raise ValueError(
                    "This native host DLL is already bound to another CUDA device"
                )
        self.loaded = loaded
        for i, p in enumerate(self.parameters):
            kind = p["kind"]
            if kind in ("ptr", "stream"):
                ctype = ctypes.c_void_p
            elif kind in _SCALARS:
                ctype = _SCALARS[kind]
            elif kind == "bool":
                ctype = ctypes.c_bool
            elif kind == "tensor":
                fields = []
                for field in p["fields"]:
                    scalar = _C_TYPES[field["ctype"]]
                    fields.append(
                        (
                            field["name"],
                            (
                                scalar * field["count"]
                                if field["count"] is not None
                                else scalar
                            ),
                        )
                    )
                descriptor = type(
                    f"Tensor_{p['name']}", (ctypes.Structure,), {"_fields_": fields}
                )
                ctype = (
                    ctypes.POINTER(descriptor)
                    if p["ctype"].endswith("*")
                    else descriptor
                )
            elif kind == "opaque":
                ctype = ctypes.c_char * loaded.library.ts_parameter_size(i)
            else:
                raise ValueError(f"Unsupported host parameter kind: {kind}")
            if ctypes.sizeof(ctype) != loaded.library.ts_parameter_size(i):
                raise ValueError(f"Host C ABI size mismatch for {p['name']}")
            self.parameter_types.append(ctype)

    def _tensor_pointer(self, p, value):
        if not isinstance(value, torch.Tensor) or value.device != self.device:
            raise ValueError(
                f"{p['name']} requires a tensor on the preloaded CUDA device"
            )
        dtype = _DTYPES.get(p["dtype"])
        if dtype is None:
            raise ValueError(f"No automatic Torch adapter for CuTe dtype {p['dtype']}")
        if value.dtype != dtype:
            raise ValueError(f"{p['name']} dtype differs from the compiled contract")
        if value.data_ptr() % p["alignment"]:
            raise ValueError(f"{p['name']} pointer violates compiled alignment")
        return value.data_ptr()

    @staticmethod
    def _dimensions(contracts, values, symbols):
        dynamic = []
        if len(contracts) != len(values):
            raise ValueError("Tensor rank differs from compiled host layout")
        for contract, value in zip(contracts, values):
            if "static" in contract:
                if value != contract["static"]:
                    raise ValueError("Tensor static shape/stride contract mismatch")
            else:
                if value % contract["divisibility"] or not 0 <= value < 2 ** (
                    contract["width"] - 1
                ):
                    raise ValueError(
                        "Tensor dimension violates symbolic divisibility/range"
                    )
                key = contract["symbol"]
                if key in symbols and symbols[key] != value:
                    raise ValueError("Shared symbolic tensor dimensions differ")
                symbols[key] = value
                dynamic.append(value)
        return dynamic

    def __call__(self, *args, **kwargs) -> None:
        """Execute on the current Torch stream using the exported host signature.

        The caller owns tensor bounds/aliasing for pointer-only signatures and
        any operation-specific requirements beyond the exported tensor layout.
        Tensor inputs and outputs must outlive asynchronous launches and graphs.
        """
        if torch.cuda.current_device() != self.device.index:
            raise ValueError("Current CUDA device differs from preloaded host program")
        bound = self.signature.bind(*args, **kwargs)
        symbols, values = {}, []
        for p, ctype in zip(self.parameters, self.parameter_types):
            kind = p["kind"]
            if kind == "stream":
                packed = ctype(torch.cuda.current_stream(self.device).cuda_stream)
            else:
                value = bound.arguments[p["name"]]
                if kind == "ptr":
                    packed = ctype(self._tensor_pointer(p, value))
                elif kind == "tensor":
                    pointer = self._tensor_pointer(p, value)
                    shapes = self._dimensions(p["shape"], value.shape, symbols)
                    strides = self._dimensions(p["stride"], value.stride(), symbols)
                    descriptor_type = (
                        ctype._type_ if p["ctype"].endswith("*") else ctype
                    )
                    descriptor = descriptor_type()
                    descriptor.data = pointer
                    for field in p["fields"]:
                        name = field["name"]
                        if name != "data":
                            dims = shapes if name == "dynamic_shapes" else strides
                            field_type = dict(descriptor_type._fields_)[name]
                            for dim in dims:
                                bits = ctypes.sizeof(field_type._type_) * 8
                                if not 0 <= dim < 2 ** (bits - 1):
                                    raise ValueError(
                                        "Tensor descriptor scalar overflow"
                                    )
                            if name == "dynamic_shapes":
                                descriptor.dynamic_shapes[:] = dims
                            else:
                                descriptor.dynamic_strides[:] = dims
                    packed = (
                        ctypes.pointer(descriptor)
                        if p["ctype"].endswith("*")
                        else descriptor
                    )
                elif kind == "opaque":
                    if not isinstance(value, bytes) or len(value) != ctypes.sizeof(
                        ctype
                    ):
                        raise ValueError(
                            "Opaque host parameter requires exact-size bytes"
                        )
                    packed = ctype.from_buffer_copy(value)
                elif kind == "bool":
                    if type(value) is not bool:
                        raise ValueError("Expected a bool host argument")
                    packed = ctype(value)
                elif kind.startswith(("i", "u")):
                    bits = ctypes.sizeof(ctype) * 8
                    low = -(2 ** (bits - 1)) if kind[0] == "i" else 0
                    high = 2 ** (bits - (kind[0] == "i")) - 1
                    if type(value) is not int or not low <= value <= high:
                        raise ValueError(f"Host scalar is outside {kind} range")
                    packed = ctype(value)
                else:
                    if type(value) not in (int, float):
                        raise ValueError("Expected a numeric host argument")
                    packed = ctype(value)
                    if math.isfinite(value) and not math.isfinite(packed.value):
                        raise ValueError("Floating host scalar overflow")
            values.append(packed)
        ptr = ctypes.c_void_p
        pointers = (ptr * len(values))(
            *[ctypes.cast(ctypes.byref(v), ptr) for v in values]
        )
        code = self.loaded.library.ts_run(pointers)
        if code:
            raise RuntimeError(
                f"CuTe native host launch failed with CUDA status {code}"
            )


def compile_host_program(
    *,
    compiler_python: Path,
    source: Path,
    factory: str,
    parameters: dict,
    arch: str,
    cache: Path,
    dependencies: tuple[Path, ...],
    cuda_home: Path,
    cxx: Path,
    device: torch.device,
) -> CuTeHostProgram:
    """Build/cache a user's CuTe host program and preload its native ARM64 DLL.

    Build arguments follow build_host_artifact. device is an explicit current
    CUDA device. The result accepts the exported positional/keyword signature;
    no manual device ABI, grid/block or stream argument is needed.
    """
    directory = build_host_artifact(
        compiler_python=compiler_python,
        source=source,
        factory=factory,
        parameters=parameters,
        arch=arch,
        cache=cache,
        dependencies=dependencies,
        cuda_home=cuda_home,
        cxx=cxx,
    )
    return CuTeHostProgram(directory, device=device, cuda_home=cuda_home)
