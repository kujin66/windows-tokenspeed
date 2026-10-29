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

"""Native Windows launch of user CuTe device artifacts, without importing cutlass.

The caller supplies the device ABI, launch geometry and operation contract.
Keep this object and all referenced tensors alive through the final graph replay.
"""

import ctypes
import math
import os
import re
from pathlib import Path

import torch
from tokenspeed_kernel.cute_dsl_compile import (
    compile_cached_artifact,
    validate_artifact,
)

_SCALARS = {
    "i8": ctypes.c_int8,
    "u8": ctypes.c_uint8,
    "i16": ctypes.c_int16,
    "u16": ctypes.c_uint16,
    "i32": ctypes.c_int32,
    "u32": ctypes.c_uint32,
    "i64": ctypes.c_int64,
    "u64": ctypes.c_uint64,
    "f32": ctypes.c_float,
    "f64": ctypes.c_double,
}


def _parameter_size(declaration: str) -> int:
    match = re.search(r"\.(?:b|u|s|f)(8|16|32|64)\b", declaration)
    if match is None:
        raise ValueError(f"Unsupported PTX parameter: {declaration}")
    array = re.search(r"\[(\d+)\]", declaration)
    return int(match[1]) // 8 * (int(array[1]) if array else 1)


class CuTeKernel:
    """Load one entry of a compile_artifact result on an explicit CUDA device.

    Args:
        directory: Validated device artifact directory, including PTX and CUBIN.
        entry: Exact exported kernel symbol (not a host-side cute.jit wrapper).
        argument_types: Device parameter types: ptr, i/u8/16/32/64, f32/f64,
            or bytes:N for an opaque by-value struct of exactly N bytes.
            ptr arguments must be CUDA tensors on the selected device. PTX
            parameter widths and declared pointer alignment are checked.
        device: Explicit device index; make it current before constructing.
        shared_memory_bytes: Maximum dynamic shared memory configured at load.
            The launch supplies its own size, bounded by this value.

    This supports ordinary Driver API launches. Cluster/PDL/cooperative launch
    attributes and host JIT wrappers are not synthesized. The caller implements
    tensor dtype, layout, bounds, aliasing and any descriptor/synchronization
    requirements of the chosen operation. close() invalidates captured graphs.
    """

    def __init__(
        self,
        directory: Path,
        *,
        entry: str,
        argument_types: tuple[str, ...],
        device: torch.device,
        shared_memory_bytes: int,
    ):
        if os.name != "nt":
            raise RuntimeError("CuTe native artifact loading requires Windows")
        self.device = torch.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("An explicit CUDA device index is required")
        if torch.cuda.current_device() != self.device.index:
            raise ValueError("Select the target CUDA device before preloading")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Load CuTe kernels before CUDA graph capture")
        if type(shared_memory_bytes) is not int or shared_memory_bytes < 0:
            raise ValueError("Dynamic shared memory must be a nonnegative integer")
        directory = Path(directory)
        self.metadata = validate_artifact(directory)
        major, minor = torch.cuda.get_device_capability(self.device)
        if not re.fullmatch(rf"sm_{major}{minor}[af]?", self.metadata["arch"]):
            raise ValueError("Artifact architecture differs from the selected device")
        ptx = (directory / "kernel.ptx").read_text(encoding="utf-8")
        matches = re.findall(
            rf"\.entry\s+{re.escape(entry)}\s*\((.*?)\)([^{{]*)\{{", ptx, re.S
        )
        if len(matches) != 1:
            raise ValueError("Expected one matching PTX entry")
        declarations, directives = matches[0]
        if re.search(r"\.(?:reqnctapercluster|explicitcluster)\b", directives):
            raise ValueError("Cluster kernels require a separate launch contract")
        self.parameters = tuple(p.strip() for p in declarations.split(",") if p.strip())
        self.argument_types = tuple(argument_types)
        if len(self.parameters) != len(self.argument_types):
            raise ValueError("Device argument count differs from PTX ABI")
        for kind, declaration in zip(self.argument_types, self.parameters):
            if kind == "ptr":
                size = ctypes.sizeof(ctypes.c_void_p)
            elif kind in _SCALARS:
                size = ctypes.sizeof(_SCALARS[kind])
            elif re.fullmatch(r"bytes:[1-9][0-9]*", kind):
                size = int(kind.split(":")[1])
            else:
                raise ValueError(f"Unsupported device argument type: {kind}")
            if size != _parameter_size(declaration):
                raise ValueError(f"Argument {kind} width differs from PTX ABI")
            if ".ptr" in declaration and kind != "ptr":
                raise ValueError("PTX pointer parameters require CUDA tensors")
        required = re.search(r"\.reqntid\s+(\d+),\s*(\d+),\s*(\d+)", directives)
        self.required_block = tuple(map(int, required.groups())) if required else None
        self.entry = entry
        self.shared_memory_bytes = shared_memory_bytes
        self.closed = False
        self.module, self.function = ctypes.c_void_p(), ctypes.c_void_p()
        self.driver = ctypes.WinDLL("nvcuda.dll")
        ptr = ctypes.c_void_p
        signatures = {
            "cuModuleLoadData": [ctypes.POINTER(ptr), ptr],
            "cuModuleGetFunction": [ctypes.POINTER(ptr), ptr, ctypes.c_char_p],
            "cuFuncSetAttribute": [ptr, ctypes.c_int, ctypes.c_int],
            "cuLaunchKernel": [
                ptr,
                *([ctypes.c_uint] * 7),
                ptr,
                ctypes.POINTER(ptr),
                ctypes.POINTER(ptr),
            ],
            "cuModuleUnload": [ptr],
            "cuGetErrorName": [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)],
        }
        for name, signature in signatures.items():
            function = getattr(self.driver, name)
            function.argtypes, function.restype = signature, ctypes.c_int
        torch.empty(0, device=self.device)
        image = ctypes.create_string_buffer((directory / "kernel.cubin").read_bytes())
        self._check(self.driver.cuModuleLoadData(ctypes.byref(self.module), image))
        try:
            self._check(
                self.driver.cuModuleGetFunction(
                    ctypes.byref(self.function), self.module, entry.encode("utf-8")
                )
            )
            if shared_memory_bytes:
                # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8.
                self._check(
                    self.driver.cuFuncSetAttribute(
                        self.function, 8, shared_memory_bytes
                    )
                )
        except Exception:
            self.driver.cuModuleUnload(self.module)
            raise

    def _check(self, code: int) -> None:
        if code:
            name = ctypes.c_char_p()
            self.driver.cuGetErrorName(code, ctypes.byref(name))
            raise RuntimeError(f"CUDA driver error {code}: {name.value}")

    def launch(
        self,
        *arguments,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        shared_memory_bytes: int,
    ) -> None:
        """Enqueue on Torch's current stream; capture performs no I/O or loading.

        Scalar values are range checked. bytes:N accepts an exact-size bytes
        value; ptr accepts tensors only. The caller owns tensor lifetimes and
        dependencies across streams, including during CUDA Graph replay.
        """
        if self.closed:
            raise RuntimeError("CuTe module is closed")
        if torch.cuda.current_device() != self.device.index:
            raise ValueError("Current CUDA device differs from preloaded device")
        if len(arguments) != len(self.argument_types):
            raise ValueError("Device argument count differs from declared ABI")
        for dimensions, name in ((grid, "grid"), (block, "block")):
            if len(dimensions) != 3 or any(
                type(v) is not int or not 0 < v <= 0x7FFFFFFF for v in dimensions
            ):
                raise ValueError(f"Expected three positive integer {name} dimensions")
        if math.prod(block) > 1024 or (
            self.required_block is not None and tuple(block) != self.required_block
        ):
            raise ValueError("Block dimensions violate the kernel launch contract")
        if (
            type(shared_memory_bytes) is not int
            or not 0 <= shared_memory_bytes <= self.shared_memory_bytes
        ):
            raise ValueError("Dynamic shared memory exceeds the preloaded contract")
        values = []
        for kind, declaration, value in zip(
            self.argument_types, self.parameters, arguments
        ):
            if kind == "ptr":
                if not isinstance(value, torch.Tensor) or value.device != self.device:
                    raise ValueError(
                        "Pointer arguments require tensors on the preloaded CUDA device"
                    )
                alignment = re.search(r"\.ptr.*?\.align\s+(\d+)", declaration)
                if alignment and value.data_ptr() % int(alignment[1]):
                    raise ValueError("Tensor pointer violates PTX alignment")
                values.append(ctypes.c_void_p(value.data_ptr()))
            elif kind.startswith("bytes:"):
                if not isinstance(value, bytes) or len(value) != int(
                    kind.split(":")[1]
                ):
                    raise ValueError(
                        "Opaque parameter requires exactly the declared bytes"
                    )
                values.append(ctypes.create_string_buffer(value, len(value)))
            elif kind.startswith(("i", "u")):
                bits = ctypes.sizeof(_SCALARS[kind]) * 8
                low = -(1 << (bits - 1)) if kind[0] == "i" else 0
                high = (1 << (bits - (kind[0] == "i"))) - 1
                if type(value) is not int or not low <= value <= high:
                    raise ValueError(f"Argument is outside {kind} range")
                values.append(_SCALARS[kind](value))
            else:
                if type(value) not in (int, float):
                    raise ValueError(f"Expected a numeric {kind} argument")
                scalar = _SCALARS[kind](value)
                if math.isfinite(value) and not math.isfinite(scalar.value):
                    raise ValueError(f"Argument is outside {kind} range")
                values.append(scalar)
        ptr = ctypes.c_void_p
        params = (ptr * len(values))(
            *[ctypes.cast(ctypes.byref(v), ptr) for v in values]
        )
        stream = ptr(torch.cuda.current_stream(self.device).cuda_stream)
        self._check(
            self.driver.cuLaunchKernel(
                self.function, *grid, *block, shared_memory_bytes, stream, params, None
            )
        )

    def close(self) -> None:
        """Synchronize and unload; captured graphs must never replay afterwards."""
        if not self.closed:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Cannot unload CuTe modules during CUDA graph capture"
                )
            with torch.cuda.device(self.device):
                torch.cuda.synchronize(self.device)
                self._check(self.driver.cuModuleUnload(self.module))
            self.closed = True


def compile_kernel(
    *,
    compiler_python: Path,
    source: Path,
    factory: str,
    parameters: dict,
    arch: str,
    cache: Path,
    dependencies: tuple[Path, ...],
    entry: str | None,
    argument_types: tuple[str, ...],
    device: torch.device,
    shared_memory_bytes: int,
) -> CuTeKernel:
    """Compile a user factory on demand and return a native loaded kernel.

    Compilation happens in the specified Windows compiler interpreter. The
    returned object runs device code in the calling ARM64 process. A verified
    cache hit skips compilation. Pass entry=None only for a single-entry CUBIN;
    otherwise name the device kernel explicitly. Other arguments follow
    compile_cached_artifact and CuTeKernel. Invoke during preparation/warmup
    and retain the result for repeated launches, capture and serving. This
    does not execute the factory's host cute.jit wrapper or infer its ABI.
    """
    directory = compile_cached_artifact(
        compiler_python=compiler_python,
        source=source,
        factory=factory,
        parameters=parameters,
        arch=arch,
        cache=cache,
        dependencies=dependencies,
    )
    if entry is None:
        entries = validate_artifact(directory)["entries"]
        if len(entries) != 1:
            raise ValueError("Select an explicit entry for a multi-kernel artifact")
        entry = entries[0]["name"]
    return CuTeKernel(
        directory,
        entry=entry,
        argument_types=argument_types,
        device=device,
        shared_memory_bytes=shared_memory_bytes,
    )
