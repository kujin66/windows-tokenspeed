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

"""Preloaded native Windows execution of one exported router kernel.

Keep loaded modules alive until every graph using them has finished its final
replay. This module imports no CuTe DSL compiler and never starts WSL.
"""

import ctypes
import hashlib
import json
import os
import tempfile
import threading
from functools import lru_cache
from pathlib import Path

import torch


def _validate_router_buffers(a, b, out, *, shape, device):
    m, n, k = shape
    if a.shape != (m, k) or b.shape != (n, k) or out.shape != (m, n):
        raise ValueError(f"Expected A[{m},{k}], B[{n},{k}], out[{m},{n}]")
    if (
        a.dtype != torch.bfloat16
        or b.dtype != torch.bfloat16
        or out.dtype != torch.float32
    ):
        raise ValueError("Expected BF16 inputs and FP32 output")
    if any(
        t.device != device or not t.is_contiguous() or t.data_ptr() % 16
        for t in (a, b, out)
    ):
        raise ValueError(
            "Expected contiguous 16-byte aligned tensors on the preloaded device"
        )
    if torch.cuda.current_device() != device.index:
        raise ValueError("Current CUDA device differs from preloaded device")
    out_start, out_end = (
        out.data_ptr(),
        out.data_ptr() + out.numel() * out.element_size(),
    )
    for value in (a, b):
        start, end = (
            value.data_ptr(),
            value.data_ptr() + value.numel() * value.element_size(),
        )
        if max(start, out_start) < min(end, out_end):
            raise ValueError("Output must not overlap an input")


def _router_operands_supported(a, b, blocks):
    return (
        os.name == "nt"
        and a.ndim == b.ndim == 2
        and a.shape[1] == b.shape[1]
        and (a.shape[0], b.shape[0], a.shape[1]) in blocks
        and a.dtype == b.dtype == torch.bfloat16
        and a.is_cuda
        and a.device == b.device
        and a.is_contiguous()
        and b.is_contiguous()
        and a.data_ptr() % 16 == b.data_ptr() % 16 == 0
        and torch.cuda.get_device_capability(a.device) == (12, 1)
    )


class RouterAOT:
    def __init__(
        self,
        directory: Path,
        *,
        shape_mnk: tuple[int, int, int],
        block_size: int,
        device: torch.device,
    ):
        self.device = torch.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("An explicit CUDA device index is required")
        if torch.cuda.current_device() != self.device.index:
            raise ValueError("Select the target CUDA device before preloading")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Preload AOT modules before CUDA graph capture")
        if torch.cuda.get_device_capability(self.device) != (12, 1):
            raise ValueError("This artifact targets sm_121a only")
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["schema"] != 1:
            raise ValueError("Unsupported manifest schema")
        matches = [
            item
            for item in manifest["kernels"]
            if tuple(item["shape_mnk"]) == shape_mnk
            and item["block"] == [block_size, 1, 1]
        ]
        if len(matches) != 1:
            raise ValueError(
                f"No unique AOT specialization for {shape_mnk}, block {block_size}"
            )
        self.metadata = matches[0]
        if Path(self.metadata["name"]).name != self.metadata["name"]:
            raise ValueError("Artifact name must be a basename")
        if (
            self.metadata["arch"] != "sm_121a"
            or self.metadata["has_bias"]
            or self.metadata["enable_pdl"]
        ):
            raise ValueError("Unsupported architecture, bias or PDL contract")
        if self.metadata["abi"] != ["a:u64", "b:u64", "c:u64", "unused_bias:u64"]:
            raise ValueError("Unsupported device ABI")
        m, n, k = shape_mnk
        if (
            min(m, n, k) < 1
            or m > 4
            or k % 8
            or block_size not in (64, 128, 256, 512)
            or self.metadata["grid"] != [n, 1, 1]
            or self.metadata["shared_memory_bytes"] != m * 4 * (block_size // 32)
            or self.metadata["input_dtype"] != "bfloat16"
            or self.metadata["output_dtype"] != "float32"
            or self.metadata["layout"] != "contiguous A[M,K], B[N,K], C[M,N]"
        ):
            raise ValueError("Unsupported specialization or launch contract")
        self.shape = shape_mnk
        self.module = ctypes.c_void_p()
        self.function = ctypes.c_void_p()
        self.closed = False
        payload = (directory / (self.metadata["name"] + ".cubin")).read_bytes()
        if hashlib.sha256(payload).hexdigest() != self.metadata["cubin_sha256"]:
            raise ValueError("CUBIN hash mismatch")
        self.driver = ctypes.WinDLL("nvcuda.dll")
        ptr = ctypes.c_void_p
        signatures = {
            "cuModuleLoadData": [ctypes.POINTER(ptr), ptr],
            "cuModuleGetFunction": [ctypes.POINTER(ptr), ptr, ctypes.c_char_p],
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
            fn = getattr(self.driver, name)
            fn.argtypes = signature
            fn.restype = ctypes.c_int
        torch.empty(0, device=self.device)
        image = ctypes.create_string_buffer(payload)
        self._check(self.driver.cuModuleLoadData(ctypes.byref(self.module), image))
        try:
            self._check(
                self.driver.cuModuleGetFunction(
                    ctypes.byref(self.function),
                    self.module,
                    self.metadata["entry"].encode(),
                )
            )
        except Exception:
            self.driver.cuModuleUnload(self.module)
            raise

    def _check(self, code):
        if code:
            name = ctypes.c_char_p()
            self.driver.cuGetErrorName(code, ctypes.byref(name))
            raise RuntimeError(f"CUDA driver error {code}: {name.value}")

    def __call__(self, a: torch.Tensor, b: torch.Tensor, *, out: torch.Tensor):
        if self.closed:
            raise RuntimeError("AOT module is closed")
        _validate_router_buffers(a, b, out, shape=self.shape, device=self.device)
        ptr = ctypes.c_void_p
        values = [
            ptr(a.data_ptr()),
            ptr(b.data_ptr()),
            ptr(out.data_ptr()),
            ptr(out.data_ptr()),
        ]
        args = (ptr * 4)(*[ctypes.cast(ctypes.byref(value), ptr) for value in values])
        stream = ptr(torch.cuda.current_stream(self.device).cuda_stream)
        self._check(
            self.driver.cuLaunchKernel(
                self.function,
                *self.metadata["grid"],
                *self.metadata["block"],
                self.metadata["shared_memory_bytes"],
                stream,
                args,
                None,
            )
        )
        return out

    def close(self):
        if not self.closed:
            # Graphs must not be replayed after explicit module close.
            with torch.cuda.device(self.device):
                torch.cuda.synchronize(self.device)
                self._check(self.driver.cuModuleUnload(self.module))
            self.closed = True


class RouterAOTCatalog:
    """Explicit artifact selection, with modules retained for the process lifetime.

    Args:
        config: JSON file containing schema=1, an artifact directory and a list
            of selections, each with shape_mnk and block_size. Relative artifact
            directories resolve against the config file, never the working dir.
    """

    def __init__(self, config: Path):
        config = config.resolve()
        policy = json.loads(config.read_text(encoding="utf-8"))
        if policy["schema"] != 1:
            raise ValueError("Unsupported AOT selection schema")
        self.directory = (config.parent / policy["directory"]).resolve()
        manifest = json.loads(
            (self.directory / "manifest.json").read_text(encoding="utf-8")
        )
        source = Path(__file__).parent / "thirdparty/cute_dsl/ll_bf16/_kernel.py"
        if manifest["source_sha256"] != hashlib.sha256(source.read_bytes()).hexdigest():
            raise ValueError("AOT artifacts do not match the installed router source")
        self.blocks: dict[tuple[int, int, int], int] = {}
        for item in policy["selections"]:
            shape = tuple(item["shape_mnk"])
            block = item["block_size"]
            if (
                len(shape) != 3
                or any(type(v) is not int or v < 1 for v in shape)
                or shape[0] > 4
                or shape[2] % 8
                or type(block) is not int
                or block not in (64, 128, 256, 512)
                or shape in self.blocks
            ):
                raise ValueError("Invalid or duplicate AOT shape/block selection")
            if (
                sum(
                    tuple(k["shape_mnk"]) == shape and k["block"] == [block, 1, 1]
                    for k in manifest["kernels"]
                )
                != 1
            ):
                raise ValueError(
                    f"Missing/ambiguous exported selection: {shape}/{block}"
                )
            self.blocks[shape] = block
        if not self.blocks:
            raise ValueError("At least one AOT selection is required")
        self.modules: dict[tuple[int, tuple[int, int, int]], RouterAOT] = {}
        self.lock = threading.Lock()

    def supports(self, a: torch.Tensor, b: torch.Tensor) -> bool:
        """Return whether the configured artifacts cover these exact operands."""
        return _router_operands_supported(a, b, self.blocks)

    def preload(self, *, device: torch.device) -> None:
        """Load every selected module before graph capture on an explicit device."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Preload AOT modules before CUDA graph capture")
        with self.lock:
            for shape, block in self.blocks.items():
                key = (device.index, shape)
                if key not in self.modules:
                    self.modules[key] = RouterAOT(
                        self.directory, shape_mnk=shape, block_size=block, device=device
                    )

    def run(
        self, a: torch.Tensor, b: torch.Tensor, *, out: torch.Tensor
    ) -> torch.Tensor:
        """Execute a covered shape; startup loads all selected modules once."""
        if not self.supports(a, b):
            raise ValueError("Operands are not covered by the configured AOT artifacts")
        key = (a.device.index, (a.shape[0], b.shape[0], a.shape[1]))
        if key not in self.modules:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm up the AOT router before graph capture")
            self.preload(device=a.device)
        return self.modules[key](a, b, out=out)


class RouterHostCatalog:
    """Schema-2 router policy selecting native DLLs with original CuTe launchers.

    Args:
        config: JSON with schema=2, explicit cuda_home and selections containing
            shape_mnk, block_size and program (a native host artifact directory).
            Paths resolve relative to the policy. Only the installed unmodified
            router factory/source with matching compile parameters is accepted.
    """

    def __init__(self, config: Path):
        from tokenspeed_kernel.cute_dsl_compile import validate_artifact

        config = Path(config).resolve()
        policy = json.loads(config.read_text(encoding="utf-8"))
        if policy["schema"] != 2:
            raise ValueError("Unsupported native router policy schema")
        self.cuda_home = (config.parent / policy["cuda_home"]).resolve()
        self.blocks = {}
        self.directories = {}
        self.modules = {}
        self.lock = threading.Lock()
        source = Path(__file__).parent / "thirdparty/cute_dsl/ll_bf16/_kernel.py"
        factory = source.with_name("_windows_host.py")
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        factory_hash = hashlib.sha256(factory.read_bytes()).hexdigest()
        for item in policy["selections"]:
            shape, block = tuple(item["shape_mnk"]), item["block_size"]
            if (
                len(shape) != 3
                or any(type(v) is not int for v in shape)
                or shape not in ((1, 896, 7168), (2, 896, 7168), (4, 896, 7168))
                or type(block) is not int
                or block not in (64, 128, 256, 512)
                or shape in self.blocks
            ):
                raise ValueError("Invalid or duplicate native router shape/block")
            directory = (config.parent / item["program"]).resolve()
            manifest = validate_artifact(directory)
            if (
                manifest["source_sha256"] != factory_hash
                or source_hash not in manifest["compile_inputs"].values()
                or manifest["factory"] != "compile_router"
                or manifest["parameters"]
                != dict(m=shape[0], n=shape[1], k=shape[2], block_size=block)
                or manifest["arch"] != "sm_121a"
            ):
                raise ValueError(
                    "Native router artifacts do not match installed source/contract"
                )
            self.blocks[shape] = block
            self.directories[shape] = directory
        if not self.blocks:
            raise ValueError("At least one native router selection is required")

    def supports(self, a, b):
        """Whether the policy covers these exact BF16 operands."""
        return _router_operands_supported(a, b, self.blocks)

    def preload(self, *, device: torch.device):
        """Load every selected host DLL before CUDA graph capture."""
        from tokenspeed_kernel.cute_dsl_host import CuTeHostProgram

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Preload native routers before CUDA graph capture")
        with self.lock:
            for shape, directory in self.directories.items():
                key = (device.index, shape)
                if key not in self.modules:
                    self.modules[key] = CuTeHostProgram(
                        directory, device=device, cuda_home=self.cuda_home
                    )

    def run(self, a, b, *, out):
        """Run the selected original CuTe host launcher; preserve output storage."""
        if not self.supports(a, b):
            raise ValueError("Operands are not covered by native router artifacts")
        shape = (a.shape[0], b.shape[0], a.shape[1])
        _validate_router_buffers(a, b, out, shape=shape, device=a.device)
        key = (a.device.index, shape)
        if key not in self.modules:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm up the native router before graph capture")
            self.preload(device=a.device)
        self.modules[key](a, b, out, out)
        return out


def prepare_router_host(
    *,
    compiler_python: Path,
    cache: Path,
    cuda_home: Path,
    cxx: Path,
    blocks: dict[int, int],
    configuration: Path,
) -> Path:
    """Build native router DLLs and atomically publish a schema-2 selection policy.

    Args:
        compiler_python, cache, cuda_home, cxx: Windows compiler/cache/native
            toolchain inputs for build_host_artifact, outside capture/serving.
        blocks: Explicit M-to-block-size selections; M is 1, 2 or 4 and block
            is 64, 128, 256 or 512. No automatic tuning/performance claim.
        configuration: New JSON policy path. Existing policies are preserved.

    Set TOKENSPEED_CUTE_DSL_AOT_CONFIG to the returned absolute path before
    starting workers, disable PDL and warm up before CUDA Graph capture.
    """
    from tokenspeed_kernel.cute_dsl_compile import _require_preparation
    from tokenspeed_kernel.cute_dsl_host import build_host_artifact

    _require_preparation()
    configuration = Path(configuration).resolve()
    if configuration.exists():
        raise FileExistsError(configuration)
    if not blocks or any(
        type(m) is not int
        or m not in (1, 2, 4)
        or type(bs) is not int
        or bs not in (64, 128, 256, 512)
        for m, bs in blocks.items()
    ):
        raise ValueError("Explicit valid native router M/block selections are required")
    source = Path(__file__).parent / "thirdparty/cute_dsl/ll_bf16/_kernel.py"
    selections = []
    for m, block in blocks.items():
        directory = build_host_artifact(
            compiler_python=compiler_python,
            source=source.with_name("_windows_host.py"),
            factory="compile_router",
            parameters=dict(m=m, n=896, k=7168, block_size=block),
            arch="sm_121a",
            cache=cache,
            dependencies=(source,),
            cuda_home=cuda_home,
            cxx=cxx,
        )
        selections.append(
            dict(shape_mnk=[m, 896, 7168], block_size=block, program=str(directory))
        )
    policy = dict(
        schema=2, cuda_home=str(Path(cuda_home).resolve()), selections=selections
    )
    configuration.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="router-policy-", dir=configuration.parent
    ) as temp:
        staging = Path(temp) / "selection.json"
        staging.write_text(json.dumps(policy, indent=2), encoding="utf-8")
        RouterHostCatalog(staging)
        staging.rename(configuration)
    return configuration


@lru_cache(maxsize=1)
def configured_router_aot() -> RouterAOTCatalog | RouterHostCatalog | None:
    """Read the process-start opt-in once; absent configuration leaves dispatch alone."""
    config = os.environ.get("TOKENSPEED_CUTE_DSL_AOT_CONFIG")
    if config is None:
        return None
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Initialize AOT configuration before graph capture")
    path = Path(config)
    policy = json.loads(path.read_text(encoding="utf-8"))
    if policy["schema"] == 2:
        return RouterHostCatalog(path)
    return RouterAOTCatalog(path)
