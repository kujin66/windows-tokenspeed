# Windows-local CuTe compilation and automatic TokenSpeed AOT routing

Follow-up: [native ARM64 host launchers](windows-arm64-cutedsl-native-host.md)
preserve the compiled host program, automatically adapt common arguments and
support a schema-2 router policy. The device-only APIs described below remain
available for callers that supply their own launch contract.

This adds a Windows-local compilation route on the ARM64 machine: an isolated
**Windows x64 Python process runs NVIDIA's official CuTe DSL 4.8.1 under Windows
system emulation** and exports GPU code. Native Windows ARM64 Python loads that
code through its CUDA driver. Compilation and execution do not invoke WSL.
This is not an ARM64 build of NVIDIA's compiler or an in-process cutlass.cute
implementation for ARM64 Python.

Before running these examples from this reporting PR, restore the
[exact tested sources](windows-arm64-cutedsl-native-host.md#restore-the-exactly-tested-optional-sources-from-this-reporting-pr)
into the reconstructed port. The [native-host follow-up](windows-arm64-cutedsl-native-host.md)
extends device-only execution with native host launchers and concrete
cluster, PDL and TMA validation.

## Implemented behavior

| Capability | Scope |
| --- | --- |
| Local CuTe compiler | Official Windows x64 wheels, in an isolated x64 venv on the ARM64 host |
| On-demand compilation | Native ARM64 callers invoke compile_artifact with a user factory, explicit JSON parameters and target architecture |
| Custom kernels | A factory returns a real cute.compile result; exported CUBIN/PTX, entry parameters, source and compiler hashes are retained |
| Cached on-demand compile/load | compile_kernel compiles a user factory on a cache miss and returns a native ARM64 CuTeKernel |
| General device launch | Explicit pointer/scalar/opaque-by-value ABI, 3D grid/block, dynamic shared-memory limit and current Torch CUDA stream |
| TokenSpeed automatic AOT use | kimi3_router_projection(..., solution="auto") selects the registered cute_dsl_aot_router when explicitly configured and eligible |
| CUDA Graph execution | Modules preload during warmup; graph capture refuses an unloaded specialization |
| Native ARM64 compiler | Still unavailable from the checked official distributions; x64 process emulation is the implemented route |

The compiler runner does not translate arbitrary Python into CUDA. A supplied
factory must use valid CuTe DSL APIs supported by the installed version and
requested GPU architecture. Exported device code does not transport Linux or
x64 host JIT functions into ARM64 Python. Consumers supply each kernel's ABI,
launch geometry and synchronization. Cluster launches, TMA descriptors, PDL,
callbacks and kernels needing an x64 CUDA host runtime are not automatically
provided by this device-artifact interface.

## Reproduce the compiler environment

Use a separate compiler environment; leave native ARM64 Torch untouched.
The setup script requires uv and installs no global Python registration.
The tested Python is CPython 3.13.17 x64; package pins include CuTe DSL 4.8.1
and CUDA bindings 13.4.4. Native execution uses the existing ARM64 Torch
2.14.0a0+git20b0f57 / CUDA 13.4 environment on RTX Spark N1X, SM12.1.

```powershell
./tools/windows/cute_dsl_aot/setup_compiler.ps1 -Destination .port/cutedsl-local -IndexUrl https://pypi.org/simple
$compilerPython = (Resolve-Path .port/cutedsl-local/venv/Scripts/python.exe).Path
```

During this machine's setup, files.pythonhosted.org failed its TLS handshake.
Installation succeeded through https://pypi.tuna.tsinghua.edu.cn/simple with
certificate verification enabled. Pass that URL explicitly if needed. No
compiler source patch or renamed wheel platform tag is used. Windows
platform.machine() reports ARM64 even in this x64 process, so artifact
provenance records the interpreter's actual PE machine: 0x8664 for x64,
0xaa64 for the native execution process.

A native ARM64 Python caller can compile a custom factory on demand:

```python
from pathlib import Path
from tokenspeed_kernel.cute_dsl_compile import compile_artifact

manifest = compile_artifact(
    compiler_python=Path(".port/cutedsl-local/venv/Scripts/python.exe"),
    source=Path("tools/windows/cute_dsl_aot/elementwise_factory.py"),
    factory="compile_elementwise",
    parameters={"operation": "multiply", "block_size": 256},
    arch="sm_121a",
    directory=Path(".port/custom-cute-multiply"),
)
```

Install the updated tokenspeed-kernel sources in the native venv, or prepend
its python directory to PYTHONPATH while testing a source checkout. The
compiler subprocess uses isolated Python mode so the kernel package's
platform.py cannot shadow Python's standard-library platform module. A
failed compile does not publish an artifact directory. Existing destinations
are rejected; choose a fresh directory for a new build. Compilation belongs
to explicit preparation/warmup, outside CUDA capture and serving timing.

## Compile and launch a custom kernel from ARM64 Python

The convenience API combines the Windows compiler process, a persistent device
artifact cache and native loading. The factory can define a new CuTe kernel;
it is not restricted to the router or the example's operations. It must return
a compiled object that exposes device PTX and CUBIN. A host wrapper containing
several launches still requires the caller to launch the exported entries in
the correct order. No host JIT wrapper is emulated inside ARM64 Python.

```python
from pathlib import Path

import torch
from tokenspeed_kernel.cute_dsl_runtime import compile_kernel

torch.cuda.set_device(0)
kernel = compile_kernel(
    compiler_python=Path(".port/cutedsl-win-x64-venv/Scripts/python.exe"),
    source=Path("tools/windows/cute_dsl_aot/elementwise_factory.py"),
    factory="compile_affine",
    parameters={"block_size": 128},
    arch="sm_121a",
    cache=Path(".port/cutedsl-cache"),
    dependencies=(),
    entry=None,  # Only valid when exactly one device entry is exported.
    argument_types=("ptr", "ptr", "i32", "i64", "f32", "f32"),
    device=torch.device("cuda:0"),
    shared_memory_bytes=0,
)
a = torch.arange(258, device="cuda", dtype=torch.float32)
out = torch.empty(129, device="cuda")
kernel.launch(
    a, out, 129, 2, 2.0, 1.0,
    grid=(2, 1, 1), block=(128, 1, 1), shared_memory_bytes=0,
)
torch.testing.assert_close(out, a[::2] * 2 + 1, atol=0, rtol=0)
kernel.close()  # Only after the last use, including captured graph replays.
```

For a compiler created by setup_compiler.ps1, use its printed interpreter path
instead of the preexisting local environment path above.

The cache key covers source content, explicit imported-file dependencies,
factory, JSON parameters, architecture, compiler executable, package versions
and installation records, worker source and the CUDA/build environment. List
all helper files or external build inputs in dependencies. Factories must be
deterministic with respect to these inputs; undeclared file/environment inputs
cannot be tracked automatically. Installed compiler packages and external
toolchains must be kept immutable or explicitly included as dependencies.
The worker supports adjacent Python helpers. Concurrent callers publish complete
entries atomically. Cache hits verify both CUBIN and PTX hashes; corruption fails
explicitly. Hashes refer to actual UTF-8 bytes, including on Windows.

CuTeKernel also loads an existing compile_artifact directory without invoking
the compiler. Scalar types are i/u8/16/32/64 and f32/f64; bytes:N passes an
opaque by-value payload of exactly N bytes. Device parameter widths, integer
ranges, declared pointer alignment, required block size, device architecture
and configured shared-memory limits are checked. Pointer arguments must be
CUDA tensors on the selected device. The operation's adapter still owns dtype,
layout, tensor bounds, aliasing and descriptor validity. Cluster, cooperative
and PDL launches require additional launch support.

Retain a loaded kernel before CUDA capture. Runtime launch reads no files and
spawns no process; dynamic scalar arguments and tensor addresses are captured
as ordinary CUDA Graph arguments. Changing their Python values after capture
does not update a captured node; changing values in the captured input tensors
does. Compilation/cache probing and loading are preparation operations.

## Build router artifacts and opt into automatic dispatch

The existing exporter can now run directly in the Windows compiler venv:

```powershell
& $compilerPython -I tools/windows/cute_dsl_aot/export_router.py --source tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cute_dsl/ll_bf16/_kernel.py --output .port/router-local --block-sizes 64 128 256 512
```

Create .port/router-local/selection.json with explicit selections. These block
choices are experimental; they are not a guarantee of better performance.

```json
{
  "schema": 1,
  "directory": ".",
  "selections": [
    {"shape_mnk": [1, 896, 7168], "block_size": 128},
    {"shape_mnk": [2, 896, 7168], "block_size": 128},
    {"shape_mnk": [4, 896, 7168], "block_size": 64}
  ]
}
```

Set TOKENSPEED_CUTE_DSL_AOT_CONFIG to its absolute path **before starting
TokenSpeed workers**, and disable PDL using the engine's --disable-pdl option.
The normal Kimi K3 router family API then selects the registered AOT kernel
for covered shapes. It preserves the optional FP32 output buffer. Explicit
solution="cute_dsl_aot" rejects an ineligible call; auto keeps the existing
fallback for uncovered shapes. With no configuration, existing dispatch
remains unchanged. The configured source hash must match the installed
vendored router source; corrupt binaries, ambiguous selections and unsupported
contracts fail rather than silently switching implementations.

The selected modules stay loaded for the process lifetime. The first ordinary
warmup call preloads all selected specializations on its device. Do this before
graph capture; no compiler process, file read or module load occurs in the
warmed captured operation. Configuration is read once per worker process.

## Validation and limitations

The evidence below covers real hardware, not mocked compiler availability:

- Windows x64 CuTe compiled the unmodified TokenSpeed router source into GPU
  CUBINs. Native ARM64 execution checks the exported shape suite against a CPU
  FP64 reference, including changed-input CUDA Graph replays and a non-default
  stream (atol=0.001, rtol=0.0001).
- The automatic family-API tests use M=1/2/4, N=896, K=7168. Each captured graph
  must contain exactly one kernel whose symbol equals the configured CUBIN's
  entry. This establishes that auto actually calls AOT, rather than a Torch
  result that happens to pass numerical comparison.
- An uncovered M=5 follows the existing auto fallback. Forced AOT with an
  unsupported shape or enabled PDL fails. Source mismatch, duplicate policy
  entries and corrupt CUBIN tests reject invalid configurations.
- The on-demand compiler tests build two different elementwise operations
  (add and multiply), then run each on native ARM64 for N=1/127/128/129/257/4096,
  with three changed-input graph replays per length and exact output checks.
  A deliberately failing factory leaves no published artifact directory.
- The general runtime tests compile an affine kernel with runtime int32/int64
  and float32 arguments and a transpose kernel with a two-dimensional launch.
  They cover tail sizes, strides, changed-input graph replays, a non-default
  stream, invalid ABI widths, scalar overflow, CPU pointers, pointer alignment,
  block/shared-memory violations and capture-time load/unload refusal.
- Cache tests execute add and multiply artifacts produced by a changed imported
  helper, verify a hit preserves the binary timestamp, invalidate on source
  changes and reject a corrupted CUBIN on both cache lookup and native loading.

Run the checked-in GPU tests from the updated source environment with
TOKENSPEED_CUTE_DSL_COMPILER_PYTHON and TOKENSPEED_CUTE_DSL_AOT_CONFIG set:

```powershell
$env:TOKENSPEED_CUTE_DSL_COMPILER_PYTHON = $compilerPython
$env:TOKENSPEED_CUTE_DSL_AOT_CONFIG = (Resolve-Path .port/router-local/selection.json).Path
$env:PYTHONPATH = (Resolve-Path tokenspeed-kernel/python).Path
.venv/Scripts/python.exe tools/windows/run.py .venv/Scripts/python.exe -m pytest tokenspeed-kernel/test/nvidia/test_cute_dsl_windows_compile.py tokenspeed-kernel/test/nvidia/test_cute_dsl_aot.py -q -p no:cacheprovider
```

Tests intentionally skip when Windows, CUDA or an explicit compiler/artifact
configuration is missing; a skip is not validation. CI needs a Windows ARM64
NVIDIA GPU runner and must prepare artifacts before running the tests.

Automatic integration currently covers this BF16 router operation, not all
TokenSpeed CuTe kernels or all MoE models. It does not broaden upstream's
supported GPU matrix. No complete Kimi K3 model or model-level throughput
improvement is claimed. Dense Qwen3-0.6B has no matching router, so the
[existing six-engine/backend comparison](windows-wsl2-engine-backend-retest.md)
records aot_router.active=false. Its throughput does not establish model-level
gains from this dispatch integration.

See the [local compiler evidence](../validation/windows-cutedsl-native-host/)
for test reports, manifests, graph dumps, dependency versions and source hashes.

To rebuild all router artifacts on Windows and retain a fresh test evidence
directory, run this from the checkout using the prepared native CUDA environment.
Both output directories must be new; no previous evidence is overwritten.
The command fails if any test fails or skips. Block choices are explicit
experimental selections, not performance guarantees.

```powershell
.venv/Scripts/python.exe tools/windows/run.py .venv/Scripts/python.exe tools/windows/cute_dsl_aot/validate_local.py --compiler-python .port/cutedsl-win-x64-venv/Scripts/python.exe --work-directory .port/cutedsl-local-validation --evidence-directory docs/validation/windows-cutedsl-native-host-new --router-blocks 128 128 64 --suite device
```

An ARM64-native compiler remains blocked by the required native backend
components: the available _cutlass_ir extensions and compiler/runtime DLLs
are Windows x64 binaries. The Python frontend's editable build consumes those
binary packages; it does not build replacement ARM64 backend components.
The implemented local route uses those official binaries through Windows
system emulation and executes only their device artifacts in native ARM64.
