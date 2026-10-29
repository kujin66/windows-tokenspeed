# CuTe host launchers compiled for native Windows ARM64

This extends the [Windows-local compiler](windows-arm64-cutedsl-local-compiler.md)
to export the original CuTe host program as a Windows ARM64 COFF object and
link it into a native DLL. The calling ARM64 Python process executes that
launcher directly. Its control flow, scalar arithmetic, device ABI, launch
order, grid/block computation, shared memory and CUDA launch attributes come
from CuTe compilation, rather than a handwritten device launch adapter.

Compilation still uses official Windows x64 CuTe DSL 4.8.1 under system
emulation. The compiler's long-form host target is
aarch64-pc-windows-msvc. The object and final DLL are checked for ARM64
architecture. An independently implemented native forwarding layer calls the
installed CUDA SDK; the x64 CuTe runtime library is not linked into the DLL.
No WSL process or renamed wheel platform tag is involved.

This establishes a local compile/native execution route. It does not create
an ARM64 build of the Python compiler extension, or make import cutlass.cute
work inside native ARM64 Python.

## Restore the exactly tested optional sources from this reporting PR

First follow the [build guide](windows-arm64-build-details.md) to reconstruct
and verify the pinned 864bcdd9-to-9a50ac82 Windows port. This reporting PR has
a different main runtime API. Its optional CuTe integration is retained as an
exact tested source snapshot, rather than silently replacing that API.

From the PR checkout, copy the snapshot into the reconstructed port:

```powershell
$snapshot = (Resolve-Path docs/validation/windows-cutedsl-native-host/sources).Path
$testedCheckout = (Resolve-Path ../tokenspeed-windows-arm64-build).Path
Get-ChildItem -LiteralPath $snapshot | Copy-Item -Destination $testedCheckout -Recurse -Force
```

The [validation summary](../validation/windows-cutedsl-native-host/summary.json)
identifies all 18 exactly tested source hashes. Five setup/microbenchmark helpers
are included as reproduction support. Original source bytes/line endings are
preserved. Apply this optional overlay after baseline source verification; its
deliberate edits no longer match the baseline-only postimage check. Run the
commands below from that restored checkout with native ARM64 dependencies
prepared and fresh output directories. External toolchain/model binaries
remain external inputs.

## Compile a program with automatic argument adaptation

Run the caller using the repository's prepared native CUDA/MSVC environment:

```powershell
$env:PYTHONPATH = (Resolve-Path tokenspeed-kernel/python).Path
.venv/Scripts/python.exe tools/windows/run.py .venv/Scripts/python.exe your_program.py
```

Example caller:

```python
import os
import shutil
from pathlib import Path

import torch
from tokenspeed_kernel.cute_dsl_host import compile_host_program

torch.cuda.set_device(0)
program = compile_host_program(
    compiler_python=Path(".port/cutedsl-win-x64-venv/Scripts/python.exe"),
    source=Path("tools/windows/cute_dsl_aot/host_factory.py"),
    factory="compile_tensor",
    parameters={"divisibility": 1},
    arch="sm_121a",
    cache=Path(".port/cutedsl-native-cache"),
    dependencies=(Path("tools/windows/cute_dsl_aot/elementwise_factory.py"),),
    cuda_home=Path(os.environ["CUDA_PATH"]),
    cxx=Path(shutil.which("cl")),
    device=torch.device("cuda:0"),
)
a = torch.arange(258, device="cuda", dtype=torch.float32)[::2]
out = torch.empty(129, device="cuda")
program(a, out)
torch.testing.assert_close(out, a * 2 + 1, atol=0, rtol=0)
```

The caller supplies neither a device ABI nor grid/block dimensions. C scalar
types and pointer dtype/alignment are exported automatically. Flat fake-tensor
arguments export their static/dynamic shapes, strides, symbolic equality and
divisibility contracts; the native caller builds the C descriptors from Torch
tensors and validates those contracts. cudaStream_t arguments use the current
Torch stream. Positional and keyword arguments follow the exported C signature;
unsupported arguments fail rather than being discarded.

The factory returns a real cute.compile result using string compiler options.
An isolated compiler-process hook preserves its options while selecting the
ARM64 host target and retaining PTX/CUBIN. Existing incompatible targets fail.
The installed compiler sources are not patched. A C-compatible CUDA-dialect
export is required. Other aggregate arguments currently require an explicit
exact-size byte payload; arbitrary Python objects are not serialized into the
native process.

The two cache layers cover source/dependencies, compiler package records and
target, then the native adapter, forwarding source, compiler/linker, CUDA import
libraries/headers and build environment. Toolchains must stay immutable and all
additional source/build inputs must be declared. Cache hits verify the device,
object, header, adapter and DLL hashes. A failed build never publishes a native
artifact directory. Compilation and native loading are preparation operations;
retain the program for ordinary calls and graph capture. Loaded DLLs/libraries
remain alive until process exit so captured graphs retain valid device entries.

Ordinary calls accept changing runtime scalar values and dynamic tensor layouts
without another compilation. CUDA Graph capture records those values and tensor
addresses; later Python argument changes do not update a captured node. Replays
may change tensor contents while retaining the captured storage and geometry.

## Automatic TokenSpeed router dispatch

Prepare the unchanged vendored BF16 router and publish an explicit policy:

```python
from tokenspeed_kernel.cute_dsl_aot import prepare_router_host

configuration = prepare_router_host(
    compiler_python=Path(".port/cutedsl-win-x64-venv/Scripts/python.exe"),
    cache=Path(".port/cutedsl-native-router-cache"),
    cuda_home=Path(os.environ["CUDA_PATH"]),
    cxx=Path(shutil.which("cl")),
    blocks={1: 128, 2: 128, 4: 64},
    configuration=Path(".port/router-native-host.json"),
)
```

The helper compiles each selected specialization's GPU code and ARM64 host
launcher, links native DLLs and atomically creates a schema-2 policy. Existing
policies are preserved. Block choices are explicit experimental selections,
not an autotuning result or performance guarantee.

Set TOKENSPEED_CUTE_DSL_AOT_CONFIG to the policy's absolute path before starting
workers and use --disable-pdl for this router. The existing
kimi3_router_projection(..., solution="auto") family API and kernel registry
then select these native launchers for covered shapes. Warmup preloads all
selected programs before capture. The loader checks installed factory/kernel
hashes, specialization, output shape/dtype/layout and input/output overlap.
Uncovered shapes retain the existing fallback; forced ineligible AOT calls fail.
The original schema-1 device-only policies remain supported.

## Hardware validation and scope

The checked-in tests exercise actual compilation and native GPU execution:

- Dynamic pointer/scalar signatures and non-default CUDA streams.
- Dynamic tensor dimensions and strides with changed-input graph replays;
  invalid dtype, alignment, symbolic equality/divisibility and strides fail.
- Runtime host loops: graph inspection counts the expected 2/3/5 kernel nodes
  for 0/1/3 additional iterations, verifying preserved control flow and order.
- Two-CTA clusters, with output depending on cluster-local block rank, and
  PDL instructions/launch attributes during ordinary and captured execution.
- A TMA global-to-shared copy, shared-memory barrier synchronization and
  native host tensor-map encoding, compared exactly after changed-input replays.
- The real K3 router through TokenSpeed auto dispatch at M=1/2/4. Captured
  symbols must match the compiled CUBIN and contain one kernel node.
- Native cache reuse, corrupted DLL rejection and capture-time build refusal.

Rebuild artifacts and retain a fresh combined validation cohort:

```powershell
.venv/Scripts/python.exe tools/windows/run.py .venv/Scripts/python.exe tools/windows/cute_dsl_aot/validate_local.py --compiler-python .port/cutedsl-win-x64-venv/Scripts/python.exe --work-directory .port/cutedsl-native-host-validation --evidence-directory docs/validation/windows-cutedsl-native-host --router-blocks 128 128 64 --suite all
```

Both output directories must be new. A skipped or failed test makes validation
fail. Evidence includes test reports, source hashes, host DLL architectures,
compiler manifests, linker logs and captured graph dumps.

This is not a claim of complete CuTe compatibility. It has been exercised on
RTX Spark N1X, SM12.1, CUDA 13.4, native ARM64 Torch/Python and CuTe DSL 4.8.1.
One DLL is currently bound to one CUDA device. Nested/custom descriptors,
TVM-FFI exports, every CUDA runtime entry point, other devices/drivers and all
upstream kernels require separate coverage; unsupported linker symbols fail
explicitly. Compiler-process startup still occurs on cache misses; a persistent
compiler service is not implemented. Automatic model integration covers the
K3 BF16 router, and these tests establish no full-model throughput improvement.
