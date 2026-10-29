# CuTe DSL AOT kernels on Windows ARM64

Follow-up: [Windows-local compilation and automatic router dispatch](windows-arm64-cutedsl-local-compiler.md)
now uses official x64 compiler wheels under Windows system emulation and a
registered, explicitly configured native ARM64 AOT route. The measurements
below remain the earlier standalone WSL2-export experiment.

The WSL2-to-Windows AOT route now executes a real TokenSpeed-vendored kernel:
the unchanged CuTe DSL BF16 router dot-product GEMM. WSL2 compiles GPU CUBINs
with CuTe DSL 4.7.0; native Windows ARM64 loads them through the CUDA Driver
API. Windows does not import Python cutlass or start WSL during execution.

The original experiment below did not install a native Windows ARM64 compiler
or change production dispatch. Its timings do not establish an end-to-end
TokenSpeed or vLLM throughput improvement. The original routed
normalization warning concerns a different, cluster-based kernel and is not
resolved by this router experiment.

## Code and build contract

- [export_router.py](../validation/windows-cutedsl-native-host/sources/tools/windows/cute_dsl_aot/export_router.py) imports
  the existing Apache-2.0-licensed
  [LLBf16Dotprod source](../validation/windows-cutedsl-native-host/sources/tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cute_dsl/ll_bf16/_kernel.py)
  directly, without modifying its body or overriding hardware exclusions.
- [windows_router_aot.py](../validation/windows-cutedsl-native-host/sources/tools/windows/cute_dsl_aot/windows_router_aot.py)
  validates the artifact SHA256, architecture, ABI, shape, dtype, layout,
  alignment and launch geometry before using the current PyTorch CUDA stream.
  Modules load before capture and remain alive through all graph replays.
  Output buffers must be preallocated and must not overlap inputs. Close a
  module only after its graphs' final replay; the caller owns tensor lifetimes
  and stream dependencies.
- [test_router_aot.py](../validation/windows-cutedsl-native-host/sources/tools/windows/cute_dsl_aot/test_router_aot.py)
  checks numerical behavior and measures CUDA graph replay. Backend selection
  and block size remain explicit; unexported shapes fail rather than silently
  changing the operation or dtype.
- [inspect_graphs.py](../validation/windows-cutedsl-native-host/sources/tools/windows/cute_dsl_aot/inspect_graphs.py)
  records the actual one-operation CUDA graph kernel names for both backends.

The source SHA256 is
f7002218589c3af7639f6c245b69209142b922abdadcd8998be3c511da9cd278.
The exported operation is C[M,N] = A[M,K] @ B[N,K].T, with contiguous BF16
inputs, FP32 accumulation/output, no bias and PDL disabled. Static tensor
layouts leave four eight-byte pointer parameters in the device ABI; the fourth
is an unused bias placeholder. The exporter rejects an unexpected PTX ABI.
Grid size is N CTAs; block sizes 64/128/256/512 are measured explicitly.
Dynamic shared memory is M * 4 * (block_size / 32) bytes. No clusters or
host-side Linux MLIR execution-engine objects are transported to Windows.

The export set has 16 specializations: M=1/2/4, N=896, K=7168 at all four
block sizes, plus M=1, N=17, K=8/320/1032/2056 at block size 256 to exercise
scalar and vector tails. This is finite shape coverage, not arbitrary JIT.
Artifacts target sm_121a specifically. The local GPU is RTX Spark N1X, SM12.1,
40 SMs and 24 MiB L2. This does not broaden upstream's supported GPU matrix.

## Non-eager measurements

Latest independent confirmation, N=896 and K=7168; microseconds per operation:

| M | Selected block | Original AOT block 256 | Tuned AOT | PyTorch/cuBLAS | cuBLAS / tuned AOT | Original / tuned AOT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 18.974 | 18.238 | 21.198 | 1.162x | 1.040x |
| 2 | 128 | 19.199 | 18.493 | 19.681 | 1.064x | 1.038x |
| 4 | 64 | 16.427 | 15.266 | 22.161 | 1.452x | 1.076x |

In this confirmation, block-size tuning reduced AOT latency by about 3.7%-7.1%
against the original block size. The cuBLAS-to-AOT ratios above are measured
kernel-operation ratios, not model token-throughput ratios.

Variability matters: the retained first confirmation selected blocks 512/128/64
and measured cuBLAS/AOT ratios 1.869x/0.944x/1.548x for M=1/2/4.
M=2 was about 5.9% slower than cuBLAS in that earlier confirmation, despite
winning in the sweep. The later confirmation is not proof of a universal win.
The M=4, block-512 sweep case also lost to cuBLAS (0.747x in the final run).
All raw samples, including these regressions, remain in the evidence files.
The maximum observed absolute error across the 16 correctness cases was
0.00006103515625. Windows still reported Python cutlass unavailable.

The comparator is the same mathematical operation through native Windows
PyTorch CUDA GEMM: torch.mm(a, b.T, out_dtype=torch.float32, out=buffer).
Both implementations read the same BF16 values and produce FP32. The baseline
does not round its result to BF16 or pay an artificial BF16-to-FP32 cast.
It is a directly selected same-operation comparator, not a claim that every
TokenSpeed model automatically falls back to this function.

For each trial, a graph captures 100 operations and replays ten times. Fifteen
paired rounds alternate execution order; CUDA events report elapsed GPU time
divided by 1,000 operations. Reported values are medians, with all samples and
min/max retained. Each graph gets 20 warmup replays, in addition to ten ordinary
warmup operations before capture. Allocations, compilation, module loading,
CPU reference calculations and correctness checks are outside timing.
The working set uses repeated weights without an L2 flush; this is a cache
reuse microbenchmark, not a cold-cache serving measurement. One operation is
not necessarily one graph kernel node: the PyTorch M=2/4 baseline uses two.

The initial sweep allocates tensors separately for each specialization. A
separate confirmation phase retests the chosen block size, original block
size 256, and cuBLAS against shared input buffers, with separate preallocated
outputs. Tuning choices can vary across runs. Retained earlier measurements
make this variability visible; no automatic production cutover is inferred
from the fastest sweep entry.

Correctness passed for all 16 specializations: three ordinary input sets and
three changed-input graph replays per specialization, output reset to NaN
before execution, plus a non-default stream test. The reference multiplies
the actual BF16 input values on CPU in FP64, then converts to FP32; tolerance
is atol=0.001, rtol=0.0001. Wrong dtypes, shapes, noncontiguous output and
overlapping output are rejected. Missing specializations and corrupted
CUBINs also fail explicitly. Additional confirmation outputs are checked
against the same CPU reference. BF16 reduced-precision reduction is disabled
for the PyTorch baseline. Ordinary launches are correctness checks only;
performance results in this report are exclusively non-eager.

## What explains the difference

The vendored kernel specializes small M and K, vector-loads BF16 operands,
reuses each loaded weight vector across M token accumulators, and performs
FP32 warp/shared-memory reduction inside one CTA per output column. This
avoids a separate global split-K reduction. Block-size tuning changes its
work distribution and reduction cost; the upstream default was measured on
a different GPU and cannot be assumed optimal here.

Actual captured graph inspection shows one CuTe kernel in all three shapes.
For M=1, PyTorch also launches one cuBLAS GEMV kernel, so fewer launches cannot
explain that case. For M=2/4, PyTorch uses a CUTLASS WMMA GEMM plus a cuBLASLt
split-K reduction, two kernels per operation. This supports a concrete
algorithmic distinction, but fewer launches alone do not guarantee a win.
No hardware-counter profile establishes the exact contributions of cache,
occupancy or clock variation. AOT removes the need for the unsupported native
Python compiler; AOT itself is not evidence that identical device code runs
faster than JIT-generated code.

## Reproduce

In WSL2 ARM64, use a separate virtual environment. Replace the checkout path
with the Windows checkout mounted under /mnt/c. The public source named by
--source must contain the LLBf16Dotprod kernel; a source hash in each manifest
records exactly what was compiled. The validated compiler environment used
CPython 3.13, CuTe DSL 4.7.0 and CUDA bindings 13.4.4. Emitted PTX identifies
NVVM/CUDA 13.3 and sm_121a.

```bash
python3 -m venv ~/venvs/tokenspeed-cutedsl47
~/venvs/tokenspeed-cutedsl47/bin/python -m pip install --index-url https://pypi.org/simple --only-binary=:all: 'nvidia-cutlass-dsl[cu13]==4.7.0' 'cuda-bindings==13.4.4'
checkout=/mnt/c/Users/user/projects/official-repos/windows-tokenspeed
LD_LIBRARY_PATH=/usr/lib/wsl/lib ~/venvs/tokenspeed-cutedsl47/bin/python "$checkout/tools/windows/cute_dsl_aot/export_router.py" --source "$checkout/tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cute_dsl/ll_bf16/_kernel.py" --output "$checkout/.port/cutedsl-history/router-aot-optimized" --block-sizes 64 128 256 512
```

In native Windows ARM64 PowerShell, use the existing native PyTorch environment
and the prepared Windows CUDA environment runner. The validated version was
torch 2.14.0a0+git20b0f57 / CUDA 13.4; Python cutlass was absent.

```powershell
$nativePython = (Resolve-Path .venv/Scripts/python.exe).Path
& $nativePython tools/windows/run.py $nativePython tools/windows/cute_dsl_aot/test_router_aot.py --directory .port/cutedsl-history/router-aot-optimized --rounds 15 --operations 100 --replays 10 --confirm-tuning yes
& $nativePython tools/windows/run.py $nativePython tools/windows/cute_dsl_aot/inspect_graphs.py --directory .port/cutedsl-history/router-aot-optimized
```

The exporter writes CUBIN/PTX files and manifest.json. The Windows test writes
windows-results.json with raw samples, source/compiler provenance and helper
hashes. The inspector writes graph-kernels.json and local DOT diagnostics.
Generated binaries stay under the ignored .port directory. Reviewable evidence
is retained under [docs/validation/windows-cutedsl-aot](../validation/windows-cutedsl-aot/).

## Coverage of the original experiment

The [native-host follow-up](windows-arm64-cutedsl-native-host.md) now validates
concrete cluster/PDL/TMA cases. This original experiment retains its narrower
scope.

The original cluster-based routed normalization, argmax, attention and full
model execution have not been ported by this change. Cluster launches need
separate Driver API configuration and hardware validation. Existing SM120+
exclusions are preserved. Other shapes, layouts, output dtypes, bias, PDL,
multiple GPUs and other drivers/architectures require additional artifacts
and tests. Do not equate this isolated router result with Qwen decode speed,
general TokenSpeed acceleration, or an OS-level Windows-versus-WSL comparison.
The follow-up implements opt-in registry integration and per-shape selection
for this router. General artifact packaging/versioning, other operations and
model-level validation remain separate work.
