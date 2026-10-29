# TokenSpeed on Windows ARM64: build and non-eager inference

## Non-eager Windows/WSL2 results

The tables report the latest available non-eager measurement for each
configuration. FlashInfer combines two independent engine-process repetitions
per platform (ten measured calls per case); vLLM and Triton each use five
measured calls per case. These measurements come from separate cohorts with
verified per-platform environments. They are a cross-cohort summary, rather
than a single paired six-configuration experiment.

| Input / batch | Win vLLM | Win TS Triton | Win TS FI | WSL2 vLLM | WSL2 TS Triton | WSL2 TS FI |
| --- | --- | --- | --- | --- | --- | --- |
| 128 / 1 | 137.79 | 143.62 | 141.63 | 143.02 | 141.98 | 138.85 |
| 128 / 4 | 585.17 | 597.74 | 569.60 | 549.26 | 571.50 | 564.50 |
| 512 / 1 | 134.78 | 130.01 | 130.94 | 134.46 | 125.93 | 132.60 |
| 512 / 4 | 509.22 | 522.49 | 490.88 | 496.45 | 499.86 | 501.26 |

Values are aggregate output tokens/s. TS means TokenSpeed; FI means FlashInfer;
Win means native Windows ARM64.

| Input / batch | vLLM Win / WSL2 | TS Triton Win / WSL2 | TS FI Win / WSL2 |
| --- | --- | --- | --- |
| 128 / 1 | 0.963x | 1.012x | 1.020x |
| 128 / 4 | 1.065x | 1.046x | 1.009x |
| 512 / 1 | 1.002x | 1.032x | 0.987x |
| 512 / 4 | 1.026x | 1.045x | 0.979x |

There is no uniform engine/backend winner. The
[complete comparison](windows-wsl2-engine-backend-retest.md) contains the
methods, per-platform environments, independent FlashInfer repetitions,
within-platform ratios, output-consistency checks and reproduction commands.
vLLM and TokenSpeed are independent engines in these measurements.

## Build and source reproduction

The [detailed build guide](windows-arm64-build-details.md) documents the
complete 66-file validated Windows ARM64 port, external dependencies,
compiler/linker flags, source reconstruction and wheel installation checks.
The [source inventory](../../ports/windows-arm64/FILES.md) identifies every
included source file. Native runtime source is pinned at 9a50ac82 over
upstream 864bcdd9. Reconstruct that runtime before reproducing throughput.
The current-main compilation unit has a different runtime API.

Use ARM64 CPython 3.13.16, MSVC toolset 14.51 (compiler 19.51.36260),
Windows SDK 10.0.26100.0, CUDA 13.4/nvcc 13.4.46, Rust 1.92.0,
CMake/Ninja and the recorded ARM64 Torch/TVM FFI wheels. Preserve the official
vcvarsall.bat layout. Build scheduler, engine, kernels and Python runtime in
that order with tools/windows/build_wheels.ps1. Audit the four installed
project wheels against their bytes and check native binaries for ARM64 PE.
Dependency-prepared builds use --no-build-isolation and --no-deps; clean-machine
and public-PyPI-only installation remain untested.

## CuTe compiler and router validation

The [native-host implementation](windows-arm64-cutedsl-native-host.md) uses
official Windows x64 CuTe DSL 4.8.1 under system emulation and compiles original
host launchers into native ARM64 COFF/DLLs. The combined suite passes 33 tests
with zero failures, errors or skips. Execution Python and all 11 host DLLs are
ARM64; compiler Python is x64. Tests cover dynamic arguments/layouts, host
loops, cluster/PDL/TMA cases, cache rejection and TokenSpeed router auto dispatch.
The guide includes the exact tested source restoration and execution commands.

Dense Qwen3-0.6B has no matching MoE router, and every model report records
AOT inactive. These inference results establish no model-level CuTe gain.
The compiler extension itself is not native ARM64, and tested kernels do not
establish support for arbitrary CuTe APIs or complete MoE models.

## Validation and limits

The [complete comparison](windows-wsl2-engine-backend-retest.md) describes
software/graph differences, ID variability, timing scope and statistical
limits. The summary validator recomputes current throughput/ratios and verifies
the published evidence. Publication checks do not rerun GPU measurements.

```bash
python -m tools.perf.summarize_engine_retest --root . --report latest-engine-summary.json
python -m pytest test/tools --noconftest -q -o pythonpath=.
pre-commit run --all-files
```
