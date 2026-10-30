# Native Windows ARM64 wheel build

This is a compilation-only step against this repository's current main
(`afb6e89b`). It builds the scheduler, Rust engine/host, CUDA kernel and Python
wheels directly from their normal source directories. The explicit `minimal`
kernel profile compiles only RMSNorm and speculative-chain sampling; it does
not claim that every runtime backend is ported.

## Reproduce

Use native ARM64 CPython 3.13.16, an ARM64 MSVC developer shell, Windows SDK
10.0.26100.0, CUDA 13.4, Rust 1.92 (`aarch64-pc-windows-msvc`), CMake 4.4.4
and Ninja 1.13.2. The tested
compiler is MSVC 19.51.36260. Preserve the official VC/Auxiliary/Build/vcvarsall.bat
layout required by nvcc; use the ARM64 host/target compiler, SDK and CUDA import
libraries. Set `CUDA_PATH` to the SDK; keep Python's base
directory on PATH so the embedded host can find `python313.dll`.

Prepare a wheelhouse with every dependency in
`requirements/windows-arm64-build.txt`, including native ARM64 CUDA PyTorch
and TVM FFI wheels. They are external binary prerequisites, not supplied by
this PR or guaranteed to resolve from public PyPI. Rust's locked crates and
xgrammar 0.2.8 source must be downloadable; offline runners need a populated
`CARGO_HOME`, `CARGO_NET_OFFLINE=true` and `FETCHCONTENT_SOURCE_DIR_XGRAMMAR`.

```powershell
python -m venv .venv
$env:PATH = (Join-Path $PWD '.venv/Scripts') + ';' + $env:PATH
$env:CARGO_BUILD_JOBS = '2'
$env:CMAKE_BUILD_PARALLEL_LEVEL = '2'
$env:MAX_JOBS = '1'
$env:CMAKE_ARGS = '-DCMAKE_TRY_COMPILE_CONFIGURATION=Release'
.venv/Scripts/python -m pip install --no-index --find-links $env:TOKENSPEED_WINDOWS_WHEELHOUSE -r requirements/windows-arm64-build.txt
.venv/Scripts/python tools/windows/build_wheels.py --cuda-arch 121a --output outputs/windows-arm64-1
```

Choose a new output directory for each build. The entrypoint builds four
wheels without dependency resolution, rejects non-ARM64 native payloads or
incorrect wheel tags, installs only those wheels into the active venv, and
checks installed packages, CNG hashes against Python SHA256, RMSNorm CUDA
graph replay at batches 1/4 and chain-verifier replay with changing inputs.
It also runs the Rust metrics unit tests, including positioned Windows writes.
The graph checks call installed DLL exports through TVM FFI. High-level
runtime imports, the Windows Triton adapter and full-engine execution require
separate runtime integration.
`build.json` records wheel SHA256, native members, source revision/diff hash
and target architecture; `tests.xml` records the actual test outcome.
This is not a clean-machine dependency installer or a full inference port.
Current verification rebuilt all four wheels from a **fresh tracked Git
checkout**, a fresh ARM64 venv and fresh Cargo/CMake/CUDA output directories.
The 18 project source/build files match the tested ac4b6b32 source exactly.
All **five native PE payloads are ARM64**; **eight Python tests** (including
three changed-input CUDA Graph cases) and **three Rust metrics tests** pass
with no skips. Every installed wheel member except pip's rewritten RECORD
was compared with the wheel bytes.

External ARM64 dependency packages were shared from prepared environments,
and the offline Cargo registry/xgrammar source were pre-provisioned. This
validates project-source completeness; clean wheelhouse bootstrapping and
a GitHub Actions runner execution remain unvalidated. Local and CI recipes
use the same entrypoint, concurrency limits and Release CMake checks.

## CI

`Windows ARM64 build` is a manually dispatched workflow on a prepared
`[self-hosted, Windows, ARM64, tokenspeed-cuda]` runner. Install the same
native Python, supported MSVC/CUDA pair, Rust and CMake; configure
`TOKENSPEED_WINDOWS_WHEELHOUSE` and CUDA/Python DLL search paths on that runner.
It uses the same entrypoint and uploads the wheels, manifest and JUnit report.
It needs a compatible NVIDIA GPU for the required graph replay checks.
The workflow is dispatch-only; this PR does not register or provision a runner.
A GitHub Actions run requires that runner and the workflow on the default branch.

## Minimal source scope

The complete change is 24 files over main afb6e89b: 18 scheduler/engine/CUDA
source or build files, and six supporting files (the build entrypoint,
dependency list, installed-wheel tests, workflow, this guide and output ignore).
The Python runtime wheel uses the existing package sources without a runtime
port or additional kernel/backend integration. No source overlay, binary
payload, benchmark archive or file from PR #1 is required to build this unit.

## Inference evaluation

The [Draft PR evaluation](https://github.com/kujin66/windows-tokenspeed/pull/2)
contains the latest selected non-eager reports, per-platform throughput,
recomputed engine/backend ratios and conclusions. Each configuration uses one
retained process with three warmups and five measured calls per case.

Those measurements use PR #1's complete Windows port 9a50ac82 and WSL2 upstream
864bcdd9 with backend adapters. This compilation-only unit has the separate
build/DLL checks described above. Its full-engine current-main integration
remains separate work.

[Selected Windows FlashInfer report](https://github.com/kujin66/windows-tokenspeed/blob/de694811af609f221c8d7be17fd5cbcc5fc098df/docs/validation/flashinfer-paired-retest-20261029/windows-flashinfer-repeat2.json) · [Selected WSL2 FlashInfer report](https://github.com/kujin66/windows-tokenspeed/blob/de694811af609f221c8d7be17fd5cbcc5fc098df/docs/validation/flashinfer-paired-retest-20261029/wsl2-flashinfer-repeat2.json).
