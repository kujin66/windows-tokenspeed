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
compiler is MSVC 19.51.36260. Set `CUDA_PATH` to the SDK; keep Python's base
directory on PATH so the embedded host can find `python313.dll`.

Prepare a wheelhouse with every dependency in
`requirements/windows-arm64-build.txt`, including native ARM64 CUDA PyTorch
and TVM FFI wheels. They are external binary prerequisites, not supplied by
this PR or guaranteed to resolve from public PyPI. Rust's locked crates and
xgrammar 0.2.8 source must be downloadable; offline runners need a populated
`CARGO_HOME`, `CARGO_NET_OFFLINE=true` and `FETCHCONTENT_SOURCE_DIR_XGRAMMAR`.

```powershell
python -m venv .venv
. .venv/Scripts/Activate.ps1
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
Local prepared-machine validation produced four wheels and five ARM64 PE
payloads: all eight Python checks (including three graph cases) and three Rust
tests passed. Dependencies were shared from an existing environment; clean
wheelhouse bootstrapping and a GitHub Actions run remain unvalidated.

## CI

`Windows ARM64 build` is a manually dispatched workflow on a prepared
`[self-hosted, Windows, ARM64, tokenspeed-cuda]` runner. Install the same
native Python, supported MSVC/CUDA pair, Rust and CMake; configure
`TOKENSPEED_WINDOWS_WHEELHOUSE` and CUDA/Python DLL search paths on that runner.
It uses the same entrypoint and uploads the wheels, manifest and JUnit report.
It needs a compatible NVIDIA GPU for the required graph replay checks.
The workflow is dispatch-only; this PR does not register or provision a runner.
A GitHub Actions run requires that runner and the workflow on the default branch.

## Non-eager reference results

The table below is historical full-engine evidence from the separately
validated Windows port `9a50ac82`; it is **not a benchmark of this build-only
change or of latest main**. Both independent engines used Qwen3-0.6B BF16 on
RTX Spark N1X, 128 generated tokens, 3 warmups and 5 measured calls per case,
with token-ID input and normal text-output APIs. Values are output tokens/s.

| Input / batch | vLLM | TokenSpeed Triton | TokenSpeed FlashInfer |
| --- | ---: | ---: | ---: |
| 128 / 1 | 144.75 | 144.22 | 139.61 |
| 128 / 4 | 582.45 | 608.83 | 564.28 |
| 512 / 1 | 133.66 | 128.36 | 125.84 |
| 512 / 4 | 509.51 | 515.84 | 502.62 |

vLLM used Inductor with FULL_AND_PIECEWISE capture; TokenSpeed captured decode
batches 1/4 with prefill graphs disabled. Compiler policies and dependency
versions differ, token outputs are not equivalent, and five samples do not
establish a stable ranking. These are native Windows results, not WSL2 results.
See the [pinned non-eager evidence](https://github.com/kujin66/windows-tokenspeed/blob/f6405a6295f4ac79e9b523c971d8a8074c670c80/docs/validation/tokenspeed-vs-vllm-20261029/README.md)
for the original method, measurements and limitations. Full-engine execution
on the new API/main remains separate work; this PR's fresh graph checks cover
the compiled RMSNorm and chain-verifier libraries only.
