# Windows ARM64: complete source reconstruction and native build details

This guide accompanies the [performance report](windows-arm64-performance-report.md).
The PR includes the **66 complete changed/added source files** that turn official
upstream `864bcdd9` into the validated Windows port `9a50ac82`, including its
Python runtime, C++ scheduler, Rust engine, CUDA/Triton kernels and regression
tests. They are readable under [the source overlay](../../ports/windows-arm64/overlay/)
and individually identified by upstream blob, port blob and SHA256 in
[source-manifest.json](../../ports/windows-arm64/source-manifest.json).
Unmodified source comes from the pinned official upstream revision. The
[file inventory](../../ports/windows-arm64/FILES.md) lists every changed path.

## 1. Obtain the complete tested source from this PR

Run the following from the PR checkout, using a Python interpreter with Git
available. Reconstruction itself requires no torch, GPU, MSVC or CUDA package.

```powershell
$py = './.venv/Scripts/python.exe'
& $py tools/windows/prepare_source_checkout.py --action verify
& $py tools/windows/prepare_source_checkout.py --action create --destination ../tokenspeed-windows-arm64-build --git-source https://github.com/lightseekorg/tokenspeed.git
& $py tools/windows/prepare_source_checkout.py --action verify --destination ../tokenspeed-windows-arm64-build
```

For offline reconstruction, `--git-source` can name a local repository containing
the pinned upstream commit. The destination must not exist. The script fetches
the exact base, checks each preimage, copies and verifies all 66 postimages,
and rejects additional source differences within the five source components.
It copies this PR's build helpers, requirements, build skill, reports and raw
evidence. Experimental FlashInfer script snapshots and the WSL environment
template are restored into `.port`, ready for the documented preparation steps.
`.port/source-reconstruction.json` records the source identities and copied
support-file hashes. It does not create venvs or install system dependencies.

The resulting checkout contains the old, tested Python `Engine` API. Its Git
HEAD remains the official upstream base, with the verified Windows changes in
the working tree. Record both HEAD and the overlay manifest when reporting a
build. The newer PR base `main` has a different runtime API; the overlay must
not be copied directly over that runtime or described as a tested latest-main
integration. No external Windows-port clone is needed to reconstruct the code.

## 2. Native toolchain and dependency preparation

| Input | Exact tested identity | Where it enters the build |
| --- | --- | --- |
| CPython | 3.13.16, native Windows ARM64 | venv, PyO3 interpreter, Python import library and DLL |
| MSVC | Toolset 14.51; compiler 19.51.36260, `_MSC_VER=1951`, `_MSC_FULL_VER=195136260` | Hostarm64/arm64 cl and linker 14.51 |
| Windows SDK | 10.0.26100.0 | ARM64 headers and import libraries |
| CUDA SDK | 13.4.0, nvcc component 13.4.46 | SM121a device compilation and ARM64 CUDA import libraries |
| Rust | 1.92.0, aarch64-pc-windows-msvc | PyO3 extension and embedded-Python host executable |
| CMake / Ninja | 4.4.4 / 1.13.2 | Scheduler and Rust/C++ dependency builds |
| setuptools / wheel | 80.10.2 / 0.48.0 | Actual non-isolated wheel backend |
| scikit-build-core / nanobind | 1.1.1 / 3.1.0 | Scheduler Python binding |
| tokenspeed-spdlog | 1.15.1 | C++ headers and CMake package |
| torch / TVM FFI | 2.14.0a0+git20b0f57 / 0.1.13.post3, native ARM64 wheels | GPU runtime, FFI headers, tvm_ffi.lib and DLL |
| Triton | triton-windows 3.8.0.post29 | SM12x JIT, including packed NVFP4 matrix multiply |
| xgrammar C++ | 0.2.8 source | Statically linked grammar core; distinct from Python xgrammar |
| GoogleTest | v1.14.0 source | Optional scheduler C++ regression executable |

Source/provenance records are included locally in
[MSVC payloads](../validation/windows-arm64-20261026/msvc-packages.json),
[download audit](../validation/windows-arm64-fp8-20261027/inventory/download-cache-audit.json),
[observed Python environment](../validation/windows-arm64-fp8-20261027/inventory/python-environment.json)
and the [dependency reference](../../skills/tokenspeed-windows-arm64-build/references/dependencies.md).
Rust's official archive is
https://static.rust-lang.org/dist/rust-1.92.0-aarch64-pc-windows-msvc.tar.xz,
SHA256 `cd9f4ceb634329b920285998d451cb16df59d7ba6e7fade49c39c9194a6981b8`.
xgrammar 0.2.8's source URL and SHA256
`8305646fd2b319d6c3003f51a1665e54cbbe9cba1fb636bc41d0b46434ccbb80`
are pinned in the overlaid `tokenspeed-engine/crates/ts-grammar/CMakeLists.txt`.

Install VS/Windows SDK/CUDA through their official distributions. Extract the
documented MSVC compiler/CRT/resource payloads into `.port/msvc`, preserving
`Contents/VC/Tools/MSVC/14.51.36231`. That directory name is not the actual
compiler version. nvcc also requires the official `VC/Auxiliary/Build/vcvarsall.bat`
layout; the runner restores its missing workspace copy from installed VS.
Missing this layout caused `Host compiler targets unsupported OS` even when
`cl.exe` itself ran correctly. The compiler frontend and NVIDIA executables
were not modified and no compiler-version macros were forged.

Create the venv using native ARM64 Python. Supply verified ARM64 CUDA torch and
TVM FFI wheels, install the documented build/runtime dependencies, unpack Rust
and xgrammar, and populate Cargo's locked dependency cache. A prepared-machine
alternative is `prepare_local.ps1`, which copies the documented toolchains,
archives, model and selected Python dependencies from an existing prepared
workspace into a new venv. It does not copy the existing venv wholesale.

The measured venv uses `local-runtime.pth` to share additional dependencies.
[Observed constraints](../../requirements/windows-arm64-tested.txt) are not a
complete lockfile or an offline wheelhouse. Native dependency wheels, SDKs,
driver installers, model weights and caches are external binary inputs; their
versions/provenance are documented rather than redistributed as project code.
The main package declares setuptools 83.0.0, while the actual non-isolated build
used 80.10.2. Python xgrammar was 0.2.1 while its separately linked C++ core was
0.2.8; grammar was disabled in the measured workload. Full dependency resolution
and a clean-machine installation have not been validated.

## 3. Compiler environment, smoke test and wheel builds

In the reconstructed checkout, `tools/windows/run.py` obtains the installed
ARM64 VS/SDK environment and substitutes the workspace MSVC 14.51 paths.
It sets Python DLL/library search paths and CUDA/cache paths. Verify actual
compiler output and PE metadata; a PATH change alone cannot rule out a cached
CMake selection of MSVC 14.44. The compiler-layout regression temporarily moves
only the workspace copy of vcvarsall.bat; do not run it alongside another build.

```powershell
$py = './.venv/Scripts/python.exe'
& $py tools/windows/run.py $py tools/windows/check_cuda_toolchain.py
$env:PATH = (Join-Path $PWD '.port/rust-toolchain/bin') + ';' + $env:PATH
$env:CARGO_HOME = Join-Path $PWD '.port/cargo'
$env:CARGO_NET_OFFLINE = 'true'
$env:PYO3_PYTHON = Join-Path $PWD '.venv/Scripts/python.exe'
$env:FETCHCONTENT_SOURCE_DIR_XGRAMMAR = Join-Path $PWD '.port/xgrammar-0.2.8'
$env:CMAKE_GENERATOR = 'Ninja'
$env:CMAKE_ARGS = '-DCMAKE_TRY_COMPILE_CONFIGURATION=Release'
$env:PYTHONPATH = ''
$env:TOKENSPEED_KERNEL_BACKEND = 'cuda'
$env:TOKENSPEED_KERNEL_WINDOWS_PROFILE = 'dense-triton'
$env:TOKENSPEED_CUDA_ARCH = '121'
$env:TOKENSPEED_KERNEL_CUDA_FRONTEND_CCBIN = ''
New-Item -ItemType Directory -Force dist | Out-Null
& $py tools/windows/run.py $py -m pip wheel ./tokenspeed-scheduler --no-build-isolation --no-deps --wheel-dir dist --config-settings=build-dir=.port/scheduler-reproduction-new
& $py tools/windows/run.py $py -m pip wheel ./tokenspeed-engine --no-build-isolation --no-deps --wheel-dir dist
& $py tools/windows/run.py $py -m pip wheel ./tokenspeed-kernel/python --no-build-isolation --no-deps --wheel-dir dist
& $py tools/windows/run.py $py -m pip wheel ./python --no-build-isolation --no-deps --wheel-dir dist
```

These commands explain the build stages. The automated
`powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools/windows/build_wheels.ps1 -BuildId reproduction-new`
performs the same sequence, checks every command's exit code, refuses a dist
directory containing anything other than four candidate wheels, installs those
four wheels and runs the native audit. Choose a new build ID when changing
compiler/generator state. Cargo's cache must already be complete for offline
mode; otherwise explicitly prepare it first.

### Scheduler: C++20, nanobind and CNG

The scheduler is built with scikit-build-core/CMake/Ninja. Windows selects
`bcrypt` and the CNG SHA256 implementation in `prefix_hasher.h` instead of
OpenSSL, preserving the hashed byte stream. `NOMINMAX`, `WIN32_LEAN_AND_MEAN`,
MSVC UTF-8/permissive options and missing standard headers address Windows
compilation. Installation includes the runtime DLL destination as well as the
library destination. GoogleTest uses the same shared CRT.

### Engine: PyO3 extension and embedded CPython host

`setup.py` runs `cargo build --release --locked` separately for `tokenspeed-py`
and `tokenspeed-host`. Only the extension sets `PYO3_BUILD_EXTENSION_MODULE=1`;
the host links the current interpreter's import library and must locate its
DLL/standard library at runtime. Windows packages `_native.dll` under the Python
extension suffix and `bin/ts-host.exe`. CMake builds the scheduler embed and
static xgrammar core with Ninja. The build handles a scheduler symlink checked
out as a text file and strips Rust's Windows verbatim-path prefix for MSVC.
Windows links bcrypt and avoids Linux-only stdc++/crypto/rpath flags.

### Kernel: ARM64 TVM FFI DLLs, SM121a and packed FP4

The `dense-triton` profile selects `requirements/windows-cuda.txt`, skips nested
pip resolution and compiles `rmsnorm_fused_parallel`, `sampling_chain` and
`cublas_fp8_tensor`. Actual target code is `compute_121a/sm_121a`. Flags include
C++17, `-O3`, `-DNDEBUG`, `-use_fast_math`, `--expt-relaxed-constexpr`,
`--use-local-env` and MSVC `/MD /EHsc /utf-8 /Zc:__cplusplus /Zc:preprocessor`.
Link inputs are CUDA's ARM64 `cudart.lib/cuda.lib`, `tvm_ffi.lib` and, for the
FP8 group, cuBLASLt. Each group records includes, flags, compiler identity and
frontend choice in build-identity metadata. Files retain the upstream `.so`
suffix but contain ARM64 Windows PE DLLs loaded by TVM FFI.

The overlay also carries the cuBLAS FP8 binding and packed NVFP4 Triton
`tl.dot_scaled` implementation for SM12x. Packed weights, 16-element scale
groups, fused projection global scales and compressed-tensors' divisor/inverse
global-scale convention are preserved. Their earlier numerical/inference tests
are historical evidence, separate from the final dense BF16 performance matrix.

### Runtime: Windows process/IPC and optional backends

The port uses inherited Windows pipe handles/PipeConnection and process-group
creation for the Rust host, guards POSIX-only signals and adjusts temporary-file
reopening for autotune caches. Optional NCCL/DeepEP/MLA/MoE paths are guarded;
single-GPU initialization does not imply these paths are ported. uvloop is
excluded on Windows. The source regression tests covering host startup,
plugin selection and FP8/NVFP4 schemes are included in the overlay.

## 4. Install, audit and exercise the built artifacts

Six project-native files are expected: scheduler extension, engine extension,
host executable and three CUDA FFI libraries. Their PE machine must be `0xaa64`,
and this build's linker must be 14.51. Verify that Python itself is native
ARM64, compare every installed project file with its wheel, and record SHA256.
Do not treat an ARM64 PE header or a successful import as a numerical test.

```powershell
$py = './.venv/Scripts/python.exe'
& $py tools/windows/run.py $py tools/windows/check_native_artifacts.py
& $py tools/windows/run.py $py -m pytest test/runtime/test_compressed_tensors_fp8.py test/runtime/test_compressed_tensors_nvfp4.py test/runtime/test_runtime_plugins.py test/runtime/execution/test_dummy_drafter.py test/runtime/test_rust_host.py -q --basetemp=.port/runtime-tmp-new -o cache_dir=.port/runtime-cache-new
```

The [validation reference](../../skills/tokenspeed-windows-arm64-build/references/validation.md)
provides scheduler C++, installed-wheel kernel tests, RMSNorm/sampling numerical
probes and BF16/NVFP4 real-model commands. Clear source PYTHONPATH for installed
runtime validation; kernel GPU tests use `--noconftest --import-mode=importlib`
to avoid a source-inserting conftest. Use repository-local, uniquely named
pytest temp/cache directories because the system shared Temp ACL previously
caused setup errors. Preserve failed attempts separately from successful reruns.

The [2026-10-28 build record](../validation/windows-wsl2-perf-20261028/README.md)
has 160 runtime, 26 GPU and 16 performance-tool passes. Earlier C++/native/FP8/FP4
records are included with a separate
[historical publication manifest](../../ports/windows-arm64/historical-evidence-manifest.json).
Do not add those earlier overlapping counts to 202, or present source
reconstruction/publication checks as new GPU/model runs. New builds may have
different wheel names and hashes due to dates, paths and ZIP timestamps.

## 5. FlashInfer, WSL2 and performance boundaries

Follow the [FlashInfer procedure](windows-flashinfer-comparison.md) only after
base wheel validation. It creates isolated kernel copies, installs the supplied
0.6.17 JIT source wheel, disables unavailable MoE registrations in both Windows
controls, registers the restricted SM12.1 FA2 prefill provider and applies the
hashed-object-name fix. TRT-LLM decode remains active. GPU numerical probes
must precede model timing. Preparation scripts are first-run recipes.

The [WSL2 build](wsl2-arm64-build.md) uses official upstream and Linux AArch64
wheels in its own filesystem/venv. The included `tools/perf/wsl-env.sh` records
the separate 13.4 SDK and runtime-library layout. Windows wheels are not reused
on Linux. The final matrix uses the same FA2 source adapter on both platforms.

The [main report](windows-arm64-performance-report.md) contains all eager and
non-eager results, graph/eager ratios, raw data links and output consistency.
Non-eager means decode graphs only. Windows FlashInfer 0.6.17 versus Linux
0.7.2, unmatched torch/CUDA/Triton versions, CPU prefill planning, five measured
rounds, unlocked clocks, missing WSL2 profiler GPU activity and unresolved
input-128 output variation prevent an OS-only performance conclusion. No full
service, grammar, multi-GPU, FlashInfer MoE/NVFP4, prefill graph or clean-machine
support claim follows from these builds.
