# Windows ARM64 native CUDA wheel build

This guide covers four project wheels: scheduler, engine, kernel and runtime.
The native build includes all 24 CUDA groups. Python runtime, model registration,
quantization, operator selection and Linux release workflows retain upstream code.

## Prepared environment

| Input | Prepared Windows ARM64 configuration |
|---|---|
| Python | Native ARM64 CPython 3.13.16 |
| Compiler | MSVC ARM64 19.51.36260; Windows SDK 10.0.26100.0 |
| CUDA | SDK 13.4, nvcc 13.4.59; tested GPU architecture 121a |
| Rust | 1.92.0, aarch64-pc-windows-msvc |
| Clang | LLVM 21.1.8 ARM64 compiler; needed by the locked ring dependency |
| CMake / generator | CMake 4.4.4 / Ninja |
| OpenSSL | 3.5.4 static libcrypto, /MD; compatible ARM64 include/lib prefix |
| Torch | 2.16.0.dev20261101+cu134, cp313-win_arm64 |
| TVM FFI | 0.1.15.post2, cp313-win_arm64 |
| Triton | tokenspeed-triton 3.9.10.post20261023, cp312-abi3-win_arm64 |
| FlashInfer build input | Public 0.6.17 wheel, extracted headers/data |

The remaining build requirements are pinned in
`requirements/windows-arm64-build.txt`. The explicit no-build-isolation recipe
uses setuptools 80.10.2; the runtime project's isolated build configuration
declares setuptools 83.0.0. This guide does not claim a clean isolated PEP 517
build with that different configuration.

Torch comes from the [NVIDIA nightly feed](https://pypi.nvidia.com/nvtorch_oot_nightly/).
TVM FFI and FlashInfer build data come from public PyPI. The vendor Triton wheel
comes from the [verified vendor build](https://github.com/chinazhangchao/triton-windows/actions/runs/39916481668),
mirrored byte-for-byte in the
[vendor build-input release](https://github.com/kujin66/windows-tokenspeed/releases/tag/windows-arm64-build-deps-20261109).
Its source is 0c4736788169767960cb1455268353fc3de79970 and wheel SHA256 is
107470c3f1269bb91513a0b5c7539cc1a324b75a98f1e60247d905ee8225b08e.
The compiled extension, AMD library, PTXAS and TinyCC payloads are ARM64.
This wheel uses the genuine tokenspeed_triton namespace. The TokenSpeed adapter
shares the vendor import and scoped third-party namespace redirect with Linux.
The optional Proton profiler is absent from this wheel and is not validated.
The preparation script verifies fixed wheel SHA256 values and patches the public
headers for MSVC. It does not install FlashInfer's Linux communication dependencies.
Patch application isolates Git repository discovery so outputs inside a checkout
receive the same fixes as standalone headers; unchanged patch targets fail
preparation instead of allowing a silently skipped compatibility patch.
Patch text and its manifest hash use canonical LF line endings, including when
Git checks the patch out with Windows CRLF line endings.
The patch command enables Git long paths only for that process, so nested build
and test directories work without modifying system or global Git settings.
The native workflow passes the same process-level Git setting through checkout
and cleanup so a reused runner can remove long-path test outputs safely.

Install Microsoft/NVIDIA SDKs from their official providers. Prepare ARM64 Python,
Rust and OpenSSL before invoking the build. Cargo's locked registry sources,
LLVM's [Windows ARM64 archive](https://github.com/llvm/llvm-project/releases/tag/llvmorg-21.1.8),
GoogleTest v1.14.0 and xgrammar v0.2.8 must also be available. The
[build-input release](https://github.com/kujin66/windows-tokenspeed/releases/tag/windows-arm64-build-deps-20261105)
contains public wheels, a transitive build wheelhouse, Cargo vendor sources,
OpenSSL source/install archives, GoogleTest and xgrammar sources, licenses and
SHA256 manifests. SDK installers are external prerequisites.

## Activate the toolchain and prepare dependencies

Use Visual Studio's native ARM64 developer PowerShell. Set CUDA_HOME/CUDA_PATH to
the ARM64 CUDA SDK. Include the LLVM compiler bin directory in PATH
so ring can find clang.exe. Set OPENSSL_ROOT_DIR to the ARM64 static /MD prefix.
Set OPENSSL_LICENSE_FILE to LICENSE.txt from the same OpenSSL source/install archive.
The source repository does not duplicate OpenSSL license texts; Windows wheels
carry the external license with their statically linked libcrypto. Missing
license input fails the Windows package build. Linux packaging is unchanged.

VS2022's `Launch-VsDevShell.ps1` accepts only x86/amd64 host arguments. The GPU
workflow uses `vcvarsall.bat arm64` and explicitly selects
`VCToolsInstallDir/bin/Hostarm64/arm64` instead. Native ARM64 host
tools must be installed; `TOKENSPEED_MSVC_ROOT` can select a prepared toolset.

```powershell
python -m venv .venv
.venv/Scripts/python tools/windows/prepare_public_dependencies.py --output outputs/public-dependencies
.venv/Scripts/python -m pip install --find-links outputs/public-dependencies -r requirements/windows-arm64-build.txt
$env:TOKENSPEED_FLASHINFER_DATA_DIR = (Resolve-Path outputs/public-dependencies/flashinfer-headers).Path
$env:OPENSSL_ROOT_DIR = 'C:/build-inputs/openssl-arm64'
$env:OPENSSL_LICENSE_FILE = 'C:/build-inputs/openssl-arm64/LICENSE.txt'
$env:CUDA_HOME = $env:CUDA_PATH
$env:PATH = 'C:/build-inputs/llvm/bin;' + $env:PATH
$env:CARGO_TARGET_DIR = 'C:/build-inputs/ts-cargo'
```

For offline preparation, place the four verified public input wheels in
`outputs/public-dependencies`, use `--verify-only` on the preparation script, and
add `--no-index --find-links C:/build-inputs/wheelhouse` to pip installation.
Extract the release wheelhouse before using it; it includes the prepared
source-built ARM64 psutil wheel and provenance.

Resolve the current Cargo.lock with `cargo fetch --locked`. For an offline build,
generate a matching snapshot with `cargo vendor --locked vendor` from
`tokenspeed-engine` and save its source replacement configuration in
`tokenspeed-engine/.cargo/config.toml`. Preserve the current Cargo.lock; the older
release vendor archive must be checked against it before reuse.
Set `FETCHCONTENT_SOURCE_DIR_XGRAMMAR` and, through CMAKE_ARGS,
`FETCHCONTENT_SOURCE_DIR_GOOGLETEST` to the extracted source directories when
building offline. They must match the documented tags. Cargo and CMake may fetch
public sources in an online prepared environment.

## Build and validate

The native workflow preserves VCToolsVersion and WindowsSDKVersion across job
steps. When selecting TOKENSPEED_MSVC_ROOT, its VCINSTALLDIR/version pair must
identify that same standard VC/Tools/MSVC/version tree. A vendor JIT preflight
runs before the full native build, then the installed-wheel suite repeats the
checks against the newly built packages. Missing toolchain-discovery variables
must fail the preflight rather than being hidden by cached compiler helpers.

Use a dedicated venv: the entrypoint installs the newly built wheels into it.
Keep checkout/output paths short. Set CARGO_TARGET_DIR to a short cache path
when a deep checkout would exceed MSVC object/PDB path limits.
Each output directory must be new. Supply the architecture explicitly.
The Windows entrypoint retries a command at most twice only when Rust reports a
temporary-archive cleanup sharing violation (OS error 32). It preserves command
output and exit codes; other build/compiler failures are not retried.

```powershell
.venv/Scripts/python tools/windows/build_wheels.py --cuda-arch 121a --output outputs/windows-arm64-1
```

The entrypoint builds the four wheels with `--no-build-isolation --no-deps
--no-index`, builds/runs scheduler CTest, audits ARM64 PE payloads and wheel tags,
requires all 24 CUDA groups, then installs its own output wheels. It runs native
package/OpenSSL SHA256 checks, changed-input CUDA Graph checks, selected Marlin
and communication-sentinel numerical checks, and Rust host/metrics/scheduler tests.
It also verifies the installed vendor compiler identity and scoped import
restoration, compiles Triton and Gluon kernels, replays both CUDA Graphs with
changed inputs, and checks TokenSpeed vendor SwiGLU for batches 1 and 4.
The existing native RMSNorm graph checks remain separate. Importing the complete
Python norm/quantization registry still requires additional runtime libraries,
including FlashInfer; the header-only build input does not provide that runtime.
Configured CUDA tests fail when no GPU is available; they do not silently skip.
Wheel inventories, source-diff hash and JUnit/log files are saved in the output.
Setuptools staging trees are retained in separate output directories to avoid
Windows file-lock errors during post-package cleanup. Remove these rebuildable
directories after their processes and indexers release them.

The complete build produces 27 ARM64 native payloads across the wheels. The
numerical tests cover selected operations, rather than every shape of every
compiled group. Retain source commit/diff hashes with results.

## CI and Linux regression

`.github/workflows/windows-arm64-build.yml` runs portable PE audit behavior tests
on Linux and Windows hosted runners. Its separate native CUDA job is an explicit
manual opt-in requiring `[self-hosted, Windows, ARM64, tokenspeed-cuda]`. Use
an additional label unique to the intended runner and supply it as runner_label
when dispatching with native_cuda=true and an explicit cuda_arch. An ephemeral
runner can execute one manual job and deregister afterward; a Windows service
installation is not required. Keep its workspace and Cargo cache paths short.
Configure the runner with the SDKs, native Python, OpenSSL prefix/license and source inputs
above. The native job calls the same build entrypoint and uploads wheels,
build manifest and JUnit. A hosted build-tool check is not native ARM64 CUDA
compilation evidence. A native job requires an available configured runner.

Existing upstream Linux workflows and runtime files are unchanged. Windows
compiler/linker choices are gated by WIN32, _MSC_VER or Python target platform.
The Linux toolchain retains dynamic crypto/stdC++ linking, original Marlin
dispatch and CUDA compilation flags. RoPE FP8 dispatch retains its original
enabled/disabled calls through macro expansion. Verify both variants when
changing those macros.

Run the upstream Rust workspace and scheduler Python/C++ suites on Linux after
changes. GPU kernel/model/distributed tests require the corresponding upstream
hardware/dependency matrix; a single SM12.1 Windows/WSL machine cannot provide
SM90/SM100/SM107 multi-GPU, AMD or Ascend coverage.

## Scope and limitations

- Building all native CUDA groups does not establish that every Python vendor
  backend, model, quantization mode or multi-GPU feature works on Windows.
- Full Windows runtime startup/inference is outside this compiler port. This
  change does not add experimental dense-Qwen model loading or comparison flags.
  Native numerical checks load the compiled libraries directly.
- The prepared Windows environment does not provide the upstream DeepEP,
  TRT-LLM GEMM or fast_hadamard_transform Python dependency paths. Their runtime
  support is unvalidated; their absence must not suppress original Linux code.
- Extracting public FlashInfer headers does not install the separate experimental
  FlashInfer 0.6.17+g2ea3c24b5dac runtime port used by earlier measurements.
- The pinned Windows vendor wheel is distributed through the linked public
  build-input release. Matching a distribution version does not establish
  support for every compiler API, GPU architecture or Proton profiling path.
- Windows Python CuTe DSL and strict Hugging Face output-sequence equivalence
  remain unvalidated/unsatisfied by the earlier experiments. Earlier throughput
  results belong to their recorded source and environment, not this reduced
  compiler-only source. New throughput claims require a new matching evaluation.

## Source baseline and validation

This port is based on upstream main at
6023794d37a8f44b48e7f04522a807c39f43a839 (2026-11-09). The Windows ARM64
compiler changes are reapplied from PR4. The upstream runtime, test-scope
selection helper and Linux release workflows come from that main snapshot.
The lint toolchain pin and opt-in Windows ARM64 build workflow supplement
upstream CI. The K8s manual dispatch menu also includes an existing MI35x
monokernel performance configuration omitted by the upstream snapshot.

Historical PR3/PR4 throughput results are not validation of this source.
Fresh native build, JIT, numerical and Linux regression results are recorded
in the pull request with their exact tested source and, when run, Actions links.
Do not infer full Windows inference support from compiler checks.
