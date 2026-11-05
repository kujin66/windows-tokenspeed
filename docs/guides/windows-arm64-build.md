# TokenSpeed Windows ARM64 wheel compilation

For the uploaded build wheelhouse, Cargo sources, OpenSSL libraries and exact
reproduction steps, see [the build-input guide](windows-arm64-reproduction.md).

The tested public dependencies are Torch 2.16.0.dev20261101+cu134 from NVIDIA's
nvtorch_oot_nightly feed and apache-tvm-ffi 0.1.15.post2 from PyPI. Use native
ARM64 CPython 3.13. Neither dependency wheel is modified or rebuilt locally.
Windows uses triton-windows 3.9.0.post30 through the central kernel _triton.py.
Native compilation also uses headers extracted from the public FlashInfer
0.6.17 wheel. Its runtime dependency graph is not installed by this build recipe.

## OpenSSL

The scheduler uses OpenSSL SHA256 on every platform; the Windows bcrypt
implementation has been reverted. Set OPENSSL_ROOT_DIR to an ARM64 static
OpenSSL prefix for the scheduler and Rust engine. CMake finds OpenSSL::Crypto;
the Rust facade receives its resolved include path, archive and system libraries
from CMake. Windows wheels link crypto statically and include its license.

OpenSSL 3.5.4 compiled with MSVC 14.44 and 14.51. Use native Strawberry Perl,
an ARM64 MSVC developer shell and a fresh source directory from the
[official release](https://github.com/openssl/openssl/releases/tag/openssl-3.5.4):

```powershell
perl Configure VC-WIN64-ARM no-shared no-module no-asm no-tests no-makedepend 'CFLAGS=/MD /O2' --prefix=C:/build/openssl-arm64 --openssldir=C:/build/openssl-arm64/ssl
nmake /NOLOGO build_libs
nmake /NOLOGO install_dev
$env:OPENSSL_ROOT_DIR = 'C:/build/openssl-arm64'
```

/MD matches Python's MSVC runtime. no-asm selects the supported ARM64 C
implementation. no-tests disables OpenSSL's own test build; TokenSpeed's
scheduler tests remain enabled separately. Record the source version, compiler
and configuration. Tests cover prefix framing, chaining, signed tokens and UTF-8
salts. A build-only x64 Perl executable does not change the ARM64 library target.

## Public dependencies and project build

Prepare an ARM64 MSVC/CUDA developer environment. The tested CUDA SDK is 13.4
with MSVC 14.51, Rust 1.92, CMake and Ninja. Preserve the official
VC/Auxiliary/Build/vcvarsall.bat layout required by nvcc. Put Python's base
directory on PATH for python313.dll discovery.

The retained scheduler build records MSVC 19.51.36260.0, ARM64, CMake 4.4.4,
Ninja and Release configuration. It links the static OpenSSL 3.5.4 installation
built with MSVC 14.44 and /MD; the separate MSVC 14.51 OpenSSL build also passed.
The tested Python build requirements are pinned in requirements/windows-arm64-build.txt.
This recipe uses --no-build-isolation with setuptools 80.10.2. The runtime
pyproject declares setuptools 83.0.0 for isolated builds; that isolated build
contract was not validated by these no-isolation measurements.

```powershell
python -m venv .venv
$env:PATH = (Join-Path $PWD '.venv/Scripts') + ';' + $env:PATH
.venv/Scripts/python tools/windows/prepare_public_dependencies.py --output outputs/public-dependencies
$env:TOKENSPEED_FLASHINFER_DATA_DIR = (Resolve-Path outputs/public-dependencies/flashinfer-headers).Path
.venv/Scripts/python -m pip install --find-links outputs/public-dependencies -r requirements/windows-arm64-build.txt
$env:CARGO_BUILD_JOBS = '2'
$env:CMAKE_BUILD_PARALLEL_LEVEL = '2'
$env:MAX_JOBS = '1'
$env:CMAKE_ARGS = '-DCMAKE_TRY_COMPILE_CONFIGURATION=Release'
.venv/Scripts/python tools/windows/build_wheels.py --cuda-arch 121a --output outputs/windows-arm64-1
```

The helper downloads Torch, TVM, Triton Windows and FlashInfer from their official indexes and
verifies pinned SHA256 hashes. Use --verify-only for existing downloads.
It extracts FlashInfer build data and records the checked-in Windows header
compatibility patch (CUDA qualifiers, integer widths, template syntax and math
helpers). Eight fmt rounding constants are reconstructed from two UTF-16 halves
to avoid MSVC/NVCC rejecting out-of-range UTF-32 escapes; their integer values
are unchanged. Torch, TVM and Triton wheels remain unmodified.
An offline runner can populate and verify the wheelhouse first, then install
with --no-index. Additional prepared dependencies can use --find-links; offline
Cargo registry and optional xgrammar source provisioning are still needed.

## Native kernel coverage

The Windows-specific profile and the later minimal/full group filter have been
removed. CUDA builds submit all 24 native groups to the normal builder.
The wheel audit requires all 24 distinct group payloads and records their names
in build.json. No group is silently omitted to produce a successful wheel.

Earlier minimal experiments only built rmsnorm_fused_parallel and sampling_chain;
the other 22 groups were excluded to bound that experiment, without per-group
compatibility evidence. The review work subsequently attempted every group.
See the [validation report](windows-arm64-validation.md) for completed builds;
successful compilation alone does not establish model or multi-GPU support.

Marlin dispatch uses flat first-match returns to avoid MSVC's 128-level block
nesting limit while keeping the same specializations. Its fatal host path uses
standard abort after FFI error reporting. Collective negative-zero sentinels
use inline device functions instead of const-qualified device variable templates;
the sign bits are checked independently for FP16, BF16 and FP32.

## Review rationale

MSVC rejects the const prefix-index conditional expressions with C2446 because
the map stores iterator while entries.end() from a const container returns
const_iterator. Separate returns permit each conversion independently; existing
prefix-index tests verify found and missing-key behavior.

PE binaries use Windows DLL search rules, including the prepared Python PATH,
rather than ELF/GNU rpath. Omitting -Wl,-rpath does not disable an engine feature.
Python DLL discovery remains an environment requirement independent of static
OpenSSL linking.

RUNTIME DESTINATION handles Windows DLL installation; LIBRARY DESTINATION
handles shared libraries on other platforms. enable_testing is platform-neutral
and already gated by the build-tests option. Rust Windows compiler definitions
now have a Windows target condition. The FlashInfer inline macro uses the
reviewed platform branches. Missing optional FlashInfer headers follow a direct
conditional fallback without raising and immediately swallowing ImportError.

## Existing release workflows

The standalone windows-arm64-build.yml has been removed. release-pypi.yml and
weekly-release.yml call .github/actions/windows-arm64-build on a prepared
[self-hosted, Windows, ARM64, tokenspeed-cuda] runner. Enable it with the
windows_arm64 dispatch input or TOKENSPEED_WINDOWS_ARM64_CI=true repository
variable. Dispatch inputs select the architecture; all CUDA groups are built.
TOKENSPEED_WINDOWS_CUDA_ARCH selects the scheduled
architecture. Provide OPENSSL_ROOT_DIR and native toolchains on the runner;
TOKENSPEED_MSVC_ROOT can select a prepared portable MSVC installation.
The Windows job is independent of the upstream-only publication plan, so a fork
can run the optional build without entering weekly release preparation.

The jobs build, audit and upload wheels, scheduler CTest results, installed-wheel
checks and collective sentinel/fmt/half-math/Marlin tests, and run Rust host/scheduler/metrics
tests. They do not add a
Windows PyPI publication step. Actual Actions execution requires a configured
runner; local YAML and PowerShell validation does not replace that evidence.
The wheel build enables C++ tests explicitly and disables GoogleTest installation,
so test dependencies do not become wheel payloads. CTest fails if no tests exist.

## Inference limits

See [the local SM12.1 integration guide](local-sm121-runtime.md) for the explicit
dense Qwen WireDriver profile. Completed generation and GPU primitive checks do
not establish strict model equivalence: the retained HF full-sequence comparison
failed. The latest Windows/WSL2 comparison matches dependency revisions and
measurement settings, with disclosed platform compatibility adaptations; it does
not isolate the operating system.
