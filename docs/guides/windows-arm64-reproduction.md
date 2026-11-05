# Reproduce the TokenSpeed Windows ARM64 wheel build

The [build-input release](https://github.com/kujin66/windows-tokenspeed/releases/tag/windows-arm64-build-deps-20261105)
contains the original public Torch/TVM/Triton/FlashInfer build-data wheels, a
pinned Python build wheelhouse, the tested OpenSSL ARM64 static installations,
public OpenSSL, GoogleTest and xgrammar source archives, locked Cargo source dependencies,
and SHA256 manifests. Package licenses remain inside the original wheels and
vendored sources; the OpenSSL packages also include LICENSE.txt.

The release is a prerelease build-input snapshot. The included project wheels
are the previously measured Windows artifacts, not a claim that every optional
runtime dependency or every native kernel path is supported.

## Install external tools

Install native ARM64 CPython 3.13, ARM64 MSVC C++ tools, Windows SDK, CUDA SDK,
Rust, CMake and Ninja. The tested versions are documented in the
[build guide](windows-arm64-build.md). Use MSVC 19.51.36260, CUDA 13.4 and Rust
1.92.0 for the recorded configuration. Preserve the official MSVC developer-shell
directory layout required by nvcc.

Obtain Microsoft tools from [Visual Studio downloads](https://visualstudio.microsoft.com/downloads/),
CUDA 13.4 from the [CUDA Toolkit archive](https://developer.nvidia.com/cuda-toolkit-archive),
and Python/Rust/CMake from their official distributions. Microsoft and NVIDIA
SDK installers are not included in the repository or release; the downloadable
package contains the redistributable build inputs, not a replacement SDK.

## Download and verify build inputs

Check out the PR branch under a short path such as C:/build/tokenspeed and open
an ARM64 MSVC developer shell at that checkout. Download every release asset:

~~~powershell
gh release download windows-arm64-build-deps-20261105 --repo kujin66/windows-tokenspeed --dir inputs
$manifest = Get-Content inputs/dependency-manifest.json -Raw | ConvertFrom-Json
foreach ($asset in $manifest.assets) {
    $path = Join-Path inputs $asset.name
    $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $asset.sha256) { throw "SHA256 mismatch: $($asset.name)" }
}
~~~

The dependency manifest excludes itself and SHA256SUMS.txt. Verify the manifest
against SHA256SUMS.txt if copying inputs through another storage service.

## Prepare Python, OpenSSL, Cargo and GoogleTest

~~~powershell
python -m venv .venv
$env:PATH = (Resolve-Path .venv/Scripts).Path + ';' + $env:PATH
Expand-Archive inputs/windows-arm64-build-wheelhouse.zip -DestinationPath inputs/wheelhouse
.venv/Scripts/python -m pip install --no-index --find-links inputs --find-links inputs/wheelhouse -r inputs/build-requirements.lock.txt
.venv/Scripts/python tools/windows/prepare_public_dependencies.py --verify-only --output inputs
$env:TOKENSPEED_FLASHINFER_DATA_DIR = (Resolve-Path inputs/flashinfer-headers).Path

Expand-Archive inputs/openssl-3.5.4-win-arm64-msvc14.44-md.zip -DestinationPath inputs/openssl-arm64
$env:OPENSSL_ROOT_DIR = (Resolve-Path inputs/openssl-arm64).Path
Expand-Archive inputs/tokenspeed-cargo-vendor.zip -DestinationPath tokenspeed-engine
$env:CARGO_NET_OFFLINE = 'true'
tar -xf inputs/xgrammar-0.2.8.tar.gz -C inputs
$env:FETCHCONTENT_SOURCE_DIR_XGRAMMAR = (Resolve-Path inputs/xgrammar-0.2.8).Path
Expand-Archive inputs/googletest-v1.14.0.zip -DestinationPath inputs/googletest
$gtestSource = (Resolve-Path inputs/googletest/googletest-1.14.0).Path
$env:CMAKE_ARGS = '-DCMAKE_TRY_COMPILE_CONFIGURATION=Release -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=' + $gtestSource
~~~

The Cargo archive supplies vendor/ and .cargo/config.toml under tokenspeed-engine;
do not overwrite an existing local Cargo configuration without preserving it.
The Python lock describes the tested explicit no-isolation build, using
setuptools 80.10.2. It does not claim validation of the runtime pyproject's
separate isolated setuptools 83.0.0 build requirement.

The OpenSSL MSVC 14.51 package is an alternative to the tested MSVC 14.44 prefix.
Both were built from unmodified OpenSSL 3.5.4 with static libraries, /MD and
no-shared no-module no-asm no-tests no-makedepend. OpenSSL's own tests were not
run. The source tarball is supplied for rebuilding instead of using these libraries.

## Build all four project wheels

Set CUDA_HOME or CUDA_PATH to the CUDA 13.4 SDK and put Python's base directory
on PATH so the embedded Rust host can discover python313.dll. Then run:

~~~powershell
$env:CARGO_BUILD_JOBS = '2'
$env:CMAKE_BUILD_PARALLEL_LEVEL = '2'
$env:MAX_JOBS = '1'
.venv/Scripts/python tools/windows/build_wheels.py --cuda-arch 121a --output outputs/windows-arm64-1
~~~

The entrypoint builds scheduler, engine, all 24 native CUDA groups and runtime
wheels. It checks ARM64 payloads, scheduler tests, installed native checks,
selected CUDA numerical tests and Rust host/metrics tests. Use a new output
directory for another build.

The recorded complete build used a prepared environment. The newly assembled
release inputs and this download/extraction workflow have been verified as
artifacts; a clean machine rebuild from the published release has not yet been
performed. This is also not a complete default inference installation: prepared
FlashInfer runtime adapters, model files and other runtime dependencies are
separate from the public FlashInfer headers needed for compilation.

See the [validation report](windows-arm64-validation.md) for the actual
measurements, failed strict HF sequence equivalence and unvalidated Windows
Python CuTe DSL support. Windows AOT-policy controls do not establish native
Python CuTe DSL compilation.
