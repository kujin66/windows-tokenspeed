# Windows ARM64 build and validation

Validated 2026-10-27 (Asia/Shanghai), upstream
`864bcdd9ef3c5f1512bbe293d970d73eb29d2c94`.

The single-GPU dense BF16 and compressed-tensors NVFP4 Python Engine work
on native Windows ARM64 / RTX Spark N1X (SM12.1). Each checkpoint passed
three sequential and two batched requests, with identical greedy token IDs
within each checkpoint. Four installable wheels are in `dist/`.

Latest evidence and artifact hashes are in
`docs/validation/windows-arm64-fp8-20261027/`. Older native-artifact and
host-detection reports remain historical snapshots before these fixes.
This profile does not establish support for all upstream models/backends.

**Direct MSVC 14.51 + CUDA SDK 13.4.0 builds now pass.** The earlier
host-detection failure came from an incomplete portable Visual Studio layout:
the extracted compiler packages omitted `VC/Auxiliary/Build/vcvarsall.bat`.
nvcc needs that file to recognize the Windows toolchain, including when
`--use-local-env` is supplied. `tools/windows/run.py` copies the official
script from the installed Visual Studio into the portable layout when missing.
The runner prepares the SDK environment; nvcc uses it with `--use-local-env`.
The copied script is not executed in this mode.

Both CUDA preprocessing and host compilation now use the real 19.51.36260
compiler. The former 14.44 frontend helper is unnecessary. No compiler version
macros or NVIDIA binaries are modified. A compiled GPU regression asserts
`_MSC_VER == 1951` and an ARM64 host, then checks all 4097 vector-add results.

最新产物路径、SHA256、依赖快照和测试证据见
[Windows ARM64 构建产物与测试总表](windows-arm64-build-record.md)；可复用的
完整流程见 [构建 skill](../../skills/tokenspeed-windows-arm64-build/SKILL.md)。

## Environment

| Component | Validated version |
| --- | --- |
| OS/Python | Windows ARM64 / CPython 3.13.16 |
| GPU | NVIDIA RTX Spark N1X, SM 12.1 |
| CUDA SDK | 13.4.0 (`version.json`), nvcc V13.4.46 |
| Host compiler/linker | MSVC 14.51, compiler 19.51.36260 |
| Windows SDK | 10.0.26100.0 |
| Rust | 1.92.0, aarch64-pc-windows-msvc |
| PyTorch | 2.14.0a0+git20b0f57 / CUDA 13.4 / win_arm64 |
| Triton | triton-windows 3.8.0.post29 |
| Transformers / FlashInfer | 5.12.0 / 0.7.2 |
| TVM FFI | 0.1.13.post3 / win_arm64 |

Official MSVC packages were SHA256-checked against the
[VS manifest](https://aka.ms/vs/18/stable/channel) and extracted to
`.port/msvc/Contents/VC/Tools/MSVC/14.51.36231`. Rust and xgrammar archives were
checked against official/pinned hashes and unpacked under `.port/`. System
VS/CUDA installations were not replaced.

The project `.venv` has a `local-runtime.pth` referencing the existing
`../vllm-bench/.venv/Lib/site-packages` for ARM64 PyTorch and other dependencies.
New packages were installed only in this project. The environment depends on
that directory: this is not a clean-machine dependency installation test.
Another host needs equivalent ARM64 PyTorch and TVM FFI wheels. Actual package
versions and artifact hashes are saved in the validation directory.

## Changes

The scheduler uses Windows CNG SHA256 instead of an external OpenSSL DLL,
preserving prefix hash framing. MSVC iterator conversions, missing includes,
diagnostic type names and extension installation were fixed. Linux keeps OpenSSL.

The Rust engine builds Windows artifacts, resolves the scheduler source when
Git checks its symlink out as text, handles console signals and uses Windows
file APIs for metrics. Python starts ranks with inherited named-pipe handles
and watches their process handles. On Windows without NCCL, world_size=1 uses
Gloo for startup; larger worlds are explicitly rejected.

The explicit Windows kernel profile is `dense-triton`: portable MHA, RMSNorm
and SwiGLU plus native RMSNorm, chain-sampling and cuBLASLt FP8 libraries. NVFP4 linears use native SM12x FP4 Tensor Cores. Linux-only vendor
packages are excluded from this profile's kernel requirements. Models load on
demand instead of importing every unrelated model. Optional vendor imports
are gated. Autotune temporary files close before FlashInfer reopens them.

## Rebuild

From the repository root in PowerShell, with the local toolchain/dependencies
already present. `tools/windows/run.py` prepares MSVC 14.51, the installed SDK,
CUDA 13.4, Python DLL lookup and limited build jobs.

For manual nvcc commands in an ordinary VS Code terminal, use the wrapper.
It prepares the compiler environment and supplies `--use-local-env`:

```powershell
& ./tools/windows/nvcc.cmd --version
```

This entry works with PowerShell script execution disabled. For an interactive
PowerShell shell with the same environment, start a child shell with a
process-local execution policy:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -NoExit -File ./tools/windows/Activate-Toolchain.ps1
```

The activation script imports the prepared compiler environment from
`run.py --print-env`. It affects this shell and its child processes. Validate
activation, native compilation and GPU execution together with:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ./tools/windows/check_toolchain_activation.ps1
```

```powershell
$py = '.venv/Scripts/python.exe'
$env:PATH = (Join-Path $PWD '.port/rust-toolchain/bin') + ';' + $env:PATH
$env:CARGO_HOME = Join-Path $PWD '.port/cargo'
$env:PYO3_PYTHON = Join-Path $PWD '.venv/Scripts/python.exe'
$env:FETCHCONTENT_SOURCE_DIR_XGRAMMAR = Join-Path $PWD '.port/xgrammar-0.2.8'
$env:CMAKE_ARGS = '-DCMAKE_TRY_COMPILE_CONFIGURATION=Release'
$schedulerBuild = Join-Path $PWD '.port/scheduler-wheel-msvc1451'
& $py tools/windows/run.py $py -m pip wheel ./tokenspeed-scheduler --no-build-isolation --no-deps --wheel-dir dist --config-settings=build-dir=$schedulerBuild
& $py tools/windows/run.py $py -m pip wheel ./tokenspeed-engine --no-build-isolation --no-deps --wheel-dir dist
$env:TOKENSPEED_KERNEL_BACKEND = 'cuda'
$env:TOKENSPEED_KERNEL_WINDOWS_PROFILE = 'dense-triton'
$env:TOKENSPEED_CUDA_ARCH = '121'
$env:TOKENSPEED_KERNEL_CUDA_FRONTEND_CCBIN = ''
& $py tools/windows/run.py $py -m pip wheel ./tokenspeed-kernel/python --no-build-isolation --no-deps --wheel-dir dist
& $py tools/windows/run.py $py -m pip wheel ./python --no-build-isolation --no-deps --wheel-dir dist
```

Native kernels record the host compiler, nvcc flags and an empty frontend
helper in their `build-identity.json`, included in the wheel. The kernel wheel is tagged
`cp313-cp313-win_arm64`; its `.so` files are ARM64 PE DLLs retaining upstream
wrapper filenames. The engine includes `ts-host.exe` and its Python `.pyd`.
The main wheel still declares vendor serving dependencies not ported to Windows.
Install into the prepared environment with `--no-deps`; this is not a fully
resolved public-PyPI installation:

```powershell
Get-ChildItem dist/*.whl | ForEach-Object { & $py -m pip install --force-reinstall --no-deps --no-index $_.FullName }
```

The runner selects Ninja unless `CMAKE_GENERATOR` is explicitly supplied.
Visual Studio's generator can select its installed 14.44 toolset despite
the 14.51 compiler on PATH. The separate scheduler build directory avoids
reusing that older generator/compiler cache. Verify the installed wheel
contents and the actual Python process architecture with:

```powershell
$env:PYTHONPATH = ''
& $py tools/windows/run.py $py tools/windows/check_native_artifacts.py
```

## CUDA toolchain regression

Run from the prepared repository environment. The source asserts the actual
MSVC version and ARM64 target at compile time. Verbose output records
`--microsoft_version=1951`, `--msvc_target_version=1951` and `--target_woa`.

```powershell
& $py tools/windows/run.py nvcc --use-local-env -std=c++17 -arch=sm_121 -Xcompiler /MD,/EHsc,/utf-8,/Zc:__cplusplus,/Zc:preprocessor tools/windows/cuda_smoke.cu -o .port/nvcc1451-cuda-smoke.exe -v
& $py tools/windows/run.py .port/nvcc1451-cuda-smoke.exe
$env:PYTHONPATH = 'python;tokenspeed-kernel/python;tokenspeed-engine/python'
& $py tools/windows/run.py $py tools/windows/check_rmsnorm.py
& $py tools/windows/run.py $py tools/windows/check_sampling_chain.py
```

The smoke result reports `msc_ver: 1951`, `msc_full_ver: 195136260` and
`cuda_runtime_version: 13040`. The direct-built native libraries pass 24
RMSNorm cases and 72 chain-verification cases, including PDL on/off. Focused
evidence is in `docs/validation/nvcc-msvc1451-20261027/`. Older failed host
probes in the previous validation directories are historical, before the
portable layout repair.

The layout regression deliberately reproduces the failure using only the
workspace copy, verifies automatic repair, and runs the same GPU smoke:

```powershell
& $py tools/windows/run.py $py tools/windows/check_cuda_toolchain.py
```

Run it without other simultaneous nvcc builds using this portable toolchain,
because it temporarily moves the workspace's copied environment script.

## Real inference

Model: `Qwen/Qwen3-0.6B`, revision
`c1899de289a04d12100db370d81485cdf75e47ca`, under `.port/models/Qwen3-0.6B`.
Checkpoint SHA256:
`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
The validation script selects BF16, Triton attention, greedy sampling, eager
execution, context 512, two sequences and 512 total KV tokens. Host KV offload
and retraction snapshots are explicitly disabled to fit this machine's RAM.

```powershell
$env:PYTHONPATH = ''
$env:HF_HUB_OFFLINE = '1'
$env:FLASHINFER_WORKSPACE_BASE = Join-Path $PWD '.port/flashinfer'
$env:TRITON_CACHE_DIR = Join-Path $PWD '.port/triton'
& $py tools/windows/run.py $py tools/windows/validate_inference.py
```

Expected: `PASS: three repeat requests and two batched requests; identical greedy token IDs`.
Prompt: `Write the number after 1: `. Output: `1, 2, 3, 4, 5, 6`. Every request
reports 8 prompt and 16 completion tokens. Results:
`.port/inference-validation.json`. These short runs establish functionality and
repeatability; they are not a performance benchmark or broad model-quality eval.

Final evidence and artifact hashes: `docs/validation/windows-arm64-20261027/`.
Earlier component evidence remains under `windows-arm64-20261026/`. Covered
tests include scheduler C++/Python, Rust wire/collectives/metrics/grammar/host,
Python host logic, RMSNorm/QK norm, SwiGLU, chain verification, model registry
and autotune persistence.

## Compressed-tensors NVFP4

The validated checkpoint is `mgoin/Qwen3-0.6B-NVFP4`, revision
`545def2dd2492f4abd75435ccde6dd93d1d93e34`, available locally under
`../vllm-bench/models/Qwen3-0.6B-NVFP4`. The new scheme accepts symmetric
4-bit floating-point tensor-group weights and dynamic local activations,
with group size 16 and `nvfp4-pack-quantized` storage. It keeps weights packed
and preserves each fused projection's own global scales. The registry
dispatches the SM120/SM121 quantized linear to native FP4 Tensor Cores.

```powershell
$env:PYTHONPATH = ''
$env:HF_HUB_OFFLINE = '1'
& $py tools/windows/run.py $py tools/windows/validate_inference.py --model ../vllm-bench/models/Qwen3-0.6B-NVFP4 --report .port/nvfp4-inference-validation.json
```

The same 16-token greedy validation is used for BF16 and NVFP4. Repetition
checks compare requests within one checkpoint; they do not require the two
models to generate identical text or establish model quality.

FlashMLA is optional at import time. Its unavailable entry points fail when
called, and DSV4 FlashMLA kernels register only when the dependency is present.
Resolving an unrelated MTP model no longer imports Inkling-specific kernels.
These changes fix the two previous plugin import failures without claiming
that FlashMLA or Inkling/MoE execution has been ported.

## FP8 and optional model backends

The previous 46 FP8/MLA configuration/dispatch failures are fixed. The
full-row-count portable token quantizer is registered for SM100-family
GPUs without TRT-LLM. Vendor quantization keeps its higher priority when
present. CuTe fusion, TRT-LLM and DeepGEMM imports and registrations check
that their optional dependencies are available; importing model classes
does not execute unavailable backends. Forced unavailable operations fail
explicitly.

The Windows kernel profile now also builds `cublas_fp8_tensor` with the
CUDA SDK's cuBLASLt library and MSVC 14.51. It exposes tensor-scaled E4M3
GEMM with FP16/BF16 outputs through the normal registry. Native GPU tests
cover numerical results, small and large batches, padded output row strides,
bias, independent CUDA streams, zero/tiny quantization inputs and the
minimum token scale. The full source and installed-wheel regressions each
pass 160 tests; 26 additional installed GPU/cache tests pass without skips.

## Validation scope

- The current tested target is native Windows ARM64 / SM12.1, single GPU,
  eager BF16 and compressed-tensors NVFP4 inference. Each checkpoint passes
  three sequential and two batched requests.
- Synthetic SM100/103 tests validate registry, quantization and loading
  contracts. They are not measurements on physical SM100/103 GPUs.
- No HTTP/gRPC/SMG serving, multi-GPU/NCCL, full DeepSeek/MLA/MoE model,
  CUDA graph, disaggregation or offload validation is claimed. Optional
  dependency handling does not implement those vendor backends.
- CUDA RoPE's full native group is outside this dense profile; portable
  Triton performs attention prologue work.
