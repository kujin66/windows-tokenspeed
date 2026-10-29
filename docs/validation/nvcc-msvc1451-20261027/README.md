# Direct MSVC 14.51 / CUDA 13.4.0 validation

The host-detection issue is resolved on native Windows ARM64. The portable
MSVC compiler archives omitted `VC/Auxiliary/Build/vcvarsall.bat`. nvcc uses
that Visual Studio layout file when recognizing the host compiler, even with
`--use-local-env`. The runner now restores the official installed VS script
into the portable layout and supplies the prepared 14.51 environment.

The previous conclusion that nvcc could not handle this compiler version was
incorrect. Both CUDA preprocessing and host compilation now use MSVC
19.51.36260. No 14.44 frontend helper, version macro changes or modified
NVIDIA executables are used in these builds. CUDA SDK `version.json` reports
13.4.0; the included nvcc component reports 13.4.46.

| Check | Result | Evidence |
| --- | --- | --- |
| Remove workspace layout file | Expected `Host compiler targets unsupported OS` reproduced | `nvcc1451-layout-negative.log` |
| Run automatic repair and direct nvcc build | Passed, exit 0 | `nvcc1451-layout-regression.json` |
| Actual compiler and target | `_MSC_VER=1951`, `_MSC_FULL_VER=195136260`, ARM64 static assertions | `nvcc1451-cuda-smoke-build.log`, `nvcc1451-cuda-smoke-result.json` |
| GPU vector addition | All 4097 elements correct | `nvcc1451-cuda-smoke-result.json` |
| VS Code PowerShell environment activation | Direct nvcc build and all 4097 GPU results passed | `nvcc1451-activated-smoke-build.log`, `nvcc1451-activated-smoke-result.json` |
| nvcc.cmd from an ordinary terminal | Direct nvcc build and all 4097 GPU results passed | `nvcc1451-cmd-smoke-build.log`, `nvcc1451-cmd-smoke-result.json` |
| Native RMSNorm | 24 cases passed, FP16/BF16, PDL on/off | `rmsnorm-results.json` |
| Native chain verifier | 72 cases passed against sequential oracle, PDL on/off | `sampling-chain-results.json` |
| Rebuilt ARM64 kernel wheel | Installed DLL bytes match wheel; frontend helper is empty | `direct-wheel-provenance.json` |
| Installed-wheel BF16 inference | Three sequential plus two batched requests; identical greedy token IDs | `inference-validation.json`, `direct-wheel-inference-host.log` |

The verbose nvcc log contains `--microsoft_version=1951`,
`--msvc_target_version=1951`, `--target_woa` and `_WIN_PLATFORM_=arm64`.
`results.json` records current wheel SHA256 hashes and embedded build
identities. The kernel wheel contains two ARM64 PE CUDA libraries; the
upstream `.so` filenames are retained.

Reproduction commands and environment prerequisites are in
[the Windows build guide](../../guides/windows-arm64.md). The layout
regression is `tools/windows/check_cuda_toolchain.py`, invoked through
`tools/windows/run.py`. It only moves the copied script within `.port`,
restores it on exit, and must run without simultaneous nvcc builds using
the same portable toolchain. System Visual Studio and CUDA are unchanged.

This verifies the `dense-triton` Windows profile on RTX Spark N1X (SM 12.1),
using Qwen3-0.6B BF16. NVFP4, other kernel groups, HTTP/SMG serving and
multi-GPU operation are outside these results. Historical failed probes and
split-toolchain results in earlier validation directories describe the
environment before the portable layout was repaired.
