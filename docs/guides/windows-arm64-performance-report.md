# TokenSpeed on Windows ARM64: Engine throughput, WSL2 and kernel backends

This report measures **TokenSpeed Engine.generate throughput** for Qwen3-0.6B
BF16 on native Windows ARM64 and WSL2. Within TokenSpeed, the adapted Windows
FlashInfer path improves eager throughput by **30.75–40.73%** over its Windows
Triton control.
With **decode CUDA Graphs enabled**, Windows Triton is faster than Windows
FlashInfer in all four cases. Windows FlashInfer is within 3.00% of WSL2
FlashInfer at batch 1 and measures 3.48–9.04% higher throughput at batch 4.
These are measurements of the tested software stacks, with unresolved output
consistency differences, rather than a claim of operating-system superiority.

This report organizes evidence collected on **2026-10-28**. Publishing it does
not rerun or increase the historical test counts.

## What is measured: TokenSpeed Engine throughput

**Every row in the final eager/decode-graph tables is a TokenSpeed model-generation
measurement.** FlashInfer and Triton identify kernel choices inside TokenSpeed,
not separate inference engines being timed in place of TokenSpeed.

The [actual benchmark source](../validation/windows-flashinfer-0617-20261028/scripts/flashinfer_comparison_bench.py)
imports `Engine` from `tokenspeed.runtime.entrypoints.engine`, constructs it with
the Qwen3-0.6B checkpoint, and calls `engine.generate(input_ids=inputs,
sampling_params=sampling)` for every warmup and measured request. Its timing loop
is around the local `generate()` helper, which performs that Engine call and
the immediate output-length/completion-metadata checks:

```python
from tokenspeed.runtime.entrypoints.engine import Engine

engine = Engine(model=str(model), **engine_config)
# generate() calls engine.generate(...) and validates returned token IDs.
start = time.perf_counter()
ids = generate()
elapsed = time.perf_counter() - start
```

The [tested Engine implementation](../../ports/windows-arm64/overlay/python/tokenspeed/runtime/entrypoints/engine.py)
launches TokenSpeed's runtime processes and routes requests through its frontend,
IPC, Rust host/C++ scheduler and model execution. The measured call includes
prefill, autoregressive decode, sampling and output return through this Engine
path. It excludes model construction/loading and explicit warmup. Inputs are
already token IDs, so this is not a text-tokenization or HTTP benchmark.

| Benchmark configuration | Inference engine | Selected backend within that engine |
| --- | --- | --- |
| TokenSpeed / Windows / Triton | Validated native Windows ARM64 TokenSpeed port | Triton attention, RMSNorm and SwiGLU control |
| TokenSpeed / Windows / FlashInfer | Same TokenSpeed port, with the documented isolated compatibility overlay | FlashInfer FA2 prefill, TRT-LLM decode, RMSNorm and SwiGLU |
| TokenSpeed / WSL2 / FlashInfer | Official pinned TokenSpeed Linux ARM64 build, with the same FA2 adapter | Linux FlashInfer attention/norm/activation path |

GEMM, attention prologue and other operations retain the corresponding stack's
existing implementations. The names do not mean every operator was replaced.
The [raw graph report](../validation/windows-flashinfer-0617-20261028/windows-flashinfer-graph.json)
records TokenSpeed Engine arguments including `enforce_eager=false` and
`attention_backend=flashinfer`; the matching [startup log](../validation/windows-flashinfer-0617-20261028/windows-flashinfer-graph.log)
records TokenSpeed `ServerArgs`. Eager and other-platform runs retain their own
configuration, timings and returned token IDs in the same evidence directory.

These measurements establish **TokenSpeed's end-to-end Engine API throughput**
for the stated workload, and the changes from backend/graph/platform selection.
For example, +30.75–40.73% means selecting the adapted FlashInfer path improves
TokenSpeed's Windows eager throughput relative to its Triton control. It is not
a measured TokenSpeed speedup over another inference engine.

**No non-TokenSpeed baseline was measured.** The report therefore cannot quantify
TokenSpeed's incremental benefit over Transformers/PyTorch generation, vLLM or
TensorRT-LLM, or isolate its scheduler's contribution. That would require a
matched end-to-end comparison against another engine or a controlled ablation.
The separate operator/synchronization microbenchmarks are diagnostic experiments;
their timings must not be presented as model-generation throughput.

## Source revisions and reproduction boundary

| Component | Tested source |
| --- | --- |
| Common upstream | [lightseekorg/tokenspeed at 864bcdd9ef3c5f1512bbe293d970d73eb29d2c94](https://github.com/lightseekorg/tokenspeed/tree/864bcdd9ef3c5f1512bbe293d970d73eb29d2c94) |
| Windows port | [jinkun_microsoft/windows-tokenspeed at 9a50ac82b59fc47044090e4441c05c88173715c1](https://github.com/jinkun_microsoft/windows-tokenspeed/tree/9a50ac82b59fc47044090e4441c05c88173715c1) |
| WSL2 | Official common upstream revision, followed by the documented experimental FlashInfer kernel overlay |

The PR now includes the complete 66-file Windows source overlay, per-file
upstream/port blob identities, build requirements, helpers, source tests and
historical build evidence. Follow the [detailed build guide](windows-arm64-build-details.md)
to reconstruct a complete checkout from official upstream using only this PR's
overlay and support files. An external Windows-port clone is not required.
The code targets the pinned Python `Engine` API. The newer `main` has a different
launcher API; this is a reproducible versioned port rather than a tested
integration with that newer runtime. See the [complete source inventory](../../ports/windows-arm64/FILES.md).

## Hardware and software

| Item | Native Windows ARM64 | WSL2 Linux ARM64 |
| --- | --- | --- |
| GPU | Same RTX Spark N1X, SM12.1, GPU 0, driver 615.89 | Same physical GPU and Windows host driver |
| Python | Native CPython 3.13.16 | Native CPython 3.13.16 |
| Host compiler | MSVC toolset 14.51; compiler 19.51.36260 | GCC 15.2; Rust 1.92.0 |
| CUDA compiler | SDK 13.4.0; nvcc 13.4.46 | Separate SDK 13.4.0rc1; nvcc 13.4.46 |
| PyTorch | 2.14.0a0+git20b0f57, CUDA 13.4 | 2.14.0+cu130, actual runtime CUDA 13.0 |
| Triton | triton-windows 3.8.0.post29 | tokenspeed-triton 3.9.10.post20261008 |
| FlashInfer, final matrix | 0.6.17+g2ea3c24b5dac | 0.7.2 |
| Other environment | Windows SDK 10.0.26100.0; TVM FFI 0.1.13.post3; Rust 1.92.0 | Ubuntu-26.04, WSL 3.0.3.0, kernel 6.18.54.1 |

The compiler SDK and actual PyTorch CUDA runtime are recorded separately.
Windows project binaries were audited as ARM64, without x64 emulation, and
identify linker 14.51.

## Windows ARM64 build process

The [Windows reproduction guide](windows-arm64-reproduction.md),
[build skill](../../skills/tokenspeed-windows-arm64-build/SKILL.md) and
[dependency references](../../skills/tokenspeed-windows-arm64-build/references/dependencies.md)
record the detailed preparation and archive provenance.

1. Prepare native ARM64 Python, Visual Studio/Windows SDK, CUDA 13.4 and driver.
   Preserve the official MSVC `VC/Auxiliary/Build/vcvarsall.bat` layout required
   by nvcc. Directory `14.51.36231` contains compiler `19.51.36260`; verify the
   compiler output and macros rather than inferring the version from its path.
2. Create the project venv. Prepare ARM64 PyTorch/TVM FFI wheels, Rust 1.92.0,
   the offline Cargo cache and xgrammar 0.2.8 source. Local preparation copies
   an already-prepared environment; it is not a clean-machine installer.
3. Use Ninja, a distinct scheduler build directory, `TOKENSPEED_CUDA_ARCH=121`
   and the `dense-triton` Windows profile. `tools/windows/run.py` prepares
   compiler/SDK/library/Python DLL paths and preserves the real MSVC frontend.
4. Build scheduler, engine, kernel and runtime wheels with `--no-build-isolation`
   and `--no-deps`; install exactly four candidates. Compare installed contents
   with wheel bytes and audit PE machine, linker and CUDA build identity.
5. Run the numerical, runtime and real-model checks recorded with the evidence.

Executed entry points in the prepared local workspace:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools/windows/prepare_local.ps1 -SourceRepo ../../windows-tokenspeed -RuntimeSitePackages ../../vllm-bench/.venv/Lib/site-packages -NativeWheelhouse ../../N1X_ARM_Package_Support/wheelhouse
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools/windows/build_wheels.ps1 -BuildId migration-20261028
$py = './.venv/Scripts/python.exe'
& $py tools/windows/run.py $py tools/windows/check_cuda_toolchain.py
& $py tools/windows/run.py $py tools/windows/check_native_artifacts.py
```

The named sibling directories are inputs from the measured workspace; supply
equivalent verified dependencies elsewhere. `.port`, `.venv` and `dist` retain
local tools, models, wheels and caches and are not published. Constraint files
[windows-arm64-tested.txt](../../requirements/windows-arm64-tested.txt) and
[windows-arm64-flashinfer0617-tested.txt](../../requirements/windows-arm64-flashinfer0617-tested.txt)
are observed versions, not complete lockfiles. The venv shares some dependencies
through a `.pth` file. Clean-machine and public-PyPI-only installs remain untested.

## WSL2 build process

The [WSL2 reproduction guide](wsl2-arm64-build.md) provides the installation,
environment variables, SDK fixes and four wheel build commands. WSL2 and
VirtualMachinePlatform were installed, followed by a host reboot and Ubuntu
ARM64 registration. Source, venv and models reside in the Linux filesystem
under `/opt/tokenspeed-work`. The official source was pinned before building:

```bash
git clone https://github.com/lightseekorg/tokenspeed.git /opt/tokenspeed-work/source
git -C /opt/tokenspeed-work/source checkout --detach 864bcdd9ef3c5f1512bbe293d970d73eb29d2c94
```

Install build tools and declared CUDA/runtime dependencies in the isolated
ARM64 venv; supply Rust 1.92.0 and xgrammar source. Configure the separate CUDA
SDK as documented, build all four project wheels with `--no-build-isolation
--no-deps`, and install the Linux wheels. The final audit byte-matched the
installed contents and identified 27 project ELF files as AArch64.

CUDA 13.0 nvcc conflicted with Ubuntu 26.04 glibc declarations. The build used
the separate official 13.4 SDK, including cuRAND headers, a `lib64` link and
unversioned CUDA library links needed by FlashInfer JIT. Failure and repair
logs are retained. The vendor cuSPARSELt wheel's `sbsa` tag still causes a
`pip check` platform warning despite verified AArch64 ELF. The local official
kernel wheel is tagged `py3-none-any` but contains native libraries; only this
Linux ARM64 installation was validated.

## FlashInfer adaptations and GPU validation

The supplied Windows JIT source wheel was
`flashinfer_python-0.6.17+g2ea3c24b5dac-py3-none-any.whl`, SHA256
`47db3e3972fda64719ceae8490d1934362380cef62f33975cb593ad086e0bcb6`.
It has no precompiled ARM64 GPU DLLs. Five cached JIT DLLs were audited as
ARM64/linker 14.51. Cached DLLs can belong to failed paths, so architecture
identity alone does not establish functional support.

The [FlashInfer guide](windows-flashinfer-comparison.md) and
[experiment record](../validation/windows-flashinfer-0617-20261028/README.md)
include source snapshots, failure logs, patches and exact commands:

- Isolated Windows kernel copies select FlashInfer or Triton for attention,
  RMSNorm and SwiGLU. The original port forces Triton norm/activation on
  Windows; installing the wheel alone would not select FlashInfer.
- Both Windows controls disable unavailable FlashInfer MoE registrations because
  0.6.17 lacks newer imported per-channel FP8 APIs. This is a dense-model
  experiment, without fabricated replacements for missing interfaces.
- TRT-LLM prefill rejects SM12.1. A narrow registry provider uses FA2 paged
  prefill on SM12.1/BF16/head dimension 128/causal full context; FlashInfer
  TRT-LLM decode remains enabled. Windows and WSL2 use the same adapter source.
  Unsupported quantization, LSE, sinks and sliding-window options are rejected.
- FA2 planning copies metadata to the CPU and constructs a wrapper on every
  prefill call. The benchmark includes this unoptimized adapter overhead.
- Windows JIT object paths exceeded the path limit. One JIT file was patched
  to use stable hashed object filenames. CUDA computation, flags and architecture
  checks were preserved; the original wheel was retained.

Numerical validation covered RMSNorm, residual RMSNorm and SwiGLU at
1/4/128/512 rows against FP32 references, plus CUDA Graph replay. Attention
decode was checked against FP32 SDPA with 16 Q heads, 8 KV heads, head dimension
128 and KV length 128. On both platforms, FA2 prefill was checked against
bottom-right causal FP32 SDPA for 128/512-token inputs, mixed prefix lengths,
partial tail pages and shuffled page tables. The wheel audit compared 4,985
files and found only the documented JIT filename change.

## Final benchmark methodology

All final tables use the same six-run matrix, without mixing earlier diagnostic
runs. Qwen3-0.6B BF16; same GPU; batch 1/4; input 128/512; fixed output 128;
three warmups and five measured rounds per case. Each configuration produced
50 measured requests and 6,400 measured output tokens. Context=704, KV token
capacity=2816, seed=20261028. Prefix cache, KV offload and prefill graphs were
disabled. Inputs were deterministic token IDs.

Samples time blocking `Engine.generate`, including prefill, decode, IPC and
immediate output-length/metadata validation. Loading and explicit warmup are
excluded. Throughput is total measured output tokens / total measured time.
Profiler collection was disabled; platforms ran sequentially without competing
benchmarks. All throughput columns are **output tokens/s**. Percentages mean
`(Windows FlashInfer / comparison throughput - 1) * 100`; positive is faster.

### Eager mode

`enforce_eager=True`; decode and prefill graphs are disabled.

| Input | Batch | TokenSpeed / Windows / Triton | TokenSpeed / Windows / FlashInfer | TokenSpeed / WSL2 / FlashInfer | Win FI vs Triton | Win FI vs WSL2 FI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 49.62 | 64.88 | 73.49 | +30.75% | -11.72% |
| 128 | 4 | 177.59 | 249.92 | 271.39 | +40.73% | -7.91% |
| 512 | 1 | 43.64 | 57.92 | 59.40 | +32.72% | -2.50% |
| 512 | 4 | 184.65 | 256.88 | 272.37 | +39.12% | -5.68% |

### Non-eager mode: decode CUDA Graphs

`enforce_eager=False`, `cudagraph_capture_sizes=[1, 4]`,
`max_cudagraph_capture_size=4`, `disable_prefill_graph=True`.
These are complete generation timings with graph-enabled decode, not isolated
replay or fully graphed prefill-and-decode measurements.

| Input | Batch | TokenSpeed / Windows / Triton | TokenSpeed / Windows / FlashInfer | TokenSpeed / WSL2 / FlashInfer | Win FI vs Triton | Win FI vs WSL2 FI | Win Triton vs WSL2 FI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 145.33 | 136.77 | 140.99 | -5.89% | -3.00% | +3.08% |
| 128 | 4 | 600.93 | 593.53 | 544.32 | -1.23% | +9.04% | +10.40% |
| 512 | 1 | 130.20 | 127.89 | 128.32 | -1.77% | -0.34% | +1.47% |
| 512 | 4 | 516.95 | 494.58 | 477.95 | -4.33% | +3.48% | +8.16% |

Windows FlashInfer is 1.23–5.89% below Windows Triton in graph mode. Relative to
WSL2 FlashInfer, batch 1 is 0.34–3.00% lower and batch 4 is 3.48–9.04% higher.
Windows Triton measures 1.47–10.40% above WSL2 FlashInfer across these cases;
this comparison changes both platform and backend.
Input-128/batch-4 Windows results have repeat-output variation in both backends;
read these results with the correctness limitations below.

### Decode graph speedup relative to eager

Each ratio is graph throughput / eager throughput for that same backend in the
final matrix. Higher ratios indicate greater measured benefit.

| Input | Batch | TokenSpeed / Windows / Triton | TokenSpeed / Windows / FlashInfer | TokenSpeed / WSL2 / FlashInfer |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 2.93x | 2.11x | 1.92x |
| 128 | 4 | 3.38x | 2.37x | 2.01x |
| 512 | 1 | 2.98x | 2.21x | 2.16x |
| 512 | 4 | 2.80x | 1.93x | 1.75x |

After preparing isolated Windows overlays and restoring the archived scripts
to their documented `.port` locations, select graph execution explicitly:

```powershell
$py = './.venv/Scripts/python.exe'
& $py tools/windows/run.py $py .port/launch_flashinfer0617.py --ops flashinfer .port/flashinfer_comparison_bench.py --model .port/models/Qwen3-0.6B --report .port/flashinfer-comparison/windows-flashinfer-graph-new.json --execution graph --profile none --attention-backend flashinfer --input-lengths 128 512
& $py tools/windows/run.py $py .port/launch_flashinfer0617.py --ops triton .port/flashinfer_comparison_bench.py --model .port/models/Qwen3-0.6B --report .port/flashinfer-comparison/windows-triton-graph-new.json --execution graph --profile none --attention-backend triton --input-lengths 128 512
```

In prepared WSL2, with the same FA2 overlay already installed:

```bash
source "$repo_mount/.port/wsl-env.sh"
export PYTHONPATH=/opt/tokenspeed-work/venv/flashinfer-comparison
python "$repo_mount/.port/flashinfer_comparison_bench.py" \
  --model /opt/tokenspeed-work/models/Qwen3-0.6B \
  --report "$repo_mount/.port/flashinfer-comparison/wsl2-flashinfer-graph-new.json" \
  --execution graph --profile none --attention-backend flashinfer --input-lengths 128 512
```

Set `repo_mount` to the Windows checkout's WSL mount path. Use new report names
for reruns. Preparation scripts are first-run recipes; inspect existing overlays
before considering rerunning them. Preserve historical evidence.

## Earlier paired baseline and investigation

The separate [post-reboot paired report](../validation/windows-wsl2-perf-20261028-after-reboot/README.md)
used Triton attention, eager execution and ten measured rounds per case.
Norm/activation defaults differed between platforms. Per platform: 100 measured
requests and 12,800 output tokens.

| Input | Batch | TokenSpeed / Windows tokens/s | TokenSpeed / WSL2 tokens/s | WSL2 relative to Windows |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 53.19 | 65.01 | +22.22% |
| 128 | 4 | 215.64 | 244.94 | +13.59% |
| 512 | 1 | 55.00 | 64.06 | +16.49% |
| 512 | 4 | 215.88 | 244.66 | +13.33% |

The [operator, synchronization and graph investigation](../validation/windows-wsl2-diagnosis-20261028/README.md)
found similar large BF16 GEMM times and no consistent Windows disadvantage in
the tested copy, synchronization or simplified thread-handoff operations.
Some small eager norm/activation calls were slower on Windows, while graph
replay sharply reduced overhead. Together with real-model graph measurements,
this prioritizes launch/wrapper overhead and backend selection for further
investigation; it does not isolate a single library, driver or OS cause.
WSL2 profiler GPU kernel activity was missing and profiling overhead differed,
preventing quantitative attribution of the unprofiled performance gap.

## Correctness, validation counts and limits

Output consistency in the final six-run matrix is recorded separately from
fixed-length throughput completion:

| Mode | Input | Batch | Windows FI repeats | Windows Triton repeats | WSL2 FI repeats | Windows FI/Triton reference IDs | Windows/WSL2 FI reference IDs |
| --- | ---: | ---: | --- | --- | --- | --- | --- |
| eager | 128 | 1 | consistent | consistent | consistent | different | different |
| eager | 128 | 4 | consistent | consistent | consistent | different | different |
| eager | 512 | 1 | consistent | consistent | consistent | match | match |
| eager | 512 | 4 | consistent | consistent | consistent | match | match |
| graph | 128 | 1 | consistent | consistent | consistent | different | different |
| graph | 128 | 4 | variation | variation | consistent | different | different |
| graph | 512 | 1 | consistent | consistent | consistent | match | match |
| graph | 512 | 4 | consistent | consistent | consistent | match | match |

Repeat checks cover three warmups plus five measurements. Reference comparisons
use the first measured output, not every possible cross-platform output pair.

- Every final request produced 128 output IDs and matching completion metadata.
  Raw JSON preserves IDs for every warmup and measured request.
- Input 128: reference IDs differed across FlashInfer/Triton and platforms,
  including batch 1. In graph mode at batch 4, both Windows backends also had
  repeat-output variation. The cause remains unresolved.
- Input 512: all final cases were repeat-consistent and the reference IDs
  matched across tested backends/platforms. This is workload-specific evidence.
- The earlier [Windows build validation](../validation/windows-wsl2-perf-20261028/README.md)
  recorded **160 runtime + 26 GPU + 16 performance-tool pytest passes = 202**.
  These are historical counts, not newly rerun tests for this report. Failure
  logs preceding successful reruns remain alongside the evidence.
- Five rounds, unmatched library versions, no clock locking and no repeated
  ABBA alternation limit causal interpretation and small percentage differences.
  Fixed-length throughput completion does not prove numerical equivalence,
  determinism or service stability.
- Scope: single-GPU dense BF16. HTTP, TTFT/ITL, prefill graphs, FlashInfer
  MoE/NVFP4, multi-GPU and long-duration stress were not measured.

| Boundary | What was established | What remains unresolved |
| --- | --- | --- |
| Source and installation | Native ARM64 source port, wheel/install byte comparisons, compiler/linker audits | Latest-main integration; clean-machine dependency resolution; reproducible binary hashes |
| GPU numerical checks | Specified normalization/activation/attention probes passed; historical FP8/FP4 checks retained | End-to-end numerical equivalence for input 128; source of graph batch-4 repeat variation |
| Backend performance | Final six configurations measured eager and decode-graph throughput | Equal-version backend comparisons; isolated per-component/OS attribution |
| Graph scope | Decode capture sizes 1/4; prefill still eager | Prefill graphs, arbitrary capture shapes, full server stability |
| Statistical scope | Five measured rounds per case; raw durations and IDs retained | Clock-locked, repeated ABBA trials and confidence intervals |
| Product scope | Dense BF16, one GPU, blocking Engine API | HTTP/TTFT/ITL, multi-GPU, grammar, FlashInfer MoE/NVFP4 and prolonged stress |

Both WSL2 and native Windows use the Windows host GPU driver. The diagnosis
does not support explaining the difference as native Windows WDDM versus
bare-Linux TCC, nor as missing NCCL for a world-size-one workload.

## Evidence index

| Record | Contents |
| --- | --- |
| [Windows build](../validation/windows-wsl2-perf-20261028/README.md) | Build/install logs, native audits, JUnit, dependencies, pre-reboot historical baseline |
| [WSL2 paired baseline](../validation/windows-wsl2-perf-20261028-after-reboot/README.md) | Linux builds/fixes, native audit, paired ten-round raw data |
| [Investigation](../validation/windows-wsl2-diagnosis-20261028/README.md) | Microbenchmarks, synchronization tests, profiler summaries, earlier graph experiments |
| [Final matrix](../validation/windows-flashinfer-0617-20261028/README.md) | Six raw runs, eight comparison rows, output IDs, probes, patches, source snapshots |

Each record has an `evidence-sha256.json` manifest with original/public hashes.
Public copies redact local user paths, hostnames and internal domains. The
FlashInfer manifest covers 59 files. Archived logs and source snapshots are
immutable: formatter hooks exclude historical evidence, while a verifier checks
hashes and recomputes throughput. Historical source links within raw evidence
refer to the tested source revisions.

## Publication verification

Run these publication checks from the PR checkout. Build and installed-wheel
validation commands in the detailed guide run from the reconstructed checkout.

The publication checks separately passed 28 GPU-free tests for performance
aggregation, report compatibility, hash integrity, derived percentages and
execution-mode labels. They also verified all 152 manifest entries and all
eight final comparison rows. Source reconstruction also verifies all 66 source
postimages and 136 additional historical build evidence files. Real local-Git
tests cover reconstruction, refusal to overwrite, corrupted input rejection
and unexpected source differences. The historical GPU/model runs above were not
rerun for this documentation change.

```bash
python -m pytest test/tools --noconftest -q -o pythonpath=.
python tools/perf/verify_evidence.py
pre-commit run --all-files
```
