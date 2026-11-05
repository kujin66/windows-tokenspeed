## Summary

Build all four TokenSpeed project wheels on native Windows ARM64 with all
24 native CUDA groups. Use public NVIDIA-feed Torch, PyPI TVM FFI and Triton
Windows wheels; restore OpenSSL hashing and integrate the Windows build into
the existing release workflows.

The proposed update also includes the experimental dense-Qwen runtime
integration used by these tests. It changes optional imports, Windows process
and transport handling, and SM12.1 attention dispatch. This broader source
scope is explicit; it does not establish a clean default runtime installation
or general SM12.1/model support.

## OpenSSL review fixes

The scheduler again uses its original OpenSSL SHA256 prefix hasher on both
platforms. The Windows bcrypt implementation was removed. Unmodified public
OpenSSL 3.5.4 was separately built for Windows ARM64 with MSVC 14.44 and 14.51.
Installed SHA256 and scheduler checks passed; OpenSSL's own test suite was not
run. The source update also uses public NVIDIA-feed Torch and PyPI TVM FFI,
removes native-group filters, and integrates Windows jobs into the existing
workflows. GitHub-hosted runner execution remains unvalidated.

**Complete original PR2 scope: 16 configurations, 64 cases, 320 timed calls.**

Main comparison: 32 cases/160 timed calls. CuTe visibility and AOT-policy supplement: 32 cases/160 timed calls.

The main TokenSpeed comparison completed 32 cases and 160 timed calls. Windows
packages and the original four project wheels were preserved. The independent
Linux environment rebuilt its dependencies and four project wheels (one
universal Python wheel and three native ARM64 wheels).
Strict HF full-sequence equivalence: **FAIL**; details below.

The tested TokenSpeed source is PR2 head
b0d36dcb9a6cf8186ba05f504814b63dc8f3d249 plus the frozen local review fixes and
experimental dense-Qwen integration (3,103 manifest files). The broader runtime
changes and strict-reference limitations do not establish a clean default runtime
installation, general model quality or general SM12.1 support.
Linux additionally applies a documented initialization-only override: the
existing Windows single-rank/no-NCCL Gloo fallback is made platform-neutral,
and USE_LIBUV=0 selects the same effective legacy TCPStore initialization.
Windows installed files remain unchanged. The model, scheduling and GPU kernel
implementations are shared; do not read this as byte-identical initializer files.

## Throughput

Aggregate output tokens per second, total generated tokens / total measured time.
Each value combines two independent processes, five timed calls per case/process.

| Input tokens | Concurrent requests | Windows / Triton ops | WSL2 / Triton ops | Windows / FlashInfer ops | WSL2 / FlashInfer ops |
|---|---:|---:|---:|---:|---:|
| 128 | 1 | 137.81 | 136.41 | 136.94 | 133.03 |
| 128 | 4 | 574.51 | 571.45 | 568.20 | 531.93 |
| 512 | 1 | 130.05 | 128.09 | 133.71 | 126.63 |
| 512 | 4 | 506.28 | 502.20 | 499.03 | 488.83 |

Windows / WSL2 ratios across the four cases: Triton ops
1.005–1.015x; FlashInfer ops
1.021–1.068x. These compare the recorded platform
builds and workload; they do not establish a general OS ranking or attribute a
difference to one library.

## Environment alignment

Both sides use Python 3.13.16, Torch 2.16.0.dev20261101+cu134, TVM FFI
0.1.15.post2, FlashInfer 0.6.17+g2ea3c24b5dac, triton-windows 3.9.0.post30,
Transformers 5.12.0, ArmPL 26.1.1 and cuDNN 9.26.0.51. The measured CUDA library
API versions and actual precision/import checks are retained in the evidence.
The native/JIT compiler selection is CUDA 13.4.59.
Compiled FlashAttention availability and the recorded default SDP backend
selection flags also match the unchanged Windows Torch snapshot.
Whitelisted TF32, cuBLAS workspace, cuDNN, allocator, launch-blocking and Triton
switches match, including unset values; OMP/MKL threads are one on both sides.

Windows Torch reports source revision 1f4d2d5e820dd870626dca4f9603e3a49879dbde;
Linux was built from that revision. FlashInfer's common base is
2ea3c24b5dac97e1bbd3834b698010bc48c52848. Triton's common tag revision is
26a4654c314a781f18c33dc32e519f89f9e12798, with its pinned LLVM revision.

That FlashInfer commit retains a stale Linux import of an NVSHMEM generator
already removed from its comm module. The Linux comparison removes only that
unused import; the repacked wheel changes only jit/__init__.py and RECORD.
No replacement implementation, numerical kernel or native payload is added.
Windows already skips that import in its platform branch.
The shared MSVC preprocessor host option is additionally guarded by platform
for GCC. Only compilation_context.py and RECORD change in that repack; all
numerical sources and optimization options remain unchanged.
XQA also needs a Linux constant-storage adjustment: two host constexpr objects
used by device reference-taking helpers become typed literal expressions.
The original float32 -100000 and int32 -1 values/bit patterns, formulas and
Windows branch are preserved. This separate header patch is disclosed in the
evidence; it is not an unmodified-upstream package claim.

The original Linux build recipe pins older Torch/FFI/FlashInfer versions and
executes nested pip even if the outer wheel build specifies --no-deps. The
controlled Linux build uses two recorded dependency-recipe overrides matching
Windows, plus pinned/no-deps/offline installation. It is a comparison environment,
not an upgrade to the repository's default Linux installation recipe.

Linux Triton needs recorded build-glue adaptations for Windows import-library
packaging and GCC diagnostics. Compiler algorithms and TokenSpeed model,
scheduling and GPU-kernel implementation source remain unchanged. A disclosed namespace adapter reuses
the same compiled triton modules as tokenspeed_triton. Platform ABI, native
toolchains, driver paths and Windows port adaptations remain different.
The rebuilt Linux Triton wheel was separately repacked to match device-tool
inputs: only PTXAS, cuda.h and RECORD changed; the compiled native compiler and
all other members were byte-identical to the original rebuilt wheel. cuda.h
and libdevice match the Windows input hashes; native PTXAS versions match.
The strict-C11 Linux host-helper source additionally defines the POSIX feature
macro before system headers so posix_memalign is declared. This separate repack
changes only driver.c and RECORD; the native GPU compiler and kernel inputs
remain byte-identical.

## Protocol and correctness

Actual TokenSpeed WireDriver.generate and pinned installed Rust host/scheduler;
Qwen3-0.6B BF16; identical model hashes and seeded token-ID inputs; input lengths 128/512;
tokenizer/generation configuration hashes and optional-file presence also match;
concurrency 1/4; 128 output tokens per request; greedy sampling with EOS ignored. There
are three warmups and five timed calls per case. Decode graphs use [1,4]; prefill
graphs, prefix caching, KVStore and PDL are disabled. GPU work is sequential and
does not overlap builds. Loading, initialization, compilation/capture and warmups
are excluded; generation, IPC, detokenization and immediate ID/count checks are
included. Backend names identify the selected TokenSpeed comparison operations.

HF BF16 SDPA greedy references were freshly generated independently on each OS;
their execution times never supply TokenSpeed throughput. Exact sequence counts
compare all 128 generated token IDs for every saved warmup/timed sequence. They
are not answer-accuracy scores, and divergence causes are not established here.

| Platform | Configuration | Exact HF sequences | Saved token-position agreement | Repeatable outputs |
|---|---|---:|---:|---|
| windows | main-triton-r1 | 40/80 | 77.19% | False |
| windows | main-flashinfer-r1 | 48/80 | 79.45% | False |
| windows | main-triton-r2 | 40/80 | 77.19% | False |
| windows | main-flashinfer-r2 | 48/80 | 79.61% | False |
| wsl | main-triton-r1 | 40/80 | 77.19% | True |
| wsl | main-flashinfer-r1 | 48/80 | 80.08% | True |
| wsl | main-triton-r2 | 40/80 | 77.19% | True |
| wsl | main-flashinfer-r2 | 48/80 | 80.08% | True |

Windows/WSL2 output-ID comparison for the same backend, repetition, input case
and saved-call index (warmups included):

| Configuration | Input | Concurrency | First warmup batch exact | Paired exact sequences | Paired token-position agreement |
|---|---:|---:|---|---:|---:|
| main-triton-r1 | 128 | 1 | True | 8/8 | 100.00% |
| main-triton-r1 | 128 | 4 | False | 25/32 | 94.87% |
| main-triton-r1 | 512 | 1 | True | 8/8 | 100.00% |
| main-triton-r1 | 512 | 4 | True | 32/32 | 100.00% |
| main-flashinfer-r1 | 128 | 1 | True | 8/8 | 100.00% |
| main-flashinfer-r1 | 128 | 4 | False | 28/32 | 89.65% |
| main-flashinfer-r1 | 512 | 1 | True | 8/8 | 100.00% |
| main-flashinfer-r1 | 512 | 4 | True | 32/32 | 100.00% |
| main-triton-r2 | 128 | 1 | True | 8/8 | 100.00% |
| main-triton-r2 | 128 | 4 | False | 29/32 | 97.80% |
| main-triton-r2 | 512 | 1 | True | 8/8 | 100.00% |
| main-triton-r2 | 512 | 4 | True | 32/32 | 100.00% |
| main-flashinfer-r2 | 128 | 1 | True | 8/8 | 100.00% |
| main-flashinfer-r2 | 128 | 4 | False | 29/32 | 92.24% |
| main-flashinfer-r2 | 512 | 1 | True | 8/8 | 100.00% |
| main-flashinfer-r2 | 512 | 4 | True | 32/32 | 100.00% |

These are paired output diagnostics, not answer-accuracy scores or evidence
that any difference is caused by the operating system.

Python CuTe DSL is unavailable on both main arms. The 24 native CUDA C++ groups
are not evidence of Python CuTe DSL compilation or a DSL speedup.

## Regression checks

| Check | Passed |
|---|---:|
| windows/installed_native | 8 |
| windows/numerical | 23 |
| windows/triton | 6 |
| windows/flashinfer | 14 |
| windows/scheduler | 662 |
| wsl/numerical | 23 |
| wsl/triton | 6 |
| wsl/flashinfer | 14 |
| wsl/scheduler | 662 |

Rust tests passed: Windows host 3, metrics 3; Linux host 3, metrics 2. The
additional positioned-write metrics test is Windows-only. The Rust scheduler
crate has no unit tests.
Wheel audits verify four project wheels, 24 native CUDA groups and 1,243 installed
members per platform (excluding pip-rewritten RECORD). Numerical coverage is
selected; it is not numerical validation of every compiled group.


## Original PR2 ratios

Ratios use unrounded aggregate throughput. Values above 1 favor the numerator.

| Input | Concurrency | Windows/WSL2 Triton | Windows/WSL2 FlashInfer | Windows FlashInfer/Triton | WSL2 FlashInfer/Triton |
|---|---:|---:|---:|---:|---:|
| 128 | 1 | 1.010x | 1.029x | 0.994x | 0.975x |
| 128 | 4 | 1.005x | 1.068x | 0.989x | 0.931x |
| 512 | 1 | 1.015x | 1.056x | 1.028x | 0.989x |
| 512 | 4 | 1.008x | 1.021x | 0.986x | 0.973x |

## Original PR2 CuTe DSL supplement, retested

The model, source, common compute packages, input IDs and measurement boundaries
match the main comparison. Each arm uses two independent processes, three
warmups and five timed calls per case/process. Arm order is off/on, then on/off;
GPU workloads run sequentially after source builds finish.

Linux reuses only frozen CuTe DSL 4.8.0 compiler files and its protobuf 6.33.6
support dependency in a private overlay. No old Torch, FFI, FlashInfer or old
site-packages root is inherited. Protobuf support is held visible in both arms;
the recorded startup audit removes only compiler paths in the hidden arm. Actual imports
verify the exact new Torch source/version and the new runtime prefix.
All 311 copied optional compiler files matched their installed distribution
RECORD hashes, including the native payloads.

Windows retains its existing packages and uses the original AOT router-policy
off/on control. Neither Windows arm has an importable Python CuTe compiler;
the dense model has no matching Kimi/MoE router. These are configuration
negative controls, not evidence of native Windows ARM64 DSL compilation.

Aggregate output tokens/s, including prefill and decode:

| Input | Concurrency | WSL2 DSL hidden | WSL2 DSL available | WSL2 change | Windows AOT off | Windows AOT on |
|---|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 133.70 | 133.15 | -0.41% | 136.99 | 137.07 |
| 128 | 4 | 545.08 | 544.23 | -0.15% | 563.15 | 570.22 |
| 512 | 1 | 131.69 | 130.24 | -1.10% | 132.94 | 133.25 |
| 512 | 4 | 484.51 | 475.91 | -1.78% | 486.82 | 485.15 |

Both Linux DSL-available configurations recorded successful cute.compile returns for RMSNormKernel and FusedAddRMSNormKernel, with no recorded compile failures. This verifies only those norm paths, not all TokenSpeed DSL operations.

| Platform | Configuration | Compiler visible | Compile returns | Compile failures | Both norm paths exercised |
|---|---|---|---:|---:|---|
| windows | supplement-fi-off-r1 | False | 0 | 0 | False |
| windows | supplement-fi-on-r1 | False | 0 | 0 | False |
| windows | supplement-fi-on-r2 | False | 0 | 0 | False |
| windows | supplement-fi-off-r2 | False | 0 | 0 | False |
| wsl | supplement-fi-off-r1 | False | 0 | 0 | False |
| wsl | supplement-fi-on-r1 | True | 2 | 0 | True |
| wsl | supplement-fi-on-r2 | True | 2 | 0 | True |
| wsl | supplement-fi-off-r2 | False | 0 | 0 | False |

The additional Linux FlashInfer primitive/graph run with DSL visible passed 14
selected checks without failures, errors or skips. This remains selected
numerical coverage; native group compilation does not validate every runtime path.

Independent HF exact-sequence checks for the supplemental calls:

| Platform | Configuration | Exact HF sequences | Saved token-position agreement | Repeatable outputs |
|---|---|---:|---:|---|
| windows | supplement-fi-off-r1 | 48/80 | 79.30% | False |
| windows | supplement-fi-on-r1 | 48/80 | 79.30% | False |
| windows | supplement-fi-on-r2 | 48/80 | 79.14% | False |
| windows | supplement-fi-off-r2 | 48/80 | 79.30% | False |
| wsl | supplement-fi-off-r1 | 48/80 | 80.08% | True |
| wsl | supplement-fi-on-r1 | 48/80 | 83.28% | True |
| wsl | supplement-fi-on-r2 | 48/80 | 83.28% | True |
| wsl | supplement-fi-off-r2 | 48/80 | 80.08% | True |

Availability/policy throughput changes are measured point estimates with unlocked
clocks. Exact HF token agreement is a diagnostic, not model answer accuracy.
Reference generation timings never contribute to TokenSpeed throughput.


[Curated measurements, build audits and evidence hashes](https://github.com/kujin66/windows-tokenspeed/blob/build/windows-arm64-review-fixes/docs/guides/windows-arm64-validation.json).
