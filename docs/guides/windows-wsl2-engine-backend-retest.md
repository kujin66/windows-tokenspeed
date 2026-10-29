# Fresh Windows ARM64 and WSL2 inference comparison

This reruns the non-eager workload referenced by
[PR #2](https://github.com/kujin66/windows-tokenspeed/pull/2): vLLM versus
TokenSpeed, and TokenSpeed Triton versus FlashInfer, on each platform.

The main tables use the latest pooled FlashInfer measurements from the
[focused confirmation](#focused-flashinfer-confirmation-2026-10-29). The four
Windows/WSL2 FlashInfer cases each combine two independent process runs
(ten measured calls per case/platform); vLLM and Triton retain their earlier
measurements. Ratios involving FlashInfer are recomputed from these values.
This is a summary across cohorts, not a new simultaneous six-configuration run.

## Latest optimization and measured runtime

At the time of this cohort, the [CuTe DSL AOT experiment](windows-arm64-cutedsl-aot.md)
ran an exported BF16 MoE-router kernel on Windows without engine registration.
The [subsequent Windows-local compiler and opt-in router integration](windows-arm64-cutedsl-local-compiler.md)
does not change these measurements. Dense Qwen3-0.6B has no matching 7168-to-896 MoE-router
projection. All fresh reports record aot_router.active=false. These reruns
cannot establish a model-level benefit from that isolated kernel optimization.

The complete Windows runtime remains the validated port from 9a50ac82; WSL2
TokenSpeed remains based on official upstream 864bcdd9, with isolated backend
adapters. PR #2's afb6e89b-based current-main change is compilation-only.
These are not full-engine throughput measurements of that compilation unit.

## Latest non-eager results

| Input / batch | Win vLLM | Win TS Triton | Win TS FI | WSL2 vLLM | WSL2 TS Triton | WSL2 TS FI |
| --- | --- | --- | --- | --- | --- | --- |
| 128 / 1 | 137.79 | 143.62 | 141.63 | 143.02 | 141.98 | 138.85 |
| 128 / 4 | 585.17 | 597.74 | 569.60 | 549.26 | 571.50 | 564.50 |
| 512 / 1 | 134.78 | 130.01 | 130.94 | 134.46 | 125.93 | 132.60 |
| 512 / 4 | 509.22 | 522.49 | 490.88 | 496.45 | 499.86 | 501.26 |

All throughput values are aggregate output tokens/s. FI means FlashInfer.

FlashInfer values above pool total measured output tokens over total elapsed
time from both confirmation runs. Per-run results and unchanged environment
checks appear in the confirmation section.

Ratios within each operating environment:

| Platform | Input / batch | TS Triton / vLLM | TS FI / vLLM | FI / Triton |
| --- | --- | --- | --- | --- |
| windows | 128 / 1 | 1.042x | 1.028x | 0.986x |
| wsl2 | 128 / 1 | 0.993x | 0.971x | 0.978x |
| windows | 128 / 4 | 1.021x | 0.973x | 0.953x |
| wsl2 | 128 / 4 | 1.040x | 1.028x | 0.988x |
| windows | 512 / 1 | 0.965x | 0.971x | 1.007x |
| wsl2 | 512 / 1 | 0.937x | 0.986x | 1.053x |
| windows | 512 / 4 | 1.026x | 0.964x | 0.940x |
| wsl2 | 512 / 4 | 1.007x | 1.010x | 1.003x |

Native Windows / WSL2 ratios for each engine/backend:

| Input / batch | vLLM Win / WSL2 | TS Triton Win / WSL2 | TS FI Win / WSL2 |
| --- | --- | --- | --- |
| 128 / 1 | 0.963x | 1.012x | 1.020x |
| 128 / 4 | 1.065x | 1.046x | 1.009x |
| 512 / 1 | 1.002x | 1.032x | 0.987x |
| 512 / 4 | 1.026x | 1.045x | 0.979x |

<!-- flashinfer-confirmation:start -->
## Focused FlashInfer confirmation (2026-10-29)

The earlier short-input Windows slowdown did not reproduce in either confirmation pair.
For input 128 / batch 1, the new Windows/WSL2 ratios are 1.054 / 0.988x; pooling all ten measured calls per platform gives 1.020x (141.63 versus 138.85 output tokens/s).

| Input / batch | Windows R1 / R2 (tokens/s) | WSL2 R1 / R2 (tokens/s) | Win / WSL2 R1 / R2 | Pooled Win / WSL2 |
| --- | --- | --- | --- | --- |
| 128 / 1 | 145.12 / 138.30 | 137.70 / 140.03 | 1.054 / 0.988 | 1.020x |
| 128 / 4 | 566.84 / 572.39 | 550.64 / 579.07 | 1.029 / 0.988 | 1.009x |
| 512 / 1 | 134.09 / 127.93 | 132.93 / 132.27 | 1.009 / 0.967 | 0.987x |
| 512 / 4 | 493.76 / 488.03 | 501.07 / 501.46 | 0.985 / 0.973 | 0.979x |

R1 and R2 are independent engine-process repetitions. Execution order was Windows R1, WSL2 R1, WSL2 R2, Windows R2, with no overlapping GPU benchmarks. Each process reran all four original cases in the same case order, with three warmups and five measured calls per case. Pooling divides all measured output tokens by all measured seconds; it does not average individual rates or discard slow calls.

Exact per-platform checks match the earlier benchmark SHA256, package versions, recorded module paths/hashes, model hashes, prompts, sampling, graph policy, GPU and engine settings. Windows still uses the guarded FlashInfer 0.6.17 overlay; WSL2 still uses FlashInfer 0.7.2. No dependency upgrade, engine patch, AOT enablement, clock lock or application shutdown was applied for this confirmation. These remain comparisons of different platform software stacks, rather than an isolated operating-system experiment.

The superseded short-input Windows run included two slower 1.5171/1.9140-second calls. Its complete original data remains archived in the earlier evidence directory. The new Windows calls span 0.8725–0.9479 seconds across both repetitions; the WSL2 calls span 0.8962–0.9413 seconds. This identifies the contribution of the old slow calls to the reported ratio, but does not establish their underlying cause. The evidence cannot distinguish CPU scheduling, GPU clocks, driver scheduling or other transient interference; no such explanation is asserted as a finding.

All four processes and all output-count checks passed (16 case/process combinations, 80 measured generate calls, plus 48 warmup calls). Small differences across these two repeats do not establish a stable ranking. 2 case/process combinations had repeated greedy-ID variation; 4 of eight paired case comparisons had matching first-warmup IDs. Completion/count checks are not a quality evaluation or a proof of output equivalence.

The [confirmation evidence](../validation/flashinfer-paired-retest-20261029/) contains all four raw reports/logs, command order, exact-version guards, pre/post GPU snapshots, lightweight CPU samples, source snapshots, recomputed comparisons and checksums. GPU telemetry was queried only before/after each process, outside generation timing. The main tables now use the pooled FlashInfer values from this confirmation and recompute every dependent ratio. Their vLLM/Triton values remain from the earlier cohort; the tables are explicitly a summary across these runs. Superseded raw measurements remain archived rather than contributing to the current pooled FlashInfer values.
<!-- flashinfer-confirmation:end -->

## Method

All six configurations use identical local Qwen3-0.6B BF16 configuration and
safetensors bytes, verified by SHA256. Inputs are deterministic token IDs:
seed 20261028 + input_length * 100 + batch_size, drawn from [1000,32000).
Input lengths are 128/512; batches are 1/4. Each request generates exactly 128
tokens with temperature zero and ignore_eos true. Each case has three warmups
and five measured calls. Model loading, compilation, capture and warmups are
excluded. Timings include prefill, decode, engine IPC, normal text-output
processing and immediate returned-ID/count checks. There is no HTTP or input
tokenization overhead. Throughput is total measured output tokens divided by
total measured wall time, not the average of individual per-call rates.

TokenSpeed uses Engine.generate, explicit attention_backend selection,
decode graph capture sizes [1,4], prefill graphs disabled, max_model_len 704,
max_num_seqs 4 and max_total_tokens 2816. Prefix caching, KV offload and
retraction snapshots are disabled. The backend control changes attention,
RMSNorm/residual RMSNorm and SwiGLU selection. Other operators retain their
stack's implementations. Windows uses the validated dense-only FlashInfer
0.6.17 overlay and SM12.1 FA2 paged-prefill adapter. WSL2 FlashInfer uses its
matching FA2 adapter. The added WSL2 Triton control is a separate kernel-package
copy: only norm/activation facade conditions change through an explicit
environment selection inherited by workers. Its patch and hashes are retained.

Client batch size four does not guarantee one GPU prefill batch: native
FlashInfer logs include 512-token inputs admitted in groups of two. These
measurements include scheduling effects and do not isolate kernel latency.

vLLM uses LLM.generate with tokenizer initialization and detokenization enabled,
max_model_len 704, max_num_seqs 4, max_num_batched_tokens 2816, explicit
335544320-byte KV capacity, prefix caching disabled, chunked prefill enabled,
automatic attention selection, and graph capture sizes [1,4]. Its resolved
compilation/attention/graph configuration is stored in each report. Graph policy
is not forced to equal TokenSpeed's decode-only policy: the old PR used vLLM's
default Inductor plus FULL_AND_PIECEWISE capture. The independent vLLM process
is checked not to import the TokenSpeed engine.

GPU workloads run sequentially. Two preliminary native attempts are rejected:
the restricted process could not read the intended FlashInfer overlay and
loaded global FlashInfer 0.7.2 instead, with unreadable version metadata. The
first also encountered a safetensors-loading access violation while dependency
installation was running; the second failed in the unintended 0.7.2 JIT's
Windows Ninja file. Neither is counted as a FlashInfer 0.6.17 measurement.
Final native measurements run under the authorized account after WSL2 testing,
with a fail-fast check of the actual FlashInfer import path and exact 0.6.17
version for both TokenSpeed configurations. No directory ACL or global package
is changed. The preliminary logs and reports are retained. Clocks remain
unlocked and configuration order is fixed; five rounds do not establish a
stable small performance ranking. No unrelated application or distribution
is stopped.

## Environments

| Package | Windows TS | Windows vLLM | WSL2 TS | WSL2 vLLM |
| --- | --- | --- | --- | --- |
| Python | 3.13.16 | 3.13.16 | 3.13.16 | 3.13.16 |
| Torch | 2.14.0a0+git20b0f57 | 2.14.0a0+git20b0f57 | 2.14.0 | 2.11.0 |
| Torch CUDA | 13.4 | 13.4 | 13.0 | 13.0 |
| FlashInfer | 0.6.17+g2ea3c24b5dac | 0.6.17+g2ea3c24b5dac | 0.7.2 | 0.6.14 |
| Triton | 3.8.0.post29 | 3.8.0.post29 | 3.8.0 | 3.6.0 |
| TokenSpeed Triton | not installed | not installed | 3.9.10.post20261008 | 3.9.10.post20261028 |
| vLLM | 0.26.0+cu134 | 0.26.0+cu134 | not installed | 0.26.0 |

WSL2 initially had no vLLM installation. Official vLLM 0.26.0 and its supported
dependencies are installed into a separate venv, without replacing TokenSpeed's
Torch/CUDA/FlashInfer packages. Linux vLLM pins Torch 2.11.0 and FlashInfer
0.6.14, whereas native Windows uses the local Torch/CUDA port. Dependency and
graph-policy differences mean cross-platform ratios compare software stacks,
not the operating system in isolation.

Default WSL2 vLLM V2 startup initially failed with "UVA is not available".
The upstream VLLM_WSL2_ENABLE_PIN_MEMORY=1 option is used after a real hardware
check: an int32 pinned CPU buffer shaped [4,704] is read by a CUDA pointwise
kernel, then by three CUDA graph replays with changed CPU values. All reads
must match exactly. This enables vLLM's supported opt-in without patching its
UVA checks, changing its runner, or falling back to eager execution. The probe,
flag, initial failure and selected retry are included in the evidence.

## Output consistency and historical PR comparison

The unchanged vLLM/Triton reports and the two FlashInfer repetitions retain
separate output-consistency diagnostics. The latest FlashInfer diagnostics
are reported in the confirmation section; earlier six-configuration ID-match
counts do not describe the updated summary tables. Inspect the raw reports
for each cohort rather than inferring output equivalence from throughput.

Returned token counts and completion metadata are validated. This is not a
quality evaluation. Raw input/output IDs, repeated-output flags, pairwise
matching positions and common-prefix lengths are retained; throughput does
not establish output equivalence.

| Stack | Input / batch | Old tokens/s | Latest tokens/s | Latest vs old |
| --- | --- | --- | --- | --- |
| windows-vllm-auto | 128 / 1 | 144.75 | 137.79 | -4.81% |
| windows-tokenspeed-triton | 128 / 1 | 144.22 | 143.62 | -0.41% |
| windows-tokenspeed-flashinfer | 128 / 1 | 139.61 | 141.63 | +1.45% |
| wsl2-tokenspeed-flashinfer | 128 / 1 | 140.99 | 138.85 | -1.52% |
| windows-vllm-auto | 128 / 4 | 582.45 | 585.17 | +0.47% |
| windows-tokenspeed-triton | 128 / 4 | 608.83 | 597.74 | -1.82% |
| windows-tokenspeed-flashinfer | 128 / 4 | 564.28 | 569.60 | +0.94% |
| wsl2-tokenspeed-flashinfer | 128 / 4 | 544.32 | 564.50 | +3.71% |
| windows-vllm-auto | 512 / 1 | 133.66 | 134.78 | +0.84% |
| windows-tokenspeed-triton | 512 / 1 | 128.36 | 130.01 | +1.28% |
| windows-tokenspeed-flashinfer | 512 / 1 | 125.84 | 130.94 | +4.05% |
| wsl2-tokenspeed-flashinfer | 512 / 1 | 128.32 | 132.60 | +3.33% |
| windows-vllm-auto | 512 / 4 | 509.51 | 509.22 | -0.06% |
| windows-tokenspeed-triton | 512 / 4 | 515.84 | 522.49 | +1.29% |
| windows-tokenspeed-flashinfer | 512 / 4 | 502.62 | 490.88 | -2.34% |
| wsl2-tokenspeed-flashinfer | 512 / 4 | 477.95 | 501.26 | +4.88% |

The old native vLLM/TokenSpeed values are PR #2's 2026-10-29 cohort. Its old
WSL2 FlashInfer values are the separate 2026-10-28 cohort. PR #2 provides no
old WSL2 vLLM or WSL2 Triton result for this matrix. Latest-versus-old changes
use pooled confirmation values for FlashInfer and earlier values for vLLM/Triton.
They are observational, not an isolated optimization effect.

## Reproduce and inspect

The [benchmark](../../tools/perf/benchmark_engine_matrix.py) runs both engines
with explicit platform/backend/mode arguments. The
[comparison validator](../../tools/perf/compare_engine_matrix.py) rejects
incomplete rounds, different weights/prompts/workloads, wrong output counts
and mismatched benchmark/GPU identities before deriving ratios.

The [evidence directory](../validation/engine-matrix-retest-20261029/) contains
the original six raw reports, logs, exact command/environment overrides, source
snapshots, dependency pins, preliminary failures and derived comparisons.
Controller snapshots resolve paths relative to .port; restore them there or
adjust root paths on another machine. Existing native/WSL2 TokenSpeed build
preparation is in the [versioned build report](windows-arm64-build-details.md).
Do not substitute compile-only current-main wheels for the historical complete
runtime without the separate runtime integration work.
