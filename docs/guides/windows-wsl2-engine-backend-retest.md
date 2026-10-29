# Windows ARM64 and WSL2 non-eager inference comparison

This compares independent vLLM and TokenSpeed engines, and TokenSpeed's
Triton and FlashInfer backends, on the same RTX Spark N1X (SM12.1, 40 SMs).
The model is dense Qwen3-0.6B BF16. The measured vLLM installation has no
TokenSpeed plugin or enable/disable switch.

The tables report the latest available non-eager measurement for each
configuration. FlashInfer combines two independent engine-process repetitions
per platform (ten measured calls per case); vLLM and Triton each use five
measured calls per case. These measurements come from separate cohorts with
verified per-platform environments. They are a cross-cohort summary, rather
than a single paired six-configuration experiment.

## Non-eager results

| Input / batch | Win vLLM | Win TS Triton | Win TS FI | WSL2 vLLM | WSL2 TS Triton | WSL2 TS FI |
| --- | --- | --- | --- | --- | --- | --- |
| 128 / 1 | 137.79 | 143.62 | 141.63 | 143.02 | 141.98 | 138.85 |
| 128 / 4 | 585.17 | 597.74 | 569.60 | 549.26 | 571.50 | 564.50 |
| 512 / 1 | 134.78 | 130.01 | 130.94 | 134.46 | 125.93 | 132.60 |
| 512 / 4 | 509.22 | 522.49 | 490.88 | 496.45 | 499.86 | 501.26 |

All throughput values are aggregate output tokens/s. TS means TokenSpeed;
FI means FlashInfer. Win means native Windows ARM64. Ratios above 1 favor
the numerator. FlashInfer throughput combines both repetitions using total
output tokens divided by total measured seconds.

Within each operating environment:

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

Native Windows / WSL2:

| Input / batch | vLLM Win / WSL2 | TS Triton Win / WSL2 | TS FI Win / WSL2 |
| --- | --- | --- | --- |
| 128 / 1 | 0.963x | 1.012x | 1.020x |
| 128 / 4 | 1.065x | 1.046x | 1.009x |
| 512 / 1 | 1.002x | 1.032x | 0.987x |
| 512 / 4 | 1.026x | 1.045x | 0.979x |

## FlashInfer measurement details

Each numbered test below is an independent repetition in a fresh engine
process, using the same WSL2 installation and the same per-platform software
environment. The numbers identify test repetitions, not WSL versions.
Throughput columns are aggregate output tokens/s.

| Input / batch | Win test 1 | Win test 2 | WSL2 test 1 | WSL2 test 2 | Win / WSL2 |
| --- | --- | --- | --- | --- | --- |
| 128 / 1 | 145.12 | 138.30 | 137.70 | 140.03 | 1.020x |
| 128 / 4 | 566.84 | 572.39 | 550.64 | 579.07 | 1.009x |
| 512 / 1 | 134.09 | 127.93 | 132.93 | 132.27 | 0.987x |
| 512 / 4 | 493.76 | 488.03 | 501.07 | 501.46 | 0.979x |

Win / WSL2 divides the combined Windows throughput by the combined WSL2
throughput. Each platform's combined throughput is total output tokens across
both tests divided by total measured seconds; individual rates are not
averaged and no slow sample is discarded.

Execution order was Windows test 1, WSL2 test 1, WSL2 test 2, Windows test 2.
GPU benchmarks did not overlap. Every process executed all four cases in the
same case order, with three warmups and five measured calls per case.
All four processes and output-count checks passed: 16 case/process
combinations, 80 measured generate calls and 48 warmup calls.

Per-platform checks verify benchmark SHA256, package versions, recorded module
paths/hashes, model hashes, prompts, sampling, graph policy, GPU and engine
settings. Windows uses guarded FlashInfer 0.6.17+g2ea3c24b5dac; WSL2 uses
FlashInfer 0.7.2. Dependencies and engine settings were fixed for the two
repetitions. Clocks were unlocked and unrelated applications remained running.
GPU telemetry was queried before/after each process, outside generation timing;
lightweight CPU samples accompany the logs.

At input 128 / batch 1, Windows calls span 0.8725–0.9479 seconds and WSL2
calls span 0.8962–0.9413 seconds. Two repetitions do not establish a stable
small-percentage performance ranking or isolate an operating-system effect.

Two case/process combinations show repeated greedy-ID variation; four of
eight cross-platform case comparisons match first-warmup IDs. Completion
and token-count checks do not establish output equivalence or answer quality.

The [measurement evidence](../validation/flashinfer-paired-retest-20261029/)
contains the four reports/logs, command order, exact-version guards, GPU/CPU
samples, source snapshots, recomputed comparisons and checksums.

## Method

All six configurations use identical local Qwen3-0.6B BF16 configuration and
safetensors bytes, verified by SHA256. Inputs are deterministic token IDs:
seed 20261028 + input_length * 100 + batch_size, drawn from [1000,32000).
Input lengths are 128/512; batches are 1/4. Each request generates exactly 128
tokens with temperature zero and ignore_eos true. Each case has three warmups
and five measured calls per engine process. FlashInfer has two independent
process repetitions per platform, as detailed above. Model loading, compilation, capture and warmups are
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
uses vLLM's default Inductor plus FULL_AND_PIECEWISE capture; it differs
from TokenSpeed's decode-only policy. The independent vLLM process
is checked not to import the TokenSpeed engine.

GPU workloads run sequentially. Native processes verify the actual FlashInfer
import path and exact 0.6.17 version before testing. Clocks remain unlocked;
configuration order is fixed. No unrelated application or distribution is
stopped. These measurements include scheduling effects and do not isolate
the contribution of a single kernel, compiler or operating system.

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

Official vLLM 0.26.0 and its supported dependencies are installed into a separate venv, without replacing TokenSpeed's
Torch/CUDA/FlashInfer packages. Linux vLLM pins Torch 2.11.0 and FlashInfer
0.6.14, whereas native Windows uses the local Torch/CUDA port. Dependency and
graph-policy differences mean cross-platform ratios compare software stacks,
not the operating system in isolation.

WSL2 vLLM uses VLLM_WSL2_ENABLE_PIN_MEMORY=1 after a hardware check:
an int32 pinned CPU buffer shaped [4,704] is read by a CUDA pointwise kernel,
then by three CUDA graph replays with changed CPU values. All reads must match
exactly. The supported opt-in enables the V2 runner without patching UVA
checks, changing its runner or falling back to eager execution. The probe,
flag and resolved runtime configuration are included in the evidence.

## Tested runtime and CuTe scope

Native Windows uses the validated complete TokenSpeed port at 9a50ac82.
WSL2 uses official upstream 864bcdd9 with recorded backend adapters. The
[build guide](windows-arm64-build-details.md) and
[source inventory](../../ports/windows-arm64/FILES.md) identify all source
and dependency inputs needed to reconstruct the measured runtime. The
compilation-only current-main change is a different integration boundary;
these measurements do not establish its complete-engine throughput.

The [Windows-local CuTe native-host implementation](windows-arm64-cutedsl-native-host.md)
passes its combined 33-test validation suite, including automatic router
dispatch. Dense Qwen3-0.6B has no matching 7168-to-896 MoE-router projection:
all model reports record aot_router.active=false. No model-throughput gain
is attributed to the CuTe router. The compiler runs as x64 under Windows
emulation; the generated host DLLs and execution Python are native ARM64.

## Limitations and reproduction

Raw reports retain input/output IDs, repeated-output flags, matching positions
and common-prefix lengths. These fixed-length single-GPU measurements do not
evaluate answer quality, HTTP serving, TTFT/ITL, continuous/variable-length
traffic, full-model MoE, quantized inference, multi-GPU operation or prolonged
stability. Software versions, attention implementations and graph policies
differ across stacks; the results do not isolate an OS-only speedup.

The [benchmark](../../tools/perf/benchmark_engine_matrix.py) uses explicit
platform/backend/mode arguments. The
[comparison validator](../../tools/perf/compare_engine_matrix.py) checks
weights, prompts, workload, counts, timing rounds and GPU/script identities.
The [summary validator](../../tools/perf/summarize_engine_retest.py) verifies
the evidence and recomputes the displayed throughput and ratios.
The [engine measurement evidence](../validation/engine-matrix-retest-20261029/)
provides vLLM/Triton reports, exact commands, environment overrides, dependency
pins and source snapshots; FlashInfer's measurement evidence is linked above.
Controller snapshots resolve paths relative to .port; restore them there or
adjust root paths when reproducing on another machine.

```bash
python -m tools.perf.summarize_engine_retest --root . --report latest-engine-summary.json
```
