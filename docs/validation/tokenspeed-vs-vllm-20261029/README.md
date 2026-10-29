# Native Windows ARM64: TokenSpeed versus vLLM

This compares **two independent inference engines**. The installed vLLM package
has no TokenSpeed plugin or enable/disable
switch. The control calls native vLLM LLM.generate without loading TokenSpeed;
the other side calls the compiled Windows ARM64 TokenSpeed Engine.generate.
FlashInfer and Triton identify kernels inside TokenSpeed, not an integration
that changes vLLM into TokenSpeed.

This new 2026-10-29 cohort reruns TokenSpeed together with vLLM. It never uses
earlier HF/WSL2 throughput as the denominator of a vLLM ratio. The primary
selection has **6 successful reports, 300 measured sequences
and 38,400 measured output tokens**. Extra token-only vLLM controls, failed
startup attempts, warmups and engine setup are outside those primary counts.

## Observed deployment throughput

Values above 1 favor TokenSpeed; values below 1 favor vLLM. A 0.90x ratio means
TokenSpeed has 10% lower throughput, not a 10% gain. These are observed ranges
over the four cases, not a universal ranking or a compiler-only improvement.

| Execution mode | TokenSpeed Triton / vLLM | TokenSpeed FI / vLLM |
| --- | ---: | ---: |
| eager | 0.89–0.99x (-11.0% to -0.6%) | 1.07–1.19x (+6.8% to +19.5%) |
| graph | 0.96–1.05x (-4.0% to +4.5%) | 0.94–0.99x (-5.9% to -1.4%) |

**Eager versus vLLM:** triton: 0.6–11.0% lower; flashinfer: 6.8–19.5% higher.

**Graph versus vLLM:** triton: -4.0% to +4.5% relative throughput; flashinfer: 1.4–5.9% lower.

The compiled port does **not** demonstrate a general throughput advantage over
vLLM across modes/backends. FI eager improves throughput in the tested cases;
the graph results are close or favor vLLM depending on backend/case. Small
graph differences need repeated, clock-controlled trials before a stable
ranking can be asserted.

The earlier 2.99–3.32x Windows FI eager ratio was relative to default HF generate.
It is a supplementary framework reference, **not an improvement over vLLM**.
Native vLLM already provides optimized attention, KV caching, normalization and
execution scheduling. The direct vLLM tables here are the relevant engine
comparison for this fixed generation workload. Neither engine's results certify
equivalent generated answers; output-ID and repeat flags remain visible below.

## Hardware, model and software

Both engines use the same physical RTX Spark N1X (SM12.1), GPU 0, on native
Windows ARM64, with world size/tensor parallel size 1. Both load the identical
Qwen3-0.6B BF16 config and safetensors checkpoint; model/config SHA256, GPU name,
PyTorch and CUDA runtime must match in every pair. No NVFP4/FP8/INT4 weights or
KV quantization are substituted. The older vLLM NVFP4 HTTP results are unrelated
and are not used in any ratio here.

vLLM uses the installed 0.26.0+cu134 Windows ARM64 port. The prior installation
audit records wheel SHA256
7c16f54474519ebb6e06ebea4da52b25120dbc56afbd64d33472ad6c7339a13a.
This is a Windows port, not a claim that upstream vLLM officially supports this
Windows configuration. TokenSpeed uses its installed validated Windows port
and isolated documented FI adapter; the source/build package and exact pinned
revisions are in the [build guide](../../guides/windows-arm64-build-details.md).
The post-run entrypoint-source-identity.json records that the installed
TokenSpeed Engine source matches the included versioned overlay after CRLF/LF
normalization; it preserves both original byte hashes.

| Package | vLLM environment | TokenSpeed environment |
| --- | --- | --- |
| torch | 2.14.0a0+git20b0f57 | 2.14.0a0+git20b0f57 |
| transformers | 5.5.3 | 5.12.0 |
| flashinfer-python | 0.6.17+g2ea3c24b5dac | 0.6.17+g2ea3c24b5dac |
| triton-windows | 3.8.0.post29 | 3.8.0.post29 |

Transformers versions differ even though torch/CUDA/FI/Triton versions match.
Native vLLM and TokenSpeed do not call the same HF generate implementation.
The comparison therefore measures deployment stacks, rather than isolating
an algorithm, compiler, host scheduler, or a single dependency's contribution.

## Workload and timing method

- Cross input lengths 128/512 with batch sizes 1/4. Each sequence generates
  exactly 128 new tokens, greedy decoding, EOS termination ignored. The same
  input IDs are constructed using Python Random with seed
  20261028 + input_length * 100 + batch_size and token IDs in [1000, 32000).
- Three warmup calls precede five measured calls for each case. Batch4 is one
  API call submitting four sequences together, not four timed HTTP clients.
  Every warmup and measured output ID is retained. The validator recomputes
  tokens/s as all measured output tokens divided by total measured wall time;
  it does not average per-call throughput. Relative throughput divides TS by
  vLLM, and relative percent subtracts one from that ratio then multiplies by 100.
- Time one normal blocking generate call, including prefill, autoregressive
  decode, scheduler/IPC, sampling, output return and immediate checks. Inputs
  are token IDs, so prompt tokenization is excluded. **The primary vLLM calls
  initialize the tokenizer and enable detokenization**, matching TokenSpeed's
  normal text-return path; returned token IDs are used for validation. Each
  engine retains its own output/control-plane implementation.
- Model loading, explicit warmup, graph setup and compilation are excluded.
  Earlier vLLM controls with skip_tokenizer_init=True/detokenize=False are
  preserved as diagnostics; they are not substituted for the primary normal
  text-output API comparison. Their selection is archived separately.
- Prefix caching and offload are disabled. OMP/MKL threads are 1. GPU jobs are
  sequential; WSL2 is stopped during native runs. This cohort does not run
  CPU test suites concurrently with its GPU measurements. Background OS
  activity and GPU clocks are not locked; configurations are grouped rather
  than repeated in an ABBA design. Five samples do not establish significance
  for small percentage changes.

### Resource limits and the failed memory-admission pilot

Both engines allow four sequences with max model length 704. TokenSpeed sets
max_total_tokens=2816 and GPU memory utilization 0.25. vLLM sets max_num_batched_tokens=2816
and a fixed **320 MiB KV allocation**, resolving to 2,912 cache tokens in the
successful initial native run. These two limits have different meanings;
max_num_batched_tokens is a per-iteration budget, not TokenSpeed's total KV cap.
Four 704-token contexts fit the configured cache. Chunked prefill remains enabled
in the primary vLLM policy; the benchmark does not force the unsupported
disable-chunked-prefill setting used by the short initial pilot.

The short pilot requested utilization 0.25. vLLM rejected startup because free
CUDA memory was 1.74/12.35 GiB, below the requested 3.09 GiB, before any timed
sample. The main vLLM attempts use utilization 0.03 with explicit KV bytes. The
vLLM log states that this fixed KV allocation bypasses its normal cache memory
profiling; utilization 0.03 **does not mean total process/device usage is 3%**.
This is a resource-fit configuration on a system with limited available shared
memory, not a whole-engine equal-memory or memory-efficiency benchmark. Failed
attempts and controlled cleanup of their own process trees are retained.

## Eager results

All throughput columns are aggregate output tokens/s. Both engines enforce
eager execution and disable decode CUDA Graphs in these rows.

| Input / batch | vLLM | TokenSpeed Triton | TokenSpeed FI | TS Triton / vLLM | TS FI / vLLM |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 / 1 | 59.00 | 58.65 | 63.02 | 0.994x | 1.068x |
| 128 / 4 | 228.44 | 212.93 | 260.88 | 0.932x | 1.142x |
| 512 / 1 | 61.60 | 54.82 | 67.90 | 0.890x | 1.102x |
| 512 / 4 | 229.22 | 222.55 | 273.89 | 0.971x | 1.195x |

## Non-eager results and actual vLLM policy

TokenSpeed captures decode sizes [1, 4], maximum 4; prefill graphs stay disabled.
vLLM uses decode capture sizes [1, 4] and records its actual resolved compile/graph
policy. The default vLLM policy is attempted first in the initial controls.
If startup fails, the explicit decode-only alternative sets FULL_DECODE_ONLY
and compilation mode 0: CUDA Graph replay without torch.compile. Such a result
is labeled with its actual policy, not described as successful default compile.
Graph boundaries, prefill compilation and scheduling differ between engines.

| Mode | vLLM attention evidence | vLLM compilation mode | vLLM CUDA Graph mode |
| --- | --- | --- | --- |
| eager | FlashAttention v2 | 0 (no torch.compile) | NONE |
| graph | FlashAttention v2 | 3 (vLLM compile/Inductor) | FULL_AND_PIECEWISE |

| Input / batch | vLLM | TokenSpeed Triton | TokenSpeed FI | TS Triton / vLLM | TS FI / vLLM |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 / 1 | 144.75 | 144.22 | 139.61 | 0.996x | 0.965x |
| 128 / 4 | 582.45 | 608.83 | 564.28 | 1.045x | 0.969x |
| 512 / 1 | 133.66 | 128.36 | 125.84 | 0.960x | 0.941x |
| 512 / 4 | 509.51 | 515.84 | 502.62 | 1.012x | 0.986x |

### Non-eager versus eager within each engine

| Input / batch | vLLM non-eager / eager | TS Triton graph / eager | TS FI graph / eager |
| --- | ---: | ---: | ---: |
| 128 / 1 | 2.454x | 2.459x | 2.216x |
| 128 / 4 | 2.550x | 2.859x | 2.163x |
| 512 / 1 | 2.170x | 2.341x | 1.853x |
| 512 / 4 | 2.223x | 2.318x | 1.835x |

These are whole-generation throughput ratios. vLLM's default non-eager path
also enables its compiler optimizations, so its ratio is not a pure CUDA Graph
ablation. This table demonstrates why graph-policy benefits should not be
attributed exclusively to TokenSpeed or presented as a gain over another engine.

The installed vLLM can use its own FlashAttention extension even where the
Windows torch wheel lacks built-in Flash SDPA. HF's math-SDPA fallback is not
an explanation of a direct vLLM result. No per-component ablation here isolates
how much of a difference comes from attention, projection/norm fusion, CPU
scheduling, graph replay, compiler optimization or output processing.

The TokenSpeed Triton control selects Triton attention, RMSNorm and SwiGLU.
Its adapted FI path uses FA2 prefill, TRT-LLM decode and FI norm/activation;
other operations keep their stack's implementations. vLLM's automatic backend
choice and actual compile/graph policy are shown above. A shared installed FI
version does not make the engines use identical kernels. Both engines already
have optimized execution paths, so the HF norm/graph ablations cannot be added
together or transferred as a claimed TokenSpeed advantage over vLLM.

## Token-ID validation and limits

All selected timed sequences complete 128 tokens. Each engine's repeat flag
compares all three warmups and five measured results for that case. Cross-engine
reference flags compare the first measured output, not every possible pair.
BF16 attention/fusion and sampling numerics can differ; fixed-length completion
and matching model hashes do not establish full numerical or language-quality
equivalence. Variations and mismatches are retained rather than omitted.

| Mode / TokenSpeed backend | vLLM repeat-stable cases | TS repeat-stable cases | First measured output matches |
| --- | ---: | ---: | ---: |
| eager / triton | 3/4 | 4/4 | 2/4 |
| eager / flashinfer | 3/4 | 3/4 | 3/4 |
| graph / triton | 3/4 | 3/4 | 1/4 |
| graph / flashinfer | 3/4 | 3/4 | 2/4 |

This is a single-GPU, fixed-length, synthetic-input, fixed-batch offline Engine
API comparison. It does not measure HTTP, streaming TTFT/ITL, realistic arrival
rates, continuous serving concurrency, prefix reuse, long prompts, quality,
MoE, quantized models, multi-GPU or prolonged stability. WSL2 vLLM is not installed
in this test setup; **these new vLLM ratios apply to native Windows only**.
Historical WSL2 results remain a separate cohort.

## Attempts and reproducibility

| Attempt | Completed benchmark | Controller result |
| --- | --- | --- |
| windows-vllm-auto-eager-default.json | True | exit 0 |
| windows-tokenspeed-triton-eager.json | True | exit 0 |
| windows-tokenspeed-flashinfer-eager.json | True | exit 0 |
| windows-vllm-auto-graph-default.json | True | exit 0 |
| windows-tokenspeed-triton-graph.json | True | exit 0 |
| windows-tokenspeed-flashinfer-graph.json | True | exit 0 |
| windows-vllm-text-eager.json | True | exit 0 |
| windows-vllm-text-graph.json | True | exit 0 |

Use the prepared native environments, exact same verified model and a fresh
report directory. The portable maintained program is tools/perf/benchmark_vllm.py;
its arguments explicitly select execution, attention, graph and output policy.
Windows needs the documented compiler/DLL environment. The exact measured
programs, Windows launchers, controller order/commands, dependency metadata,
token-only diagnostics and failures are archived under sources and JSON/logs.
Restore archived versions when reproducing their exact source SHA256 identities;
the maintained tools may be formatted or consolidated after measurement.

After applying that native compiler/DLL launcher environment and selecting the
vLLM virtual environment, the maintained module's explicit normal-output form
is:

```bash
python -m tools.perf.benchmark_vllm --model /path/to/Qwen3-0.6B --report /path/to/new-vllm-eager.json --execution eager --attention auto --graph-policy default --output-policy text --memory-utilization 0.03 --input-lengths 128 512 --batch-sizes 1 4 --output-length 128 --warmup 3 --rounds 5
```

Repeat with execution graph and a new filename; retain the selected graph
policy in the report. The archived measured normal-output program fixes the
same two output choices directly; the maintained CLI consolidates them into
the required output-policy argument. The exact Windows wrapper commands are
preserved in commands.json and text-commands.json.

Run the GPU-free comparison validator from the PR checkout:

```bash
python -m tools.perf.summarize_vllm --directory docs/validation/tokenspeed-vs-vllm-20261029 --report recomputed-vllm-results.json
```

The strict validator checks actual vLLM engine identity, no TokenSpeed imports,
normal text-output policy, matching model/GPU/torch/CUDA/precision, modes,
prefix-cache policy, input IDs, warmups/rounds, output lengths and all raw timing
arithmetic. It rejects a graph result resolved to NONE and preserves throughput
losses. It recomputes comparison values and consistency flags without a GPU.
The original/public evidence SHA256 manifest covers 41 files. Public
copies redact local user/host/vendor-build paths and use UTF-8/LF. Two initial
pilot metadata files had a literal EOF newline escape; the manifest explicitly
records its repair in public JSON without changing values. Raw originals stay
local. README.md and comparisons.csv are derived from results.json and are
outside the evidence manifest.
