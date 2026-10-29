# TokenSpeed versus Transformers/PyTorch without TokenSpeed

This is a fresh, same-platform comparison of the full TokenSpeed Engine API
against Transformers/PyTorch generation without importing or calling TokenSpeed.
It uses the same Qwen3-0.6B BF16 weights, GPU, input IDs, output lengths and
within-platform torch/CUDA installations. All primary TokenSpeed measurements
were rerun on **2026-10-29**; 2026-10-28 throughput is not used in these ratios.

The main matrix has **30 raw report files**, representing 24 equivalent complete
configurations, **1,200 measured sequences and 153,600 measured output tokens**.
API calls returning a batch count as multiple sequences. Warmup, graph setup,
graph validation, CPU-profiler requests and followup checks are not part of
these measured totals. Each complete configuration uses input 128/512, output
128, batch 1/4, three warmups and five measured rounds per case.

- **windows: TokenSpeed FI eager is 2.99–3.32x stock HF generate**, and 1.57–1.94x the static eager control with raw FI RMSNorm. In decode-graph mode, it is **1.14–1.80x** the request-sized HF graph with FI RMSNorm.
  Selecting the fastest measured no-TokenSpeed graph control separately for each case gives 1.13–1.80x. This conservative comparison includes the math-flag variant; it does not establish a causal effect of that flag.
- **wsl2: TokenSpeed FI eager is 1.76–2.19x stock HF generate**, and 1.31–1.60x the static eager control with raw FI RMSNorm. In decode-graph mode, it is **1.11–1.65x** the request-sized HF graph with FI RMSNorm.
  Selecting the fastest measured no-TokenSpeed graph control separately for each case gives 1.11–1.65x. This conservative comparison includes the math-flag variant; it does not establish a causal effect of that flag.

The optimized no-TokenSpeed controls substantially narrow the apparent gain.
TokenSpeed still has higher measured throughput in these paired cases, but
the stock-HF ratio must not be presented as an isolated TokenSpeed algorithm
or scheduler improvement. The ablations below explain which gains can also
be obtained without TokenSpeed and which differences remain unisolated.

## The no-TokenSpeed condition

The baseline loads `transformers.Qwen3ForCausalLM` via AutoModelForCausalLM and
executes the real 28-layer model on CUDA, with BF16 parameters and SDPA attention.
It runs in separate processes from TokenSpeed. The existing environment can
contain installed TokenSpeed wheels, but the baseline checks at entry and exit
that **no module whose name begins with tokenspeed was imported**. Every
completed baseline records an empty `tokenspeed_modules_loaded` list. The
FlashInfer normalization controls call the raw FlashInfer package, without
TokenSpeed kernel/runtime imports.

TokenSpeed uses its installed Windows port or pinned official WSL2 build, full
Engine.generate request path and documented Triton/FI kernel configuration.
The no-TokenSpeed controls return CPU token IDs; TokenSpeed's normal API also
includes its own frontend/IPC/output processing. This is a deployment-stack
comparison, not a pure scheduler or equal-software-boundary microbenchmark.

## Eager results

All throughput is **aggregate output tokens/s**, computed from total measured
output tokens divided by total measured seconds. Ratios above 1 favor TokenSpeed.
The FI-normalization eager control also uses a static cache, so its difference
from stock generate must not be attributed solely to normalization. The
like-for-like normalization comparison appears in the ablation table below.

| Platform | Input / batch | HF generate | HF static eager + FI RMSNorm | TokenSpeed Triton | TokenSpeed FI | TS FI / HF generate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| windows | 128 / 1 | 21.13 | 40.27 | 53.15 | 63.11 | 2.987x |
| windows | 128 / 4 | 84.65 | 140.32 | 205.84 | 272.00 | 3.213x |
| windows | 512 / 1 | 21.77 | 40.83 | 54.44 | 65.38 | 3.003x |
| windows | 512 / 4 | 68.47 | 138.66 | 202.39 | 227.31 | 3.320x |
| wsl2 | 128 / 1 | 31.95 | 42.98 | not measured | 56.21 | 1.759x |
| wsl2 | 128 / 4 | 117.74 | 151.14 | not measured | 226.33 | 1.922x |
| wsl2 | 512 / 1 | 30.99 | 43.46 | not measured | 62.95 | 2.031x |
| wsl2 | 512 / 4 | 110.82 | 152.13 | not measured | 242.82 | 2.191x |

## Non-eager results: decode CUDA Graphs

HF graphs capture the manual model step, argmax, output write and GPU counters.
TokenSpeed uses its normal decode-graph implementation. Both leave prefill
outside the decode graph. Graph capture/validation is excluded from throughput.
Graph boundaries and control-plane responsibilities differ between the engines.

The fixed KV control allocates 704 positions per sequence. Request-sized
controls allocate input+output capacity: 256 or 640. They retain the same
logical generation task and validate graph output against their own eager loop.
This avoids making TokenSpeed look faster solely by selecting an oversized
static-cache baseline. The separate math flag is a PyTorch accumulation policy;
its effect depends on whether the selected SDPA backend actually uses math.

| Platform | Input / batch | HF graph, 704, Torch norm | HF graph, fit KV, FI norm | HF fit KV, FI norm, math flag | TokenSpeed Triton | TokenSpeed FI | TS FI / HF fit KV + FI norm |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| windows | 128 / 1 | 98.64 | 123.81 | 125.24 | 143.44 | 141.60 | 1.144x |
| windows | 128 / 4 | 235.85 | 372.61 | 384.67 | 606.79 | 581.75 | 1.561x |
| windows | 512 / 1 | 87.95 | 108.25 | 111.60 | 130.66 | 126.10 | 1.165x |
| windows | 512 / 4 | 227.29 | 274.18 | 267.71 | 506.77 | 493.94 | 1.802x |
| wsl2 | 128 / 1 | 99.84 | 126.97 | 126.93 | not measured | 140.64 | 1.108x |
| wsl2 | 128 / 4 | 254.35 | 403.98 | 396.61 | not measured | 536.36 | 1.328x |
| wsl2 | 512 / 1 | 95.82 | 110.37 | 110.20 | not measured | 130.92 | 1.186x |
| wsl2 | 512 / 4 | 240.19 | 283.56 | 283.24 | not measured | 467.21 | 1.648x |

### Complete no-TokenSpeed control matrix

All values below are aggregate output tokens/s. Static configurations use
704 KV positions unless labeled bucketed (request-sized). FlashInfer in these
HF labels replaces RMSNorm only; the HF attention backend remains SDPA.

| Platform | No-TokenSpeed configuration | 128 / 1 | 128 / 4 | 512 / 1 | 512 / 4 |
| --- | --- | ---: | ---: | ---: | ---: |
| windows | generate-torch | 21.13 | 84.65 | 21.77 | 68.47 |
| windows | loop-dynamic-torch | 22.18 | 90.12 | 18.72 | 68.48 |
| windows | loop-static-eager-torch | 22.40 | 83.14 | 24.01 | 85.21 |
| windows | loop-static-graph-torch | 98.64 | 235.85 | 87.95 | 227.29 |
| windows | loop-static-eager-flashinfer | 40.27 | 140.32 | 40.83 | 138.66 |
| windows | loop-static-graph-flashinfer | 108.78 | 277.42 | 105.57 | 257.98 |
| windows | loop-static-graph-bucketed-torch | 103.39 | 341.18 | 97.25 | 251.72 |
| windows | loop-static-graph-bucketed-flashinfer | 123.81 | 372.61 | 108.25 | 274.18 |
| windows | loop-static-graph-bucketed-flashinfer-reduced | 125.24 | 384.67 | 111.60 | 267.71 |
| wsl2 | generate-torch | 31.95 | 117.74 | 30.99 | 110.82 |
| wsl2 | loop-dynamic-torch | 32.12 | 129.54 | 32.25 | 119.53 |
| wsl2 | loop-static-eager-torch | 25.86 | 99.72 | 29.00 | 98.50 |
| wsl2 | loop-static-graph-torch | 99.84 | 254.35 | 95.82 | 240.19 |
| wsl2 | loop-static-eager-flashinfer | 42.98 | 151.14 | 43.46 | 152.13 |
| wsl2 | loop-static-graph-flashinfer | 112.03 | 288.74 | 109.51 | 269.67 |
| wsl2 | loop-static-graph-bucketed-torch | 113.39 | 367.94 | 102.07 | 258.93 |
| wsl2 | loop-static-graph-bucketed-flashinfer | 126.97 | 403.98 | 110.37 | 283.56 |
| wsl2 | loop-static-graph-bucketed-flashinfer-reduced | 126.93 | 396.61 | 110.20 | 283.24 |

## What the controlled experiments establish

### Windows versus WSL2 in the fresh cohort

These are WSL2/Windows throughput ratios; values above 1 favor the tested WSL2
stack. Each ratio uses the same named engine/control and execution mode, from
this fresh cohort only. PyTorch/CUDA/FlashInfer builds differ across platforms.

| Input / batch | WSL2 / Windows HF generate | WSL2 / Windows HF fit-KV graph + FI norm | WSL2 / Windows TS FI eager | WSL2 / Windows TS FI graph |
| --- | ---: | ---: | ---: | ---: |
| 128 / 1 | 1.512x | 1.026x | 0.891x | 0.993x |
| 128 / 4 | 1.391x | 1.084x | 0.832x | 0.922x |
| 512 / 1 | 1.424x | 1.020x | 0.963x | 1.038x |
| 512 / 4 | 1.619x | 1.034x | 1.068x | 0.946x |

### Ablations within the no-TokenSpeed implementation

| Platform | Change within the no-TokenSpeed implementation | Throughput ratio range over four cases |
| --- | --- | ---: |
| windows | Manual dynamic loop / HF generate | 0.860x–1.065x |
| windows | Static cache/mask policy / dynamic loop (coupled changes) | 0.923x–1.282x |
| windows | Decode graph / eager, same 704 cache and Torch norm | 2.667x–4.404x |
| windows | FI RMSNorm / Torch RMSNorm, same static eager loop | 1.627x–1.798x |
| windows | FI RMSNorm / Torch RMSNorm, same fixed-cache graph | 1.103x–1.200x |
| windows | Request-sized KV / 704 KV, FI norm graph | 1.025x–1.343x |
| windows | Math-reduction flag enabled / disabled, same fit-KV graph | 0.976x–1.032x |
| wsl2 | Manual dynamic loop / HF generate | 1.005x–1.100x |
| wsl2 | Static cache/mask policy / dynamic loop (coupled changes) | 0.770x–0.899x |
| wsl2 | Decode graph / eager, same 704 cache and Torch norm | 2.439x–3.861x |
| wsl2 | FI RMSNorm / Torch RMSNorm, same static eager loop | 1.498x–1.662x |
| wsl2 | FI RMSNorm / Torch RMSNorm, same fixed-cache graph | 1.122x–1.143x |
| wsl2 | Request-sized KV / 704 KV, FI norm graph | 1.008x–1.399x |
| wsl2 | Math-reduction flag enabled / disabled, same fit-KV graph | 0.982x–1.000x |

These are observed ratios between specified implementations on four cases,
not an additive decomposition of TokenSpeed's speedup. In particular:

1. **Graph replay is valuable without TokenSpeed too.** The fixed-cache
   eager/graph pair keeps model, cache, precision and normalization constant.
   It tests the combined reduction in per-token Python/operator dispatch,
   allocations and kernel-submission overhead from capture/replay.
2. **Fused RMSNorm is a measured contributor, rather than an engine-only
   feature.** Only the 113 HF RMSNorm modules are replaced, retaining their
   parameters and epsilon. Attention, MLP and cache are unchanged in the paired
   static-loop controls. FP32-reference probes are recorded for the replacement.
   Fused FI and HF's intermediate BF16 rounding differ, so this is not a
   bitwise-equivalent numerical implementation or an attribution of that same
   percentage to TokenSpeed.
3. **Attention dispatcher and cache policy matter.** Static versus dynamic
   changes masks, GQA handling, padding and allocation together. Request-sized
   graph buffers isolate the capacity choice within the same graph/norm path.
   A math-reduction flag is not an optimization of a graph using a different
   SDPA backend; small differences in such a control cannot establish a
   precision-related speedup.

### Actual attention paths, not just the requested name sdpa

| Platform | Flash SDPA compiled | Stock HF CPU diagnostic attention operator | Fixed static eager attention operator |
| --- | --- | --- | --- |
| windows | False | _scaled_dot_product_attention_math | _scaled_dot_product_efficient_attention |
| wsl2 | True | _scaled_dot_product_flash_attention | _scaled_dot_product_efficient_attention |

The Windows wheel reports no compiled Flash SDPA. The representative backend
probe also tests forced efficient/cuDNN/math variants against FP32 attention,
recording both supported and rejected combinations. The WSL2 installation has
different torch/CUDA libraries and backend availability; cross-OS differences
therefore cannot be isolated to OS overhead.

The selected full-model forced-cuDNN control (input128/batch1) and its CPU
operator summary are retained as a separate diagnostic. Successful bare-tensor
backend support does not by itself prove full-model layout support. Failed
forced-backend attempts are retained as failures, not substituted with a fake
or fallback implementation. These diagnostic requests are outside the primary
matrix totals.

| Platform | Selected forced-cuDNN full-model control, input128 / batch1 |
| --- | --- |
| windows | 22.36 output tok/s (1.058x default HF; TS FI eager / this control = 2.822x); _scaled_dot_product_cudnn_attention |
| wsl2 | 34.16 output tok/s (1.069x default HF; TS FI eager / this control = 1.645x); _scaled_dot_product_cudnn_attention |

Forcing cuDNN is supported in both selected full-model checks, but does not
eliminate their Windows/WSL2 difference. Missing compiled Flash SDPA is therefore
an observed stack difference, not a sufficient single-cause explanation.
cuDNN, GEMM, PyTorch builds and host dispatch still differ; these measurements
do not independently quantify each source of the remaining cross-platform gap.

### Operator evidence for normalization fusion

The following are CPU-dispatched ATen operation counts for one separately
profiled fixed-cache eager request, input128/batch1/output128. They are not GPU
kernel counts or unprofiled latency fractions. Profiling occurred after timing.

| Platform / fixed eager norm | pow | mean | rsqrt |
| --- | ---: | ---: | ---: |
| windows / torch | 14464 | 14464 | 14464 |
| windows / flashinfer | 0 | 0 | 0 |
| wsl2 / torch | 14464 | 14464 | 14464 |
| wsl2 / flashinfer | 0 | 0 | 0 |

HF's observed SDPA wrapper also expands 8 KV heads to 16 heads when its native
GQA path is not selected for masked static attention. TokenSpeed's paged
attention/decode path and fused QK/RoPE/KV-write, packed projections and
activation paths differ. These source-level differences are consistent with
remaining gains but were not each independently ablated; **the native C++/Rust
scheduler's individual contribution has not been measured**.
The pinned [Qwen3 model implementation](https://github.com/lightseekorg/tokenspeed/blob/864bcdd9ef3c5f1512bbe293d970d73eb29d2c94/python/tokenspeed/runtime/models/qwen3.py)
shows the packed QKV and gate/up projections and attention QK-norm contract;
the [experimental backend record](../windows-flashinfer-0617-20261028/README.md)
documents the actual FI attention/norm/activation adapter used in the reruns.

## Correctness and graph-state verification

Every timed request produces exactly 128 new tokens; all warmup/measured IDs
are retained. Each HF graph configuration first compares full generated IDs
with the matching eager loop. Additional representative GPU audits compare
all KV tensor contents, graph-side cache lengths, position and output-index
counters against eager execution, including a request-sized capacity boundary.
Those checks are recorded separately and are not throughput samples.

See `comparisons.csv` for every repeat-consistency and cross-engine reference-ID
flag. Reference comparison uses the first measured output, not every possible
pair. Different BF16 norm/attention implementations can produce different
greedy IDs. Fixed-length completion and component FP32 tolerances do not prove
language quality or full-engine numerical equivalence. Prior TokenSpeed output
variation is preserved in historical evidence and fresh flags remain visible.

| Platform / TokenSpeed backend / mode | HF generate or fit-KV FI graph repeat-stable cases | TokenSpeed repeat-stable cases | First measured output matches HF |
| --- | ---: | ---: | ---: |
| windows / flashinfer / eager | 4/4 | 3/4 | 3/4 |
| windows / flashinfer / graph | 4/4 | 4/4 | 3/4 |
| windows / triton / eager | 4/4 | 4/4 | 2/4 |
| windows / triton / graph | 4/4 | 3/4 | 2/4 |
| wsl2 / flashinfer / eager | 4/4 | 4/4 | 2/4 |
| wsl2 / flashinfer / graph | 4/4 | 4/4 | 3/4 |

Repeat-stable means all three warmups and five measured outputs were identical
within that case. These flags describe token IDs, independently of timing and
fixed-length completion. Cross-engine mismatches remain unresolved; the
throughput comparison does not certify equivalent generated answers.

## Limits of the conclusion

- Improvements are relative to the named, tested Transformers controls, not
  every possible implementation without TokenSpeed. Torch compile, TensorRT,
  vLLM and other engines are not benchmarked here.
- Backend/fusion/cache/control-plane differences are part of the stack-level
  comparison. Their individual effects cannot be obtained by subtracting
  timings from different engines or summing profiler entries.
- This is a fixed-length, greedy, synthetic-token, single-GPU, fixed-batch test.
  Prefix cache, offload and prefill graphs are disabled. HTTP, TTFT/ITL,
  continuous batching, variable-length serving, grammar, MoE, quantized models,
  multi-GPU and long-duration stability are outside this comparison.
- Five measured rounds, sequential configuration groups, unlocked clocks and
  no repeated ABBA trials limit significance claims for small differences.
- Background CPU/OS activity was not isolated. GPU-free tool validation
  overlapped part of the WSL2 TokenSpeed eager group; GPU jobs themselves were
  sequential. Treat the cross-platform eager ratios as descriptive observations,
  not an isolated OS effect or a precise ranking.
- HF uses known request/batch configurations for graph setup and request-sized
  buffers. The setup cost and broader serving capability are not measured.

## Reproduce and inspect

The portable programs are under `tools/perf`: benchmark_without_tokenspeed.py,
run_without_tokenspeed_suite.py, run_bucketed_without_tokenspeed.py,
probe_sdpa_backends.py, probe_hf_graph_state.py and the comparison validators.
Run module entry points from the PR checkout. Supply the native dependency
versions and verified local model described in the build guides. All behavior
choices are explicit CLI arguments; use a new report directory for reruns.

For example, after activating the platform's prepared virtual environment:

```bash
python -m tools.perf.run_without_tokenspeed_suite \
  --model /path/to/Qwen3-0.6B --report-dir /path/to/new-results \
  --platform-label wsl2 --diagnostics cpu \
  --configurations generate-torch loop-dynamic-torch loop-static-eager-torch loop-static-graph-torch loop-static-eager-flashinfer loop-static-graph-flashinfer
python -m tools.perf.run_bucketed_without_tokenspeed \
  --model /path/to/Qwen3-0.6B --report-dir /path/to/new-results \
  --platform-label wsl2 --normalizations torch flashinfer \
  --input-lengths 128 512 --batch-sizes 1 4 --output-length 128 \
  --warmup 3 --rounds 5 --math-reduction full
python -m tools.perf.summarize_without_tokenspeed \
  --directory /path/to/complete-results --report /path/to/summary.json
```

On Windows, invoke the selected venv Python through tools/windows/run.py with
the FlashInfer-only environment recorded in windows-commands.json. Run the
reduced-math controls separately with normalization flashinfer and
math-reduction reduced. The strict summarizer requires all 30 primary reports,
including fresh TokenSpeed reports; it validates input/model hashes, platform
torch/CUDA identity, modes, token counts, graph checks and raw timing arithmetic.
It also runs against the committed evidence directory without a GPU.

Verify all public hashes and recompute every comparison without a GPU from
the PR checkout:

```python
import hashlib
import json
from pathlib import Path
from tools.perf.summarize_without_tokenspeed import summarize

directory = Path("docs/validation/tokenspeed-vs-transformers-20261029")
manifest = json.loads((directory / "evidence-sha256.json").read_text())
for entry in manifest:
    actual = hashlib.sha256((directory / entry["file"]).read_bytes()).hexdigest()
    assert actual == entry["published_sha256"], entry["file"]
assert summarize(directory) == json.loads((directory / "results.json").read_text())
print("Evidence hashes and all derived comparisons verified")
```

Exact measured source snapshots, launch commands, raw reports, capability and
graph-state probes, CPU summaries, failures and original/public hashes are
preserved alongside this record. Restore archived benchmark modules to
`tools/perf` and the TokenSpeed launcher scripts to their documented `.port`
locations before executing them. Windows and WSL2 GPU jobs ran sequentially;
the task's WSL2 distro was stopped for Windows runs. Public copies redact local
user/host/vendor-build paths and use UTF-8/LF. Raw originals remain local.
The source-sha256 manifest identifies the exact original measured bytes;
evidence-sha256 records both original and sanitized public hashes. Archived
sources precede final formatting of the maintained tools; use the archived
versions to reproduce those identities. README.md and comparisons.csv are
derived presentations of results.json and are outside the evidence manifest.
