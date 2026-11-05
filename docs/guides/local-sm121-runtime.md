# Local Windows ARM64 / SM12.1 inference integration

This experimental checkout extends the public dependency build with a dense
Qwen runtime profile. It is not a release of general SM12.1 support.

Use `TOKENSPEED_RUNTIME_PROFILE=dense-qwen` explicitly. The profile lazily loads
the requested model and excludes unrelated quantization and MoE implementations.
Windows uses `triton-windows` through the single kernel `_triton.py` boundary;
Linux retains `tokenspeed-triton`. Windows transport uses multiprocessing pipe
handles and avoids POSIX-only signals and file-descriptor inheritance.

For comparisons, set `TOKENSPEED_COMPARISON_OPS=triton` or `flashinfer` explicitly
alongside the matching attention backend. This also selects the RMSNorm and
SwiGLU implementations. Q/K normalization retains the portable Triton kernel.
The fresh retest uses all 24 native CUDA groups on both Windows and Linux
ARM64. Earlier minimal-profile measurements remain historical evidence and
are not used in the replacement tables. FlashInfer and Triton still supply
operations selected by the explicit runtime profile.

On SM12.1, BF16 causal paged prefill with head dimension 128 uses FlashInfer FA2
because this FlashInfer revision's TRTLLM context runner rejects the architecture.
This local adapter requires full context, no LSE, no sinks, no logit cap, and no
quantized Q/K/V. It builds CSR metadata on the host by downloading device
metadata, and allocates and plans a wrapper per call. Those costs are included
in end-to-end latency. It does not satisfy the production unified-path host
mirror rule; moving planning into the existing host metadata contract remains
necessary before proposing it as a production kernel. Prefill graphs are disabled.
The existing FlashInfer TRTLLM decode entrypoint routes SM12.1 to XQA; decode
uses the fixed graph ladder.

The current Qwen3 dense path has no Kimi router projection. Setting a historical
router AOT policy does not establish CuTe execution, and policy off/on timing is
only a negative control unless dispatch evidence demonstrates an active path.
CuTe availability in Linux is audited separately. Do not infer router or MoE
performance from these Qwen timings.
The matched Linux FlashInfer 0.6.17+g2ea3c24b5dac selects CuTe RMSNorm and
fused-add RMSNorm when CuTe DSL 4.8.0 is available; removing its dependency path selects CUDA norm instead. Those
norm paths require separate correctness checks and are an actual execution A/B.

Run `test/windows/test_runtime_cuda.py` for BF16 references and changed-input
CUDA graph checks. The retained external WireDriver harness measured input
lengths 128/512, concurrency 1/4 and 128 output tokens with three warmups and
five timed calls per case, including detokenization and count checks. That
external harness is not distributed by this PR. See the
[validation report](windows-arm64-validation.md) for cohort boundaries and
known correctness limits.
