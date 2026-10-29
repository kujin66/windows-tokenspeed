# Fresh non-eager engine/backend evidence

See [the methods and complete tables](../../guides/windows-wsl2-engine-backend-retest.md).

The six top-level *-graph.json files are the selected fresh cohort. comparisons.json is derived only after matching model/script/GPU/workload/input identities and validating all output counts and measurement rounds. Raw reports retain sampled token IDs and elapsed times.

The preliminary/ directory is excluded from selected ratios. Two restricted native attempts loaded unintended global FlashInfer 0.7.2 because the intended 0.6.17 overlay was unreadable. The first also had a safetensors startup access violation during dependency installation; the second failed in the unintended JIT. The final native cohort ran after WSL2 testing under the authorized account, with explicit import-path and exact-version checks. No directory ACL or global package was changed.

Initial default WSL2 vLLM startup failed its UVA availability check. The selected retry uses the official VLLM_WSL2_ENABLE_PIN_MEMORY=1 option after pinned-memory CUDA reads and three changed-input graph replays passed on real hardware. No vLLM source or hardware gate was patched. Probe results, initial failure, selected retry, command overrides, source snapshots, isolated dependency pins and wheel provenance are included. No kernel-level AOT speedup is attributed to these dense-model results.

Publication normalizes user/build paths and ANSI formatting. Numerical values, sampled IDs and durations are retained. evidence-sha256.json preserves original/public identities; checksums.sha256 verifies published bytes. Tested native-host source snapshots retain their original bytes and line endings.
