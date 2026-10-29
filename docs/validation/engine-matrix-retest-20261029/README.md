# vLLM and TokenSpeed Triton measurements

The four *-graph.json reports are the selected vLLM/Triton measurements for
Windows ARM64 and WSL2. Each case has three warmups and five timed calls.
Reports retain all sampled IDs, durations, source/package/model identities
and resolved graph settings. FlashInfer measurements are in the adjacent
flashinfer-paired-retest-20261029 directory.

The pinned-memory probe validates the supported VLLM_WSL2_ENABLE_PIN_MEMORY=1
option using changed-input CUDA graph replays. Exact commands, dependency
pins, backend adapters and setup source snapshots accompany the reports.
Paths and ANSI formatting are normalized for publication; checksums verify
the retained bytes. See [methods and results](../../guides/windows-wsl2-engine-backend-retest.md).
