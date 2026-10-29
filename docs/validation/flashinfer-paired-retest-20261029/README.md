# Paired FlashInfer confirmation

See [the report](../../guides/windows-wsl2-engine-backend-retest.md) for the original cohort and this confirmation. Four sequential process runs retain every measured time and output ID. Both repeats pass process, package, source, model, input, graph, output-count and throughput validation against the original per-platform environment. The two Windows FlashInfer import/version guard reports also match the original exactly. comparisons.json is recomputed from all ten measured calls per case/platform, with no samples removed. commands.json and outcomes.json retain execution order, pre/post GPU snapshots and lightweight CPU samples; no GPU telemetry is sampled during generation. These observations cannot isolate OS or compiler effects or establish output equivalence.

Publication normalizes user/build paths and ANSI formatting. Numerical values, sampled IDs and durations are retained. evidence-sha256.json preserves original/public identities; checksums.sha256 verifies published bytes. Tested native-host source snapshots retain their original bytes and line endings.

Recompute from the repository root with python -m tools.perf.summarize_engine_retest --root . --report latest-engine-summary.json. This verifies all five new evidence manifests and the recorded CuTe test counts/architectures.
