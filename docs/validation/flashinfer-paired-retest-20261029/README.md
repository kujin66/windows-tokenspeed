# FlashInfer measurements

Four sequential fresh engine processes, two per platform, with three warmups
and five timed calls per case. Reports retain every measured duration and
input/output ID. commands.json and outcomes.json record execution order and
pre/post GPU plus lightweight CPU samples.

comparisons.json combines total output tokens over total elapsed seconds.
Per-platform dependencies, module/source hashes, model and settings match
across both repetitions. Publication normalizes user/build paths and ANSI
formatting; hashes verify the published bytes.

See [methods and results](../../guides/windows-wsl2-engine-backend-retest.md).
Run python -m tools.perf.summarize_engine_retest --root . --report latest-engine-summary.json
from the PR checkout to verify current evidence and recompute the comparison.
