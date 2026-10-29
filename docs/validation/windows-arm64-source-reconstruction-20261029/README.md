# Windows ARM64 source reconstruction verification

This publication adds the complete validated Windows source overlay and 136
historical build evidence files. It does not rerun or increase historical
GPU/build/performance counts. See [results.json](results.json) and the
[detailed build guide](../../guides/windows-arm64-build-details.md).

Fresh checks:

- Reconstruct a real checkout from the pinned upstream commit, overlay all 66
  source files, and verify modified/added and unchanged source boundaries.
- Compare the source inventory with the actual upstream-to-port Git diff and
  every overlay postimage with the validated port's Git blob. No relevant
  source file in the five source components is omitted.
- Verify 136 new historical evidence hashes, 152 previously published evidence
  hashes and all eight final performance-comparison rows.
- Run 28 GPU-free tests, including a real-Git reconstruction fixture,
  non-overwrite behavior, corrupt-source rejection, extra-source detection,
  path containment, performance arithmetic and report-integrity checks.
- Run the repository's exact full pre-commit command. Immutable historical
  logs and overlay source bytes are excluded only from modifying formatters;
  syntax, private-key, merge-conflict and other applicable checks still run.

The main branch's newer runtime API is not the tested port. The reconstruction
is an explicit versioned source workflow. Windows ARM64 SDKs/third-party native
wheels remain external inputs; clean-machine dependency resolution, latest-main
runtime integration and new model/GPU benchmarks were not performed here.
