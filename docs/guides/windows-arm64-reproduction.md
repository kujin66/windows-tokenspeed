# Reproduce Windows ARM64 TokenSpeed

Follow the [complete build guide](windows-arm64-build-details.md) to restore
the source overlay and external ARM64 dependency inputs, then build and audit
the scheduler, engine, kernel and runtime wheels.

Prepare the [FlashInfer adapter](windows-flashinfer-comparison.md) and
[separate WSL2 build](wsl2-arm64-build.md) before following the current
[non-eager workload and comparison](windows-wsl2-engine-backend-retest.md).
Use the [build skill](../../skills/tokenspeed-windows-arm64-build/SKILL.md)
for local toolchain preparation and failure diagnosis.
