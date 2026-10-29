# Validated Windows ARM64 source port

This bundle includes all 66 modified/added build, runtime, kernel and source-test
files from port 9a50ac82 relative to official upstream 864bcdd9. The overlay is
kept separate from the newer main-branch runtime because its launcher API has
changed. This is the complete tested versioned port, not a latest-main migration.

- [Complete source inventory](FILES.md)
- [Source identity manifest](source-manifest.json)
- [136 historical build records and original/public hashes](historical-evidence-manifest.json)
- [Detailed English reconstruction and native build guide](../../docs/guides/windows-arm64-build-details.md)
- [Eager, decode-graph, FlashInfer/Triton and WSL2 results and limits](../../docs/guides/windows-arm64-performance-report.md)

Run tools/windows/prepare_source_checkout.py from this PR to reconstruct a full
checkout from the pinned official upstream revision. The script verifies source
hashes/preimages, refuses an existing destination, and copies build helpers,
dependencies, skills, evidence and experimental scripts. The source overlay
requires upstream unchanged files; it is not itself a standalone repository.

System SDKs, toolchains, native third-party wheels and models remain external
inputs documented by version, source and hash. They are not redistributed here.
