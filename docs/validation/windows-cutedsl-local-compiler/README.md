# Windows-local CuTe compiler validation

This cohort rebuilds the TokenSpeed BF16 router with official Windows x64
CuTe DSL 4.8.1 and executes its CUBINs in native Windows ARM64 Python on
RTX Spark N1X (SM12.1). Windows system emulation runs the compiler; WSL is not
invoked. It does not establish an ARM64-native compiler or model throughput.

- [summary.json](summary.json) records process architectures, GPU/runtime
  versions, test counts and SHA256 hashes of all tested sources.
- [tests.log](tests.log) and [tests.xml](tests.xml) retain the full GPU test result.
- [router-manifest.json](router-manifest.json) records all 16 freshly compiled
  router specializations; [router-selection.json](router-selection.json) is
  the explicit M=1/2/4 block policy (128/128/64).
- [artifacts](artifacts/) retains custom compiler manifests and CUDA Graph DOT
  dumps. Negative tests deliberately create invalid manifests/artifacts; these
  are raw test evidence, not a distributable kernel package. CUBINs stay local.
- [compiler-binaries.json](compiler-binaries.json) records the PE architecture
  of the installed compiler extensions and CuTe runtime DLLs.
- [checksums.sha256](checksums.sha256) covers the evidence bytes.

The tests cover custom add/multiply factories, an affine kernel with runtime
int32/int64/float32 arguments, a two-dimensional transpose, tail sizes,
non-default streams and changed-input graph replays. Cache tests verify reuse,
source/helper invalidation (including stale bytecode with unchanged file size
and timestamp), and corrupted-device-code rejection. Router family API tests
check the captured device symbol and a single kernel node, proving that
solution="auto" uses the configured AOT operation rather than a numerical
fallback. Unsupported router shapes and enabled PDL exercise the fallback or
explicit rejection contracts.

Reproduce with the [local compiler guide](../../guides/windows-arm64-cutedsl-local-compiler.md).
Choose fresh output directories; the validation command refuses to overwrite
existing evidence and treats skipped tests as a validation failure.

Publication normalizes user/build paths and ANSI formatting. Numerical values, sampled IDs and durations are retained. evidence-sha256.json preserves original/public identities; checksums.sha256 verifies published bytes. Tested native-host source snapshots retain their original bytes and line endings.
