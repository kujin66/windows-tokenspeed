# Windows ARM64 native CuTe host validation

This cohort rebuilds device artifacts with the official Windows x64 CuTe DSL
compiler, cross-compiles its original host launchers to ARM64 COFF, links native
ARM64 DLLs with the installed CUDA SDK and executes them from native ARM64
Python. It does not invoke WSL or link the x64 CuTe runtime into those DLLs.
It does not establish an ARM64-native Python compiler backend or model speedups.

- [summary.json](summary.json) records test counts, process/DLL architectures,
  runtime versions and all tested source hashes.
- [tests.log](tests.log) and [tests.xml](tests.xml) retain the complete combined
  device-only and native-host test result. A skip is a validation failure.
- [router-manifest.json](router-manifest.json) and
  [router-selection.json](router-selection.json) describe the original schema-1
  router cohort rebuilt for the device-only regression tests.
- [ready-router-selection.json](ready-router-selection.json) records the durable
  schema-2 policy built outside the test cache. Its M=1/2/4 native programs were
  separately checked against CPU FP64 references and changed-input graph replays.
- [artifacts](artifacts/) contains custom/host manifests, native linker logs,
  schema-2 native router policies and CUDA Graph dumps. Negative tests also
  deliberately create invalid manifests; these are raw test evidence.
  Device binaries and native DLLs remain in the local ignored work directory.
- [checksums.sha256](checksums.sha256) covers the evidence bytes.

Native-host cases validate automatic scalar/pointer/tensor arguments, dynamic
strides, shared symbolic dimensions, non-default streams, runtime host loops,
two-CTA clusters, PDL launches and a dependent two-kernel PDL sequence. The TMA
case performs a global-to-shared transfer and compares exact output after
changed-input graph replays. Its issuing warp participates collectively and
reconverges before barrier polling; bounded waits produce a failing sentinel
instead of allowing a defective test kernel to wait forever.

The native K3 router tests use the real TokenSpeed family API with
solution="auto" for M=1/2/4. Each graph must contain exactly one kernel whose
symbol matches the compiled artifact. CPU FP64 references check results;
unsupported shapes, enabled router PDL, overlap, duplicate policies, mismatched
factory provenance and corrupted DLLs exercise rejection/fallback behavior.

See the [native host guide](../../guides/windows-arm64-cutedsl-native-host.md)
for the exact reproduction command and current compatibility limits. This is
coverage of these concrete operations on RTX Spark N1X/SM12.1, not a declaration
that every upstream CuTe kernel, tensor format, host API or model is supported.

Publication normalizes user/build paths and ANSI formatting. Numerical values, sampled IDs and durations are retained. evidence-sha256.json preserves original/public identities; checksums.sha256 verifies published bytes. Tested native-host source snapshots retain their original bytes and line endings.

The [sources](sources/) directory contains the exact tested optional runtime and supporting setup/microbenchmark helpers. Restore it into the reconstructed pinned port using the guide; this PR main has a different runtime API.
