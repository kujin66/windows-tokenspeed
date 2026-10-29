# Windows ARM64 CuTe DSL AOT evidence

See the [method, results and reproduction commands](../../guides/windows-arm64-cutedsl-aot.md).

- manifest.json records 16 architecture-specific GPU exports, source/exporter
  hashes, compiler version, launch geometry, pointer ABI and CUBIN/PTX hashes.
- windows-results.json is the final native Windows ARM64 run: correctness,
  rejected inputs/artifacts, raw non-eager samples and independent tuning
  confirmation. Helper hashes identify the exact tested source files.
- windows-results-first-confirmation.json retains the previous full run,
  including its M=2 regression. The earlier helper called the count of captured
  operations graph_nodes; the count is 100 operations, not 100 physical kernel
  nodes for every backend. It used the same timing method and tolerances.
- graph-kernels.json contains actual one-operation graph kernel labels from
  the final graph inspection helper. It confirms one CuTe launch in each case,
  one PyTorch/cuBLAS launch at M=1, and two at M=2/4.
- checksums.sha256 hashes the four JSON evidence files.

CUBINs, PTX and full graph DOT files are generated under the ignored
.port/cutedsl-history/router-aot-optimized directory. They are reproduced by
the exporter rather than committed. All 16 final CUBIN hashes matched those
from the first confirmation run. Timing varied across runs despite identical
device binaries; these observations do not establish fixed-clock performance,
cold-cache behavior or end-to-end model speedup.

Publication normalizes user/build paths and ANSI formatting. Numerical values, sampled IDs and durations are retained. evidence-sha256.json preserves original/public identities; checksums.sha256 verifies published bytes. Tested native-host source snapshots retain their original bytes and line endings.
