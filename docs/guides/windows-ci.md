# Windows ARM64 CI and releases

Windows uses the upstream CI task scanner, executor, result format and reusable
task workflow. Platform preparation activates MSVC and prepares the pinned
dependencies described in [the build guide](windows-arm64-build.md).

## Pull requests and test coverage

Relevant upstream pull-request workflows accept both main and
baseline/upstream-* target branches. The baseline branch keeps the compiler
port review independent of subsequent upstream changes; selecting this target
must not suppress the ordinary regression checks.

| Check | Linux | Windows ARM64 | Execution resource |
| --- | --- | --- | --- |
| Lint and build-tool regression | Yes | Build tools | GitHub-hosted |
| Scheduler C++ tests | Full suite | Same suite | GitHub-hosted |
| Scheduler Python tests | Full suite | Same suite | GitHub-hosted |
| Engine Rust tests | Full workspace | Host, metrics and scheduler libraries/binaries | GitHub-hosted |
| Native wheel build and CUDA checks | Existing upstream tasks | Four wheels, 24 CUDA groups and selected numerical/JIT checks | Prepared Windows GPU runner |
| Model, distributed and performance tests | Existing hardware-specific tasks | Not established by this compiler port | Corresponding GPU/model infrastructure |
| AMD, Ascend, K8s and Slurm | Existing upstream workflows | Not Windows coverage | Corresponding Linux hardware/cluster |

The upstream GPU workflows retain their repository and Draft-PR gates. Their
presence or a skipped result does not establish hardware coverage. Linux task
scans exclude Windows tasks; the Windows scan explicitly selects the
windows-arm64 runner group. Windows serving tasks are rejected because the
POSIX server lifecycle is not a validated Windows runtime implementation.

The native build task lives in test/ci/ut/ut-windows-arm64-build.yaml. It runs
the vendor preflight, builds and installs wheels using the package backends,
runs CTest, installed-package tests, selected CUDA numerical tests and Rust
tests. The Windows packaging helper owns packaging and PE auditing; test
selection and failure reporting belong to the shared task executor.

## Configure native GPU execution

Configure these repository variables only after preparing an available runner:

| Variable | Meaning |
| --- | --- |
| TOKENSPEED_WINDOWS_ARM64_RUNNER_LABEL | Additional label selecting the prepared Windows ARM64 CUDA runner |
| TOKENSPEED_WINDOWS_CUDA_ARCH | Explicit nvcc architecture, for example 121a |
| TOKENSPEED_CI_REPOSITORY | This repository's owner/name, when enabling upstream GPU workflows that allow an alternate CI repository |

The runner must also have self-hosted, Windows, ARM64 and tokenspeed-cuda
labels. It needs native Python 3.13, PowerShell 7.3 or later, Rust, MSVC,
CUDA, static ARM64 OpenSSL with its license and the public dependency inputs.
Keep its checkout and Cargo target directories short. The job prepares its
own Python environment; Linux host cleanup never runs on this Windows host.

When a runner label and architecture are configured, relevant pushes and
same-repository pull requests automatically select native tasks. This also
works for Draft PRs. Cross-repository pull requests run hosted checks only.
Without a configured label, the scan summary explicitly says that native
CUDA validation was not run.

For temporary verification, manually dispatch Windows ARM64 build validation
with native_cuda=true, an explicit runner_label and cuda_arch. An ephemeral
runner can execute the native job once and deregister; it does not constitute
persistent automatic CI capacity. A manual native request with incomplete
configuration fails rather than silently skipping the build.

## Releases

The existing Release TokenSpeed workflow accepts windows_arm64=true. It calls
the same Windows CI workflow, waits for hosted and native checks, downloads
the successful run's artifact and verifies the source SHA, complete task
result, clean tracked source, four-package inventory and wheel SHA256 hashes
before creating a release. Failed or partially executed tasks cannot publish.

The release target is the current repository by default. Set
TOKENSPEED_RELEASE_REPOSITORY to an owner/repository wheelhouse and provide
the TOKENSPEED_RELEASE_TOKEN secret when publication needs cross-repository
write access. The default repository uses the workflow's GitHub token. Source
provenance is retained in release notes, build.json and result.json.

For a pip Simple API index, also select publish_windows_index=true. This
reuses the upstream index publisher, validates GitHub asset digests against
the tested wheels and writes only windows-arm64/package/index.html and its
Windows package listing. Existing Linux nightly indexes are preserved.

| Variable | Meaning |
| --- | --- |
| TOKENSPEED_RELEASE_REPOSITORY | GitHub repository receiving verified release assets; defaults to this repository |
| TOKENSPEED_RELEASE_INDEX_REPOSITORY | GitHub Pages index repository; defaults to the release repository |
| TOKENSPEED_RELEASE_INDEX_BRANCH | Existing index branch; defaults to gh-pages |

The configured index branch must exist and have Pages deployment configured.
Index publication requires TOKENSPEED_RELEASE_TOKEN with write access to that
repository, including when it is the source repository: pushes using the
default GITHUB_TOKEN do not trigger a Pages build
([GitHub token behavior](https://docs.github.com/en/actions/concepts/security/github_token)).
Its pip index URL is the Pages site URL followed by
windows-arm64/. The publisher changes no root Pages document. On an index
publication failure, rerun the failed index job; verified release assets do
not need to be rebuilt or replaced.

This Windows path publishes GitHub release assets and an optional configured
Windows index. It does not publish to the official PyPI projects. Existing
Linux nightly/index controllers retain their upstream destinations and
credentials; enabling those in a fork requires ownership of the corresponding
package/index targets.

Adding workflows on a PR branch does not update the default branch. The
Actions sidebar can include historical workflow paths and dynamic entries;
its item count is not a test-coverage metric. Scheduled workflows require the
default-branch deployment and appropriate resources.
