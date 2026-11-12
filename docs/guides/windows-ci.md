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
finish jobs list skipped stages as not executed, and require selected stages
to succeed when the source is eligible. A failed scanner cannot pass the gate.
Their presence or a skipped result does not establish hardware coverage. Linux task
scans exclude Windows tasks; the Windows scan explicitly selects the
windows-arm64 runner group. Windows serving tasks are rejected because the
POSIX server lifecycle is not a validated Windows runtime implementation.

Forks must explicitly opt in to upstream Linux GPU execution through
TOKENSPEED_CI_REPOSITORY after preparing those runners. Making a PR ready for
review does not enable unavailable AMD/NVIDIA hardware. Hosted scans still run
and their summaries state that GPU/model coverage was not executed. The Windows
required native gate is independent and remains mandatory for relevant trusted
sources even while the PR is a Draft.

The native build task lives in test/ci/ut/ut-windows-arm64-build.yaml. It runs
the vendor preflight, builds and installs wheels using the package backends,
runs CTest, installed-package tests, selected CUDA numerical tests and Rust
tests. The Windows packaging helper owns packaging and PE auditing; test
selection and failure reporting belong to the shared task executor.
Wheel builds and the native Rust test command share bounded recovery for the
specific Windows Rust temporary-archive sharing violation (OS error 32).
Compilation errors, test failures and persistent locks still fail validation.
Rerunning a failed native job replaces its CI artifact; published release
assets remain immutable.

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
Without a configured label, a relevant trusted push or PR fails the Windows
ARM64 required validation gate with missing-resource coverage. An untrusted
cross-repository PR or an explicitly hosted-only manual run reports native
coverage as not applicable. Draft status does not waive Windows validation.

For temporary verification, manually dispatch Windows ARM64 build validation
with native_cuda=true, an explicit runner_label and cuda_arch. An ephemeral
runner can execute the native job once and deregister; it does not constitute
persistent automatic CI capacity. A manual native request with incomplete
configuration fails rather than silently skipping the build.

## Component builds and releases

The upstream Release TokenSpeed workflow (release-tokenspeed.yml) keeps its
original tag/Release-page role. Windows package validation and publication do
not replace that entry point or its default behavior.

The existing Build and Release TokenSpeed workflow (release-pypi.yml) accepts
windows_arm64=true. Its Windows path selects the date once, calls the shared
Windows CI task once, then consumes the successful artifact through the existing
kernel, scheduler and engine release workflows, followed by its runtime job.
Those component workflows expose an explicit Windows workflow_call contract;
their original Linux workflow_dispatch inputs, builds and PyPI publication
remain separate. Official Linux publication stays restricted to the official
repository, and Linux scheduled component jobs are not run in an unconfigured
fork. No unavailable Linux GPU, NPU or cluster is represented as Windows coverage.

The shared native task owns the component sequence: prepare source metadata,
build scheduler, engine, kernel and runtime through their project backends, then
finalize/audit/install the four wheels before all existing C++/package/CUDA/Rust
tests. The Windows build helper adapts backend arguments and paths; it does not
select tasks or silently orchestrate another four-package build loop. Component
receipts reject stale source/interpreter/date state, missing components and
wheel modification before finalization.
The runtime backend receives README/LICENSE in a separate source staging area,
matching the upstream runtime build. This supplies package metadata without
changing tracked checkout files or archiving unrelated native source for an
ordinary runtime wheel.

The component release jobs share a platform composite action that downloads the
tested package set, verifies its source SHA, complete task result, clean tracked
source, four-package inventory, wheel metadata and hashes, then selects one
component wheel plus its build/test evidence. A failed or partial native task
cannot publish any component. The component calls are sequenced to avoid
concurrent index writes. Their concurrency groups must not equal the enclosing
release-pypi group: reusable workflows inherit the caller's github.workflow.

The caller supplies platform=windows, source_run_id, nightly_date (or empty),
dry_run and publish_windows_index explicitly. Internal calls use the completed
shared build from the same parent run; requiring whole-run success would be
circular while its publication jobs run. A different run must be a successful
exact-source Windows validation run before download, and the artifact itself
must still pass verification. A PR merge artifact cannot be relabeled as an
unrelated release commit. Development wheels are not promoted to stable.

Windows release validation remains dry-run by default. A dry run writes a
separate publication-plan artifact for each component and creates no tags,
releases or indexes. The original component entry points still provide their
Linux build paths; Windows calls consume the validated shared artifact instead
of independently rebuilding the same package set in every release job.

### Windows-specific version policy retained for review

This CI-only refactor does not silently change the existing Windows package
dependency policy. Runtime/kernel nightly formatting calls the upstream helper.
The existing Windows adapter also gives engine/scheduler the same date/RC suffix
and pins all three native components from the runtime. These extra two suffixes
are Windows-specific behavior, not an unchanged upstream Linux rule. They keep
the tested same-source package set explicit. An alternative that consumes
separately published stable Windows dependencies must be reviewed and validated
before changing this contract. Retain the current policy only as an explicit
review item before enabling actual publication; changing it requires its own
compatibility evidence.

Nightly metadata is prepared in a Git archive of the exact source revision;
the original checkout remains clean. Archive symlinks are staged as link-text
files, matching Windows core.symlinks=false and the Rust scheduler source
fallback. Tests run against installed wheels. Publication verifies all four
dates, source and runtime pins, including forwarded release candidates.
Windows replace_nightly requests remain rejected before building.

Each component uses a platform-qualified tag:
component-nightly-vVERSION-win_arm64, or component-vVERSION-win_arm64 for an
ordinary release. This is separate from the upstream runtime tag/Release page.
An existing component tag/source or asset digest cannot be overwritten.
Rerunning a failed publication/index job may add missing assets or retain
identical bytes from its original artifact; rebuilding and replacing a published
wheel with different bytes is rejected. No migration or rewrite of old tags
is performed by this refactor.

The ordinary index remains windows-arm64/ and the nightly index remains
nightly/windows-arm64/. The upstream index writer verifies GitHub asset digests
and updates only the selected component links. Linux indexes are preserved.
On index failure, rerun the failed release/index job with the original artifact;
do not rebuild or overwrite published assets.

| Variable | Meaning |
| --- | --- |
| TOKENSPEED_WINDOWS_RELEASES | Include Windows in the upstream daily runtime release schedule |
| TOKENSPEED_WINDOWS_PUBLISH | Publish scheduled Windows assets; otherwise scheduled runs are dry runs |
| TOKENSPEED_WINDOWS_INDEX | Publish scheduled Windows index links; requires target and token |
| TOKENSPEED_RELEASE_REPOSITORY | Owned asset repository; defaults to the current repository |
| TOKENSPEED_RELEASE_INDEX_REPOSITORY | Owned Pages index repository; defaults to the asset repository |
| TOKENSPEED_RELEASE_INDEX_BRANCH | Existing index branch; defaults to gh-pages |

Cross-repository assets require TOKENSPEED_RELEASE_TOKEN with write access.
Actual index publication requires that token even for the source repository:
default GITHUB_TOKEN pushes do not trigger a Pages build
([GitHub token behavior](https://docs.github.com/en/actions/concepts/security/github_token)).
Pages and the index branch must already exist. Windows packages are not uploaded
to the official PyPI projects by this path. All actual publication/schedule
switches remain off during branch validation and human review.

### Cache boundary

Keep the prepared Cargo cache and the project backends' own cache checks. This
change does not introduce a cache of test results or blindly restore compiled
CUDA libraries across jobs. The current backend identity plus source mtimes is
not a sufficient cross-job cache key for changed headers. A future binary cache
must include all native source/header inputs, dependency content, compiler/SDK,
Python/ABI and architecture, and must still run every required test. Until that
key is independently verified, correctness takes precedence over cache hits.

## Deploy and verify the default branch

First validate the PR source, its hosted jobs and the required Windows GPU
task. Run the existing runtime release workflow in dry-run mode on that same
branch, including the nightly channel. The already-registered parent can load
its reusable component workflows and platform action from the selected commit;
they do not need an earlier default-branch merge. A missing Actions menu entry
is not permission to deploy unvalidated code. The compiler PR targets
an upstream baseline; merging it there alone does not deploy main. A deployment
PR to main must expose the upstream synchronization separately from Windows
adaptation, because the fork's older main is not the compiler baseline.
During fork synchronization, the attribution check fetches the official main
history and preserves commits reachable from its merge base with the PR source.
It still checks newly contributed commits and verifies the current PR head.
Historical upstream co-author addresses are not rewritten to satisfy a fork's
current account lookup.

Keep the upstream fake-PD-peer test agreement fix (43c3e740) with the CI
baseline: held cancellation must retire on the same round across ranks. This
prevents the known Rust serving-test collective desynchronization without
changing production control-plane code or weakening pause/memory assertions.

After deployment, inspect the default-branch workflow inventory and Actions API
registration separately. Dispatch non-publishing validation workflows as needed;
do not dispatch publication or maintenance workflows solely to populate the
sidebar. Enable schedules only after their runner and destination prerequisites
are available. Keep unavailable AMD, NPU and cluster coverage explicit.

Adding workflows on a PR branch does not update the default branch. The
Actions sidebar can include historical workflow paths and dynamic entries;
its item count is not a test-coverage metric. Scheduled workflows require the
default-branch deployment and appropriate resources.
