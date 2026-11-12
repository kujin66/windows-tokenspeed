# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import hashlib
import json
import runpy
import shutil
import subprocess
import tomllib
import zipfile
from pathlib import Path

import pytest
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[2]
RELEASE = runpy.run_path(str(ROOT / ".github/scripts/windows_release.py"))
PREPARE = runpy.run_path(str(ROOT / "tools/windows/prepare_nightly.py"))["prepare"]
VERIFY = runpy.run_path(str(ROOT / ".github/scripts/verify_windows_release.py"))[
    "verify"
]
SHA = "a" * 40
STAGE = runpy.run_path(str(ROOT / "tools/windows/build_wheels.py"))["stage_git_source"]


def test_nightly_archive_preserves_windows_git_link_blobs(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()

    def git(*arguments):
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={repository.as_posix()}", *arguments],
            cwd=repository,
            text=True,
        ).strip()

    git("init", "--quiet")
    scheduler = repository / "tokenspeed-scheduler"
    scheduler.mkdir()
    (scheduler / "CMakeLists.txt").write_text("project(scheduler)\n")
    link_name = "tokenspeed-engine/crates/ts-scheduler/scheduler"
    link = repository / link_name
    link.parent.mkdir(parents=True)
    link.write_text("../../../tokenspeed-scheduler")
    git("add", ".")
    blob = git("hash-object", "-w", link_name)
    git("update-index", "--cacheinfo", "120000", blob, link_name)
    git(
        "-c",
        "user.name=CI fixture",
        "-c",
        "user.email=ci@example.invalid",
        "commit",
        "--quiet",
        "--no-gpg-sign",
        "--signoff",
        "-m",
        "Archive fixture",
    )
    destination = tmp_path / "source"
    destination.mkdir()
    STAGE(repository, destination)
    # Match a Windows Git checkout with core.symlinks=false: the Rust build
    # uses the root scheduler when the bundled path is a link-text file.
    staged_link = destination / link_name
    assert staged_link.read_text() == "../../../tokenspeed-scheduler"
    assert not (staged_link / "CMakeLists.txt").is_file()
    scheduler_source = staged_link.parent / staged_link.read_text()
    assert scheduler_source.resolve(strict=True) == (
        destination / "tokenspeed-scheduler"
    ).resolve(strict=True)
    assert (scheduler_source / "CMakeLists.txt").read_text() == "project(scheduler)\n"


def release_plan(**overrides):
    values = dict(
        repository="example/source",
        ref="refs/heads/main",
        windows=True,
        nightly=True,
        date="20261111",
        dry_run=True,
        index=False,
        token_configured=False,
        source_run="",
        release_candidate="",
        replace_nightly=False,
    )
    values.update(overrides)
    return RELEASE["plan"](**values)


@pytest.mark.parametrize(
    "overrides",
    [
        {"date": "20260230"},
        {"date": "2026-11-11"},
        {"nightly": False},
        {"source_run": "1; echo bad"},
        {"windows": False},
        {"dry_run": False, "ref": "refs/heads/build/test"},
        {"dry_run": False, "index": True},
    ],
)
def test_invalid_release_request_is_rejected_before_build_or_publication(overrides):
    with pytest.raises(ValueError):
        release_plan(**overrides)


def test_dry_run_allows_review_branch_without_publishing_credentials():
    assert release_plan(ref="refs/heads/build/test", index=True) == "20261111"


def test_main_publication_requires_index_credentials_only_when_index_is_selected():
    assert release_plan(dry_run=False) == "20261111"
    assert release_plan(dry_run=False, index=True, token_configured=True) == "20261111"


@pytest.fixture
def nightly_project(tmp_path):
    for name in (
        "python/pyproject.toml",
        "python/tokenspeed/version.py",
        "tokenspeed-engine/pyproject.toml",
        "tokenspeed-scheduler/pyproject.toml",
        ".github/scripts/prepare-tokenspeed-nightly.py",
    ):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    return tmp_path


@pytest.mark.parametrize("kernel", ["0.1.4.post20261111", "0.1.4rc2.post20261111"])
def test_windows_nightly_reuses_upstream_versions_and_pins_all_three_components(
    nightly_project, kernel
):
    versions = PREPARE(nightly_project, date="20261111", kernel_version=kernel)
    assert versions["tokenspeed-kernel"] == kernel
    assert len(versions) == 4
    for package, version in versions.items():
        assert version.endswith(".post20261111")
    project = tomllib.loads((nightly_project / "python/pyproject.toml").read_text())[
        "project"
    ]
    dependencies = {
        r.name: str(r.specifier)
        for value in project["dependencies"]
        for r in [Requirement(value)]
    }
    for package in ("tokenspeed-kernel", "tokenspeed-engine", "tokenspeed-scheduler"):
        assert dependencies[package] == f"=={versions[package]}"
    assert (
        runpy.run_path(str(nightly_project / "python/tokenspeed/version.py"))[
            "__version__"
        ]
        == versions["tokenspeed"]
    )


def test_mismatched_nightly_leaves_all_metadata_unchanged(nightly_project):
    before = {
        path: path.read_bytes() for path in nightly_project.rglob("*") if path.is_file()
    }
    with pytest.raises(ValueError):
        PREPARE(nightly_project, date="20261111", kernel_version="0.1.4.post20261110")
    assert before == {path: path.read_bytes() for path in before}


@pytest.fixture
def nightly_bundle(tmp_path):
    versions = {
        "tokenspeed": "0.1.1.post20261111",
        "tokenspeed-kernel": "0.1.4.post20261111",
        "tokenspeed-engine": "0.1.0.post20261111",
        "tokenspeed-scheduler": "0.1.52.post20261111",
    }
    inventory = []
    for package, version in versions.items():
        name = package.replace("-", "_")
        filename = f"{name}-{version}-cp313-cp313-win_arm64.whl"
        metadata = f"Name: {package}\nVersion: {version}\n"
        if package == "tokenspeed":
            metadata += "".join(
                f"Requires-Dist: {p}=={v}\n"
                for p, v in versions.items()
                if p != package
            )
        with zipfile.ZipFile(tmp_path / filename, "w") as archive:
            archive.writestr(f"{name}-{version}.dist-info/METADATA", metadata)
        inventory.append(
            {
                "wheel": filename,
                "sha256": hashlib.sha256(
                    (tmp_path / filename).read_bytes()
                ).hexdigest(),
            }
        )
    (tmp_path / "build.json").write_text(
        json.dumps(
            dict(
                revision=SHA,
                nightly_date="20261111",
                package_versions=versions,
                wheels=inventory,
                kernel_groups=[str(i) for i in range(24)],
                tracked_diff_sha256=hashlib.sha256(b"").hexdigest(),
            )
        )
    )
    (tmp_path / "result.json").write_text(
        json.dumps(
            dict(
                source_sha=SHA,
                ok=True,
                executed_stages=["install", "ut"],
                task="ut-windows-arm64-build",
                runner="windows-arm64-cuda",
            )
        )
    )
    return tmp_path


def test_coordinated_nightly_bundle_is_accepted(nightly_bundle):
    assert len(VERIFY(nightly_bundle, source_sha=SHA, nightly_date="20261111")) == 6


@pytest.mark.parametrize("date", ["", "20261110"])
def test_wrong_release_channel_or_date_cannot_promote_a_nightly(nightly_bundle, date):
    with pytest.raises(ValueError, match="channel/date"):
        VERIFY(nightly_bundle, source_sha=SHA, nightly_date=date)


def test_rehashed_but_mismatched_runtime_pin_is_rejected(nightly_bundle):
    path = next(nightly_bundle.glob("tokenspeed-*.whl"))
    with zipfile.ZipFile(path) as archive:
        name = archive.namelist()[0]
        metadata = (
            archive.read(name)
            .decode()
            .replace(
                "tokenspeed-kernel==0.1.4.post20261111", "tokenspeed-kernel>=0.1.4"
            )
        )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(name, metadata)
    manifest_path = nightly_bundle / "build.json"
    manifest = json.loads(manifest_path.read_text())
    next(item for item in manifest["wheels"] if item["wheel"] == path.name)[
        "sha256"
    ] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="pin"):
        VERIFY(nightly_bundle, source_sha=SHA, nightly_date="20261111")


@pytest.mark.parametrize(
    "change",
    [
        {"head_sha": "b" * 40},
        {"conclusion": "failure"},
        {"path": ".github/workflows/other.yml"},
        {"event": "pull_request_target"},
        {"repository": {"full_name": "other/source"}},
    ],
)
def test_cross_source_or_failed_artifact_run_cannot_be_promoted(change):
    run = dict(
        head_sha=SHA,
        conclusion="success",
        status="completed",
        path=".github/workflows/windows-arm64-build.yml",
        event="push",
        repository={"full_name": "example/source"},
    )
    RELEASE["verify_run"](run, source_sha=SHA, repository="example/source")
    run.update(change)
    with pytest.raises(ValueError):
        RELEASE["verify_run"](run, source_sha=SHA, repository="example/source")


@pytest.mark.parametrize(
    "overrides",
    [
        {"release_candidate": "bad"},
        {"release_candidate": "rc1", "nightly": False, "date": ""},
        {"replace_nightly": True},
        {"nightly": False, "date": "", "dry_run": False},
    ],
)
def test_unsupported_publication_inputs_fail_before_building(overrides):
    with pytest.raises(ValueError):
        release_plan(**overrides)


def test_release_candidate_and_stable_release_branch_follow_upstream_rules():
    assert release_plan(release_candidate="rc2") == "20261111"
    assert (
        release_plan(
            nightly=False, date="", dry_run=False, ref="refs/heads/release/windows"
        )
        == ""
    )


def test_stable_publication_cannot_relabel_an_ordinary_ci_development_wheel():
    paths = [
        Path("tokenspeed-0.1.1-py3-none-any.whl"),
        Path(
            "tokenspeed_kernel-0.1.4.dev20261111+gitaaaaaaaa-cp313-cp313-win_arm64.whl"
        ),
    ]
    with pytest.raises(ValueError, match="development wheel"):
        RELEASE["publication_version"](paths, nightly_date="", dry_run=False)
    assert (
        RELEASE["publication_version"](paths, nightly_date="", dry_run=True) == "0.1.1"
    )
