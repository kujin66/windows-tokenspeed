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
import os
import runpy
import shutil
import sys
import zipfile
from pathlib import Path

import pytest
import yaml
from pipeline import build_matrix, execute_task, merge_env, shell_run, validate_task

ROOT = Path(__file__).resolve().parents[2]
BUILD = runpy.run_path(str(ROOT / "tools/windows/build_wheels.py"))
plan = runpy.run_path(str(ROOT / ".github/scripts/windows_ci_plan.py"))["plan"]
verify = runpy.run_path(str(ROOT / ".github/scripts/verify_windows_release.py"))[
    "verify"
]
SHA = "a" * 40


@pytest.fixture
def prepared_backend_output(tmp_path, monkeypatch):
    monkeypatch.delenv("TOKENSPEED_WINDOWS_NIGHTLY_DATE", raising=False)
    output = tmp_path / "backend-output"
    output.mkdir()
    (output / "wheels").mkdir()
    (output / "build-state.json").write_text(
        json.dumps(
            dict(
                revision=BUILD["revision"](),
                python=sys.executable,
                cuda_arch="121a",
                cuda_home=str(tmp_path / "sdk"),
                nightly_date="",
                release_candidate=os.environ.get(
                    "TOKENSPEED_KERNEL_RELEASE_CANDIDATE", ""
                ),
                source="root",
                package_versions={},
                completed={},
            )
        ),
        encoding="utf-8",
    )
    return output


def test_prepared_backend_state_cannot_be_overwritten(prepared_backend_output):
    before = (prepared_backend_output / "build-state.json").read_bytes()
    with pytest.raises(FileExistsError):
        BUILD["prepare_build"](
            prepared_backend_output, cuda_arch="100a", cuda_home="other-sdk"
        )
    assert (prepared_backend_output / "build-state.json").read_bytes() == before


def test_backend_preparation_rejects_candidate_without_nightly(tmp_path, monkeypatch):
    monkeypatch.delenv("TOKENSPEED_WINDOWS_NIGHTLY_DATE", raising=False)
    monkeypatch.setenv("TOKENSPEED_KERNEL_RELEASE_CANDIDATE", "rc1")
    output = tmp_path / "not-created"
    with pytest.raises(ValueError, match="requires an explicit nightly date"):
        BUILD["prepare_build"](output, cuda_arch="121a", cuda_home="sdk")
    assert not output.exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision", "b" * 40),
        ("python", "other-python.exe"),
        ("source", "other-layout"),
    ],
)
def test_component_build_rejects_stale_prepared_state(
    prepared_backend_output, field, value
):
    path = prepared_backend_output / "build-state.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    state[field] = value
    path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError):
        BUILD["load_context"](prepared_backend_output)


def test_component_build_rejects_changed_channel(prepared_backend_output, monkeypatch):
    monkeypatch.setenv("TOKENSPEED_WINDOWS_NIGHTLY_DATE", "20261112")
    with pytest.raises(ValueError, match="channel/date"):
        BUILD["load_context"](prepared_backend_output)


def test_component_build_rejects_changed_release_candidate(
    prepared_backend_output, monkeypatch
):
    monkeypatch.setenv("TOKENSPEED_KERNEL_RELEASE_CANDIDATE", "rc7")
    with pytest.raises(ValueError, match="release candidate"):
        BUILD["load_context"](prepared_backend_output)


def test_component_environment_does_not_inherit_another_channel(
    prepared_backend_output, monkeypatch
):
    state = BUILD["load_context"](prepared_backend_output)
    monkeypatch.setenv("TOKENSPEED_KERNEL_NIGHTLY", "true")
    monkeypatch.setenv("TOKENSPEED_KERNEL_VERSION_DATE", "19990101")
    env = BUILD["build_environment"](state, prepared_backend_output)
    assert env["TOKENSPEED_KERNEL_NIGHTLY"] == "false"
    assert env["TOKENSPEED_KERNEL_VERSION_DATE"] == ""


def test_component_finalization_requires_all_project_backends(prepared_backend_output):
    with pytest.raises(ValueError, match="Every project component"):
        BUILD["finalize_build"](prepared_backend_output)


def test_component_cannot_be_built_twice(prepared_backend_output):
    path = prepared_backend_output / "build-state.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    state["completed"]["tokenspeed-scheduler"] = {}
    path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="already built"):
        BUILD["build_component"](
            prepared_backend_output, component="tokenspeed-scheduler"
        )


def test_component_finalization_rejects_a_modified_completed_wheel(
    prepared_backend_output,
):
    state_path = prepared_backend_output / "build-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    wheels = prepared_backend_output / "wheels"
    for package in BUILD["COMPONENTS"]:
        path = wheels / f"{package.replace('-', '_')}-0.1.0-cp313-cp313-win_arm64.whl"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("test.txt", package)
        state["completed"][package] = BUILD["audit_wheel"](path)
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with zipfile.ZipFile(next(wheels.glob("*.whl")), "a") as archive:
        archive.writestr("changed.txt", "changed after successful backend build")
    with pytest.raises(ValueError, match="wheel changed"):
        BUILD["finalize_build"](prepared_backend_output)


update_windows_index = runpy.run_path(
    str(ROOT / ".github/scripts/update-nightly-index.py")
)["update_windows_index"]


def native_plan(
    *,
    event="pull_request",
    requested=False,
    trusted=True,
    runner_label="prepared-arm64",
    cuda_arch="121a",
):
    return plan(
        repo_root=ROOT,
        event=event,
        requested=requested,
        trusted=trusted,
        runner_label=runner_label,
        cuda_arch=cuda_arch,
    )


@pytest.mark.parametrize("event", ["push", "pull_request", "workflow_dispatch"])
def test_native_plan_uses_upstream_scanner(event):
    result = native_plan(event=event, requested=event == "workflow_dispatch")
    assert result["enabled"] is True
    assert [(item["name"], item["runner"]) for item in result["matrix"]["include"]] == [
        ("ut-windows-arm64-build", "windows-arm64-cuda")
    ]


@pytest.mark.parametrize("group", ["nvidia", "nvidia-arm", "nvidia-x86", "amd", "npu"])
def test_linux_scans_do_not_dispatch_windows_tasks(group):
    matrix = build_matrix(ROOT / "test/ci", ROOT, "manual", runner_group=group)
    assert all(item["runner"] != "windows-arm64-cuda" for item in matrix["include"])


def test_unconfigured_and_untrusted_sources_do_not_dispatch_gpu():
    missing = native_plan(runner_label="")
    assert missing["enabled"] is False
    assert missing["required"] is True
    assert missing["coverage"] == "missing-resource"
    assert native_plan(trusted=False)["enabled"] is False
    assert native_plan(event="workflow_dispatch")["enabled"] is False
    with pytest.raises(ValueError, match="trusted"):
        native_plan(requested=True, trusted=False)


@pytest.mark.parametrize(
    ("runner", "arch"),
    [("", "121a"), ("ready", ""), ("ready\nother", "121a"), ("ready", "121a; exit 0")],
)
def test_explicit_native_requests_require_valid_configuration(runner, arch):
    with pytest.raises(ValueError):
        native_plan(requested=True, runner_label=runner, cuda_arch=arch)


def test_windows_serving_task_is_rejected_before_posix_process_management():
    task = yaml.safe_load(
        (ROOT / "test/ci/ut/ut-windows-arm64-build.yaml").read_text(encoding="utf-8")
    )
    task["type"] = "server_smoke"
    with pytest.raises(ValueError, match="unit/build"):
        validate_task(task, Path("task.yaml"))


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="PowerShell is required")
def test_powershell_native_failure_cannot_be_hidden_by_later_command(tmp_path):
    env = merge_env({"CI_RUNNER_LABEL": "windows-arm64-cuda"})
    with pytest.raises(RuntimeError, match="command failed"):
        shell_run(
            "python -c 'import sys; sys.exit(7)'; Write-Output 'must not pass'",
            env=env,
            cwd=tmp_path,
            dry_run=False,
        )


@pytest.mark.skipif(os.name != "nt", reason="Native Windows execution")
@pytest.mark.parametrize("exit_code", [0, 7])
def test_native_task_records_real_process_result(tmp_path, exit_code):
    config = {
        "api_version": "ci.tokenspeed.io/v1",
        "name": "native-process",
        "type": "ut",
        "triggers": ["manual"],
        "runner": {"labels": ["windows-arm64-cuda"]},
        "ut": {
            "commands": [f"python -c 'import sys; print(123); sys.exit({exit_code})'"]
        },
    }
    (tmp_path / "task.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    result_path = tmp_path / "result.json"
    status = execute_task(
        config="task.yaml",
        runner="windows-arm64-cuda",
        work_dir=str(tmp_path),
        dry_run=False,
        print_plan=False,
        result_json=str(result_path),
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert status == (1 if exit_code else 0)
    assert result["ok"] is (exit_code == 0)
    assert result["executed_stages"] == ["ut"]
    if not exit_code:
        assert "123" in result["command_results"][0]["output"]


@pytest.fixture
def release_bundle(tmp_path):
    inventories = []
    for package in (
        "tokenspeed",
        "tokenspeed_engine",
        "tokenspeed_kernel",
        "tokenspeed_scheduler",
    ):
        filename = f"{package}-0.1.0-cp313-cp313-win_arm64.whl"
        with zipfile.ZipFile(tmp_path / filename, "w") as archive:
            archive.writestr(
                f"{package}-0.1.0.dist-info/METADATA",
                f"Name: {package}\nVersion: 0.1.0\n",
            )
        payload = (tmp_path / filename).read_bytes()
        inventories.append(
            {"wheel": filename, "sha256": hashlib.sha256(payload).hexdigest()}
        )
    manifest = {
        "revision": SHA,
        "wheels": inventories,
        "kernel_groups": [f"group{i}" for i in range(24)],
        "tracked_diff_sha256": hashlib.sha256(b"").hexdigest(),
    }
    result = {
        "source_sha": SHA,
        "ok": True,
        "executed_stages": ["install", "ut"],
        "task": "ut-windows-arm64-build",
        "runner": "windows-arm64-cuda",
    }
    (tmp_path / "build.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return tmp_path


def test_verified_release_contains_only_manifest_wheels_and_evidence(release_bundle):
    paths = verify(release_bundle, source_sha=SHA)
    assert len(paths) == 6
    assert len([path for path in paths if path.suffix == ".whl"]) == 4


@pytest.mark.parametrize(
    "mutation", ["failed", "wrong-source", "partial", "corrupt", "extra"]
)
def test_release_rejects_failed_stale_partial_or_modified_artifacts(
    release_bundle, mutation
):
    result_path = release_bundle / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if mutation == "failed":
        result["ok"] = False
    elif mutation == "wrong-source":
        result["source_sha"] = "b" * 40
    elif mutation == "partial":
        result["executed_stages"] = ["install"]
    elif mutation == "corrupt":
        next(release_bundle.glob("*.whl")).write_bytes(b"corrupt")
    elif mutation == "extra":
        (release_bundle / "unlisted.whl").write_bytes(b"extra")
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError):
        verify(release_bundle, source_sha=SHA)


def index_release(directory, repository):
    return {
        "assets": [
            {
                "name": path.name,
                "browser_download_url": f"https://github.com/{repository}/releases/download/v0.1.0/{path.name}",
                "digest": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in directory.glob("*.whl")
        ]
    }


def test_windows_index_reuses_upstream_writer_and_preserves_linux_history(
    release_bundle, tmp_path
):
    wheelhouse = tmp_path / "index"
    linux = wheelhouse / "nightly/index.html"
    linux.parent.mkdir(parents=True)
    linux.write_text("existing Linux index", encoding="utf-8")
    release = index_release(release_bundle, "example/wheels")
    for package in (
        "tokenspeed",
        "tokenspeed-engine",
        "tokenspeed-kernel",
        "tokenspeed-scheduler",
    ):
        update_windows_index(
            wheelhouse,
            release,
            release_bundle,
            package,
            "example/wheels",
            False,
            nightly=False,
        )
        index = wheelhouse / "windows-arm64" / package / "index.html"
        first = index.read_text(encoding="utf-8")
        assert "https://github.com/example/wheels/releases/download/" in first
        assert "#sha256=" in first
        update_windows_index(
            wheelhouse,
            release,
            release_bundle,
            package,
            "example/wheels",
            False,
            nightly=False,
        )
        assert index.read_text(encoding="utf-8") == first
    assert linux.read_text(encoding="utf-8") == "existing Linux index"
    assert (wheelhouse / "windows-arm64/index.html").read_text().count("<a href=") == 4


@pytest.mark.parametrize(
    "problem", ["repository", "digest", "missing-asset", "platform"]
)
def test_windows_index_rejects_unverified_release_before_writing(
    release_bundle, tmp_path, problem
):
    wheelhouse = tmp_path / "index"
    if problem == "platform":
        original = next(release_bundle.glob("tokenspeed_kernel-*.whl"))
        original.rename(
            original.with_name(
                original.name.replace("win_arm64", "manylinux_2_28_aarch64")
            )
        )
    release = index_release(release_bundle, "example/wheels")
    kernel = next(
        asset
        for asset in release["assets"]
        if asset["name"].startswith("tokenspeed_kernel-")
    )
    if problem == "repository":
        kernel["browser_download_url"] = kernel["browser_download_url"].replace(
            "example/wheels", "other/wheels"
        )
    elif problem == "digest":
        kernel["digest"] = "sha256:" + "b" * 64
    elif problem == "missing-asset":
        release["assets"].remove(kernel)
    with pytest.raises(ValueError):
        update_windows_index(
            wheelhouse,
            release,
            release_bundle,
            "tokenspeed-kernel",
            "example/wheels",
            False,
            nightly=False,
        )
    assert not wheelhouse.exists()


@pytest.mark.parametrize("event", ["push", "pull_request"])
def test_missing_gpu_resource_cannot_disable_required_native_coverage(event):
    missing = native_plan(event=event, runner_label="")
    assert missing["required"] is True
    assert missing["enabled"] is False
    assert missing["coverage"] == "missing-resource"


def test_cross_repository_and_hosted_manual_runs_report_not_applicable():
    for result in (native_plan(trusted=False), native_plan(event="workflow_dispatch")):
        assert result["required"] is False
        assert result["coverage"] == "not-applicable"


def test_windows_nightly_index_does_not_change_stable_or_linux_indexes(
    release_bundle, tmp_path
):
    wheelhouse = tmp_path / "index"
    stable = wheelhouse / "windows-arm64/tokenspeed/index.html"
    linux = wheelhouse / "nightly/index.html"
    for path in (stable, linux):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("existing index", encoding="utf-8")
    update_windows_index(
        wheelhouse,
        index_release(release_bundle, "example/wheels"),
        release_bundle,
        "tokenspeed",
        "example/wheels",
        False,
        nightly=True,
    )
    assert stable.read_text() == linux.read_text() == "existing index"
    assert (
        "#sha256="
        in (wheelhouse / "nightly/windows-arm64/tokenspeed/index.html").read_text()
    )
