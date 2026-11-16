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

"""Validate Windows release policy and reuse verified upstream CI artifacts."""

import argparse
import hashlib
import json
import os
import re
import runpy
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from packaging.utils import parse_wheel_filename

ROOT = Path(__file__).resolve().parents[2]


def publication_version(
    paths: list[Path], *, component: str, nightly_date: str, dry_run: bool
) -> str:
    versions = {
        str(name): version
        for path in paths
        if path.suffix == ".whl"
        for name, version, _, _ in [parse_wheel_filename(path.name)]
    }
    if component not in COMPONENTS:
        raise ValueError(f"Unsupported Windows component: {component}")
    relevant = versions.values() if component == "tokenspeed" else [versions[component]]
    if (
        not nightly_date
        and not dry_run
        and any(version.is_prerelease or version.local for version in relevant)
    ):
        raise ValueError(
            "An ordinary CI development wheel cannot be promoted as a stable release"
        )
    return str(versions[component])


def plan(
    *,
    repository: str,
    ref: str,
    windows: bool,
    nightly: bool,
    date: str,
    dry_run: bool,
    index: bool,
    token_configured: bool,
    source_run: str,
    release_candidate: str,
    replace_nightly: bool,
) -> str:
    if not windows:
        raise ValueError("This adapter only validates Windows component publication")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9_.-]+", repository):
        raise ValueError("An explicit source owner/repository is required")
    if replace_nightly:
        raise ValueError(
            "Windows publication is immutable; select a new date or release candidate"
        )
    if release_candidate and (
        not nightly
        or not re.fullmatch(r"(?:rc)?[0-9]+", release_candidate, flags=re.IGNORECASE)
    ):
        raise ValueError(
            "A Windows release candidate requires nightly mode and a number such as 1 or rc1"
        )
    if source_run and (not windows or not re.fullmatch(r"[1-9][0-9]*", source_run)):
        raise ValueError("An artifact source must be a positive Windows CI run ID")
    if (nightly or index) and not windows:
        raise ValueError("Nightly/index inputs require Windows ARM64")
    if date and not nightly:
        raise ValueError("A version date requires nightly mode")
    if (
        not dry_run
        and ref != "refs/heads/main"
        and not ref.startswith("refs/heads/release/")
    ):
        raise ValueError("Publish only from main or a release branch")
    if (
        windows
        and not nightly
        and not dry_run
        and not ref.startswith("refs/heads/release/")
    ):
        raise ValueError(
            "Stable Windows package builds require a release branch, as the upstream kernel backend does"
        )
    if index and not dry_run and not token_configured:
        raise ValueError("Pages publication requires TOKENSPEED_RELEASE_TOKEN")
    if not nightly:
        return ""
    selected = date or datetime.now(timezone.utc).strftime("%Y%m%d")
    if not re.fullmatch(r"[0-9]{8}", selected):
        raise ValueError("Nightly date must be YYYYMMDD")
    datetime.strptime(selected, "%Y%m%d")
    return selected


def verify_run(run: dict, *, source_sha: str, repository: str) -> None:
    if (
        run.get("head_sha") != source_sha
        or run.get("conclusion") != "success"
        or run.get("status") != "completed"
        or run.get("path") != ".github/workflows/windows-arm64-build.yml"
        or run.get("repository", {}).get("full_name") != repository
        or run.get("event") not in {"push", "pull_request", "workflow_dispatch"}
    ):
        raise ValueError(
            "The artifact must come from successful Windows CI for this exact source/repository"
        )


COMPONENTS = {
    "tokenspeed",
    "tokenspeed-kernel",
    "tokenspeed-engine",
    "tokenspeed-scheduler",
}


def component_metadata(
    directory: Path,
    *,
    component: str,
    source_sha: str,
    nightly_date: str,
    dry_run: bool,
) -> dict:
    if component not in COMPONENTS:
        raise ValueError(f"Unsupported Windows component: {component}")
    verify = runpy.run_path(str(ROOT / ".github/scripts/verify_windows_release.py"))[
        "verify"
    ]
    paths = verify(directory, source_sha=source_sha, nightly_date=nightly_date)
    version = publication_version(
        paths, component=component, nightly_date=nightly_date, dry_run=dry_run
    )
    selected = [
        p
        for p in paths
        if p.suffix != ".whl" or str(parse_wheel_filename(p.name)[0]) == component
    ]
    if len(selected) != 3:
        raise ValueError(
            "A component release requires one verified wheel, build metadata and test result"
        )
    prefix = f"{component}-nightly-v" if nightly_date else f"{component}-v"
    return dict(
        version=version, tag_name=f"{prefix}{version}-win_arm64", paths=selected
    )


def missing_release_assets(
    release: dict, paths: list[Path], *, source_url: str
) -> list[Path]:
    """Allow idempotent retry/addition, never replacement of published bytes."""
    if source_url not in (release.get("body") or ""):
        raise ValueError("Existing component release does not record this source")
    assets = {asset["name"]: asset for asset in release["assets"]}
    missing = []
    for path in paths:
        asset = assets.get(path.name)
        if asset is None:
            missing.append(path)
        elif (
            asset.get("digest")
            != "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        ):
            raise ValueError(f"Published asset is immutable: {path.name}")
    return missing


def gh(*arguments: str) -> str:
    return subprocess.check_output(["gh", *arguments], text=True, encoding="utf-8")


def optional_api(path: str) -> dict | None:
    result = subprocess.run(
        ["gh", "api", path], capture_output=True, text=True, encoding="utf-8"
    )
    if result.returncode == 0:
        return json.loads(result.stdout)
    if "(HTTP 404)" in result.stderr:
        return None
    raise subprocess.CalledProcessError(
        result.returncode, result.args, output=result.stdout, stderr=result.stderr
    )


def check_source_run(run_id: str) -> None:
    if not re.fullmatch(r"[1-9][0-9]*", run_id):
        raise ValueError("A positive source run ID is required")
    if run_id == os.environ["GITHUB_RUN_ID"]:
        # The caller depends on the completed shared native build and gate.
        # Whole-run success would be circular while publication jobs run.
        return
    run = json.loads(
        gh("api", f"repos/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{run_id}")
    )
    verify_run(
        run,
        source_sha=os.environ["GITHUB_SHA"],
        repository=os.environ["GITHUB_REPOSITORY"],
    )


def bool_env(name: str) -> bool:
    value = os.environ[name]
    if value not in {"true", "false"}:
        raise ValueError(f"{name} must explicitly be true or false")
    return value == "true"


def validate_component_request(*, nightly_date: str, dry_run: bool) -> None:
    plan(
        repository=os.environ["GITHUB_REPOSITORY"],
        ref=os.environ["GITHUB_REF"],
        windows=True,
        nightly=bool(nightly_date),
        date=nightly_date,
        dry_run=dry_run,
        index=bool_env("PUBLISH_INDEX"),
        token_configured=bool_env("INDEX_TOKEN_CONFIGURED"),
        source_run="",
        release_candidate="",
        replace_nightly=False,
    )


def publish_component(metadata: dict, *, source_sha: str, nightly_date: str) -> None:
    repository = os.environ["RELEASE_REPOSITORY"]
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9_.-]+", repository
    ) or repository.split("/")[-1] in {".", ".."}:
        raise ValueError("An explicit owner/repository is required")
    # A transport/authentication error is never interpreted as an absent tag.
    target = json.loads(gh("api", f"repos/{repository}"))
    if target["full_name"].casefold() != repository.casefold():
        raise ValueError("Release repository lookup returned another target")
    repository = target["full_name"]
    tag = metadata["tag_name"]
    source_url = f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}/commit/{source_sha}"
    release = optional_api(f"repos/{repository}/releases/tags/{tag}")
    if release is None:
        options = ["--prerelease"] if nightly_date else []
        if repository.casefold() == os.environ["GITHUB_REPOSITORY"].casefold():
            ref = optional_api(f"repos/{repository}/git/ref/tags/{tag}")
            if ref is None:
                gh(
                    "api",
                    "--method",
                    "POST",
                    f"repos/{repository}/git/refs",
                    "-f",
                    f"ref=refs/tags/{tag}",
                    "-f",
                    f"sha={source_sha}",
                )
            elif (
                ref["object"]["type"] != "commit" or ref["object"]["sha"] != source_sha
            ):
                raise ValueError(
                    "Existing component tag does not match the release source"
                )
            options.append("--verify-tag")
        gh(
            "release",
            "create",
            tag,
            *[str(p) for p in metadata["paths"]],
            "--repo",
            repository,
            "--title",
            f"{os.environ['COMPONENT']} {metadata['version']} (Windows ARM64)",
            "--notes",
            f"Source: {source_url}",
            *options,
        )
    else:
        missing = missing_release_assets(
            release, metadata["paths"], source_url=source_url
        )
        if missing:
            gh(
                "release",
                "upload",
                tag,
                *[str(p) for p in missing],
                "--repo",
                repository,
            )
    published = json.loads(gh("api", f"repos/{repository}/releases/tags/{tag}"))
    if missing_release_assets(published, metadata["paths"], source_url=source_url):
        raise ValueError("Component publication is incomplete")
    Path("published-release.json").write_text(
        json.dumps(published, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Windows metadata and artifact adaptation for the official component release jobs"
    )
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("plan")
    run = sub.add_parser("verify-run")
    run.add_argument("run", type=Path)
    source = sub.add_parser("source-run")
    source.add_argument("run_id")
    for name in ("metadata", "publish"):
        command = sub.add_parser(name)
        command.add_argument("--directory", type=Path, required=True)
        command.add_argument("--component", choices=sorted(COMPONENTS), required=True)
    args = parser.parse_args()
    if args.action == "source-run":
        check_source_run(args.run_id)
        return
    if args.action == "verify-run":
        verify_run(
            json.loads(args.run.read_text(encoding="utf-8")),
            source_sha=os.environ["GITHUB_SHA"],
            repository=os.environ["GITHUB_REPOSITORY"],
        )
        return
    if args.action == "plan":
        date = plan(
            repository=os.environ["GITHUB_REPOSITORY"],
            ref=os.environ["GITHUB_REF"],
            windows=bool_env("WINDOWS_ARM64"),
            nightly=bool_env("NIGHTLY"),
            date=os.environ["VERSION_DATE"],
            dry_run=bool_env("DRY_RUN"),
            index=bool_env("PUBLISH_INDEX"),
            token_configured=bool_env("INDEX_TOKEN_CONFIGURED"),
            source_run=os.environ["SOURCE_RUN_ID"],
            release_candidate=os.environ["RELEASE_CANDIDATE"],
            replace_nightly=bool_env("REPLACE_NIGHTLY"),
        )
        with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as target:
            print(f"nightly_date={date}", file=target)
        return
    dry_run = bool_env("DRY_RUN")
    if args.action == "publish" and dry_run:
        raise ValueError("A dry run cannot publish")
    date = os.environ["NIGHTLY_DATE"]
    validate_component_request(nightly_date=date, dry_run=dry_run)
    metadata = component_metadata(
        args.directory,
        component=args.component,
        source_sha=os.environ["GITHUB_SHA"],
        nightly_date=date,
        dry_run=dry_run,
    )
    if args.action == "publish":
        publish_component(
            metadata, source_sha=os.environ["GITHUB_SHA"], nightly_date=date
        )
        return
    Path("windows-assets.txt").write_text(
        "".join(f"{p.as_posix()}\n" for p in metadata["paths"]), encoding="utf-8"
    )
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as target:
        print(f"tag_name={metadata['tag_name']}", file=target)
        print(f"version={metadata['version']}", file=target)


if __name__ == "__main__":
    main()
