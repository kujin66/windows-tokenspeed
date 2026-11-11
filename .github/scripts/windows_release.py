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
import json
import os
import re
import runpy
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from packaging.utils import parse_wheel_filename

ROOT = Path(__file__).resolve().parents[2]


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
) -> str:
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
    if not windows and repository != "lightseekorg/tokenspeed":
        raise ValueError(
            "The upstream release page requires the official source repository"
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("plan")
    run = sub.add_parser("verify-run")
    run.add_argument("run", type=Path)
    bundle = sub.add_parser("metadata")
    bundle.add_argument("--directory", type=Path)
    args = parser.parse_args()
    if args.action == "verify-run":
        verify_run(
            json.loads(args.run.read_text()),
            source_sha=os.environ["GITHUB_SHA"],
            repository=os.environ["GITHUB_REPOSITORY"],
        )
        return
    output = Path(os.environ["GITHUB_OUTPUT"])
    if args.action == "plan":
        date = plan(
            repository=os.environ["GITHUB_REPOSITORY"],
            ref=os.environ["GITHUB_REF"],
            windows=os.environ["WINDOWS_ARM64"] == "true",
            nightly=os.environ["NIGHTLY"] == "true",
            date=os.environ["VERSION_DATE"],
            dry_run=os.environ["DRY_RUN"] == "true",
            index=os.environ["PUBLISH_INDEX"] == "true",
            token_configured=os.environ["INDEX_TOKEN_CONFIGURED"] == "true",
            source_run=os.environ["SOURCE_RUN_ID"],
        )
        with output.open("a", encoding="utf-8") as target:
            print(f"nightly_date={date}", file=target)
        return
    paths = []
    if args.directory:
        verify = runpy.run_path(
            str(ROOT / ".github/scripts/verify_windows_release.py")
        )["verify"]
        paths = verify(
            args.directory,
            source_sha=os.environ["GITHUB_SHA"],
            nightly_date=os.environ["NIGHTLY_DATE"],
        )
        runtime = next(path for path in paths if path.name.startswith("tokenspeed-"))
        version = str(parse_wheel_filename(runtime.name)[1])
    else:
        version = tomllib.loads((ROOT / "python/pyproject.toml").read_text())[
            "project"
        ]["version"]
    prefix = "tokenspeed-windows-arm64-nightly-v" if os.environ["NIGHTLY_DATE"] else "v"
    Path("windows-assets.txt").write_text(
        "".join(f"{path.as_posix()}\n" for path in paths), encoding="utf-8"
    )
    with output.open("a", encoding="utf-8") as target:
        print(f"tag_name={prefix}{version}", file=target)
        print(f"version={version}", file=target)


if __name__ == "__main__":
    main()
