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

"""Select Windows tasks with the same scanner used by the Linux workflows."""

import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "test/ci_system"))

from pipeline import build_matrix


def plan(
    *,
    repo_root: Path,
    event: str,
    requested: bool,
    trusted: bool,
    runner_label: str,
    cuda_arch: str,
) -> dict:
    automatic = event in {"push", "pull_request"}
    required = trusted and (requested or automatic)
    enabled = required and bool(runner_label)
    if requested and not trusted:
        raise ValueError("Native builds require a trusted same-repository source")
    if enabled or requested:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", runner_label):
            raise ValueError("Configure a prepared Windows ARM64 runner label")
        if not re.fullmatch(r"[0-9]{2,3}[af]?", cuda_arch):
            raise ValueError(
                "Configure an explicit nvcc architecture, for example 121a"
            )
    trigger = (
        "per-commit"
        if event == "pull_request"
        else "post-submit" if event == "push" else "manual"
    )
    matrix = (
        build_matrix(
            repo_root / "test/ci",
            repo_root,
            trigger,
            runner_group="windows-arm64",
            workflow_stage="unit-test",
        )
        if enabled
        else {"include": []}
    )
    if enabled and not matrix["include"]:
        raise ValueError(
            "No Windows tasks selected; refusing an empty native validation"
        )
    return {
        "required": required,
        "enabled": enabled,
        "coverage": (
            "selected"
            if enabled
            else "missing-resource" if required else "not-applicable"
        ),
        "runner_label": runner_label,
        "cuda_arch": cuda_arch,
        "matrix": matrix,
    }


def main() -> None:
    result = plan(
        repo_root=REPO_ROOT,
        event=os.environ["GITHUB_EVENT_NAME"],
        requested=os.environ.get("NATIVE_REQUESTED") == "true",
        trusted=os.environ["TRUSTED_PR"] == "true",
        runner_label=os.environ.get("REQUESTED_RUNNER", ""),
        cuda_arch=os.environ.get("REQUESTED_ARCH", ""),
    )
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
        for key, value in result.items():
            print(
                f"{key}={json.dumps(value, separators=(',', ':')) if isinstance(value, (dict, bool)) else value}",
                file=output,
            )
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as summary:
        message = (
            "Native ARM64 CUDA tasks selected using the shared CI pipeline."
            if result["enabled"]
            else "Native CUDA validation was not run. Configure TOKENSPEED_WINDOWS_ARM64_RUNNER_LABEL and "
            "TOKENSPEED_WINDOWS_CUDA_ARCH for automatic trusted-source CI, or explicitly dispatch a prepared runner. "
            "Hosted build-tool checks do not establish native CUDA coverage."
        )
        print(message, file=summary)


if __name__ == "__main__":
    main()
