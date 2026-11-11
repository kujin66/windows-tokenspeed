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

"""Verify native CI evidence before publishing its Windows distributions."""

import argparse
import hashlib
import json
from pathlib import Path

PACKAGES = {
    "tokenspeed",
    "tokenspeed_engine",
    "tokenspeed_scheduler",
    "tokenspeed_kernel",
}


def verify(directory: Path, *, source_sha: str) -> list[Path]:
    def read_one(name: str) -> tuple[Path, dict]:
        matches = list(directory.rglob(name))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {name}; found {len(matches)}")
        return matches[0], json.loads(matches[0].read_text(encoding="utf-8"))

    manifest_path, manifest = read_one("build.json")
    result_path, result = read_one("result.json")
    if manifest["revision"] != source_sha or result.get("source_sha") != source_sha:
        raise ValueError("Build and test evidence must match the release source SHA")
    if manifest.get("tracked_diff_sha256") != hashlib.sha256(b"").hexdigest():
        raise ValueError(
            "Release wheels must be built from an unmodified source checkout"
        )
    if result.get("ok") is not True or result.get("executed_stages") != [
        "install",
        "ut",
    ]:
        raise ValueError("The complete native task must pass before publication")
    if (
        result.get("task") != "ut-windows-arm64-build"
        or result.get("runner") != "windows-arm64-cuda"
    ):
        raise ValueError("Expected the native Windows build task result")
    if len(set(manifest["kernel_groups"])) != 24:
        raise ValueError("Expected all 24 native CUDA groups")
    inventories = manifest["wheels"]
    if len(inventories) != 4:
        raise ValueError("Expected four project wheels")
    paths = []
    packages = set()
    for item in inventories:
        filename = item["wheel"]
        if Path(filename).name != filename or not filename.endswith(".whl"):
            raise ValueError("Invalid wheel filename in manifest")
        matches = list(directory.rglob(filename))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {filename}")
        path = matches[0]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Wheel SHA256 mismatch: {filename}")
        package = filename.split("-", 1)[0]
        if package in packages:
            raise ValueError(f"Duplicate wheel package: {package}")
        packages.add(package)
        if package != "tokenspeed" and not filename.endswith("-win_arm64.whl"):
            raise ValueError(f"Expected a Windows ARM64 native wheel: {filename}")
        paths.append(path)
    if packages != PACKAGES or len(list(directory.rglob("*.whl"))) != 4:
        raise ValueError(
            "Release inventory must contain exactly the four project packages"
        )
    return [*paths, manifest_path, result_path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    for path in verify(args.directory, source_sha=args.source_sha):
        print(path.as_posix())


if __name__ == "__main__":
    main()
