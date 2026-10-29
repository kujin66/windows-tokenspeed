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

"""Reconstruct the validated Windows source from its official base and overlay."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path, PurePosixPath


def git(root, *args):
    return subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={root.as_posix()}",
            "-c",
            "core.autocrlf=false",
            "-C",
            str(root),
            *args,
        ]
    )


def relative_path(root, name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError(f"Invalid relative path: {name}")
    result = root.joinpath(*path.parts)
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes source tree: {name}")
    return result


def verify_overlay(bundle):
    manifest = json.loads((bundle / "source-manifest.json").read_text(encoding="utf-8"))
    names = set()
    for entry in manifest["files"]:
        name = entry["path"]
        if (
            name in names
            or PurePosixPath(name).parts[0] not in manifest["source_prefixes"]
        ):
            raise ValueError(f"Duplicate or unexpected source: {name}")
        names.add(name)
        contents = relative_path(bundle / "overlay", name).read_bytes()
        blob = hashlib.sha1(f"blob {len(contents)}\0".encode() + contents).hexdigest()
        if (
            hashlib.sha256(contents).hexdigest() != entry["sha256"]
            or blob != entry["port_blob"]
        ):
            raise ValueError(f"Source hash mismatch: {name}")
    actual = {
        p.relative_to(bundle / "overlay").as_posix()
        for p in (bundle / "overlay").rglob("*")
        if p.is_file()
    }
    if actual != names:
        raise ValueError("Overlay inventory does not match the manifest")
    return manifest


def verify_history(repository):
    entries = json.loads(
        (
            repository / "ports/windows-arm64/historical-evidence-manifest.json"
        ).read_text(encoding="utf-8")
    )
    for entry in entries:
        path = relative_path(repository, entry["file"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["published_sha256"]:
            raise ValueError(f"Historical evidence hash mismatch: {entry['file']}")
    return len(entries)


def verify_checkout(bundle, destination):
    manifest = verify_overlay(bundle)
    if (
        git(destination, "rev-parse", "HEAD").decode().strip()
        != manifest["upstream_base"]
    ):
        raise ValueError("Checkout is not at the pinned upstream revision")
    changed = set(
        git(
            destination,
            "diff",
            "--name-only",
            manifest["upstream_base"],
            "--",
            *manifest["source_prefixes"],
        )
        .decode()
        .splitlines()
    )
    # New overlay files are untracked until the caller chooses to commit them.
    changed.update(
        git(
            destination,
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            *manifest["source_prefixes"],
        )
        .decode()
        .splitlines()
    )
    expected = {entry["path"] for entry in manifest["files"]}
    # test/tools is current PR support; port tests under test/runtime are pinned.
    changed = {name for name in changed if not name.startswith("test/tools/")}
    if changed != expected:
        raise ValueError(f"Unexpected source differences: {sorted(changed ^ expected)}")
    for entry in manifest["files"]:
        if (
            hashlib.sha256(
                relative_path(destination, entry["path"]).read_bytes()
            ).hexdigest()
            != entry["sha256"]
        ):
            raise ValueError(f"Restored source mismatch: {entry['path']}")
    return manifest


def prepare_checkout(repository, destination, git_source):
    bundle = repository / "ports/windows-arm64"
    manifest = verify_overlay(bundle)
    verify_history(repository)
    destination = destination.resolve()
    if destination.exists():
        raise ValueError(
            "Destination must not exist; existing checkouts are never overwritten"
        )
    destination.mkdir(parents=True)
    subprocess.run(["git", "init", str(destination)], check=True)
    git(
        destination,
        "-c",
        "core.autocrlf=false",
        "fetch",
        "--depth=1",
        "--no-tags",
        git_source,
        manifest["upstream_base"],
    )
    git(destination, "-c", "core.autocrlf=false", "checkout", "--detach", "FETCH_HEAD")
    for entry in manifest["files"]:
        name = entry["path"]
        if entry["base_blob"] is not None:
            before = git(destination, "rev-parse", f"HEAD:{name}").decode().strip()
            if before != entry["base_blob"]:
                raise ValueError(f"Upstream source mismatch: {name}")
        path = relative_path(destination, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bundle / "overlay" / name, path)
    support = [
        repository / name
        for name in (
            "tools/windows",
            "tools/perf",
            "test/tools",
            "skills/tokenspeed-windows-arm64-build",
            "ports/windows-arm64",
            "docs/validation",
        )
    ]
    support += list((repository / "docs/guides").glob("windows*.md"))
    support += [
        repository / "docs/guides/wsl2-arm64-build.md",
        repository / ".gitattributes",
    ]
    support += list((repository / "requirements").glob("windows-arm64*.txt"))
    copied = {}
    for item in support:
        paths = item.rglob("*") if item.is_dir() else [item]
        for path in paths:
            if (
                not path.is_file()
                or "__pycache__" in path.parts
                or path.suffix == ".pyc"
            ):
                continue
            name = path.relative_to(repository).as_posix()
            output = relative_path(destination, name)
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, output)
            copied[name] = hashlib.sha256(output.read_bytes()).hexdigest()
    with (destination / ".gitignore").open("a", encoding="utf-8") as stream:
        stream.write(
            "\n.port/\ntokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/objs/\n"
        )
    (destination / ".port").mkdir()
    snapshots = repository / "docs/validation/windows-flashinfer-0617-20261028/scripts"
    for path in snapshots.glob("*"):
        if path.is_file():
            shutil.copyfile(path, destination / ".port" / path.name)
    environment = repository / "tools/perf/wsl-env.sh"
    if environment.is_file():
        shutil.copyfile(environment, destination / ".port/wsl-env.sh")
    verify_checkout(bundle, destination)
    record = dict(
        upstream_base=manifest["upstream_base"],
        validated_port=manifest["validated_port"],
        source_files=len(manifest["files"]),
        support_files=copied,
    )
    (destination / ".port/source-reconstruction.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("verify", "create"), required=True)
    parser.add_argument("--destination", type=Path)
    parser.add_argument(
        "--git-source",
        help="Official upstream URL or a local Git repository containing the pinned base",
    )
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    if args.action == "create":
        if args.destination is None or args.git_source is None:
            parser.error("create requires --destination and --git-source")
        record = prepare_checkout(repository, args.destination, args.git_source)
        print(
            json.dumps(
                {
                    "destination": str(args.destination),
                    "source_files": record["source_files"],
                }
            )
        )
    else:
        manifest = (
            verify_checkout(repository / "ports/windows-arm64", args.destination)
            if args.destination is not None
            else verify_overlay(repository / "ports/windows-arm64")
        )
        print(
            json.dumps(
                {
                    "verified_source_files": len(manifest["files"]),
                    "historical_evidence_files": verify_history(repository),
                    "validated_port": manifest["validated_port"],
                }
            )
        )


if __name__ == "__main__":
    main()
