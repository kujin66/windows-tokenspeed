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

"""Add published wheels to the platform Simple API indexes."""

import argparse
import hashlib
import json
import re
from html import escape
from pathlib import Path


def update_index(
    wheelhouse: Path,
    release: dict,
    distributions: Path,
    package: str,
    variant: str,
    replace: bool = False,
) -> None:
    if variant not in ("cu130", "rocm72"):
        raise ValueError(f"Unsupported nightly variant: {variant}")
    if package == "tokenspeed-kernel-amd" and variant != "rocm72":
        raise ValueError("AMD kernel nightlies require the ROCm index")
    if package not in {"tokenspeed", "tokenspeed-kernel", "tokenspeed-kernel-amd"}:
        raise ValueError(f"Unsupported Linux nightly package: {package}")
    pattern = {
        "tokenspeed": "tokenspeed-dist/*.whl",
        "tokenspeed-kernel-amd": "tokenspeed-kernel-amd-dist/*.whl",
        "tokenspeed-kernel": (
            "tokenspeed-kernel-wheel-cu130-*/*.whl"
            if variant == "cu130"
            else "tokenspeed-kernel-rocm72-wheel-*/*.whl"
        ),
    }[package]
    expected = {path.name for path in distributions.glob(pattern)}
    _update_links(
        wheelhouse, release, expected, package, variant, replace, "lightseekorg/whl"
    )


def update_windows_index(
    wheelhouse: Path,
    release: dict,
    distributions: Path,
    package: str,
    repository: str,
    replace: bool,
    *,
    nightly: bool,
) -> None:
    if package not in {
        "tokenspeed",
        "tokenspeed-engine",
        "tokenspeed-kernel",
        "tokenspeed-scheduler",
    }:
        raise ValueError(f"Unsupported Windows package: {package}")
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9_.-]+", repository
    ) or repository.split("/")[-1] in {".", ".."}:
        raise ValueError(
            "An explicit owner/repository is required for the Windows index"
        )
    wheels = list(distributions.rglob(package.replace("-", "_") + "-*.whl"))
    assets = {asset["name"]: asset for asset in release["assets"]}
    for wheel in wheels:
        if package != "tokenspeed" and not wheel.name.endswith("-win_arm64.whl"):
            raise ValueError(f"Expected a Windows ARM64 wheel: {wheel.name}")
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
        if assets.get(wheel.name, {}).get("digest") != f"sha256:{digest}":
            raise ValueError(
                f"Published digest differs from the verified wheel: {wheel.name}"
            )
    _update_links(
        wheelhouse,
        release,
        {wheel.name for wheel in wheels},
        package,
        "win_arm64_nightly" if nightly else "win_arm64",
        replace,
        repository,
    )


def _update_links(
    wheelhouse: Path,
    release: dict,
    expected: set[str],
    package: str,
    variant: str,
    replace: bool,
    repository: str,
) -> None:
    assets = {asset["name"]: asset for asset in release["assets"]}
    if not expected or not expected <= assets.keys():
        raise ValueError("Release is missing expected wheels")

    nightly = wheelhouse / ("windows-arm64" if variant == "win_arm64" else "nightly")
    if variant == "rocm72":
        nightly /= "rocm7.2"
    elif variant == "win_arm64_nightly":
        nightly /= "windows-arm64"
    index = nightly / package / "index.html"
    previous = index.read_text() if index.exists() else "<!DOCTYPE html>\n"
    entries = []
    for name in sorted(expected):
        asset = assets[name]
        url = asset["browser_download_url"]
        digest = asset["digest"]
        if not url.startswith(f"https://github.com/{repository}/releases/download/"):
            raise ValueError(f"Wheel URL must belong to {repository}")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest or ""):
            raise ValueError(f"Missing SHA256 digest for {name}")
        existing = re.search(
            rf'<a href="[^"]+#sha256=([0-9a-f]{{64}})">{re.escape(escape(name))}</a><br>\n',
            previous,
        )
        if existing is not None:
            if existing.group(1) == digest.removeprefix("sha256:"):
                continue
            if not replace:
                raise ValueError(
                    f"Published digest changed for {name}; replacement required"
                )
            previous = previous.replace(existing.group(0), "")
            # A fresh URL also invalidates pip's cached HTTP response body.
            url += f"?sha256={digest.removeprefix('sha256:')}"
        entries.append(
            f'<a href="{escape(url)}#sha256={digest.removeprefix("sha256:")}">'
            f"{escape(name)}</a><br>\n"
        )

    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(previous + "".join(entries))

    root = nightly / "index.html"
    previous_root = root.read_text() if root.exists() else "<!DOCTYPE html>\n"
    package_link = f'<a href="{package}/">{package}</a><br>\n'
    if package_link not in previous_root:
        root.write_text(previous_root + package_link)

    if variant == "rocm72":
        nightly_root = wheelhouse / "nightly" / "index.html"
        previous_nightly_root = (
            nightly_root.read_text() if nightly_root.exists() else "<!DOCTYPE html>\n"
        )
        rocm_link = '<a href="rocm7.2/">rocm7.2</a><br>\n'
        if rocm_link not in previous_nightly_root:
            nightly_root.write_text(previous_nightly_root + rocm_link)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        choices=(
            "tokenspeed",
            "tokenspeed-kernel",
            "tokenspeed-kernel-amd",
            "tokenspeed-engine",
            "tokenspeed-scheduler",
        ),
        required=True,
    )
    parser.add_argument(
        "--variant",
        choices=("cu130", "rocm72", "win_arm64", "win_arm64_nightly"),
        required=True,
    )
    parser.add_argument(
        "--repository", help="Required source release repository for Windows indexes"
    )
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("wheelhouse", type=Path)
    parser.add_argument("release_json", type=Path)
    parser.add_argument("distributions", type=Path)
    args = parser.parse_args()
    release = json.loads(args.release_json.read_text())
    if args.variant in {"win_arm64", "win_arm64_nightly"}:
        if not args.repository:
            parser.error("--repository is required for Windows indexes")
        update_windows_index(
            args.wheelhouse,
            release,
            args.distributions,
            args.package,
            args.repository,
            args.replace,
            nightly=args.variant == "win_arm64_nightly",
        )
    else:
        if args.repository:
            parser.error("Linux nightlies retain their official release repository")
        update_index(
            args.wheelhouse,
            release,
            args.distributions,
            args.package,
            args.variant,
            args.replace,
        )
