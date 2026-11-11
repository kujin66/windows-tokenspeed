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

"""Prepare dated Windows packages with the upstream runtime nightly formatter."""

import re
import runpy
import tomllib
from pathlib import Path

from packaging.version import Version


def prepare(root: Path, *, date: str, kernel_version: str) -> dict[str, str]:
    runtime = runpy.run_path(
        str(root / ".github/scripts/prepare-tokenspeed-nightly.py")
    )["prepare_nightly"]
    # Validate date/kernel pairing before changing any component metadata.
    version = runtime(root, date, kernel_version)
    kernel = Version(kernel_version)
    candidate = "" if kernel.pre is None else f"rc{kernel.pre[1]}"
    versions = {"tokenspeed": version, "tokenspeed-kernel": str(kernel)}
    for package in ("tokenspeed-engine", "tokenspeed-scheduler"):
        path = root / package / "pyproject.toml"
        text = path.read_text(encoding="utf-8")
        project = tomllib.loads(text)["project"]
        if project["name"] != package:
            raise ValueError(f"Expected {package} project")
        nightly = f"{Version(project['version']).base_version}{candidate}.post{date}"
        text, count = re.subn(
            r'^version = "[^"\n]+"$', f'version = "{nightly}"', text, flags=re.MULTILINE
        )
        if count != 1:
            raise ValueError(f"Expected one {package} version")
        path.write_text(text, encoding="utf-8")
        versions[package] = nightly
    path = root / "python/pyproject.toml"
    text = path.read_text(encoding="utf-8")
    for package in ("tokenspeed-engine", "tokenspeed-scheduler"):
        text, count = re.subn(
            '"' + package + r'[^"\n]*"', f'"{package}=={versions[package]}"', text
        )
        if count != 1:
            raise ValueError(f"Expected one {package} dependency")
    path.write_text(text, encoding="utf-8")
    return versions
