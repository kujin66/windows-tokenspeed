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

"""Copy the original venv's extra build/runtime distributions, never project packages."""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path

NAMES = (
    "scikit-build-core",
    "nanobind",
    "tokenspeed-spdlog",
    "wheel",
    "pathspec",
    "packaging",
    "transformers",
    "flashinfer-python",
    "viztracer",
    "objprint",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-site", type=Path, required=True)
    parser.add_argument("--target-site", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_site.resolve()
    target = args.target_site.resolve()
    if source == target:
        parser.error("Source and target must differ")
    distributions = {
        dist.metadata["Name"].lower().replace("_", "-"): dist
        for dist in importlib.metadata.distributions(path=[str(source)])
    }
    records = []
    for name in NAMES:
        dist = distributions[name]
        files = []
        for entry in dist.files or []:
            path = Path(dist.locate_file(entry)).resolve()
            if not path.is_relative_to(source) or path.suffix == ".pyc":
                continue
            relative = path.relative_to(source)
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            files.append({"file": relative.as_posix(), "sha256": digest})
        records.append({"name": name, "version": dist.version, "files": files})
    args.manifest.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"Copied {len(records)} distributions; manifest: {args.manifest}")


if __name__ == "__main__":
    main()
