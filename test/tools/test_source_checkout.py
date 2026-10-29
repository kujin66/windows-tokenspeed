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

"""Exercise reconstruction and rejection behavior with a real local Git source."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from tools.windows.prepare_source_checkout import (
    git,
    prepare_checkout,
    relative_path,
    verify_checkout,
)


@pytest.fixture
def source_repository(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    git(root, "config", "user.name", "Reconstruction Test")
    git(root, "config", "user.email", "test@example.com")
    (root / "python").mkdir()
    (root / "python/port.py").write_bytes(b"value = 'upstream'\n")
    (root / "python/unchanged.py").write_bytes(b"value = 1\n")
    git(root, "add", ".")
    git(root, "commit", "-m", "upstream")
    base = git(root, "rev-parse", "HEAD").decode().strip()
    before = git(root, "rev-parse", "HEAD:python/port.py").decode().strip()
    (root / "python/port.py").write_bytes(b"value = 'windows'\n")
    (root / "python/added.py").write_bytes(b"value = 'added'\n")
    git(root, "add", ".")
    git(root, "commit", "-m", "port")
    port = git(root, "rev-parse", "HEAD").decode().strip()
    bundle = root / "ports/windows-arm64"
    entries = []
    for name in ("python/port.py", "python/added.py"):
        content = (root / name).read_bytes()
        path = bundle / "overlay" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        entries.append(
            dict(
                path=name,
                base_blob=before if name == "python/port.py" else None,
                port_blob=git(root, "rev-parse", f"HEAD:{name}").decode().strip(),
                sha256=hashlib.sha256(content).hexdigest(),
            )
        )
    (bundle / "source-manifest.json").write_text(
        json.dumps(
            dict(
                upstream_base=base,
                validated_port=port,
                source_prefixes=["python"],
                files=entries,
            )
        ),
        encoding="utf-8",
    )
    (bundle / "historical-evidence-manifest.json").write_text("[]", encoding="utf-8")
    return root


def test_reconstructs_modified_added_and_unchanged_sources(source_repository, tmp_path):
    destination = tmp_path / "reconstructed"
    prepare_checkout(source_repository, destination, str(source_repository))
    assert (destination / "python/port.py").read_bytes() == b"value = 'windows'\n"
    assert (destination / "python/added.py").read_bytes() == b"value = 'added'\n"
    assert (destination / "python/unchanged.py").read_bytes() == b"value = 1\n"
    verify_checkout(source_repository / "ports/windows-arm64", destination)


def test_refuses_existing_destination_without_overwriting(source_repository, tmp_path):
    destination = tmp_path / "existing"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="must not exist"):
        prepare_checkout(source_repository, destination, str(source_repository))
    assert marker.read_text(encoding="utf-8") == "keep"


def test_rejects_bad_overlay_before_creating_checkout(source_repository, tmp_path):
    (source_repository / "ports/windows-arm64/overlay/python/port.py").write_bytes(
        b"wrong\n"
    )
    destination = tmp_path / "rejected"
    with pytest.raises(ValueError, match="Source hash mismatch"):
        prepare_checkout(source_repository, destination, str(source_repository))
    assert not destination.exists()


def test_detects_modification_to_previously_unchanged_source(
    source_repository, tmp_path
):
    destination = tmp_path / "changed"
    prepare_checkout(source_repository, destination, str(source_repository))
    (destination / "python/unchanged.py").write_bytes(b"value = 2\n")
    with pytest.raises(ValueError, match="Unexpected source differences"):
        verify_checkout(source_repository / "ports/windows-arm64", destination)


@pytest.mark.parametrize(
    "name", ["../escape.py", "/absolute.py", "C:/outside.py", "..\\escape.py"]
)
def test_rejects_escaping_paths(tmp_path, name):
    with pytest.raises(ValueError, match="Invalid relative path"):
        relative_path(tmp_path, name)
