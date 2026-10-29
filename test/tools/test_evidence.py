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
import shutil
from pathlib import Path

import pytest

from tools.perf.verify_evidence import verify


@pytest.fixture
def evidence(tmp_path):
    source = Path(__file__).resolve().parents[2] / "docs/validation"
    for path in source.glob("windows-*20261028*"):
        shutil.copytree(path, tmp_path / "docs/validation" / path.name)
    return tmp_path


def update_manifest(path):
    manifest_path = path.parent / "evidence-sha256.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest:
        if entry["file"] == path.name:
            entry["published_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_published_evidence_recomputes(evidence):
    assert sum(verify(evidence).values()) == 152


def test_changed_artifact_rejected(evidence):
    path = (
        evidence
        / "docs/validation/windows-flashinfer-0617-20261028/attention-probe.json"
    )
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Hash mismatch"):
        verify(evidence)


def test_incorrect_derived_percentage_rejected(evidence):
    path = evidence / "docs/validation/windows-flashinfer-0617-20261028/results.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["rows"][0]["windows_flashinfer_vs_triton_percent"] += 1
    path.write_text(json.dumps(summary), encoding="utf-8")
    update_manifest(path)
    with pytest.raises(ValueError, match="Percentage mismatch"):
        verify(evidence)


def test_mislabeled_execution_mode_rejected_even_with_matching_hash(evidence):
    directory = evidence / "docs/validation/windows-flashinfer-0617-20261028"
    path = directory / "windows-flashinfer-graph.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["engine"]["enforce_eager"] = True
    path.write_text(json.dumps(report), encoding="utf-8")
    update_manifest(path)
    with pytest.raises(ValueError, match="Wrong mode"):
        verify(evidence)
