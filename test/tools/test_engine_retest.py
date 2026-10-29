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

"""Validate real published cohorts, aggregation and incompatible-result rejection."""

import json
import shutil
from pathlib import Path

import pytest

from tools.perf.summarize_engine_retest import EVIDENCE, summarize


@pytest.fixture
def evidence(tmp_path):
    root = Path(__file__).resolve().parents[2]
    for name in EVIDENCE:
        shutil.copytree(
            root / "docs/validation" / name, tmp_path / "docs/validation" / name
        )
    return tmp_path


def rewrite(path, value):
    import hashlib

    path.write_text(json.dumps(value), encoding="utf-8")
    directory = path.parent
    checks = directory / "checksums.sha256"
    lines = checks.read_text().splitlines()
    checks.write_text(
        "\n".join(
            (
                hashlib.sha256(path.read_bytes()).hexdigest() + "  " + path.name
                if line.split("  ", 1)[1] == path.name
                else line
            )
            for line in lines
        )
        + "\n"
    )


def test_pooled_rates_use_all_measured_seconds(evidence):
    result = summarize(evidence)
    reports = [
        json.loads(
            (
                evidence
                / "docs/validation"
                / EVIDENCE[1]
                / f"windows-flashinfer-repeat{r}.json"
            ).read_text()
        )
        for r in (1, 2)
    ]
    for row in result["throughput"]:
        key = row["input_length"], row["batch_size"]
        samples = [
            sample
            for report in reports
            for case in report["cases"]
            if (case["input_length"], case["batch_size"]) == key
            for sample in case["samples"]
        ]
        expected = (
            128 * key[1] * len(samples) / sum(sample["seconds"] for sample in samples)
        )
        assert row["windows-tokenspeed-flashinfer"] == expected
    assert result["cute_validation"][EVIDENCE[4]]["tests"] == 33


@pytest.mark.parametrize("field", ["packages", "engine_config"])
def test_different_environment_or_graph_policy_rejected(evidence, field):
    path = (
        evidence / "docs/validation" / EVIDENCE[1] / "windows-flashinfer-repeat1.json"
    )
    report = json.loads(path.read_text())
    if field == "packages":
        report[field]["torch"] = "different-build"
    else:
        report[field]["enforce_eager"] = True
    rewrite(path, report)
    with pytest.raises(ValueError, match="Changed environment/workload"):
        summarize(evidence)


def test_short_generation_rejected_even_with_updated_file_hash(evidence):
    path = evidence / "docs/validation" / EVIDENCE[1] / "wsl2-flashinfer-repeat2.json"
    report = json.loads(path.read_text())
    report["cases"][0]["samples"][0]["output_ids"][0].pop()
    rewrite(path, report)
    with pytest.raises(ValueError, match="Wrong output count"):
        summarize(evidence)


def test_evidence_corruption_rejected(evidence):
    path = evidence / "docs/validation" / EVIDENCE[1] / "windows-flashinfer-repeat1.log"
    path.write_text("corrupted")
    with pytest.raises(ValueError, match="Hash mismatch"):
        summarize(evidence)


def test_x64_compiler_not_misreported_as_native_arm64(evidence):
    path = evidence / "docs/validation" / EVIDENCE[4] / "summary.json"
    report = json.loads(path.read_text())
    report["compiler_python_pe_machine"] = "0xaa64"
    rewrite(path, report)
    with pytest.raises(ValueError, match="Wrong compiler/execution architecture"):
        summarize(evidence)
