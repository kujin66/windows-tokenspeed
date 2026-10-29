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

"""Validate archived evidence and derive the latest cross-cohort performance summary."""

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from tools.perf.compare_engine_matrix import compare

EVIDENCE = (
    "engine-matrix-retest-20261029",
    "flashinfer-paired-retest-20261029",
    "windows-cutedsl-aot",
    "windows-cutedsl-local-compiler",
    "windows-cutedsl-native-host",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_hashes(directory):
    count = 0
    for line in (directory / "checksums.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = (directory / name).resolve()
        require(path.is_relative_to(directory.resolve()), "Invalid evidence path")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == expected, f"Hash mismatch: {directory.name}/{name}")
        count += 1
    return count


def case_map(report):
    cases = {(c["input_length"], c["batch_size"]): c for c in report["cases"]}
    require(len(cases) == len(report["cases"]) == 4, "Missing/duplicate cases")
    require(set(cases) == {(128, 1), (128, 4), (512, 1), (512, 4)}, "Wrong cases")
    return cases


def validate_repeat(report, baseline):
    require(report["passed"], "Incomplete repeat")
    for key in (
        "engine_name",
        "backend",
        "execution",
        "platform",
        "benchmark_sha256",
        "packages",
        "sources",
        "model_files",
        "gpu",
        "seed",
        "warmup",
        "rounds",
        "input_lengths",
        "batch_sizes",
        "output_length",
        "engine_config",
        "sampling",
        "graph_policy",
        "source_root_override",
        "python",
        "os",
        "torch_cuda",
    ):
        require(report[key] == baseline[key], f"Changed environment/workload: {key}")
    require(report["warmup"] == 3 and report["rounds"] == 5, "Wrong rounds")
    require(not report["aot_router"]["active"], "Unexpected AOT inference")
    originals = case_map(baseline)
    cases = case_map(report)
    for key, case in cases.items():
        require(case["input_ids"] == originals[key]["input_ids"], "Different input IDs")
        require(
            len(case["warmup_ids"]) == 3 and len(case["samples"]) == 5,
            "Incomplete rounds",
        )
        outputs = case["warmup_ids"] + [s["output_ids"] for s in case["samples"]]
        for ids in outputs:
            require(
                len(ids) == key[1] and all(len(row) == 128 for row in ids),
                "Wrong output count",
            )
        durations = [s["seconds"] for s in case["samples"]]
        require(all(math.isfinite(s) and s > 0 for s in durations), "Invalid duration")
        require(
            math.isclose(
                key[1] * 128 * 5 / sum(durations),
                case["tokens_per_second"],
                rel_tol=1e-12,
            ),
            "Wrong throughput",
        )
        require(
            case["repeat_consistent"] == all(ids == outputs[0] for ids in outputs),
            "Wrong consistency flag",
        )
    return cases


def summarize(root):
    """Return validated throughput and ratios, keeping distinct cohorts explicit."""
    validation = root / "docs/validation"
    hashes = {name: verify_hashes(validation / name) for name in EVIDENCE}
    original_dir = validation / EVIDENCE[0]
    repeated_dir = validation / EVIDENCE[1]
    original = compare(original_dir)
    require(
        original == read(original_dir / "comparisons.json"), "Stale original comparison"
    )
    outcomes = read(repeated_dir / "outcomes.json")
    require(
        [o["name"] for o in outcomes]
        == [
            "windows-flashinfer-repeat1",
            "wsl2-flashinfer-repeat1",
            "wsl2-flashinfer-repeat2",
            "windows-flashinfer-repeat2",
        ],
        "Wrong process order",
    )
    require(
        all(
            o["passed"] and o["returncode"] == 0 and not o["environment_changes"]
            for o in outcomes
        ),
        "Failed/changed repeat",
    )
    repeats = {}
    for platform in ("windows", "wsl2"):
        baseline = read(original_dir / f"{platform}-tokenspeed-flashinfer-graph.json")
        for repetition in (1, 2):
            name = f"{platform}-flashinfer-repeat{repetition}"
            report = read(repeated_dir / (name + ".json"))
            repeats[(platform, repetition)] = validate_repeat(report, baseline)
            if platform == "windows":
                require(
                    read(repeated_dir / (name + "-environment.json"))
                    == read(
                        original_dir
                        / "windows-tokenspeed-flashinfer-graph-environment.json"
                    ),
                    "Different FlashInfer import/version",
                )
    latest = [dict(row) for row in original["throughput"]]
    archived_comparison = read(repeated_dir / "comparisons.json")
    for row in latest:
        key = row["input_length"], row["batch_size"]
        archived = next(
            r
            for r in archived_comparison["rows"]
            if (r["input_length"], r["batch_size"]) == key
        )
        for platform in ("windows", "wsl2"):
            seconds = sum(
                s["seconds"]
                for repetition in (1, 2)
                for s in repeats[(platform, repetition)][key]["samples"]
            )
            rate = key[1] * 128 * 10 / seconds
            require(
                math.isclose(
                    rate, archived["platforms"][platform]["pooled_tps"], rel_tol=1e-12
                ),
                "Stale pooled comparison",
            )
            row[f"{platform}-tokenspeed-flashinfer"] = rate
    ratios = []
    for previous in original["ratios"]:
        row = next(
            r
            for r in latest
            if r["input_length"] == previous["input_length"]
            and r["batch_size"] == previous["batch_size"]
        )
        a, b = previous["numerator"], previous["denominator"]
        value = row[a] / row[b]
        ratios.append(
            dict(
                input_length=row["input_length"],
                batch_size=row["batch_size"],
                numerator=a,
                denominator=b,
                ratio=value,
                percent=100 * (value - 1),
            )
        )
    cute = {}
    for name in EVIDENCE[3:]:
        directory = validation / name
        summary = read(directory / "summary.json")
        suites = ET.parse(directory / "tests.xml").getroot().findall(".//testsuite")
        for field in ("tests", "failures", "errors", "skipped"):
            require(
                summary[field] == sum(int(s.attrib[field]) for s in suites),
                f"Different JUnit count: {name}/{field}",
            )
        require(
            summary["test_exit_code"] == 0
            and not any(summary[k] for k in ("failures", "errors", "skipped")),
            "Incomplete CuTe validation",
        )
        require(
            summary["execution_python_pe_machine"] == "0xaa64"
            and summary["compiler_python_pe_machine"] == "0x8664",
            "Wrong compiler/execution architecture",
        )
        cute[name] = {k: summary[k] for k in ("tests", "failures", "errors", "skipped")}
    source_summary = read(validation / EVIDENCE[4] / "summary.json")
    for name, expected in source_summary["source_sha256"].items():
        source = validation / EVIDENCE[4] / "sources" / name
        require(
            hashlib.sha256(source.read_bytes()).hexdigest() == expected,
            f"Changed tested source snapshot: {name}",
        )
    require(
        all(v == "0xaa64" for v in source_summary["native_host_dlls"].values()),
        "Non-ARM64 host DLL",
    )
    return dict(
        schema=1,
        scope="Cross-cohort summary: original vLLM/Triton and two-repeat pooled FlashInfer; no AOT model-throughput attribution",
        throughput=latest,
        ratios=ratios,
        evidence_hashes=hashes,
        cute_validation=cute,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root.resolve())
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "evidence_hashes": result["evidence_hashes"],
                "cute_validation": result["cute_validation"],
            },
            indent=2,
        )
    )
