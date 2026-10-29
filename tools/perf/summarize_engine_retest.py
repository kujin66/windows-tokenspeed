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

"""Validate the retained current measurements and derive cross-cohort comparisons."""

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

EVIDENCE = (
    "engine-matrix-retest-20261029",
    "flashinfer-paired-retest-20261029",
    "windows-cutedsl-native-host",
    "windows-arm64-build-inputs",
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
    actual = {
        p.relative_to(directory).as_posix()
        for p in directory.rglob("*")
        if p.is_file() and p.name not in ("checksums.sha256", "evidence-sha256.json")
    }
    expected_names = {
        line.split("  ", 1)[1]
        for line in (directory / "checksums.sha256").read_text().splitlines()
    }
    require(
        len(expected_names) == count and actual == expected_names,
        "Unexpected/missing evidence file",
    )
    return count


def case_map(report):
    cases = {(c["input_length"], c["batch_size"]): c for c in report["cases"]}
    require(len(cases) == len(report["cases"]) == 4, "Missing/duplicate cases")
    require(set(cases) == {(128, 1), (128, 4), (512, 1), (512, 4)}, "Wrong cases")
    return cases


def validate_repeat(report, baseline):
    require(report["execution"] == "graph", "Expected graph execution")
    require(report["warmup"] == 3 and report["rounds"] == 5, "Wrong rounds")
    require(report["output_length"] == 128, "Wrong output length")
    require(not report["aot_router"]["active"], "Unexpected AOT inference")
    require(
        all(v is not None for v in report["packages"].values()),
        "Unreadable package metadata",
    )
    settings = report["engine_config"]
    require(
        not settings["enforce_eager"] and settings["dtype"] == "bfloat16",
        "Expected non-eager BF16 generation",
    )
    if report["engine_name"] == "tokenspeed":
        require(
            settings["attention_backend"] == report["backend"]
            and settings["disable_prefill_graph"]
            and settings["cudagraph_capture_sizes"] == [1, 4],
            "Wrong TokenSpeed backend/graph policy",
        )
    else:
        require(
            not settings["skip_tokenizer_init"]
            and report["sampling"]["detokenize"]
            and not report["resolved"]["cudagraph_mode"].upper().endswith("NONE")
            and not report["tokenspeed_engine_modules_loaded"],
            "Wrong vLLM text/graph/engine policy",
        )
    require(report["passed"], "Incomplete repeat")
    keys = (
        "engine_name",
        "backend",
        "execution",
        "platform",
        "benchmark_sha256",
        "packages",
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
        "python",
        "os",
        "torch_cuda",
    )
    if report["engine_name"] == "tokenspeed":
        keys += ("sources", "graph_policy", "source_root_override")
    for key in keys:
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
    """Validate current reports, repetition identities, all samples and CuTe evidence."""
    validation = root / "docs/validation"
    hashes = {name: verify_hashes(validation / name) for name in EVIDENCE}
    matrix, repeated = (validation / name for name in EVIDENCE[:2])
    reports = {}
    cases = {}
    reference = None
    shared = (
        "benchmark_sha256",
        "seed",
        "input_lengths",
        "batch_sizes",
        "output_length",
        "warmup",
        "rounds",
        "model_files",
        "gpu",
    )

    def add(name, report, platform, engine, backend, baseline):
        nonlocal reference
        require(
            report["platform"] == f"{platform}-arm64"
            and report["engine_name"] == engine
            and report["backend"] == backend,
            "Wrong platform/engine/backend",
        )
        cells = validate_repeat(report, baseline)
        if reference is None:
            reference = report
        for field in shared:
            require(report[field] == reference[field], f"Incomparable {field}")
        reference_cases = case_map(reference)
        for key, case in cells.items():
            require(
                case["input_ids"] == reference_cases[key]["input_ids"],
                "Different input IDs",
            )
        reports[name] = report
        cases[name] = cells

    for platform in ("windows", "wsl2"):
        for engine, backend in (("vllm", "auto"), ("tokenspeed", "triton")):
            name = f"{platform}-{engine}-{backend}"
            report = read(matrix / f"{name}-graph.json")
            add(name, report, platform, engine, backend, report)
        baseline = read(repeated / f"{platform}-flashinfer-repeat2.json")
        for repetition in (1, 2):
            name = f"{platform}-flashinfer-repeat{repetition}"
            report = read(repeated / f"{name}.json")
            add(name, report, platform, "tokenspeed", "flashinfer", baseline)
            expected_fi = "0.6.17+g2ea3c24b5dac" if platform == "windows" else "0.7.2"
            require(
                report["packages"]["flashinfer-python"] == expected_fi,
                "Wrong FlashInfer version",
            )
        if platform == "windows":
            require(
                read(repeated / "windows-flashinfer-repeat1-environment.json")
                == read(repeated / "windows-flashinfer-repeat2-environment.json"),
                "Different FlashInfer import/version",
            )

    benchmark = matrix / "sources/benchmark_engine_matrix.py"
    require(
        hashlib.sha256(benchmark.read_bytes()).hexdigest()
        == reference["benchmark_sha256"],
        "Different benchmark source",
    )
    outcomes = read(repeated / "outcomes.json")
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
    latest = []
    for key in ((128, 1), (128, 4), (512, 1), (512, 4)):
        row = dict(input_length=key[0], batch_size=key[1])
        for platform in ("windows", "wsl2"):
            for engine, backend in (("vllm", "auto"), ("tokenspeed", "triton")):
                name = f"{platform}-{engine}-{backend}"
                row[name] = cases[name][key]["tokens_per_second"]
            seconds = sum(
                s["seconds"]
                for repetition in (1, 2)
                for s in cases[f"{platform}-flashinfer-repeat{repetition}"][key][
                    "samples"
                ]
            )
            rate = key[1] * 128 * 10 / seconds
            row[f"{platform}-tokenspeed-flashinfer"] = rate
            comparison = next(
                r
                for r in read(repeated / "comparisons.json")["rows"]
                if (r["input_length"], r["batch_size"]) == key
            )
            require(
                math.isclose(
                    rate, comparison["platforms"][platform]["pooled_tps"], rel_tol=1e-12
                ),
                "Stale FlashInfer comparison",
            )
        latest.append(row)

    pairs = []
    for platform in ("windows", "wsl2"):
        pairs += [
            (f"{platform}-tokenspeed-{backend}", f"{platform}-vllm-auto")
            for backend in ("triton", "flashinfer")
        ]
        pairs.append(
            (f"{platform}-tokenspeed-flashinfer", f"{platform}-tokenspeed-triton")
        )
    pairs += [
        (f"windows-{engine}-{backend}", f"wsl2-{engine}-{backend}")
        for engine, backend in (
            ("vllm", "auto"),
            ("tokenspeed", "triton"),
            ("tokenspeed", "flashinfer"),
        )
    ]
    ratios = [
        dict(
            input_length=row["input_length"],
            batch_size=row["batch_size"],
            numerator=a,
            denominator=b,
            ratio=row[a] / row[b],
            percent=100 * (row[a] / row[b] - 1),
        )
        for row in latest
        for a, b in pairs
    ]

    directory = validation / EVIDENCE[2]
    summary = read(directory / "summary.json")
    suites = ET.parse(directory / "tests.xml").getroot().findall(".//testsuite")
    for field in ("tests", "failures", "errors", "skipped"):
        require(
            summary[field] == sum(int(s.attrib[field]) for s in suites),
            f"Different JUnit count: {field}",
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
    for name, expected in summary["source_sha256"].items():
        require(
            hashlib.sha256((directory / "sources" / name).read_bytes()).hexdigest()
            == expected,
            f"Changed tested source snapshot: {name}",
        )
    require(
        all(v == "0xaa64" for v in summary["native_host_dlls"].values()),
        "Non-ARM64 host DLL",
    )
    return dict(
        schema=2,
        scope="Current non-eager measurements from separate vLLM/Triton and two-repetition FlashInfer cohorts; AOT inactive for dense inference",
        throughput=latest,
        ratios=ratios,
        evidence_hashes=hashes,
        cute_validation={
            EVIDENCE[2]: {
                k: summary[k] for k in ("tests", "failures", "errors", "skipped")
            }
        },
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
