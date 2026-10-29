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

"""Validate and compare a complete Windows/WSL2 non-eager engine matrix."""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

CELLS = [
    (os, engine, backend)
    for os in ("windows", "wsl2")
    for engine, backend in (
        ("vllm", "auto"),
        ("tokenspeed", "triton"),
        ("tokenspeed", "flashinfer"),
    )
]
HISTORICAL = {
    "windows-vllm-auto": [144.75, 582.45, 133.66, 509.51],
    "windows-tokenspeed-triton": [144.22, 608.83, 128.36, 515.84],
    "windows-tokenspeed-flashinfer": [139.61, 564.28, 125.84, 502.62],
    "wsl2-tokenspeed-flashinfer": [140.99, 544.32, 128.32, 477.95],
}
HISTORICAL_CASES = [(128, 1), (128, 4), (512, 1), (512, 4)]


def compare(directory):
    reports = {}
    baseline = None
    all_cases = {}
    artifacts = {}
    for os, engine, backend in CELLS:
        name = f"{os}-{engine}-{backend}"
        path = directory / (name + "-graph.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        if (
            not data["passed"]
            or data["execution"] != "graph"
            or data["engine_name"] != engine
            or data["backend"] != backend
        ):
            raise ValueError(f"Incomplete/wrong configuration: {name}")
        if data["platform"] != ("windows-arm64" if os == "windows" else "wsl2-arm64"):
            raise ValueError(f"Wrong platform: {name}")
        if any(version is None for version in data["packages"].values()):
            raise ValueError(f"Unreadable package metadata: {name}")
        if (
            os == "windows"
            and engine == "tokenspeed"
            and data["packages"]["flashinfer-python"] != "0.6.17+g2ea3c24b5dac"
        ):
            raise ValueError(f"Wrong Windows FlashInfer wheel: {name}")
        settings = data["engine_config"]
        if settings["enforce_eager"] or settings["dtype"] != "bfloat16":
            raise ValueError(f"Expected non-eager BF16 generation: {name}")
        if engine == "tokenspeed":
            if (
                settings["attention_backend"] != backend
                or not settings["disable_prefill_graph"]
                or settings["cudagraph_capture_sizes"] != [1, 4]
            ):
                raise ValueError(f"Wrong TokenSpeed backend/graph policy: {name}")
        elif (
            settings["skip_tokenizer_init"]
            or not data["sampling"]["detokenize"]
            or data["resolved"]["cudagraph_mode"].upper().endswith("NONE")
            or data["tokenspeed_engine_modules_loaded"]
        ):
            raise ValueError(f"Wrong vLLM text/graph/engine policy: {name}")
        if baseline is None:
            baseline = data
        for field in (
            "benchmark_sha256",
            "seed",
            "input_lengths",
            "batch_sizes",
            "output_length",
            "warmup",
            "rounds",
            "model_files",
            "gpu",
        ):
            if data[field] != baseline[field]:
                raise ValueError(f"Incomparable {field}: {name}")
        expected = set(itertools.product(data["input_lengths"], data["batch_sizes"]))
        cases = {(c["input_length"], c["batch_size"]): c for c in data["cases"]}
        if set(cases) != expected or len(data["cases"]) != len(expected):
            raise ValueError(f"Missing/duplicate cases: {name}")
        for key, case in cases.items():
            if (
                len(case["samples"]) != data["rounds"]
                or len(case["warmup_ids"]) != data["warmup"]
            ):
                raise ValueError(f"Incomplete repetitions: {name}/{key}")
            for ids in case["warmup_ids"] + [s["output_ids"] for s in case["samples"]]:
                if len(ids) != key[1] or any(
                    len(row) != data["output_length"] for row in ids
                ):
                    raise ValueError(f"Wrong token count: {name}/{key}")
            durations = [s["seconds"] for s in case["samples"]]
            if any(not math.isfinite(value) or value <= 0 for value in durations):
                raise ValueError(f"Invalid elapsed time: {name}/{key}")
            recomputed = (
                key[1] * data["output_length"] * data["rounds"] / sum(durations)
            )
            if not math.isclose(case["tokens_per_second"], recomputed, rel_tol=1e-12):
                raise ValueError(f"Throughput mismatch: {name}/{key}")
        reports[name] = data
        all_cases[name] = cases
        artifacts[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    keys = list(all_cases[next(iter(all_cases))])
    rows, ratios, consistency, historical = [], [], [], []
    for key in keys:
        input_reference = next(iter(all_cases.values()))[key]["input_ids"]
        row = dict(input_length=key[0], batch_size=key[1])
        for name, cases in all_cases.items():
            case = cases[key]
            if case["input_ids"] != input_reference:
                raise ValueError(f"Input IDs differ: {name}/{key}")
            row[name] = case["tokens_per_second"]
            outputs = case["warmup_ids"] + [s["output_ids"] for s in case["samples"]]
            same = all(ids == outputs[0] for ids in outputs)
            if same != case["repeat_consistent"]:
                raise ValueError("Stored repeat consistency differs from raw IDs")
            consistency.append(
                dict(
                    cell=name,
                    input_length=key[0],
                    batch_size=key[1],
                    repeat_consistent=same,
                )
            )
            if name in HISTORICAL and key in HISTORICAL_CASES:
                old = HISTORICAL[name][HISTORICAL_CASES.index(key)]
                historical.append(
                    dict(
                        cell=name,
                        input_length=key[0],
                        batch_size=key[1],
                        old_tps=old,
                        new_tps=case["tokens_per_second"],
                        new_over_old=case["tokens_per_second"] / old,
                    )
                )
        rows.append(row)
        pairs = []
        for os in ("windows", "wsl2"):
            pairs += [
                (f"{os}-tokenspeed-{backend}", f"{os}-vllm-auto")
                for backend in ("triton", "flashinfer")
            ]
            pairs.append((f"{os}-tokenspeed-flashinfer", f"{os}-tokenspeed-triton"))
        pairs += [
            (f"windows-{engine}-{backend}", f"wsl2-{engine}-{backend}")
            for engine, backend in (
                ("vllm", "auto"),
                ("tokenspeed", "triton"),
                ("tokenspeed", "flashinfer"),
            )
        ]
        for numerator, denominator in pairs:
            a, b = all_cases[numerator][key], all_cases[denominator][key]
            a_ids, b_ids = a["warmup_ids"][0], b["warmup_ids"][0]
            prefixes = []
            for left, right in zip(a_ids, b_ids, strict=True):
                prefixes.append(
                    next(
                        (
                            i
                            for i, (x, y) in enumerate(zip(left, right, strict=True))
                            if x != y
                        ),
                        len(left),
                    )
                )
            equal = sum(
                x == y
                for left, right in zip(a_ids, b_ids, strict=True)
                for x, y in zip(left, right, strict=True)
            )
            ratio = a["tokens_per_second"] / b["tokens_per_second"]
            ratios.append(
                dict(
                    input_length=key[0],
                    batch_size=key[1],
                    numerator=numerator,
                    denominator=denominator,
                    ratio=ratio,
                    percent=(ratio - 1) * 100,
                    reference_ids_match=a_ids == b_ids,
                    matching_positions=equal,
                    total_positions=key[1] * baseline["output_length"],
                    common_prefix_lengths=prefixes,
                )
            )
    return dict(
        schema=1,
        workload={
            field: baseline[field]
            for field in (
                "seed",
                "input_lengths",
                "batch_sizes",
                "output_length",
                "warmup",
                "rounds",
                "model_files",
                "gpu",
                "benchmark_sha256",
            )
        },
        throughput=rows,
        ratios=ratios,
        consistency=consistency,
        historical=historical,
        reports_sha256=artifacts,
        scope="Six fresh independent-engine/backend configurations; AOT router is inactive and inapplicable to this dense model; different package/graph policies prevent an OS-only or AOT speedup attribution.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.directory)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for row in result["throughput"]:
        print(row)
