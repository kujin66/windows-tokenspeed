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

"""Verify published benchmark evidence without loading GPU packages."""

import hashlib
import json
import math
from pathlib import Path


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root):
    validation = root / "docs/validation"
    names = (
        "windows-wsl2-perf-20261028",
        "windows-wsl2-perf-20261028-after-reboot",
        "windows-wsl2-diagnosis-20261028",
        "windows-flashinfer-0617-20261028",
    )
    counts = {}
    for name in names:
        directory = validation / name
        manifest = load_json(directory / "evidence-sha256.json")
        for entry in manifest:
            path = directory / entry["file"]
            require(
                hashlib.sha256(path.read_bytes()).hexdigest()
                == entry["published_sha256"],
                f"Hash mismatch: {path}",
            )
        for path in directory.rglob("*.json"):
            load_json(path)
        counts[name] = len(manifest)

    directory = validation / names[-1]
    summary = load_json(directory / "results.json")
    require(len(summary["rows"]) == 8, "Expected eight comparison rows")
    reports = {}
    for backend in ("windows-flashinfer", "windows-triton", "wsl2-flashinfer"):
        for mode in ("eager", "graph"):
            report = load_json(directory / f"{backend}-{mode}.json")
            require(report["passed"], f"Incomplete {backend}/{mode}")
            require(report["rounds"] == 5 and report["warmup"] == 3, "Wrong rounds")
            require(report["model_files"] == summary["model_files"], "Model mismatch")
            require(
                report["script_sha256"] == summary["benchmark_sha256"],
                "Benchmark source mismatch",
            )
            require(
                report["engine"]["enforce_eager"] == (mode == "eager"), "Wrong mode"
            )
            require(
                report["engine"]["disable_prefill_graph"], "Unexpected prefill graph"
            )
            cases = {(c["input_length"], c["batch_size"]): c for c in report["cases"]}
            require(
                set(cases) == {(128, 1), (128, 4), (512, 1), (512, 4)}, "Wrong cases"
            )
            for case in cases.values():
                require(len(case["samples"]) == 5, "Missing measured samples")
                require(len(case["warmup_ids"]) == 3, "Missing warmup samples")
                outputs = case["warmup_ids"] + [
                    s["output_ids"] for s in case["samples"]
                ]
                for batch in outputs:
                    require(len(batch) == case["batch_size"], "Wrong output batch")
                    require(
                        all(len(ids) == 128 for ids in batch), "Wrong output length"
                    )
                require(
                    all(s["seconds"] > 0 for s in case["samples"]), "Invalid duration"
                )
            require(
                sum(c["batch_size"] * 5 for c in cases.values()) == 50, "Wrong requests"
            )
            reports[backend, mode] = cases

    for row in summary["rows"]:
        values = {}
        references = {}
        for backend, prefix in (
            ("windows-flashinfer", "windows_flashinfer"),
            ("windows-triton", "windows_triton"),
            ("wsl2-flashinfer", "wsl2_flashinfer"),
        ):
            case = reports[backend, row["mode"]][row["input_length"], row["batch_size"]]
            throughput = (
                case["batch_size"]
                * 128
                * 5
                / sum(s["seconds"] for s in case["samples"])
            )
            require(
                math.isclose(throughput, row[f"{prefix}_tok_s"], rel_tol=1e-12),
                "Throughput mismatch",
            )
            values[backend] = throughput
            outputs = case["warmup_ids"] + [s["output_ids"] for s in case["samples"]]
            consistent = all(ids == outputs[0] for ids in outputs)
            require(consistent == row[f"{prefix}_consistent"], "Consistency mismatch")
            references[backend] = case["samples"][0]["output_ids"]
        for comparison, percent_field, reference_field in (
            (
                "windows-triton",
                "windows_flashinfer_vs_triton_percent",
                "windows_flashinfer_vs_triton_reference_ids_match",
            ),
            (
                "wsl2-flashinfer",
                "windows_flashinfer_vs_wsl2_percent",
                "windows_vs_wsl2_reference_ids_match",
            ),
        ):
            percent = 100 * (values["windows-flashinfer"] / values[comparison] - 1)
            require(
                math.isclose(percent, row[percent_field], abs_tol=1e-9),
                "Percentage mismatch",
            )
            require(
                (references["windows-flashinfer"] == references[comparison])
                == row[reference_field],
                "Reference ID mismatch",
            )
    return counts


if __name__ == "__main__":
    counts = verify(Path(__file__).resolve().parents[2])
    print(
        json.dumps(
            {
                "published_hashes": counts,
                "total": sum(counts.values()),
                "matrix_rows": 8,
            },
            indent=2,
        )
    )
