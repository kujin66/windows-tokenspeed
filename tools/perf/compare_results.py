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

"""Compare completed Windows and WSL2 benchmarks only when workloads match."""

import argparse
import json
from pathlib import Path

from tools.perf.benchmark_tokenspeed import summarize


def compare(windows, wsl):
    if windows["platform"] != "windows-arm64" or wsl["platform"] != "wsl2-arm64":
        raise ValueError("Expected Windows and WSL2 reports, in that order")
    if not windows["passed"] or not wsl["passed"]:
        raise ValueError("Both benchmarks must have passed")
    for key in (
        "schema",
        "benchmark_sha256",
        "upstream_base",
        "workload",
        "model_files",
        "gpu",
    ):
        if windows[key] != wsl[key]:
            raise ValueError(f"Incomparable {key}")
    if len(windows["cases"]) != len(wsl["cases"]) or not windows["cases"]:
        raise ValueError("Case counts must match and be nonzero")
    workload = windows["workload"]
    expected_cases = {
        (length, batch)
        for length in workload["input_lengths"]
        for batch in workload["batch_sizes"]
    }
    for report in (windows, wsl):
        actual_cases = [
            (case["input_length"], case["batch_size"]) for case in report["cases"]
        ]
        if set(actual_cases) != expected_cases or len(actual_cases) != len(
            expected_cases
        ):
            raise ValueError("Missing or duplicate benchmark cases")
        for case in report["cases"]:
            if len(case["samples"]) != workload["rounds"]:
                raise ValueError("Incomplete measurement rounds")
            expected_tokens = case["batch_size"] * workload["output_length"]
            if any(
                sample["output_tokens"] != expected_tokens for sample in case["samples"]
            ):
                raise ValueError("Incorrect measured token count")
    rows = []
    for native, linux in zip(windows["cases"], wsl["cases"], strict=True):
        for key in ("input_length", "batch_size", "input_ids_sha256"):
            if native[key] != linux[key]:
                raise ValueError(f"Incomparable case {key}")
        summaries = [summarize(case["samples"]) for case in (native, linux)]
        native_tps, linux_tps = [
            summary["output_tokens_per_second"] for summary in summaries
        ]
        rows.append(
            {
                "input_length": native["input_length"],
                "batch_size": native["batch_size"],
                "windows_output_tokens_per_second": native_tps,
                "wsl2_output_tokens_per_second": linux_tps,
                "windows_relative_to_wsl2_percent": 100 * (native_tps / linux_tps - 1),
                "greedy_output_ids_match": native["output_ids"] == linux["output_ids"],
                "windows_within_case_greedy_consistent": native[
                    "greedy_output_ids_consistent"
                ],
                "wsl2_within_case_greedy_consistent": linux[
                    "greedy_output_ids_consistent"
                ],
            }
        )
    return {
        "cases": rows,
        "interpretation": "End-to-end comparison of these two software stacks on the same GPU model; dependencies and Windows portability changes can also affect the difference.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--wsl2", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        json.loads(args.windows.read_text(encoding="utf-8")),
        json.loads(args.wsl2.read_text(encoding="utf-8")),
    )
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for row in result["cases"]:
        print(
            f"input={row['input_length']} batch={row['batch_size']}: Windows {row['windows_relative_to_wsl2_percent']:+.2f}% vs WSL2"
        )


if __name__ == "__main__":
    main()
