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

"""Summarize complete fresh Windows/WSL2 no-TokenSpeed comparison matrices."""

import argparse
import json
from pathlib import Path

from tools.perf.compare_without_tokenspeed import compare, summarize_case


def summarize(directory):
    fixed = (
        "generate-torch",
        "loop-dynamic-torch",
        "loop-static-eager-torch",
        "loop-static-graph-torch",
        "loop-static-eager-flashinfer",
        "loop-static-graph-flashinfer",
    )
    reports = {}
    configurations = {}
    comparisons = []
    diagnostics = {}
    for platform in ("windows", "wsl2"):
        backends = (
            ("flashinfer", "triton") if platform == "windows" else ("flashinfer",)
        )
        engines = {}
        for backend in backends:
            for mode in ("eager", "graph"):
                name = f"{platform}-tokenspeed-{backend}-{mode}"
                engines[backend, mode] = json.loads(
                    (directory / f"{name}.json").read_text(encoding="utf-8")
                )
                reports[name] = engines[backend, mode]
        baseline_files = [(f"hf-{name}", f"{platform}-hf-{name}") for name in fixed]
        baseline_files += [
            (
                f"hf-loop-static-graph-bucketed-{normalization}",
                f"{platform}-hf-loop-static-graph-bucketed-{normalization}-in{length}",
            )
            for normalization in ("torch", "flashinfer")
            for length in (128, 512)
        ]
        baseline_files += [
            (
                "hf-loop-static-graph-bucketed-flashinfer-reduced",
                f"{platform}-hf-loop-static-graph-bucketed-flashinfer-reduced-in{length}",
            )
            for length in (128, 512)
        ]
        configurations[platform] = {}
        for label, name in baseline_files:
            baseline = json.loads(
                (directory / f"{name}.json").read_text(encoding="utf-8")
            )
            reports[name] = baseline
            if (
                baseline["rounds"] != 5
                or baseline["warmup"] != 3
                or baseline["output_length"] != 128
            ):
                raise ValueError(f"Unexpected workload: {name}")
            graph = baseline["implementation"] == "loop-static-graph"
            mode = "graph" if graph else "eager"
            if "bucketed" in label:
                if (
                    len(baseline["input_lengths"]) != 1
                    or baseline["capacity"] != baseline["input_lengths"][0] + 128
                ):
                    raise ValueError(f"Wrong request-sized cache: {name}")
                if baseline["sdpa_math_reduction_allowed"] != label.endswith(
                    "-reduced"
                ):
                    raise ValueError(f"Wrong math precision policy: {name}")
            elif baseline["capacity"] != 704:
                raise ValueError(f"Wrong fixed cache: {name}")
            entries = configurations[platform].setdefault(label, [])
            for backend in backends:
                rows = compare(baseline, engines[backend, mode])
                for row in rows:
                    hf, ts = row["without_tokenspeed"], row["tokenspeed"]
                    compact = dict(
                        platform=platform,
                        baseline=label,
                        mode=mode,
                        tokenspeed_backend=backend,
                        input_length=row["input_length"],
                        batch_size=row["batch_size"],
                        baseline_tokens_per_second=hf["tokens_per_second"],
                        tokenspeed_tokens_per_second=ts["tokens_per_second"],
                        tokenspeed_over_baseline_ratio=row[
                            "tokenspeed_over_baseline_ratio"
                        ],
                        tokenspeed_relative_percent=row["tokenspeed_relative_percent"],
                        baseline_repeat_consistent=hf["repeat_consistent"],
                        tokenspeed_repeat_consistent=ts["repeat_consistent"],
                        reference_ids_match=row["reference_ids_match"],
                        baseline_sample_seconds=hf["sample_seconds"],
                        tokenspeed_sample_seconds=ts["sample_seconds"],
                    )
                    comparisons.append(compact)
                    if backend == "flashinfer":
                        entries.append(compact)
            if baseline.get("cpu_diagnostic") is not None:
                diagnostics[name] = baseline["cpu_diagnostic"]
        for label, cases in configurations[platform].items():
            keys = {(row["input_length"], row["batch_size"]) for row in cases}
            if keys != {(128, 1), (128, 4), (512, 1), (512, 4)} or len(cases) != 4:
                raise ValueError(f"Incomplete case coverage: {platform}/{label}")
    measured_requests = 0
    measured_tokens = 0
    hf_hashes = set()
    ts_hashes = set()
    for name, report in reports.items():
        if not report["passed"]:
            raise ValueError(f"Incomplete report: {name}")
        baseline = "-hf-" in name
        output_length = (
            report["output_length"]
            if baseline
            else report["sampling"]["max_new_tokens"]
        )
        for case in report["cases"]:
            values = summarize_case(
                case, output_length, report["rounds"], report["warmup"]
            )
            measured_requests += values["measured_requests"]
            measured_tokens += values["measured_output_tokens"]
        if baseline:
            hf_hashes.add(report["script_sha256"])
        else:
            ts_hashes.add(report["script_sha256"])
    if len(reports) != 30 or len(hf_hashes) != 1 or len(ts_hashes) != 1:
        raise ValueError("Missing reports or mixed benchmark source versions")
    if measured_requests != 1200 or measured_tokens != 153600:
        raise ValueError("Incomplete measured workload")
    return dict(
        status="complete",
        reports=len(reports),
        measured_requests=measured_requests,
        measured_output_tokens=measured_tokens,
        hf_benchmark_sha256=hf_hashes.pop(),
        tokenspeed_benchmark_sha256=ts_hashes.pop(),
        comparisons=comparisons,
        configurations=configurations,
        cpu_diagnostics=diagnostics,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.directory)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ("comparisons", "configurations", "cpu_diagnostics")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
