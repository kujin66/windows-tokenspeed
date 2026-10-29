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

"""Recompute matched independent-engine vLLM/TokenSpeed results without a GPU."""

import argparse
import json
from pathlib import Path

from tools.perf.compare_vllm import compare
from tools.perf.compare_without_tokenspeed import summarize_case


def summarize(directory):
    selected = json.loads(
        (directory / "selected-vllm.json").read_text(encoding="utf-8")
    )
    outcomes = json.loads(
        (directory / "controller-outcomes.json").read_text(encoding="utf-8")
    )
    text_outcomes = directory / "text-controller-outcomes.json"
    if text_outcomes.is_file():
        outcomes += json.loads(text_outcomes.read_text(encoding="utf-8"))
    statuses = {row["report"]: row for row in outcomes}
    rows = []
    reports = {}
    missing = []
    for mode in ("eager", "graph"):
        if mode not in selected:
            missing.append(f"No successful vLLM {mode} report")
            continue
        name = selected[mode]
        if not statuses[name]["passed"]:
            raise ValueError(f"Selected failed vLLM report: {name}")
        vllm = json.loads((directory / name).read_text(encoding="utf-8"))
        if vllm["warmup"] != 3 or vllm["rounds"] != 5 or vllm["output_length"] != 128:
            raise ValueError("Unexpected vLLM protocol")
        if set(vllm["input_lengths"]) != {128, 512} or set(vllm["batch_sizes"]) != {
            1,
            4,
        }:
            raise ValueError("Unexpected vLLM case matrix")
        reports[name] = vllm
        for backend in ("triton", "flashinfer"):
            name = f"windows-tokenspeed-{backend}-{mode}.json"
            if not statuses[name]["passed"]:
                missing.append(f"No successful TokenSpeed {backend}/{mode} report")
                continue
            tokenspeed = json.loads((directory / name).read_text(encoding="utf-8"))
            startup_log = (directory / Path(name).with_suffix(".log")).read_text(
                encoding="utf-8"
            )
            if "skip_tokenizer_init=False" not in startup_log:
                raise ValueError(
                    "TokenSpeed normal text-output policy is not established by its startup log"
                )
            reports[name] = tokenspeed
            for row in compare(vllm, tokenspeed):
                left, right = row["vllm"], row["tokenspeed"]
                rows.append(
                    dict(
                        mode=mode,
                        backend=backend,
                        input_length=row["input_length"],
                        batch_size=row["batch_size"],
                        vllm_tokens_per_second=left["tokens_per_second"],
                        tokenspeed_tokens_per_second=right["tokens_per_second"],
                        tokenspeed_over_vllm_ratio=row["tokenspeed_over_vllm_ratio"],
                        tokenspeed_relative_percent=row["tokenspeed_relative_percent"],
                        vllm_repeat_consistent=left["repeat_consistent"],
                        tokenspeed_repeat_consistent=right["repeat_consistent"],
                        reference_ids_match=row["reference_ids_match"],
                        vllm_sample_seconds=left["sample_seconds"],
                        tokenspeed_sample_seconds=right["sample_seconds"],
                    )
                )
    requests = 0
    output_tokens = 0
    for report in reports.values():
        for case in report["cases"]:
            values = summarize_case(case, 128, 5, 3)
            requests += values["measured_requests"]
            output_tokens += values["measured_output_tokens"]
    if not missing and (
        len(reports) != 6
        or len(rows) != 16
        or requests != 300
        or output_tokens != 38400
    ):
        raise ValueError("Incomplete primary engine matrix")
    return dict(
        status="complete" if not missing else "partial",
        comparison="independent vLLM versus independent TokenSpeed; no integration toggle",
        platform="windows-arm64",
        selected_vllm=selected,
        primary_reports=len(reports),
        measured_sequences=requests,
        measured_output_tokens=output_tokens,
        missing=missing,
        rows=rows,
        attempts=outcomes,
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
                if key not in ("rows", "attempts")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
