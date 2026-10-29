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

"""Measure no-TokenSpeed decode graphs with request-sized static KV buffers."""

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--platform-label", required=True)
    parser.add_argument("--normalizations", nargs="+", choices=("torch", "flashinfer"), required=True)
    parser.add_argument("--input-lengths", type=int, nargs="+", required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", required=True)
    parser.add_argument("--output-length", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--math-reduction", choices=("full", "reduced"), required=True)
    args = parser.parse_args()
    from tools.perf import benchmark_without_tokenspeed as baseline
    baseline.torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(args.math_reduction == "reduced")

    for normalization in args.normalizations:
        for length in args.input_lengths:
            capacity = length + args.output_length
            suffix = "-reduced" if args.math_reduction == "reduced" else ""
            report = args.report_dir / f"{args.platform_label}-hf-loop-static-graph-bucketed-{normalization}{suffix}-in{length}.json"
            sys.argv = [str(Path(baseline.__file__)), "--model", str(args.model), "--report", str(report), "--implementation", "loop-static-graph", "--normalization", normalization, "--diagnostics", "none", "--input-lengths", str(length), "--batch-sizes", *map(str, args.batch_sizes), "--output-length", str(args.output_length), "--warmup", str(args.warmup), "--rounds", str(args.rounds), "--capacity", str(capacity)]
            print(f"Starting bucketed graph {args.platform_label}, input={length}, capacity={capacity}, RMSNorm={normalization}", flush=True)
            baseline.main()
            record = json.loads(report.read_text(encoding="utf-8"))
            record["sdpa_math_reduction_allowed"] = baseline.torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
            record["controller_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            report.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            gc.collect()
            baseline.torch.cuda.empty_cache()
            print(f"Completed {report.name}", flush=True)


if __name__ == "__main__":
    main()
