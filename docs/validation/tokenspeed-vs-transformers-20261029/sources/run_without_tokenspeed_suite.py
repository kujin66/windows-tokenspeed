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

"""Run explicit no-TokenSpeed configurations sequentially in one Python process."""

import argparse
import gc
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--platform-label", required=True)
    parser.add_argument("--diagnostics", choices=("none", "cpu"), required=True)
    parser.add_argument(
        "--configurations",
        nargs="+",
        choices=(
            "generate-torch",
            "loop-dynamic-torch",
            "loop-static-eager-torch",
            "loop-static-graph-torch",
            "loop-static-eager-flashinfer",
            "loop-static-graph-flashinfer",
        ),
        required=True,
    )
    args = parser.parse_args()
    from tools.perf import benchmark_without_tokenspeed as baseline

    configurations = [
        configuration.rsplit("-", 1) for configuration in args.configurations
    ]
    for implementation, normalization in configurations:
        report = (
            args.report_dir
            / f"{args.platform_label}-hf-{implementation}-{normalization}.json"
        )
        sys.argv = [
            str(Path(baseline.__file__)),
            "--model",
            str(args.model),
            "--report",
            str(report),
            "--implementation",
            implementation,
            "--normalization",
            normalization,
            "--diagnostics",
            args.diagnostics,
            "--input-lengths",
            "128",
            "512",
            "--batch-sizes",
            "1",
            "4",
            "--output-length",
            "128",
            "--warmup",
            "3",
            "--rounds",
            "5",
            "--capacity",
            "704",
        ]
        print(
            f"Starting {args.platform_label} {implementation} RMSNorm={normalization}",
            flush=True,
        )
        baseline.main()
        gc.collect()
        baseline.torch.cuda.empty_cache()
        print(f"Completed {report.name}", flush=True)


if __name__ == "__main__":
    main()
