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

"""Run selected dispatcher and graph-state checks after the primary matrices."""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--platform-label", required=True)
    args = parser.parse_args()
    from tools.perf import benchmark_without_tokenspeed as baseline
    from tools.perf import probe_hf_graph_state

    control = args.report_dir / f"{args.platform_label}-diagnostic-hf-generate-cudnn.json"
    sys.argv = [str(Path(baseline.__file__)), "--model", str(args.model), "--report", str(control), "--implementation", "generate", "--normalization", "torch", "--diagnostics", "cpu", "--input-lengths", "128", "--batch-sizes", "1", "--output-length", "128", "--warmup", "3", "--rounds", "5", "--capacity", "704"]
    try:
        with baseline.torch.nn.attention.sdpa_kernel(baseline.torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
            baseline.main()
    except RuntimeError as error:
        if not any(term in str(error).lower() for term in ("cudnn", "execution plans", "available kernel", "backend")):
            raise
        print(f"Selected forced-cuDNN control unsupported: {error}", flush=True)
    else:
        record = json.loads(control.read_text(encoding="utf-8"))
        record["forced_sdpa_backend"] = "cudnn"
        record["included_in_primary_matrix"] = False
        control.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    baseline.torch.cuda.empty_cache()
    print('Starting graph KV-content/counter audit', flush=True)
    report = args.report_dir / f"{args.platform_label}-hf-graph-state.json"
    sys.argv = [str(Path(probe_hf_graph_state.__file__)), "--model", str(args.model), "--report", str(report)]
    probe_hf_graph_state.main()


if __name__ == "__main__":
    main()
