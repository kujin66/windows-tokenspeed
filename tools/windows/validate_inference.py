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

import argparse
import faulthandler
import json
from pathlib import Path

faulthandler.dump_traceback_later(120, repeat=True)
from tokenspeed.runtime.entrypoints.engine import Engine

root = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=root / ".port/models/Qwen3-0.6B")
    parser.add_argument(
        "--report", type=Path, default=root / ".port/inference-validation.json"
    )
    args = parser.parse_args()
    engine = Engine(
        model=str(args.model.resolve()),
        dtype="bfloat16",
        world_size=1,
        max_model_len=512,
        max_num_seqs=2,
        max_total_tokens=512,
        gpu_memory_utilization=0.25,
        attention_backend="triton",
        enforce_eager=True,
        disable_prefill_graph=True,
        grammar_backend="none",
        sampling_backend="greedy",
        disable_kvstore=True,
        retraction_snapshot_ratio=0,
        log_level="info",
    )
    try:
        prompt = "Write the number after 1: "
        params = {"max_new_tokens": 16, "temperature": 0}
        results = [engine.generate(prompt, params) for _ in range(3)]
        assert all(r["output_ids"] == results[0]["output_ids"] for r in results)
        batched = engine.generate([prompt, prompt], params)
        assert len(batched) == 2
        assert all(r["output_ids"] == results[0]["output_ids"] for r in batched)
        assert all(r["meta_info"]["completion_tokens"] == 16 for r in results + batched)
        assert all(len(r["output_ids"]) == 16 for r in results + batched)
        report = {
            "model": str(args.model.resolve()),
            "quantization_format": json.loads(
                (args.model / "config.json").read_text(encoding="utf-8")
            )
            .get("quantization_config", {})
            .get("format"),
            "dtype": "bfloat16",
            "sequential": results,
            "batch2": batched,
            "passed": True,
        }
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            "PASS: three repeat requests and two batched requests; identical greedy token IDs",
            flush=True,
        )
    finally:
        engine.shutdown()
