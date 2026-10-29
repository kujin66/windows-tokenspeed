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

"""Validate graph replay KV contents and counters against the same eager loop."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from tools.perf import benchmark_without_tokenspeed as baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    baseline.assert_no_tokenspeed()
    torch.set_num_threads(1)
    model = (
        AutoModelForCausalLM.from_pretrained(
            str(args.model),
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        .to("cuda")
        .eval()
    )
    report = dict(
        throughput_samples=False,
        torch=torch.__version__,
        torch_cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),
        baseline_source_sha256=hashlib.sha256(
            Path(baseline.__file__).read_bytes()
        ).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cases=[],
        passed=False,
    )
    with torch.inference_mode():
        for normalization, length, capacity, reduced in (
            ("torch", 512, 704, False),
            ("flashinfer", 128, 256, True),
        ):
            if normalization == "flashinfer":
                report["rmsnorm_ablation"] = baseline.replace_rmsnorm(model)
            torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(reduced)
            inputs = baseline.prompt_ids(length, 4)
            runner = baseline.StaticRunner(model, length, 4, 128, capacity, "graph")
            runner.prepare(inputs)
            graph_ids = runner.generate(inputs)
            graph_lengths = [
                int(layer.get_seq_length().item()) for layer in runner.cache.layers
            ]
            graph_position = int(runner.position.item())
            graph_output_index = int(runner.output_index.item())
            tensors = [
                (layer.keys.clone(), layer.values.clone())
                for layer in runner.cache.layers
            ]
            runner.execution = "eager"
            eager_ids = runner.generate(inputs)
            max_key_error = 0.0
            max_value_error = 0.0
            for layer, (keys, values) in zip(runner.cache.layers, tensors, strict=True):
                torch.testing.assert_close(layer.keys, keys, rtol=0.02, atol=0.02)
                torch.testing.assert_close(layer.values, values, rtol=0.02, atol=0.02)
                max_key_error = max(
                    max_key_error,
                    (layer.keys.float() - keys.float()).abs().max().item(),
                )
                max_value_error = max(
                    max_value_error,
                    (layer.values.float() - values.float()).abs().max().item(),
                )
            if (
                graph_ids != eager_ids
                or any(value != length + 127 for value in graph_lengths)
                or graph_position != length + 127
                or graph_output_index != 128
            ):
                raise ValueError("Graph replay output/counter validation failed")
            report["cases"].append(
                dict(
                    normalization=normalization,
                    input_length=length,
                    batch_size=4,
                    capacity=capacity,
                    math_reduction=reduced,
                    output_ids_match=True,
                    graph_cache_lengths=graph_lengths,
                    graph_position=graph_position,
                    graph_output_index=graph_output_index,
                    max_key_abs_error=max_key_error,
                    max_value_abs_error=max_value_error,
                    kv_numerical_check_passed=True,
                )
            )
            del tensors, runner
            torch.cuda.empty_cache()
    baseline.assert_no_tokenspeed()
    report["tokenspeed_modules_loaded"] = []
    report["passed"] = True
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "rmsnorm_ablation"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
