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

"""Compare distinct vLLM and TokenSpeed engines; there is no integration toggle."""

import random

from tools.perf.compare_without_tokenspeed import summarize_case


def compare(vllm, tokenspeed):
    if not vllm["passed"] or not tokenspeed["passed"]:
        raise ValueError("Both engine reports must be complete")
    if vllm["engine_name"] != "vllm" or not vllm["actual_engine_class"].startswith(
        "vllm."
    ):
        raise ValueError("Baseline is not the native vLLM engine")
    if vllm["tokenspeed_integration"] or vllm["tokenspeed_modules_loaded"]:
        raise ValueError("vLLM baseline must not load TokenSpeed")
    if vllm["engine"]["skip_tokenizer_init"] or not vllm["sampling"]["detokenize"]:
        raise ValueError("Primary comparison requires normal text-output APIs")
    for field in ("gpu", "torch", "model_files"):
        if vllm[field] != tokenspeed[field]:
            raise ValueError(f"Incomparable {field}")
    if (
        vllm["engine"]["dtype"] != tokenspeed["engine"]["dtype"]
        or vllm["resolved"]["quantization"] is not None
    ):
        raise ValueError("Incomparable precision/quantization")
    if vllm["resolved"]["dtype"] != "torch.bfloat16":
        raise ValueError("vLLM did not resolve to BF16")
    if (
        vllm["engine"]["enforce_eager"] != tokenspeed["engine"]["enforce_eager"]
        or vllm["resolved"]["enforce_eager"] != vllm["engine"]["enforce_eager"]
    ):
        raise ValueError("Incomparable execution modes")
    if (vllm["execution"] == "eager") != vllm["engine"]["enforce_eager"]:
        raise ValueError("Incorrect execution label")
    graph_mode = str(vllm["resolved"]["cudagraph_mode"]).removeprefix("CUDAGraphMode.")
    if vllm["execution"] == "graph" and graph_mode not in (
        "FULL",
        "PIECEWISE",
        "FULL_DECODE_ONLY",
        "FULL_AND_PIECEWISE",
    ):
        raise ValueError("vLLM graph silently disabled or unresolved")
    if (
        vllm["engine"]["enable_prefix_caching"]
        or vllm["resolved"]["prefix_caching"]
        or tokenspeed["engine"]["enable_prefix_caching"]
    ):
        raise ValueError("Prefix caching must be disabled")
    if vllm["seed"] != tokenspeed["engine"]["seed"]:
        raise ValueError("Incomparable input seed")
    if vllm["warmup"] != tokenspeed["warmup"] or vllm["rounds"] != tokenspeed["rounds"]:
        raise ValueError("Incomparable measurement protocol")
    if (
        vllm["output_length"] != tokenspeed["sampling"]["max_new_tokens"]
        or vllm["output_length"] != vllm["sampling"]["max_tokens"]
        or not vllm["sampling"]["ignore_eos"]
        or not tokenspeed["sampling"]["ignore_eos"]
    ):
        raise ValueError("Incomparable fixed output policy")
    cases = {(case["input_length"], case["batch_size"]): case for case in vllm["cases"]}
    ts_cases = {
        (case["input_length"], case["batch_size"]): case for case in tokenspeed["cases"]
    }
    expected = {
        (length, batch)
        for length in vllm["input_lengths"]
        for batch in vllm["batch_sizes"]
    }
    if (
        set(cases) != expected
        or set(ts_cases) != expected
        or len(cases) != len(vllm["cases"])
        or len(ts_cases) != len(tokenspeed["cases"])
    ):
        raise ValueError("Missing or duplicate benchmark cases")
    rows = []
    for (length, batch), case in cases.items():
        rng = random.Random(20261028 + length * 100 + batch)
        inputs = [
            [rng.randrange(1000, 32000) for _ in range(length)] for _ in range(batch)
        ]
        if case["input_ids"] != inputs:
            raise ValueError("Incomparable input token IDs")
        left = summarize_case(
            case, vllm["output_length"], vllm["rounds"], vllm["warmup"]
        )
        right = summarize_case(
            ts_cases[length, batch],
            vllm["output_length"],
            vllm["rounds"],
            vllm["warmup"],
        )
        ratio = right["tokens_per_second"] / left["tokens_per_second"]
        rows.append(
            dict(
                input_length=length,
                batch_size=batch,
                vllm=left,
                tokenspeed=right,
                tokenspeed_over_vllm_ratio=ratio,
                tokenspeed_relative_percent=100 * (ratio - 1),
                reference_ids_match=left["reference_ids"] == right["reference_ids"],
            )
        )
    return rows
