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

"""Validate and compare fresh paired engine reports from the same platform."""

import math
import random


def summarize_case(case, output_length, rounds, warmup):
    batch = case["batch_size"]
    samples = case["samples"]
    if len(samples) != rounds or len(case["warmup_ids"]) != warmup:
        raise ValueError("Incomplete measurement or warmup rounds")
    for sample in samples:
        if not math.isfinite(sample["seconds"]) or sample["seconds"] <= 0:
            raise ValueError("Invalid measurement duration")
    outputs = case["warmup_ids"] + [sample["output_ids"] for sample in samples]
    for ids in outputs:
        if len(ids) != batch or any(len(row) != output_length for row in ids):
            raise ValueError("Output length or batch mismatch")
        if any(
            type(token) is not int or not 0 <= token < 151936
            for row in ids
            for token in row
        ):
            raise ValueError("Invalid output token")
    seconds = sum(sample["seconds"] for sample in samples)
    throughput = batch * output_length * rounds / seconds
    if not math.isclose(throughput, case["tokens_per_second"], rel_tol=1e-12):
        raise ValueError("Stored throughput does not match raw measurements")
    return dict(
        tokens_per_second=throughput,
        measured_requests=batch * rounds,
        measured_output_tokens=batch * rounds * output_length,
        repeat_consistent=all(ids == outputs[0] for ids in outputs),
        reference_ids=samples[0]["output_ids"],
        sample_seconds=[sample["seconds"] for sample in samples],
    )


def compare(hf, tokenspeed):
    if not hf["passed"] or not tokenspeed["passed"]:
        raise ValueError("Both engine reports must be complete")
    if hf["tokenspeed_modules_loaded"]:
        raise ValueError("Baseline imported TokenSpeed modules")
    if hf["implementation"] not in (
        "generate",
        "loop-dynamic",
        "loop-static-eager",
        "loop-static-graph",
    ):
        raise ValueError("Unknown baseline implementation")
    for key in ("model_files", "gpu", "torch", "torch_cuda"):
        left = hf[key]
        right = (
            tokenspeed["torch"]["version"]
            if key == "torch"
            else tokenspeed["torch"]["cuda"] if key == "torch_cuda" else tokenspeed[key]
        )
        if left != right:
            raise ValueError(f"Incomparable {key}")
    if hf["dtype"] != tokenspeed["engine"]["dtype"]:
        raise ValueError("Incomparable dtype")
    if hf["attention_backend"] != "sdpa" or hf["actual_attention_backend"] != "sdpa":
        raise ValueError("Baseline is not the specified SDPA implementation")
    if hf["rounds"] != tokenspeed["rounds"] or hf["warmup"] != tokenspeed["warmup"]:
        raise ValueError("Incomparable measurement protocol")
    if (
        hf["output_length"] != tokenspeed["sampling"]["max_new_tokens"]
        or not tokenspeed["sampling"]["ignore_eos"]
    ):
        raise ValueError("Incomparable fixed output policy")
    hf_graph = hf["implementation"] == "loop-static-graph"
    if hf_graph == tokenspeed["engine"]["enforce_eager"]:
        raise ValueError("Compare eager with eager, graph with graph")
    output_length, rounds, warmup = hf["output_length"], hf["rounds"], hf["warmup"]
    hf_cases = {
        (case["input_length"], case["batch_size"]): case for case in hf["cases"]
    }
    ts_cases = {
        (case["input_length"], case["batch_size"]): case for case in tokenspeed["cases"]
    }
    expected = {
        (length, batch) for length in hf["input_lengths"] for batch in hf["batch_sizes"]
    }
    if (
        set(hf_cases) != expected
        or len(hf_cases) != len(hf["cases"])
        or len(ts_cases) != len(tokenspeed["cases"])
        or not set(hf_cases).issubset(ts_cases)
    ):
        raise ValueError("Missing or duplicate benchmark cases")
    rows = []
    for key, case in hf_cases.items():
        length, batch = key
        rng = random.Random(20261028 + length * 100 + batch)
        expected_inputs = [
            [rng.randrange(1000, 32000) for _ in range(length)] for _ in range(batch)
        ]
        if (
            hf["seed"] != tokenspeed["engine"]["seed"]
            or case["input_ids"] != expected_inputs
        ):
            raise ValueError("Incomparable input token IDs")
        if hf_graph:
            proof = case["graph_validation"]
            expected_length = length + output_length - 1
            if (
                not proof["exact_match"]
                or proof["graph_ids"] != proof["eager_ids"]
                or any(value != expected_length for value in proof["cache_lengths"])
            ):
                raise ValueError("Unvalidated baseline CUDA Graph")
        left = summarize_case(case, output_length, rounds, warmup)
        right = summarize_case(ts_cases[key], output_length, rounds, warmup)
        ratio = right["tokens_per_second"] / left["tokens_per_second"]
        rows.append(
            dict(
                input_length=length,
                batch_size=batch,
                without_tokenspeed=left,
                tokenspeed=right,
                tokenspeed_over_baseline_ratio=ratio,
                tokenspeed_relative_percent=100 * (ratio - 1),
                reference_ids_match=left["reference_ids"] == right["reference_ids"],
            )
        )
    return rows
