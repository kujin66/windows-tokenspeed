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

import copy
import random

import pytest

from tools.perf.compare_vllm import compare


def reports():
    rng = random.Random(20261028 + 2 * 100 + 1)
    case = dict(
        input_length=2,
        batch_size=1,
        input_ids=[[rng.randrange(1000, 32000) for _ in range(2)]],
        warmup_ids=[[[10, 20]]],
        samples=[
            dict(seconds=2.0, output_ids=[[10, 20]]),
            dict(seconds=4.0, output_ids=[[10, 20]]),
        ],
        tokens_per_second=4 / 6,
    )
    vllm = dict(
        passed=True,
        engine_name="vllm",
        actual_engine_class="vllm.v1.engine.llm_engine.LLMEngine",
        tokenspeed_integration=False,
        tokenspeed_modules_loaded=[],
        gpu="gpu",
        torch=dict(version="torch", cuda="cuda"),
        model_files=dict(weights="same"),
        engine=dict(dtype="bfloat16", enforce_eager=True, enable_prefix_caching=False),
        resolved=dict(
            dtype="torch.bfloat16",
            quantization=None,
            enforce_eager=True,
            prefix_caching=False,
            cudagraph_mode="CUDAGraphMode.NONE",
        ),
        seed=20261028,
        execution="eager",
        input_lengths=[2],
        batch_sizes=[1],
        output_length=2,
        warmup=1,
        rounds=2,
        sampling=dict(max_tokens=2, ignore_eos=True),
        cases=[case],
    )
    ts = dict(
        passed=True,
        gpu="gpu",
        torch=dict(version="torch", cuda="cuda"),
        model_files=dict(weights="same"),
        engine=dict(
            dtype="bfloat16",
            enforce_eager=True,
            enable_prefix_caching=False,
            seed=20261028,
        ),
        sampling=dict(max_new_tokens=2, ignore_eos=True),
        warmup=1,
        rounds=2,
        cases=[copy.deepcopy(case)],
    )
    vllm["engine"]["skip_tokenizer_init"] = False
    vllm["sampling"]["detokenize"] = True
    return vllm, ts


def test_actual_throughput_loss_is_reported():
    vllm, ts = reports()
    for sample in ts["cases"][0]["samples"]:
        sample["seconds"] *= 2
    ts["cases"][0]["tokens_per_second"] /= 2
    row = compare(vllm, ts)[0]
    assert row["tokenspeed_over_vllm_ratio"] == 0.5
    assert row["tokenspeed_relative_percent"] == -50


@pytest.mark.parametrize("field", ["gpu", "torch", "model_files"])
def test_mismatched_deployment_pair_rejected(field):
    vllm, ts = reports()
    vllm[field] = "different"
    with pytest.raises(ValueError, match="Incomparable"):
        compare(vllm, ts)


def test_fake_tokenspeed_toggle_rejected():
    vllm, ts = reports()
    vllm["tokenspeed_integration"] = True
    with pytest.raises(ValueError, match="must not load"):
        compare(vllm, ts)


@pytest.mark.parametrize("mode_name", ["NONE", "CUDAGraphMode.NONE"])
def test_silent_graph_fallback_rejected(mode_name):
    vllm, ts = reports()
    vllm["engine"]["enforce_eager"] = False
    vllm["resolved"]["enforce_eager"] = False
    ts["engine"]["enforce_eager"] = False
    vllm["execution"] = "graph"
    vllm["resolved"]["cudagraph_mode"] = mode_name
    with pytest.raises(ValueError, match="silently disabled"):
        compare(vllm, ts)


def test_real_decode_graph_policy_is_accepted():
    vllm, ts = reports()
    vllm["execution"] = "graph"
    vllm["engine"]["enforce_eager"] = False
    vllm["resolved"]["enforce_eager"] = False
    vllm["resolved"]["cudagraph_mode"] = "FULL_DECODE_ONLY"
    ts["engine"]["enforce_eager"] = False
    assert compare(vllm, ts)[0]["tokenspeed_over_vllm_ratio"] == 1


def test_quantized_baseline_rejected():
    vllm, ts = reports()
    vllm["resolved"]["quantization"] = "nvfp4"
    with pytest.raises(ValueError, match="quantization"):
        compare(vllm, ts)


def test_different_generated_ids_retained():
    vllm, ts = reports()
    ts["cases"][0]["samples"][1]["output_ids"] = [[30, 40]]
    row = compare(vllm, ts)[0]
    assert not row["tokenspeed"]["repeat_consistent"]
    assert row["reference_ids_match"]


def test_cached_repeated_prompt_rejected():
    vllm, ts = reports()
    vllm["resolved"]["prefix_caching"] = True
    with pytest.raises(ValueError, match="caching"):
        compare(vllm, ts)


def test_output_length_mismatch_rejected():
    vllm, ts = reports()
    ts["cases"][0]["samples"][0]["output_ids"] = [[10]]
    with pytest.raises(ValueError, match="Output length"):
        compare(vllm, ts)


@pytest.mark.parametrize("policy", ["skip_tokenizer", "no_detokenization"])
def test_unequal_output_policy_rejected(policy):
    vllm, ts = reports()
    if policy == "skip_tokenizer":
        vllm["engine"]["skip_tokenizer_init"] = True
    else:
        vllm["sampling"]["detokenize"] = False
    with pytest.raises(ValueError, match="text-output"):
        compare(vllm, ts)
