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

"""Exercise same-platform engine comparison integrity without a GPU."""

import copy
import random

import pytest

from tools.perf.compare_without_tokenspeed import compare


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
    hf = dict(
        passed=True,
        tokenspeed_modules_loaded=[],
        implementation="generate",
        model_files={"weights": "same"},
        gpu="same",
        torch="same",
        torch_cuda="same",
        dtype="bfloat16",
        attention_backend="sdpa",
        actual_attention_backend="sdpa",
        rounds=2,
        warmup=1,
        output_length=2,
        input_lengths=[2],
        batch_sizes=[1],
        seed=20261028,
        cases=[case],
    )
    ts = dict(
        passed=True,
        model_files={"weights": "same"},
        gpu="same",
        torch=dict(version="same", cuda="same"),
        engine=dict(dtype="bfloat16", enforce_eager=True, seed=20261028),
        sampling=dict(max_new_tokens=2, ignore_eos=True),
        rounds=2,
        warmup=1,
        cases=[copy.deepcopy(case)],
    )
    return hf, ts


def test_aggregate_engine_ratio_recomputed():
    hf, ts = reports()
    for sample in ts["cases"][0]["samples"]:
        sample["seconds"] /= 2
    ts["cases"][0]["tokens_per_second"] *= 2
    row = compare(hf, ts)[0]
    assert row["tokenspeed_over_baseline_ratio"] == 2
    assert row["tokenspeed_relative_percent"] == 100
    assert row["reference_ids_match"]


@pytest.mark.parametrize(
    "field", ["model_files", "gpu", "torch", "torch_cuda", "dtype"]
)
def test_mismatched_environment_rejected(field):
    hf, ts = reports()
    hf[field] = "different"
    with pytest.raises(ValueError, match="Incomparable"):
        compare(hf, ts)


def test_imported_tokenspeed_invalidates_baseline():
    hf, ts = reports()
    hf["tokenspeed_modules_loaded"] = ["tokenspeed"]
    with pytest.raises(ValueError, match="imported TokenSpeed"):
        compare(hf, ts)


def test_eager_graph_mismatch_rejected():
    hf, ts = reports()
    ts["engine"]["enforce_eager"] = False
    with pytest.raises(ValueError, match="eager with eager"):
        compare(hf, ts)


def test_bad_prompt_rejected():
    hf, ts = reports()
    hf["cases"][0]["input_ids"][0][0] += 1
    with pytest.raises(ValueError, match="input token IDs"):
        compare(hf, ts)


def test_incomplete_rounds_rejected():
    hf, ts = reports()
    hf["cases"][0]["samples"].pop()
    with pytest.raises(ValueError, match="Incomplete"):
        compare(hf, ts)


def test_inconsistent_output_is_reported_without_hiding_throughput():
    hf, ts = reports()
    hf["cases"][0]["samples"][0]["output_ids"] = [[30, 40]]
    row = compare(hf, ts)[0]
    assert not row["without_tokenspeed"]["repeat_consistent"]
    assert not row["reference_ids_match"]
    assert row["tokenspeed_over_baseline_ratio"] == 1


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_invalid_duration_rejected(duration):
    hf, ts = reports()
    hf["cases"][0]["samples"][0]["seconds"] = duration
    with pytest.raises(ValueError, match="duration"):
        compare(hf, ts)


def test_graph_proof_required():
    hf, ts = reports()
    hf["implementation"] = "loop-static-graph"
    ts["engine"]["enforce_eager"] = False
    hf["cases"][0]["graph_validation"] = dict(
        exact_match=False, graph_ids=[[1, 2]], eager_ids=[[3, 4]], cache_lengths=[3]
    )
    with pytest.raises(ValueError, match="Unvalidated"):
        compare(hf, ts)
