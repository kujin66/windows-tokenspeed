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

import pytest

from tools.perf.benchmark_tokenspeed import check_consistency, summarize, verify_outputs
from tools.perf.compare_results import compare


def reports():
    base = {
        "passed": True,
        "schema": 1,
        "benchmark_sha256": "script",
        "upstream_base": "base",
        "workload": {
            "rounds": 2,
            "input_lengths": [128],
            "batch_sizes": [1],
            "output_length": 2,
        },
        "model_files": {"weights": "sha"},
        "gpu": {"name": "test"},
        "cases": [
            {
                "input_length": 128,
                "batch_size": 1,
                "input_ids_sha256": "prompt",
                "output_ids": [[7, 8]],
                "greedy_output_ids_consistent": True,
                "samples": [
                    {"seconds": 2, "output_tokens": 2},
                    {"seconds": 4, "output_tokens": 2},
                ],
            }
        ],
    }
    windows, wsl = copy.deepcopy(base), copy.deepcopy(base)
    windows["platform"] = "windows-arm64"
    wsl["platform"] = "wsl2-arm64"
    return windows, wsl


def test_aggregate_throughput_uses_total_time():
    result = summarize(
        [{"seconds": 1, "output_tokens": 10}, {"seconds": 3, "output_tokens": 10}]
    )
    assert result["output_tokens_per_second"] == 5
    assert result["median_seconds"] == 2
    assert result["p95_seconds"] == 3


@pytest.mark.parametrize(
    "samples",
    [[], [{"seconds": 0, "output_tokens": 1}], [{"seconds": 1, "output_tokens": 0}]],
)
def test_invalid_samples_rejected(samples):
    with pytest.raises(ValueError):
        summarize(samples)


def test_short_or_wrong_batch_completion_rejected():
    result = {"output_ids": [1], "meta_info": {"completion_tokens": 1}}
    assert verify_outputs(result, 1, 1) == [[1]]
    with pytest.raises(ValueError):
        verify_outputs(result, 1, 2)
    with pytest.raises(ValueError):
        verify_outputs(result, 2, 1)


def test_difference_recomputed_from_raw_samples():
    windows, wsl = reports()
    for sample in windows["cases"][0]["samples"]:
        sample["seconds"] /= 2
    result = compare(windows, wsl)["cases"][0]
    assert result["windows_relative_to_wsl2_percent"] == 100
    assert result["greedy_output_ids_match"]


@pytest.mark.parametrize(
    "key",
    ["schema", "benchmark_sha256", "upstream_base", "workload", "model_files", "gpu"],
)
def test_mismatched_environment_rejected(key):
    windows, wsl = reports()
    wsl[key] = "different"
    with pytest.raises(ValueError, match=key):
        compare(windows, wsl)


def test_incomplete_report_rejected():
    windows, wsl = reports()
    wsl["passed"] = False
    with pytest.raises(ValueError):
        compare(windows, wsl)


def test_shortened_measurements_rejected():
    windows, wsl = reports()
    wsl["cases"][0]["samples"].pop()
    with pytest.raises(ValueError, match="rounds"):
        compare(windows, wsl)


def test_incorrect_token_count_rejected():
    windows, wsl = reports()
    wsl["cases"][0]["samples"][0]["output_tokens"] = 1
    with pytest.raises(ValueError, match="token count"):
        compare(windows, wsl)


def test_record_policy_preserves_numerical_variation():
    assert check_consistency([[1]], [[1]], policy="record")
    assert not check_consistency([[1]], [[2]], policy="record")
    with pytest.raises(ValueError):
        check_consistency([[1]], [[2]], policy="require")
    windows, wsl = reports()
    windows["cases"][0]["greedy_output_ids_consistent"] = False
    result = compare(windows, wsl)["cases"][0]
    assert not result["windows_within_case_greedy_consistent"]
    assert result["windows_relative_to_wsl2_percent"] == 0
