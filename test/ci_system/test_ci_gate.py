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

import pytest
from ci_gate import evaluate


@pytest.mark.parametrize("result", ["skipped", "failure", "cancelled", "missing"])
def test_required_test_must_actually_succeed(result):
    jobs = {} if result == "missing" else {"native": {"result": result}}
    failures, summary = evaluate(jobs, required={"native"})
    assert failures == ["native"]
    assert "yes" in summary


def test_optional_skip_is_visible_and_does_not_claim_coverage():
    failures, summary = evaluate(
        {"scan": {"result": "success"}, "models": {"result": "skipped"}},
        required={"scan"},
    )
    assert failures == []
    assert "models | not executed | no" in summary


def test_failed_optional_job_still_fails_the_gate():
    failures, _ = evaluate({"models": {"result": "failure"}}, required=set())
    assert failures == ["models"]


def test_all_required_jobs_pass():
    failures, _ = evaluate(
        {"scan": {"result": "success"}, "native": {"result": "success"}},
        required={"scan", "native"},
    )
    assert failures == []
