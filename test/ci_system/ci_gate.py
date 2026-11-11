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

"""Report skipped coverage explicitly and reject missing required CI results."""

import json
import os
from pathlib import Path


def evaluate(results: dict, *, required: set[str]) -> tuple[list[str], str]:
    failures = []
    lines = ["| Job | Result | Required |", "| --- | --- | --- |"]
    for name in sorted(results.keys() | required):
        value = results.get(name, {}).get("result", "missing")
        mandatory = name in required
        if (mandatory and value != "success") or value not in {"success", "skipped"}:
            failures.append(name)
        coverage = "not executed" if value == "skipped" else value
        lines.append(f"| {name} | {coverage} | {'yes' if mandatory else 'no'} |")
    lines.append("\nSkipped jobs provide no test coverage.")
    return failures, "\n".join(lines) + "\n"


def main() -> None:
    failures, summary = evaluate(
        json.loads(os.environ["CI_RESULTS"]),
        required=set(os.environ["CI_REQUIRED_JOBS"].split()),
    )
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as output:
        output.write(summary)
    print(summary)
    if failures:
        raise SystemExit(f"Required CI did not pass: {', '.join(failures)}")


if __name__ == "__main__":
    main()
