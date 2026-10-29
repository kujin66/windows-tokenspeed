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

"""Check the native chain verifier against a sequential acceptance oracle."""

import json
from pathlib import Path

import torch
from tokenspeed_kernel.thirdparty.cuda.sampling_chain import verify_chain_greedy

torch.manual_seed(42)
cases = []
for batch in (1, 7, 32):
    for width in (1, 2, 4, 8):
        for pattern in ("all", "none", "partial"):
            target = (
                torch.arange(batch * width, dtype=torch.int64).view(batch, width) + 100
            )
            candidates = torch.full((batch, width), -7, dtype=torch.int32)
            candidates[:, 1:] = target[:, :-1].to(torch.int32)
            for row in range(batch):
                mismatch = (
                    width
                    if pattern == "all"
                    else (0 if pattern == "none" else row % width)
                )
                if mismatch < width - 1:
                    candidates[row, mismatch + 1] = -2
            expected_predicts = torch.full((batch * width,), 777, dtype=torch.int32)
            expected_indices = torch.full((batch, width), -1, dtype=torch.int32)
            expected_count = torch.zeros(batch, dtype=torch.int32)
            for row in range(batch):
                count = 0
                while (
                    count < width - 1
                    and candidates[row, count + 1] == target[row, count]
                ):
                    count += 1
                expected_count[row] = count
                for position in range(count + 1):
                    index = row * width + position
                    expected_indices[row, position] = index
                    expected_predicts[index] = target[row, position]
            for pdl in (False, True):
                predicts = torch.full(
                    (batch * width,), 777, dtype=torch.int32, device="cuda"
                )
                indices = torch.full(
                    (batch, width), -1, dtype=torch.int32, device="cuda"
                )
                count = torch.full((batch,), 99, dtype=torch.int32, device="cuda")
                verify_chain_greedy(
                    predicts,
                    indices,
                    count,
                    candidates.cuda(),
                    target.cuda(),
                    batch,
                    width,
                    enable_pdl=pdl,
                )
                torch.cuda.synchronize()
                for actual, expected in (
                    (predicts, expected_predicts),
                    (indices, expected_indices),
                    (count, expected_count),
                ):
                    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
                cases.append(
                    {
                        "batch": batch,
                        "width": width,
                        "pattern": pattern,
                        "pdl": pdl,
                        "passed": True,
                    }
                )
root = Path(__file__).resolve().parents[2]
report = {"device": torch.cuda.get_device_name(), "cases": cases}
(root / ".port/sampling-chain-results.json").write_text(json.dumps(report, indent=2))
print(f"{len(cases)} native sampling-chain cases passed")
