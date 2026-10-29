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

import difflib
import hashlib
import json
import shutil
from pathlib import Path

root = (
    Path(__file__)
    .resolve()
    .parents[3 if Path(__file__).parent.name == "flashinfer" else 1]
)
out = root / ".port/flashinfer-comparison"
out.mkdir(parents=True, exist_ok=True)
module = root / ".port/fi_sm121_fa2.py"
header = (
    (root / "tools/perf/benchmark_tokenspeed.py")
    .read_text(encoding="utf-8")
    .split("\n\n", 1)[0]
)
content = header + "\n\n" + module.read_text(encoding="utf-8")
manifest = []
for name in ("flashinfer0617-kernel", "flashinfer0617-kernel-triton"):
    directory = root / ".venv" / name / "tokenspeed_kernel/ops/attention/mha"
    (directory / "_sm121_fa2.py").write_text(content, encoding="utf-8", newline="\n")
    path = directory / "flashinfer.py"
    before = path.read_text(encoding="utf-8")
    after = (
        before
        + "\n# Experimental SM12.1 FA2 prefill; architecture checks remain intact.\nimport tokenspeed_kernel.ops.attention.mha._sm121_fa2  # noqa: E402,F401\n"
    )
    path.write_text(after, encoding="utf-8", newline="\n")
    manifest.append(
        dict(
            package=name,
            adapter_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            registration_sha256=hashlib.sha256(after.encode("utf-8")).hexdigest(),
        )
    )
    if name == "flashinfer0617-kernel":
        (out / "fa2-registration.patch").write_text(
            "".join(
                difflib.unified_diff(
                    before.splitlines(True),
                    after.splitlines(True),
                    fromfile="installed/ops/attention/mha/flashinfer.py",
                    tofile="experiment/ops/attention/mha/flashinfer.py",
                )
            ),
            encoding="utf-8",
        )
(out / "fa2-overlay.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
print(
    "Installed architecture-specific FA2 prefill in both isolated packages; TRTLLM decode remains unchanged"
)
