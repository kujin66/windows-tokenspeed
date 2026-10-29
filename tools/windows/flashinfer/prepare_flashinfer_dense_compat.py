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
import importlib.util
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
source = Path(importlib.util.find_spec("tokenspeed_kernel").origin).parent
control = root / ".venv/flashinfer0617-kernel-triton/tokenspeed_kernel"
control.parent.mkdir(exist_ok=True)
shutil.copytree(source, control, ignore=shutil.ignore_patterns("__pycache__"))
relative = "ops/moe/flashinfer.py"
original = (source / relative).read_text(encoding="utf-8")
assert original.count("if platform.is_nvidia:") == 10
assert original.count("if not platform.is_nvidia:") == 1
adapted = original.replace(
    "platform = current_platform()",
    "platform = current_platform()\n\n# Dense-only comparison: this older wheel lacks newer MoE APIs.\n_ENABLE_FLASHINFER_MOE = False",
)
adapted = adapted.replace(
    "if platform.is_nvidia:", "if platform.is_nvidia and _ENABLE_FLASHINFER_MOE:"
)
adapted = adapted.replace(
    "if not platform.is_nvidia:",
    "if not platform.is_nvidia or not _ENABLE_FLASHINFER_MOE:",
)
for package in (control, root / ".venv/flashinfer0617-kernel/tokenspeed_kernel"):
    (package / relative).write_text(adapted, encoding="utf-8", newline="\n")
patch = "".join(
    difflib.unified_diff(
        original.splitlines(True),
        adapted.splitlines(True),
        fromfile="installed/" + relative,
        tofile="experiment/" + relative,
    )
)
(out / "dense-compat.patch").write_text(patch, encoding="utf-8", newline="\n")
manifest = []
for original_file in source.rglob("*"):
    if not original_file.is_file() or "__pycache__" in original_file.parts:
        continue
    name = original_file.relative_to(source).as_posix()
    hashes = [
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            original_file,
            control / name,
            root / ".venv/flashinfer0617-kernel/tokenspeed_kernel" / name,
        )
    ]
    assert hashes[0] == hashes[1] or name == relative, name
    assert hashes[1] == hashes[2] or name in (
        "ops/activation/__init__.py",
        "ops/layernorm/__init__.py",
    ), name
    manifest.append(
        dict(
            file=name,
            installed_sha256=hashes[0],
            control_sha256=hashes[1],
            flashinfer_sha256=hashes[2],
        )
    )
(out / "dense-compat-overlay.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
print(
    "Created matching dense-only MoE compatibility overlays; two numerical facades differ between control and FlashInfer"
)
