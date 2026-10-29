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
source = Path(importlib.util.find_spec("tokenspeed_kernel").origin).parent
target = root / ".venv/flashinfer0617-kernel/tokenspeed_kernel"
target.parent.mkdir(exist_ok=True)
shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
changes = {
    "ops/activation/__init__.py": [('        or sys.platform == "win32"\n', "")],
    "ops/layernorm/__init__.py": [
        ('elif _platform.is_amd or sys.platform == "win32":', "elif _platform.is_amd:"),
        ('if _platform.is_amd or sys.platform == "win32":', "if _platform.is_amd:"),
    ],
}
patch = []
manifest = []
for path in source.rglob("*"):
    if not path.is_file() or "__pycache__" in path.parts:
        continue
    relative = path.relative_to(source).as_posix()
    copied = target / relative
    before = path.read_bytes()
    if relative in changes:
        original = before.decode("utf-8").replace("\r\n", "\n")
        value = original
        for old, new in changes[relative]:
            assert value.count(old) == 1, (relative, old)
            value = value.replace(old, new)
        copied.write_bytes(value.encode("utf-8"))
        patch.extend(
            difflib.unified_diff(
                original.splitlines(True),
                value.splitlines(True),
                fromfile="installed/" + relative,
                tofile="experiment/" + relative,
            )
        )
    else:
        assert before == copied.read_bytes(), relative
    manifest.append(
        dict(
            file=relative,
            installed_sha256=hashlib.sha256(before).hexdigest(),
            experimental_sha256=hashlib.sha256(copied.read_bytes()).hexdigest(),
            changed=relative in changes,
        )
    )
out = root / ".port/flashinfer-comparison"
out.mkdir(parents=True, exist_ok=True)
(out / "kernel-overlay.patch").write_text(
    "".join(patch), encoding="utf-8", newline="\n"
)
(out / "kernel-overlay.json").write_text(
    json.dumps(dict(source=str(source), target=str(target), files=manifest), indent=2)
    + "\n",
    encoding="utf-8",
)
print(
    f"Created isolated kernel overlay: {len(manifest)} files, exactly {len(changes)} changed facades; all native binaries unchanged"
)
