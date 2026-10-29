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
from pathlib import Path

root = (
    Path(__file__)
    .resolve()
    .parents[3 if Path(__file__).parent.name == "flashinfer" else 1]
)
out = root / ".port/flashinfer-comparison"
out.mkdir(parents=True, exist_ok=True)
path = root / ".venv/flashinfer0617/flashinfer/jit/cpp_ext.py"
before = path.read_text(encoding="utf-8")
old = '        obj_name = f"{source.parent.name}_{source.stem}{object_suffix}"'
assert before.count(old) == 1
after = before.replace("import functools\n", "import functools\nimport hashlib\n")
after = after.replace(
    old,
    "        if is_windows:\n            obj_name = hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:16] + object_suffix\n        else:\n"
    + old.replace("        ", "            ", 1),
)
path.write_text(after, encoding="utf-8", newline="\n")
(out / "windows-jit-object-paths.patch").write_text(
    "".join(
        difflib.unified_diff(
            before.splitlines(True),
            after.splitlines(True),
            fromfile="wheel/flashinfer/jit/cpp_ext.py",
            tofile="experiment/flashinfer/jit/cpp_ext.py",
        )
    ),
    encoding="utf-8",
)
print(
    "Shortened Windows JIT object filenames by stable path hashes; kernel code, flags and architecture checks unchanged"
)
