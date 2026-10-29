import difflib
import hashlib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / '.port/flashinfer-comparison'
path = root / '.venv/flashinfer0617/flashinfer/jit/cpp_ext.py'
before = path.read_text(encoding='utf-8')
old = '        obj_name = f"{source.parent.name}_{source.stem}{object_suffix}"'
assert before.count(old) == 1
after = before.replace('import functools\n', 'import functools\nimport hashlib\n')
after = after.replace(old, '        if is_windows:\n            obj_name = hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:16] + object_suffix\n        else:\n' + old.replace('        ', '            ', 1))
path.write_text(after, encoding='utf-8', newline='\n')
(out / 'windows-jit-object-paths.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile='wheel/flashinfer/jit/cpp_ext.py', tofile='experiment/flashinfer/jit/cpp_ext.py')), encoding='utf-8')
for suffix in ('.json', '.log'):
    old_path = out / ('fa2-prefill-probe' + suffix)
    target = out / ('fa2-prefill-probe-long-path-failure' + suffix)
    if old_path.exists() and not target.exists():
        old_path.rename(target)
print('Shortened Windows JIT object filenames by stable path hashes; kernel code, flags and architecture checks unchanged')
