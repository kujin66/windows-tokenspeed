import difflib
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / '.port/flashinfer-comparison'
source = Path(importlib.util.find_spec('tokenspeed_kernel').origin).parent
control = root / '.venv/flashinfer0617-kernel-triton/tokenspeed_kernel'
control.parent.mkdir(exist_ok=True)
shutil.copytree(source, control, ignore=shutil.ignore_patterns('__pycache__'))
relative = 'ops/moe/flashinfer.py'
original = (source / relative).read_text(encoding='utf-8')
assert original.count('if platform.is_nvidia:') == 10
assert original.count('if not platform.is_nvidia:') == 1
adapted = original.replace('platform = current_platform()', 'platform = current_platform()\n\n# Dense-only comparison: this older wheel lacks newer MoE APIs.\n_ENABLE_FLASHINFER_MOE = False')
adapted = adapted.replace('if platform.is_nvidia:', 'if platform.is_nvidia and _ENABLE_FLASHINFER_MOE:')
adapted = adapted.replace('if not platform.is_nvidia:', 'if not platform.is_nvidia or not _ENABLE_FLASHINFER_MOE:')
for package in (control, root / '.venv/flashinfer0617-kernel/tokenspeed_kernel'):
    (package / relative).write_text(adapted, encoding='utf-8', newline='\n')
patch = ''.join(difflib.unified_diff(original.splitlines(True), adapted.splitlines(True), fromfile='installed/' + relative, tofile='experiment/' + relative))
(out / 'dense-compat.patch').write_text(patch, encoding='utf-8', newline='\n')
manifest = []
for original_file in source.rglob('*'):
    if not original_file.is_file() or '__pycache__' in original_file.parts:
        continue
    name = original_file.relative_to(source).as_posix()
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in (original_file, control / name, root / '.venv/flashinfer0617-kernel/tokenspeed_kernel' / name)]
    assert hashes[0] == hashes[1] or name == relative, name
    assert hashes[1] == hashes[2] or name in ('ops/activation/__init__.py', 'ops/layernorm/__init__.py'), name
    manifest.append(dict(file=name, installed_sha256=hashes[0], control_sha256=hashes[1], flashinfer_sha256=hashes[2]))
(out / 'dense-compat-overlay.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
for suffix in ('.json', '.log'):
    old = out / ('windows-flashinfer-eager' + suffix)
    target = out / ('windows-flashinfer-eager-moe-import-failure' + suffix)
    if old.exists() and not target.exists():
        old.rename(target)
shutil.copy2(root / '.port/flashinfer_comparison_bench.py', out / 'benchmark-before-dense-compat.py')
print('Created matching dense-only MoE compatibility overlays; two numerical facades differ between control and FlashInfer')
