import difflib
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = Path(importlib.util.find_spec('tokenspeed_kernel').origin).parent
target = root / '.venv/flashinfer0617-kernel/tokenspeed_kernel'
target.parent.mkdir(exist_ok=True)
shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__'))
changes = {
    'ops/activation/__init__.py': [('        or sys.platform == "win32"\n', '')],
    'ops/layernorm/__init__.py': [('elif _platform.is_amd or sys.platform == "win32":', 'elif _platform.is_amd:'), ('if _platform.is_amd or sys.platform == "win32":', 'if _platform.is_amd:')],
}
patch = []
manifest = []
for path in source.rglob('*'):
    if not path.is_file() or '__pycache__' in path.parts:
        continue
    relative = path.relative_to(source).as_posix()
    copied = target / relative
    before = path.read_bytes()
    if relative in changes:
        original = before.decode('utf-8').replace('\r\n', '\n')
        value = original
        for old, new in changes[relative]:
            assert value.count(old) == 1, (relative, old)
            value = value.replace(old, new)
        copied.write_bytes(value.encode('utf-8'))
        patch.extend(difflib.unified_diff(original.splitlines(True), value.splitlines(True), fromfile='installed/' + relative, tofile='experiment/' + relative))
    else:
        assert before == copied.read_bytes(), relative
    manifest.append(dict(file=relative, installed_sha256=hashlib.sha256(before).hexdigest(), experimental_sha256=hashlib.sha256(copied.read_bytes()).hexdigest(), changed=relative in changes))
out = root / '.port/flashinfer-comparison'
(out / 'kernel-overlay.patch').write_text(''.join(patch), encoding='utf-8', newline='\n')
(out / 'kernel-overlay.json').write_text(json.dumps(dict(source=str(source), target=str(target), files=manifest), indent=2) + '\n', encoding='utf-8')
print(f'Created isolated kernel overlay: {len(manifest)} files, exactly {len(changes)} changed facades; all native binaries unchanged')
