import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

wheel = Path(__file__).resolve().parent / 'wheelhouse/flashinfer_python-0.6.17+g2ea3c24b5dac-py3-none-any.whl'
out = Path(__file__).resolve().parent / 'flashinfer-comparison'
out.mkdir(exist_ok=True)
with ZipFile(wheel) as archive:
    names = archive.namelist()
    metadata = {name: archive.read(name).decode('utf-8') for name in names if name.endswith(('/METADATA', '/WHEEL'))}
    native = [name for name in names if name.endswith(('.dll', '.pyd', '.so', '.lib'))]
    windows_files = [name for name in names if 'windows' in name.lower() or 'msvc' in name.lower()]
    report = dict(wheel=str(wheel), bytes=wheel.stat().st_size, sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(), metadata=metadata, native_files=native, windows_files=windows_files)
    (out / 'wheel-audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    for name in ('flashinfer/jit/env.py', 'flashinfer/jit/cpp_ext.py', 'flashinfer/jit/core.py'):
        if name in names:
            path = out / ('wheel-' + name.rsplit('/', 1)[-1])
            path.write_bytes(archive.read(name))
