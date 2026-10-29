import hashlib
import json
import shutil
import struct
from pathlib import Path
from zipfile import ZipFile

root = Path(__file__).resolve().parents[1]
out = root / '.port/flashinfer-comparison'
wheel = root / '.port/wheelhouse/flashinfer_python-0.6.17+g2ea3c24b5dac-py3-none-any.whl'
assert hashlib.sha256(wheel.read_bytes()).hexdigest() == '47db3e3972fda64719ceae8490d1934362380cef62f33975cb593ad086e0bcb6'
package = root / '.venv/flashinfer0617'
changed = []
checked = 0
with ZipFile(wheel) as archive:
    for name in archive.namelist():
        if name.endswith('/') or name.endswith('.dist-info/RECORD'):
            continue
        path = package / name
        assert path.resolve().is_relative_to(package.resolve())
        expected = archive.read(name)
        actual = path.read_bytes()
        if expected != actual:
            changed.append(dict(file=name, wheel_sha256=hashlib.sha256(expected).hexdigest(), installed_sha256=hashlib.sha256(actual).hexdigest()))
        checked += 1
assert [item['file'] for item in changed] == ['flashinfer/jit/cpp_ext.py']
native = []
cache = root.parents[1] / 'fij0617'
backup = out / 'native-dlls'
backup.mkdir(exist_ok=True)
for index, path in enumerate(sorted(cache.rglob('*.dll'))):
    raw = path.read_bytes()
    pe = struct.unpack_from('<I', raw, 0x3c)[0]
    assert raw[pe:pe+4] == b'PE\0\0'
    machine = struct.unpack_from('<H', raw, pe + 4)[0]
    assert machine == 0xaa64, path
    linker = list(raw[pe+26:pe+28])
    assert linker == [14, 51], (path, linker)
    dest = backup / f'{index:02}-{path.parent.name[:40]}.dll'
    shutil.copy2(path, dest)
    assert dest.read_bytes() == raw
    native.append(dict(path=str(path), backup=dest.relative_to(root).as_posix(), machine=hex(machine), linker=linker, sha256=hashlib.sha256(raw).hexdigest()))
assert len(native) >= 3
report = dict(passed=True, wheel=str(wheel), wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(), files_compared=checked, changed_files=changed, native=native)
(out / 'final-native-audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(f'Validated {checked} wheel files (one documented JIT path patch), {len(native)} ARM64/MSVC14.51 DLLs and local binary backups')
