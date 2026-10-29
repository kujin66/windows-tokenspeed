import difflib
from pathlib import Path

base = Path('/opt/tokenspeed-work/venv')
source = base / 'flashinfer-comparison/tokenspeed_kernel'
target = base / 'flashinfer-comparison-triton-retest/tokenspeed_kernel'
patch = []
for rel in ('ops/layernorm/__init__.py','ops/activation/__init__.py'):
    patch.extend(difflib.unified_diff((source/rel).read_text().splitlines(True),(target/rel).read_text().splitlines(True),fromfile='flashinfer/'+rel,tofile='triton/'+rel))
out = Path(__file__).resolve().parent/'engine-matrix-retest-20261029/wsl-triton-overlay.patch'
out.write_text(''.join(patch),encoding='utf-8')
print(f'Wrote {out.name}')
