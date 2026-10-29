import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
source = Path(importlib.util.find_spec('tokenspeed_kernel').origin).parent
target = Path('/opt/tokenspeed-work/venv/flashinfer-comparison/tokenspeed_kernel')
target.parent.mkdir(exist_ok=True)
shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__'))
header = (repo / 'tools/perf/benchmark_tokenspeed.py').read_text(encoding='utf-8').split('\n\n', 1)[0]
adapter = header + '\n\n' + (repo / '.port/fi_sm121_fa2.py').read_text(encoding='utf-8')
directory = target / 'ops/attention/mha'
(directory / '_sm121_fa2.py').write_text(adapter, encoding='utf-8')
path = directory / 'flashinfer.py'
path.write_text(path.read_text(encoding='utf-8') + '\nimport tokenspeed_kernel.ops.attention.mha._sm121_fa2  # noqa: E402,F401\n', encoding='utf-8')
manifest = []
for path in source.rglob('*'):
    if path.is_file() and '__pycache__' not in path.parts:
        relative = path.relative_to(source)
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        after = hashlib.sha256((target / relative).read_bytes()).hexdigest()
        assert before == after or relative.as_posix() == 'ops/attention/mha/flashinfer.py'
        manifest.append(dict(file=relative.as_posix(), original_sha256=before, overlay_sha256=after))
result = dict(source=str(source), target=str(target), adapter_sha256=hashlib.sha256(adapter.encode('utf-8')).hexdigest(), files=manifest)
(repo / '.port/flashinfer-comparison/wsl2-overlay.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print('Prepared WSL2 official kernel copy with matching SM12.1 FA2 prefill provider')
