import difflib
import hashlib
import json
import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / '.port/flashinfer-comparison'
module = root / '.port/fi_sm121_fa2.py'
header = (root / 'tools/perf/benchmark_tokenspeed.py').read_text(encoding='utf-8').split('\n\n', 1)[0]
content = header + '\n\n' + module.read_text(encoding='utf-8')
manifest = []
for name in ('flashinfer0617-kernel', 'flashinfer0617-kernel-triton'):
    directory = root / '.venv' / name / 'tokenspeed_kernel/ops/attention/mha'
    (directory / '_sm121_fa2.py').write_text(content, encoding='utf-8', newline='\n')
    path = directory / 'flashinfer.py'
    before = path.read_text(encoding='utf-8')
    after = before + '\n# Experimental SM12.1 FA2 prefill; architecture checks remain intact.\nimport tokenspeed_kernel.ops.attention.mha._sm121_fa2  # noqa: E402,F401\n'
    path.write_text(after, encoding='utf-8', newline='\n')
    manifest.append(dict(package=name, adapter_sha256=hashlib.sha256(content.encode('utf-8')).hexdigest(), registration_sha256=hashlib.sha256(after.encode('utf-8')).hexdigest()))
    if name == 'flashinfer0617-kernel':
        (out / 'fa2-registration.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile='installed/ops/attention/mha/flashinfer.py', tofile='experiment/ops/attention/mha/flashinfer.py')), encoding='utf-8')
(out / 'fa2-overlay.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
for suffix in ('.json', '.log'):
    path = out / ('windows-flashinfer-eager' + suffix)
    target = out / ('windows-flashinfer-eager-trtllm-prefill-failure' + suffix)
    if path.exists() and not target.exists():
        path.rename(target)
shutil.copy2(root / '.port/flashinfer_comparison_bench.py', out / 'benchmark-before-fa2.py')
print('Installed architecture-specific FA2 prefill in both isolated packages; TRTLLM decode remains unchanged')
