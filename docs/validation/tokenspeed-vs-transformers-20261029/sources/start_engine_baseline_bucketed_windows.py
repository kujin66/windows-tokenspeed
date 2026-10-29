import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
publication = root.with_name('windows-tokenspeed-pr')
out = root / '.port/engine-baseline-20261029'
env = dict(os.environ, HF_HUB_OFFLINE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONPATH=str(root / '.venv/flashinfer0617'), CUDA_VISIBLE_DEVICES='0', TOKENIZERS_PARALLELISM='false')
cache = root.parents[1] / 'fij0617'
for key in ('FLASHINFER_CACHE_DIR', 'FLASHINFER_WORKSPACE_BASE', 'FLASHINFER_AUTOTUNE_CACHE', 'TOKENSPEED_FLASHINFER_AUTOTUNE_CACHE_DIR'):
    env[key] = str(cache)
env['FLASHINFER_EXTRA_LDFLAGS'] = ' '.join('/LIBPATH:' + (root.parents[1] / 'vllm-bench/.venv/Lib/site-packages' / name / 'lib').as_posix() for name in ('torch', 'tvm_ffi'))
command = [sys.executable, str(root / 'tools/windows/run.py'), sys.executable, '-m', 'tools.perf.run_bucketed_without_tokenspeed', '--model', str(root / '.port/models/Qwen3-0.6B'), '--report-dir', str(out), '--platform-label', 'windows', '--normalizations', 'torch', 'flashinfer', '--input-lengths', '128', '512', '--batch-sizes', '1', '4', '--output-length', '128', '--warmup', '3', '--rounds', '5', '--math-reduction', 'full']
probe = [sys.executable, str(root / 'tools/windows/run.py'), sys.executable, '-m', 'tools.perf.probe_sdpa_backends', '--report', str(out / 'windows-sdpa-backends.json')]
with (out / 'windows-sdpa-backends.log').open('wb') as log:
    result = subprocess.run(probe, cwd=publication, env=env, stdout=log, stderr=subprocess.STDOUT)
if result.returncode:
    raise SystemExit(result.returncode)
with (out / 'windows-hf-bucketed-suite.log').open('wb') as log:
    result = subprocess.run(command, cwd=publication, env=env, stdout=log, stderr=subprocess.STDOUT)
if result.returncode:
    raise SystemExit(result.returncode)
reduced = [sys.executable, str(root / 'tools/windows/run.py'), sys.executable, '-m', 'tools.perf.run_bucketed_without_tokenspeed', '--model', str(root / '.port/models/Qwen3-0.6B'), '--report-dir', str(out), '--platform-label', 'windows', '--normalizations', 'flashinfer', '--input-lengths', '128', '512', '--batch-sizes', '1', '4', '--output-length', '128', '--warmup', '3', '--rounds', '5', '--math-reduction', 'reduced']
with (out / 'windows-hf-bucketed-reduced-suite.log').open('wb') as log:
    result = subprocess.run(reduced, cwd=publication, env=env, stdout=log, stderr=subprocess.STDOUT)
raise SystemExit(result.returncode)
