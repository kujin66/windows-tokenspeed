import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

root = Path(__file__).resolve().parents[1]
publication = root.with_name('windows-tokenspeed-pr')
out = root / '.port/engine-baseline-20261029'
fixed = ('generate-torch', 'loop-dynamic-torch', 'loop-static-eager-torch', 'loop-static-graph-torch', 'loop-static-eager-flashinfer', 'loop-static-graph-flashinfer')
names = [f'{platform}-hf-{label}.json' for platform in ('windows', 'wsl2') for label in fixed]
names += [f'{platform}-hf-loop-static-graph-bucketed-{label}-in{length}.json' for platform in ('windows', 'wsl2') for label in ('torch', 'flashinfer', 'flashinfer-reduced') for length in (128, 512)]
names += [f'{platform}-tokenspeed-{backend}-{mode}.json' for platform, backends in [('windows', ('triton', 'flashinfer')), ('wsl2', ('flashinfer',))] for backend in backends for mode in ('eager', 'graph')]

def previous_running():
    for process in psutil.process_iter(['cmdline']):
        try:
            if any(argument.endswith('continue_engine_baseline.py') for argument in process.info['cmdline'] or []):
                return True
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    return False

print('Waiting for primary matrices to finish before graph-state/dispatcher checks', flush=True)
while True:
    ready = True
    for name in names:
        try:
            data = json.loads((out / name).read_text(encoding='utf-8'))
        except (FileNotFoundError, json.JSONDecodeError):
            ready = False
            continue
        if data.get('error'):
            raise SystemExit(f'Primary matrix failed: {name}: {data["error"]}')
        ready = ready and data.get('passed', False)
    if ready and not previous_running():
        break
    time.sleep(2)
print('Starting WSL2 followup checks', flush=True)
with (out / 'wsl2-followup-checks.log').open('wb') as log:
    result = subprocess.run(['C:/Program Files/WSL/wsl.exe', '-d', 'Ubuntu-26.04', '--', 'bash', '/mnt/c/Users/user/projects/official-repos/windows-tokenspeed/.port/run_wsl_baseline_followup.sh'], cwd=root, stdout=log, stderr=subprocess.STDOUT)
if result.returncode:
    raise SystemExit(f'WSL2 followup failed: {result.returncode}')
subprocess.run(['C:/Program Files/WSL/wsl.exe', '--terminate', 'Ubuntu-26.04'], check=True, stdout=subprocess.DEVNULL)
env = dict(os.environ, HF_HUB_OFFLINE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONPATH=str(root / '.venv/flashinfer0617'), CUDA_VISIBLE_DEVICES='0', TOKENIZERS_PARALLELISM='false')
cache = root.parents[1] / 'fij0617'
for key in ('FLASHINFER_CACHE_DIR', 'FLASHINFER_WORKSPACE_BASE', 'FLASHINFER_AUTOTUNE_CACHE', 'TOKENSPEED_FLASHINFER_AUTOTUNE_CACHE_DIR'):
    env[key] = str(cache)
env['FLASHINFER_EXTRA_LDFLAGS'] = ' '.join('/LIBPATH:' + (root.parents[1] / 'vllm-bench/.venv/Lib/site-packages' / name / 'lib').as_posix() for name in ('torch', 'tvm_ffi'))
print('Starting Windows followup checks after stopping WSL2', flush=True)
command = [sys.executable, str(root / 'tools/windows/run.py'), sys.executable, '-m', 'tools.perf.run_followup_baseline_checks', '--model', str(root / '.port/models/Qwen3-0.6B'), '--report-dir', str(out), '--platform-label', 'windows']
with (out / 'windows-followup-checks.log').open('wb') as log:
    result = subprocess.run(command, cwd=publication, env=env, stdout=log, stderr=subprocess.STDOUT)
if result.returncode:
    raise SystemExit(f'Windows followup failed: {result.returncode}')
print('All primary and followup GPU experiments completed', flush=True)
