import json
import subprocess
import sys
import time
from pathlib import Path

import psutil

root = Path(__file__).resolve().parents[1]
out = root / '.port/engine-baseline-20261029'
windows_files = [f'windows-hf-{name}.json' for name in ('generate-torch', 'loop-dynamic-torch', 'loop-static-eager-torch', 'loop-static-graph-torch', 'loop-static-eager-flashinfer', 'loop-static-graph-flashinfer')]
windows_files += [f'windows-tokenspeed-{ops}-{mode}.json' for ops in ('triton', 'flashinfer') for mode in ('eager', 'graph')]

def controller_running():
    for process in psutil.process_iter(['cmdline']):
        try:
            if any(argument.endswith('start_engine_baseline_windows.py') for argument in process.info['cmdline'] or []):
                return True
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    return False

print('Waiting for current Windows matrix to finish before further GPU work', flush=True)
while True:
    ready = True
    for name in windows_files:
        path = out / name
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except (FileNotFoundError, json.JSONDecodeError):
            ready = False
            continue
        if data.get('error'):
            raise SystemExit(f'Current matrix failed: {name}: {data["error"]}')
        ready = ready and data.get('passed', False)
    if ready and not controller_running():
        break
    time.sleep(2)
print('Starting Windows request-sized KV graph controls', flush=True)
result = subprocess.run([sys.executable, str(root / '.port/start_engine_baseline_bucketed_windows.py')], cwd=root)
if result.returncode:
    raise SystemExit(result.returncode)
print('Starting WSL2 complete HF and fresh TokenSpeed matrix', flush=True)
with (out / 'wsl2-phase.log').open('wb') as log:
    result = subprocess.run(['C:/Program Files/WSL/wsl.exe', '-d', 'Ubuntu-26.04', '--', 'bash', '/mnt/c/Users/user/projects/official-repos/windows-tokenspeed/.port/start_engine_baseline_wsl.sh'], cwd=root, stdout=log, stderr=subprocess.STDOUT)
if result.returncode:
    raise SystemExit(f'WSL2 matrix failed: {result.returncode}')
print('All sequential Windows/WSL2 comparisons completed', flush=True)
