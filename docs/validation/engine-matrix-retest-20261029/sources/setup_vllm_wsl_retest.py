import json
import subprocess
from pathlib import Path

base = Path('/opt/tokenspeed-work')
env = base / 'vllm-0.26-retest-venv'
out = Path('/mnt/c/Users/user/projects/official-repos/windows-tokenspeed/.port/engine-matrix-retest-20261029')
out.mkdir(exist_ok=True)
if not env.exists():
    subprocess.run([str(base / 'venv/bin/python'), '-m', 'venv', str(env)], check=True)
python = str(env / 'bin/python')
result = subprocess.run([python, '-m', 'pip', 'install', '--index-url', 'https://pypi.org/simple', '--only-binary=:all:', '--report', str(out / 'wsl-vllm-install.json'), 'vllm==0.26.0'])
if result.returncode == 0:
    (out / 'wsl-vllm-freeze.txt').write_text(subprocess.check_output([python, '-m', 'pip', 'freeze'], text=True))
raise SystemExit(result.returncode)
