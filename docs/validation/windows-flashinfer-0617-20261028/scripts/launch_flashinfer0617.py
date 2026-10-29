import argparse
import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--ops', choices=('triton', 'flashinfer'), required=True)
parser.add_argument('command', nargs=argparse.REMAINDER)
args = parser.parse_args()
assert args.command
env = dict(os.environ)
env['PYTHONPATH'] = str(root / '.venv/flashinfer0617')
if args.ops == 'flashinfer':
    env['PYTHONPATH'] = str(root / '.venv/flashinfer0617-kernel') + os.pathsep + env['PYTHONPATH']
else:
    env['PYTHONPATH'] = str(root / '.venv/flashinfer0617-kernel-triton') + os.pathsep + env['PYTHONPATH']
env['TOKENSPEED_COMPARISON_OPS'] = args.ops
cache = root.parents[1] / 'fij0617'
env['FLASHINFER_CACHE_DIR'] = str(cache)
env['FLASHINFER_WORKSPACE_BASE'] = str(cache)
env['FLASHINFER_AUTOTUNE_CACHE'] = str(cache / 'autotune')
env['TOKENSPEED_FLASHINFER_AUTOTUNE_CACHE_DIR'] = str(cache / 'autotune')
env['HF_HUB_OFFLINE'] = '1'
env['FLASHINFER_EXTRA_LDFLAGS'] = ' '.join('/LIBPATH:' + (root.parents[1] / 'vllm-bench/.venv/Lib/site-packages' / name / 'lib').as_posix() for name in ('torch', 'tvm_ffi'))
raise SystemExit(subprocess.call([sys.executable, *args.command], env=env))
