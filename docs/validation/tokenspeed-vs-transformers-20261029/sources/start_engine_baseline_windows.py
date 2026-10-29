import json
import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
publication = root.with_name('windows-tokenspeed-pr')
out = root / '.port/engine-baseline-20261029'
out.mkdir(exist_ok=True)
env = dict(os.environ, HF_HUB_OFFLINE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONPATH=str(root / '.venv/flashinfer0617'), CUDA_VISIBLE_DEVICES='0', TOKENIZERS_PARALLELISM='false')
cache = root.parents[1] / 'fij0617'
for key in ('FLASHINFER_CACHE_DIR', 'FLASHINFER_WORKSPACE_BASE', 'FLASHINFER_AUTOTUNE_CACHE', 'TOKENSPEED_FLASHINFER_AUTOTUNE_CACHE_DIR'):
    env[key] = str(cache)
env['FLASHINFER_EXTRA_LDFLAGS'] = ' '.join('/LIBPATH:' + (root.parents[1] / 'vllm-bench/.venv/Lib/site-packages' / name / 'lib').as_posix() for name in ('torch', 'tvm_ffi'))
configurations = ['generate-torch', 'loop-dynamic-torch', 'loop-static-eager-torch', 'loop-static-graph-torch', 'loop-static-eager-flashinfer', 'loop-static-graph-flashinfer']
commands = []

def run(command, log, cwd, environment):
    commands.append(command)
    (out / 'windows-commands.json').write_text(json.dumps(commands, indent=2) + '\n', encoding='utf-8')
    print(f'Starting {log.name}', flush=True)
    with log.open('wb') as stream:
        result = subprocess.run(command, cwd=cwd, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit(f'Failed {log.name}: {result.returncode}')
    print(f'Completed {log.name}', flush=True)

hf_command = [sys.executable, str(root / 'tools/windows/run.py'), sys.executable, '-m', 'tools.perf.run_without_tokenspeed_suite', '--model', str(root / '.port/models/Qwen3-0.6B'), '--report-dir', str(out), '--platform-label', 'windows', '--diagnostics', 'cpu', '--configurations', *configurations]
run(hf_command, out / 'windows-hf-suite.log', publication, env)
for ops, execution in [('triton', 'eager'), ('flashinfer', 'eager'), ('triton', 'graph'), ('flashinfer', 'graph')]:
    report = out / f'windows-tokenspeed-{ops}-{execution}.json'
    assert not report.exists(), report
    environment = dict(env, PYTHONPATH='')
    command = [sys.executable, str(root / 'tools/windows/run.py'), sys.executable, str(root / '.port/launch_flashinfer0617.py'), '--ops', ops, str(root / '.port/flashinfer_comparison_bench.py'), '--model', str(root / '.port/models/Qwen3-0.6B'), '--report', str(report), '--execution', execution, '--profile', 'none', '--attention-backend', ops, '--input-lengths', '128', '512']
    run(command, report.with_suffix('.log'), root, environment)
print('All Windows HF baselines and fresh TokenSpeed configurations completed', flush=True)
