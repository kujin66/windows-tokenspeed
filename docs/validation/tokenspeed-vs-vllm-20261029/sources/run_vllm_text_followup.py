import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import psutil

root = Path(__file__).resolve().parents[1]
publication = root.with_name('windows-tokenspeed-pr')
out = root / '.port/vllm-comparison-20261029'
vllm_root = root.parents[1] / 'vllm-bench'
py = vllm_root / '.venv/Scripts/python.exe'

def previous_running():
    for process in psutil.process_iter(['cmdline']):
        try:
            if any(argument.endswith('run_vllm_comparison.py') for argument in process.info['cmdline'] or []):
                return True
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    return False

print('Waiting for current native GPU matrix to finish before normal-text API followup', flush=True)
while not (out / 'selected-vllm.json').is_file():
    time.sleep(10)
while previous_running():
    time.sleep(2)
prior = json.loads((out / 'selected-vllm.json').read_text(encoding='utf-8'))
destination = out / 'selection-token-only.json'
assert not destination.exists(), 'Preserve prior selection'
shutil.copyfile(out / 'selected-vllm.json', destination)
manifest = json.loads((out / 'source-sha256.json').read_text(encoding='utf-8'))
for name in ('benchmark_vllm_text.py', 'launch_vllm_text_comparison.py', 'run_vllm_text_followup.py'):
    source = root / '.port' / name
    destination = out / 'sources' / name
    assert not destination.exists(), destination
    shutil.copyfile(source, destination)
    manifest.append(dict(file='sources/' + name, sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
(out / 'source-sha256.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')

env = dict(os.environ, PYTHONPATH=str(publication), HF_HUB_OFFLINE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', VLLM_WORKER_MULTIPROC_METHOD='spawn', VLLM_USE_FLASHINFER_SAMPLER='0', CUDA_VISIBLE_DEVICES='0', CUDA_DEVICE_ORDER='PCI_BUS_ID', VLLM_CACHE_ROOT=str(vllm_root / '.cache/vllm'), TRITON_CACHE_DIR=str(vllm_root / '.cache/triton'), VLLM_CONFIG_ROOT=str(vllm_root / '.config/vllm'))
cache = root.parents[1] / 'fij0617'
for key in ('FLASHINFER_CACHE_DIR', 'FLASHINFER_WORKSPACE_BASE', 'FLASHINFER_AUTOTUNE_CACHE'):
    env[key] = str(cache)
env['FLASHINFER_EXTRA_LDFLAGS'] = ' '.join('/LIBPATH:' + (vllm_root / '.venv/Lib/site-packages' / name / 'lib').as_posix() for name in ('torch', 'tvm_ffi'))
commands = []
outcomes = []
selected = {}
for mode in ('eager', 'graph'):
    previous = json.loads((out / prior[mode]).read_text(encoding='utf-8')) if mode in prior else None
    attention = previous['attention_requested'] if previous else 'auto'
    policy = previous['graph_policy'] if previous else ('default' if mode == 'eager' else 'decode-only')
    report = out / f'windows-vllm-text-{mode}.json'
    assert not report.exists(), report
    command = [str(py), str(root / 'tools/windows/run.py'), str(py), str(root / '.port/launch_vllm_text_comparison.py'), '--model', str(root / '.port/models/Qwen3-0.6B'), '--report', str(report), '--execution', mode, '--attention', attention, '--graph-policy', policy, '--memory-utilization', '0.03', '--input-lengths', '128', '512', '--batch-sizes', '1', '4', '--output-length', '128', '--warmup', '3', '--rounds', '5']
    commands.append(dict(command=command, cwd=str(publication), report=report.name))
    (out / 'text-commands.json').write_text(json.dumps(commands, indent=2) + '\n', encoding='utf-8')
    print(f'Starting matched normal-text vLLM {mode}', flush=True)
    log = report.with_suffix('.log')
    start = time.monotonic()
    reason = None
    failed_at = None
    with log.open('wb') as stream:
        process = subprocess.Popen(command, cwd=publication, env=env, stdout=stream, stderr=subprocess.STDOUT)
        while process.poll() is None:
            contents = log.read_text(encoding='utf-8', errors='replace')
            if 'EngineCore failed to start.' in contents and failed_at is None:
                failed_at = time.monotonic()
            if (failed_at is not None and time.monotonic() - failed_at > 10) or time.monotonic() - start > 1800:
                reason = 'Recorded EngineCore startup failure' if failed_at else '30-minute timeout'
                parent = psutil.Process(process.pid)
                for child in reversed(parent.children(recursive=True)):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                parent.kill()
                break
            time.sleep(2)
        code = process.wait()
    record = json.loads(report.read_text(encoding='utf-8')) if report.exists() else {}
    success = code == 0 and record.get('passed', False)
    outcomes.append(dict(report=report.name, passed=success, returncode=code, seconds=time.monotonic()-start, controller_stop=reason, error=record.get('error')))
    (out / 'text-controller-outcomes.json').write_text(json.dumps(outcomes, indent=2) + '\n', encoding='utf-8')
    if success:
        assert not record['engine']['skip_tokenizer_init'] and record['sampling']['detokenize']
        selected[mode] = report.name
    print(f'Completed normal-text {mode}: passed={success}', flush=True)
(out / 'selected-vllm.json').write_text(json.dumps(selected, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(normal_text_vllm=selected, matched_output_policy=True)), flush=True)
raise SystemExit(0 if len(selected) == 2 else 1)
