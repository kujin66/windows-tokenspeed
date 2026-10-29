import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

root = Path(__file__).resolve().parents[1]
publication = root.with_name('windows-tokenspeed-pr')
out = root / '.port/vllm-comparison-20261029'
out.mkdir(exist_ok=True)
vllm_root = root.parents[1] / 'vllm-bench'
vllm_py = vllm_root / '.venv/Scripts/python.exe'
ts_py = root / '.venv/Scripts/python.exe'
commands = []
outcomes = []
sources = [publication / 'tools/perf/benchmark_vllm.py', root / '.port/flashinfer_comparison_bench.py', root / '.port/launch_flashinfer0617.py', root / '.port/launch_vllm_comparison.py', Path(__file__)]
manifest = []
for path in sources:
    destination = out / 'sources' / path.name
    contents = path.read_bytes()
    if destination.exists() and destination.read_bytes() != contents:
        raise ValueError(f'Preserve source: {destination}')
    shutil.copyfile(path, destination)
    manifest.append(dict(file='sources/' + path.name, sha256=hashlib.sha256(contents).hexdigest()))
(out / 'source-sha256.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')

versions = {}
names = ('torch', 'transformers', 'flashinfer-python', 'triton-windows', 'vllm', 'tokenspeed', 'tokenspeed-kernel')
code = "import importlib.metadata as m,json; names=" + repr(names) + "; print(json.dumps({n:m.version(n) for n in names if m.packages_distributions() and any(d.metadata['Name'].lower().replace('_','-')==n for d in m.distributions())}))"
for name, py in [('vllm', vllm_py), ('tokenspeed', ts_py)]:
    metadata_env = dict(os.environ)
    if name == 'tokenspeed':
        metadata_env['PYTHONPATH'] = str(root / '.venv/flashinfer0617')
    versions[name] = json.loads(subprocess.check_output([str(py), '-c', code], env=metadata_env))
(out / 'package-versions.json').write_text(json.dumps(versions, indent=2) + '\n', encoding='utf-8')

env = dict(os.environ, PYTHONPATH=str(publication), HF_HUB_OFFLINE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', VLLM_WORKER_MULTIPROC_METHOD='spawn', VLLM_USE_FLASHINFER_SAMPLER='0', CUDA_VISIBLE_DEVICES='0', CUDA_DEVICE_ORDER='PCI_BUS_ID', VLLM_CACHE_ROOT=str(vllm_root / '.cache/vllm'), TRITON_CACHE_DIR=str(vllm_root / '.cache/triton'), VLLM_CONFIG_ROOT=str(vllm_root / '.config/vllm'))
cache = root.parents[1] / 'fij0617'
for key in ('FLASHINFER_CACHE_DIR', 'FLASHINFER_WORKSPACE_BASE', 'FLASHINFER_AUTOTUNE_CACHE'):
    env[key] = str(cache)
env['FLASHINFER_EXTRA_LDFLAGS'] = ' '.join('/LIBPATH:' + (vllm_root / '.venv/Lib/site-packages' / name / 'lib').as_posix() for name in ('torch', 'tvm_ffi'))

def stop_owned_tree(process):
    try:
        parent = psutil.Process(process.pid)
        for child in reversed(parent.children(recursive=True)):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        parent.kill()
    except psutil.NoSuchProcess:
        pass

def run(command, report, cwd, environment):
    if report.exists():
        raise FileExistsError(report)
    commands.append(dict(command=command, cwd=str(cwd), report=report.name))
    (out / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n', encoding='utf-8')
    log = report.with_suffix('.log')
    print(f'Starting {report.name}', flush=True)
    start = time.monotonic()
    failure_seen = None
    stopped = None
    with log.open('wb') as stream:
        process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        while process.poll() is None:
            text = log.read_text(encoding='utf-8', errors='replace')
            if 'EngineCore failed to start.' in text:
                if failure_seen is None:
                    failure_seen = time.monotonic()
                if time.monotonic() - failure_seen > 10:
                    stopped = 'Explicit EngineCore startup failure; stopped only this failed job tree during cleanup'
                    stop_owned_tree(process)
                    break
            if time.monotonic() - start > 1800:
                stopped = '30-minute job timeout'
                stop_owned_tree(process)
                break
            time.sleep(2)
        code = process.wait()
    data = json.loads(report.read_text(encoding='utf-8')) if report.exists() else {}
    success = code == 0 and data.get('passed', False)
    outcomes.append(dict(report=report.name, passed=success, returncode=code, seconds=time.monotonic()-start, controller_stop=stopped, error=data.get('error')))
    (out / 'controller-outcomes.json').write_text(json.dumps(outcomes, indent=2) + '\n', encoding='utf-8')
    print(f'Completed {report.name}: passed={success} exit={code}', flush=True)
    return success

def vllm(execution, attention, policy):
    label = attention.lower().replace('_attn', '')
    report = out / f'windows-vllm-{label}-{execution}-{policy}.json'
    command = [str(vllm_py), str(root / 'tools/windows/run.py'), str(vllm_py), str(root / '.port/launch_vllm_comparison.py'), '--model', str(root / '.port/models/Qwen3-0.6B'), '--report', str(report), '--execution', execution, '--attention', attention, '--graph-policy', policy, '--memory-utilization', '0.03', '--input-lengths', '128', '512', '--batch-sizes', '1', '4', '--output-length', '128', '--warmup', '3', '--rounds', '5']
    return run(command, report, publication, env), report.name

def tokenspeed(backend, execution):
    report = out / f'windows-tokenspeed-{backend}-{execution}.json'
    environment = dict(env, PYTHONPATH='')
    command = [str(ts_py), str(root / 'tools/windows/run.py'), str(ts_py), str(root / '.port/launch_flashinfer0617.py'), '--ops', backend, str(root / '.port/flashinfer_comparison_bench.py'), '--model', str(root / '.port/models/Qwen3-0.6B'), '--report', str(report), '--execution', execution, '--profile', 'none', '--attention-backend', backend, '--input-lengths', '128', '512']
    return run(command, report, root, environment)

selection = {}
eager_ok, name = vllm('eager', 'auto', 'default')
attention = 'auto'
if not eager_ok:
    attention = 'TRITON_ATTN'
    eager_ok, name = vllm('eager', attention, 'default')
if eager_ok:
    selection['eager'] = name
ts_ok = tokenspeed('triton', 'eager')
ts_ok = tokenspeed('flashinfer', 'eager') and ts_ok
graph_ok, name = vllm('graph', attention, 'default')
if not graph_ok:
    graph_ok, name = vllm('graph', attention, 'decode-only')
if graph_ok:
    selection['graph'] = name
ts_ok = tokenspeed('triton', 'graph') and ts_ok
ts_ok = tokenspeed('flashinfer', 'graph') and ts_ok
(out / 'selected-vllm.json').write_text(json.dumps(selection, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(selected_vllm=selection, tokenspeed_passed=ts_ok, all_required_passed=eager_ok and graph_ok and ts_ok)), flush=True)
raise SystemExit(0 if eager_ok and graph_ok and ts_ok else 1)
