import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
out = repo / '.port/engine-matrix-retest-20261029'
base = Path('/opt/tokenspeed-work')
source = base / 'venv/flashinfer-comparison/tokenspeed_kernel'
target = base / 'venv/flashinfer-comparison-triton-retest/tokenspeed_kernel'
if target.exists():
    raise FileExistsError(target)
shutil.copytree(source,target,ignore=shutil.ignore_patterns('__pycache__'))
changes = {}
for rel in ('ops/layernorm/__init__.py','ops/activation/__init__.py'):
    path = target / rel
    before = path.read_text()
    after = before.replace('import torch\n','import os\nimport torch\n',1)
    if 'layernorm' in rel:
        assert before.count('_platform.is_amd')==2
        after = after.replace('_platform.is_amd',"(_platform.is_amd or os.environ.get('TOKENSPEED_COMPARISON_OPS') == 'triton')")
    else:
        assert before.count('or current_platform().is_amd')==1
        after = after.replace('or current_platform().is_amd',"or os.environ.get('TOKENSPEED_COMPARISON_OPS') == 'triton'\n        or current_platform().is_amd")
    assert before != after
    path.write_text(after)
    changes[rel] = dict(before_sha256=hashlib.sha256(before.encode()).hexdigest(),after_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
(out/'wsl-triton-overlay.json').write_text(json.dumps(dict(source=str(source),target=str(target),changes=changes),indent=2))

# Only one process uses the shared GPU at a time; installation is CPU-only.
deadline = time.monotonic()+3600
while True:
    native = out/'windows-outcomes.json'
    ready = native.exists() and len(json.loads(native.read_text()))==3
    installed = (out/'wsl-vllm-freeze.txt').exists()
    if ready and installed:
        break
    if time.monotonic()>deadline:
        raise TimeoutError('Waiting for native matrix and isolated vLLM installation')
    time.sleep(3)
outcomes,commands = [],[]
for engine,backend in (('tokenspeed','triton'),('tokenspeed','flashinfer'),('vllm','auto')):
    name = f'wsl2-{engine}-{backend}-graph'
    report = out/(name+'.json')
    if report.exists():
        raise FileExistsError(report)
    venv = base/('venv' if engine=='tokenspeed' else 'vllm-0.26-retest-venv')
    env = dict(os.environ,VIRTUAL_ENV=str(venv),HF_HUB_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_USE_FLASHINFER_SAMPLER='0',TOKENSPEED_COMPARISON_OPS=backend)
    if engine=='tokenspeed':
        env['PYTHONPATH'] = str(target.parent if backend=='triton' else source.parent)
    else:
        env['PYTHONPATH'] = ''
        env['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib:'+str(base/'cuda-sdk-13.4/nvidia/cu13/lib')
        env['TRITON_CACHE_DIR'] = str(base/'cache/vllm-retest/triton')
        env['VLLM_CACHE_ROOT'] = str(base/'cache/vllm-retest')
        env['PATH'] = str(venv/'bin')+':'+env['PATH']
    command = [str(venv/'bin/python'),str(repo/'tools/perf/benchmark_engine_matrix.py'),'--engine',engine,'--backend',backend,'--platform','wsl2-arm64','--execution','graph','--model',str(base/'models/Qwen3-0.6B'),'--report',str(report),'--input-lengths','128','512','--batch-sizes','1','4','--output-length','128','--warmup','3','--rounds','5']
    commands.append(dict(command=command,environment={k:env.get(k,'') for k in ('VIRTUAL_ENV','PYTHONPATH','LD_LIBRARY_PATH','TRITON_CACHE_DIR','VLLM_CACHE_ROOT','TOKENSPEED_COMPARISON_OPS')}))
    (out/'wsl-commands.json').write_text(json.dumps(commands,indent=2))
    print(f'Starting {name}',flush=True)
    start = time.monotonic()
    with (out/(name+'.log')).open('wb') as log:
        child = subprocess.Popen(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code = child.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL)
            code = child.wait()
    data = json.loads(report.read_text()) if report.exists() else {}
    outcome = dict(name=name,returncode=code,passed=code==0 and data.get('passed',False),seconds=time.monotonic()-start,error=data.get('error'))
    outcomes.append(outcome)
    (out/'wsl-outcomes.json').write_text(json.dumps(outcomes,indent=2))
    print(outcome,flush=True)
raise SystemExit(0 if all(row['passed'] for row in outcomes) else 1)
