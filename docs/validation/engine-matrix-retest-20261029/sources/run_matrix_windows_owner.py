import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

root = Path(__file__).resolve().parents[1]
out = root / '.port/engine-matrix-retest-20261029/native-owner'
out.mkdir(exist_ok=True)
vllm_root = root.parents[1] / 'vllm-bench'
ts_python = root / '.venv/Scripts/python.exe'
vllm_python = vllm_root / '.venv/Scripts/python.exe'
bench = root / 'tools/perf/benchmark_engine_matrix.py'
env = dict(os.environ,PYTHONPATH='',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_USE_FLASHINFER_SAMPLER='0',CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',VLLM_CACHE_ROOT=str(vllm_root/'.cache/vllm'),TRITON_CACHE_DIR=str(vllm_root/'.cache/triton'),VLLM_CONFIG_ROOT=str(vllm_root/'.config/vllm'))
for key in ('FLASHINFER_CACHE_DIR','FLASHINFER_WORKSPACE_BASE','FLASHINFER_AUTOTUNE_CACHE'):
    env[key] = str(root.parents[1]/'fij0617')
env['FLASHINFER_EXTRA_LDFLAGS'] = ' '.join('/LIBPATH:'+(vllm_root/'.venv/Lib/site-packages'/name/'lib').as_posix() for name in ('torch','tvm_ffi'))
while not (root / '.port/engine-matrix-retest-20261029/wsl-vllm-retry-outcome.json').exists():
    time.sleep(3)
outcomes = []
commands = []
for engine,backend in (('tokenspeed','triton'),('tokenspeed','flashinfer'),('vllm','auto')):
    name = f'windows-{engine}-{backend}-graph'
    report = out / (name+'.json')
    if report.exists():
        raise FileExistsError(report)
    python = ts_python if engine=='tokenspeed' else vllm_python
    launch = [str(root/'.port/launch_flashinfer0617.py'),'--ops',backend] if engine=='tokenspeed' else [str(root/'.port/launch_matrix_vllm.py')]
    if engine=='tokenspeed':
        launch += [str(root/'.port/run_checked_benchmark.py'),str(out/(name+'-environment.json'))]
    command = [str(python),str(root/'tools/windows/run.py'),str(python),*launch,str(bench),'--engine',engine,'--backend',backend,'--platform','windows-arm64','--execution','graph','--model',str(root/'.port/models/Qwen3-0.6B'),'--report',str(report),'--input-lengths','128','512','--batch-sizes','1','4','--output-length','128','--warmup','3','--rounds','5']
    commands.append(command)
    (out/'windows-commands.json').write_text(json.dumps(commands,indent=2))
    print(f'Starting {name}',flush=True)
    start = time.monotonic()
    with (out/(name+'.log')).open('wb') as log:
        child = subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        while child.poll() is None:
            if time.monotonic()-start > 1800:
                parent = psutil.Process(child.pid)
                for owned in reversed(parent.children(recursive=True)):
                    try:
                        owned.kill()
                    except psutil.NoSuchProcess:
                        pass
                parent.kill()
                break
            time.sleep(2)
        code = child.wait()
    data = json.loads(report.read_text()) if report.exists() else {}
    outcome = dict(name=name,returncode=code,passed=code==0 and data.get('passed',False),seconds=time.monotonic()-start,error=data.get('error'))
    outcomes.append(outcome)
    (out/'windows-outcomes.json').write_text(json.dumps(outcomes,indent=2))
    print(outcome,flush=True)
print('Native Windows matrix finished',flush=True)
raise SystemExit(0 if all(row['passed'] for row in outcomes) else 1)
