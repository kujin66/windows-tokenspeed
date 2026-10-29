import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

root = Path(__file__).resolve().parents[1]
out = root / '.port/flashinfer-paired-retest-20261029'
out.mkdir(exist_ok=True)
bench = root / 'tools/perf/benchmark_engine_matrix.py'
baseline = root / 'docs/validation/engine-matrix-retest-20261029'
expected_sha = json.loads((baseline/'windows-tokenspeed-flashinfer-graph.json').read_text())['benchmark_sha256']
assert hashlib.sha256(bench.read_bytes()).hexdigest() == expected_sha
vllm_root = root.parents[1] / 'vllm-bench'
env = dict(os.environ,PYTHONPATH='',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_USE_FLASHINFER_SAMPLER='0',CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',VLLM_CACHE_ROOT=str(vllm_root/'.cache/vllm'),TRITON_CACHE_DIR=str(vllm_root/'.cache/triton'),VLLM_CONFIG_ROOT=str(vllm_root/'.config/vllm'))
for key in ('FLASHINFER_CACHE_DIR','FLASHINFER_WORKSPACE_BASE','FLASHINFER_AUTOTUNE_CACHE'):
    env[key] = str(root.parents[1]/'fij0617')
env['FLASHINFER_EXTRA_LDFLAGS'] = ' '.join('/LIBPATH:'+(vllm_root/'.venv/Lib/site-packages'/name/'lib').as_posix() for name in ('torch','tvm_ffi'))
for key in ('TOKENSPEED_CUTE_DSL_AOT_CONFIG','TOKENSPEED_CUTE_DSL_COMPILER_PYTHON'):
    env.pop(key,None)

def snapshot():
    query = subprocess.run(['nvidia-smi','--id=0','--query-gpu=timestamp,name,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,utilization.gpu,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True)
    return dict(timestamp=datetime.now(timezone.utc).isoformat(),gpu=query.stdout.strip(),gpu_error=query.stderr.strip(),cpu_percent=psutil.cpu_percent(),available_memory=psutil.virtual_memory().available)

outcomes,commands = [],[]
for platform,repeat in [('windows',1),('wsl2',1),('wsl2',2),('windows',2)]:
    name = f'{platform}-flashinfer-repeat{repeat}'
    report = out / (name+'.json')
    if report.exists():
        raise FileExistsError(report)
    if platform=='windows':
        command = json.loads((root/'.port/engine-matrix-retest-20261029/native-owner/windows-commands.json').read_text())[1]
        old_report = command.index('--report')+1
        command[old_report] = str(report)
        guard = command.index(str(root/'.port/run_checked_benchmark.py'))+1
        command[guard] = str(out/(name+'-environment.json'))
    else:
        wsl_report = '/mnt/c/' + report.as_posix()[3:]
        command = ['wsl.exe','-d','Ubuntu-26.04','-u','root','--','bash','/mnt/c/Users/user/projects/official-repos/windows-tokenspeed/.port/retest_fi_wsl.sh',wsl_report]
    before = snapshot()
    commands.append(dict(name=name,command=command,before=before))
    (out/'commands.json').write_text(json.dumps(commands,indent=2))
    print(f'Starting {name}',flush=True)
    start = time.monotonic()
    samples=[]
    with (out/(name+'.log')).open('wb') as log:
        child = subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        while child.poll() is None:
            samples.append(dict(elapsed=time.monotonic()-start,cpu_percent=psutil.cpu_percent(),available_memory=psutil.virtual_memory().available))
            if time.monotonic()-start>1800:
                # Only this controller's native child tree is owned here.
                if platform=='windows':
                    parent=psutil.Process(child.pid)
                    for process in reversed(parent.children(recursive=True)):
                        try: process.kill()
                        except psutil.NoSuchProcess: pass
                    parent.kill()
                raise TimeoutError(name)
            time.sleep(2)
        code=child.wait()
    data=json.loads(report.read_text()) if report.exists() else {}
    original=json.loads((baseline/(('windows' if platform=='windows' else 'wsl2')+'-tokenspeed-flashinfer-graph.json')).read_text())
    changes={k:dict(old=original[k],new=data.get(k)) for k in ('benchmark_sha256','packages','sources','engine_config','sampling','graph_policy','model_files','gpu','seed','warmup','rounds','input_lengths','batch_sizes','output_length') if original[k]!=data.get(k)}
    result=dict(name=name,returncode=code,passed=code==0 and data.get('passed',False) and not changes,environment_changes=changes,seconds=time.monotonic()-start,before=before,after=snapshot(),cpu_samples=samples,error=data.get('error'))
    outcomes.append(result)
    (out/'outcomes.json').write_text(json.dumps(outcomes,indent=2))
    print({k:v for k,v in result.items() if k not in ('cpu_samples','before','after')},flush=True)
    if not result['passed']:
        raise RuntimeError(f'Rejected run: {name}')
print('Four paired runs completed',flush=True)
