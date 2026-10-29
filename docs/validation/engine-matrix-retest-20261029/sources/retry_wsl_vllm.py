import json
import os
import subprocess
import time
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
out = repo/'.port/engine-matrix-retest-20261029'
while True:
    native = out/'native-confirm/windows-outcomes.json'
    if native.exists() and len(json.loads(native.read_text()))==3:
        break
    time.sleep(3)
venv = Path('/opt/tokenspeed-work/vllm-0.26-retest-venv')
env = dict(os.environ,VIRTUAL_ENV=str(venv),PYTHONPATH='',LD_LIBRARY_PATH='/usr/lib/wsl/lib:/opt/tokenspeed-work/cuda-sdk-13.4/nvidia/cu13/lib',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_USE_FLASHINFER_SAMPLER='0',VLLM_WSL2_ENABLE_PIN_MEMORY='1',TRITON_CACHE_DIR='/opt/tokenspeed-work/cache/vllm-retest/triton',VLLM_CACHE_ROOT='/opt/tokenspeed-work/cache/vllm-retest')
env['PATH'] = str(venv/'bin')+':'+env['PATH']
probe = [str(venv/'bin/python'),str(repo/'.port/probe_wsl_vllm_uva.py')]
with (out/'wsl-uva-probe.log').open('wb') as log:
    result = subprocess.run(probe,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
if result.returncode:
    raise RuntimeError('Actual pinned/UVA ordinary and graph checks failed; inspect before selecting an alternative runner')
report = out/'wsl2-vllm-auto-graph-pinned.json'
command = [str(venv/'bin/python'),str(repo/'tools/perf/benchmark_engine_matrix.py'),'--engine','vllm','--backend','auto','--platform','wsl2-arm64','--execution','graph','--model','/opt/tokenspeed-work/models/Qwen3-0.6B','--report',str(report),'--input-lengths','128','512','--batch-sizes','1','4','--output-length','128','--warmup','3','--rounds','5']
(out/'wsl-vllm-retry-command.json').write_text(json.dumps(dict(command=command,environment={key:env[key] for key in ('VLLM_WSL2_ENABLE_PIN_MEMORY','VIRTUAL_ENV','LD_LIBRARY_PATH','TRITON_CACHE_DIR','VLLM_CACHE_ROOT')}),indent=2))
print('Starting WSL2 vLLM with validated official pinned-memory opt-in',flush=True)
with report.with_suffix('.log').open('wb') as log:
    result = subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
data = json.loads(report.read_text()) if report.exists() else {}
(out/'wsl-vllm-retry-outcome.json').write_text(json.dumps(dict(returncode=result.returncode,passed=result.returncode==0 and data.get('passed',False),error=data.get('error')),indent=2))
raise SystemExit(result.returncode)
