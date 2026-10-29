import json
import os
from pathlib import Path

import torch
from vllm.utils.platform_utils import is_pin_memory_available,is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

out = Path(__file__).resolve().parent/'engine-matrix-retest-20261029/wsl-uva-probe.json'
report = dict(flag=os.environ.get('VLLM_WSL2_ENABLE_PIN_MEMORY'),torch=torch.__version__,cuda=torch.version.cuda,pin_available=is_pin_memory_available(),uva_available=is_uva_available(),passed=False)
try:
    torch.cuda.set_device(0)
    cpu = torch.arange(4*704,dtype=torch.int32,pin_memory=True).reshape(4,704)
    view = get_accelerator_view_from_cpu_tensor(cpu)
    gpu = torch.empty_like(view)
    torch.add(view,0,out=gpu)
    torch.testing.assert_close(gpu.cpu(),cpu,atol=0,rtol=0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.add(view,0,out=gpu)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.add(view,0,out=gpu)
    for i in range(3):
        torch.cuda.synchronize()
        cpu.add_(i+1)
        graph.replay()
        torch.testing.assert_close(gpu.cpu(),cpu,atol=0,rtol=0)
    report.update(passed=True,cpu_pinned=cpu.is_pinned(),view_device=str(view.device),graph_replays=3)
except Exception as error:
    report['error'] = f'{type(error).__name__}: {error}'
    raise
finally:
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
