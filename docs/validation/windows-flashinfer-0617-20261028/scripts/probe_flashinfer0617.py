import hashlib
import importlib.metadata
import json
import os
import struct
import traceback
from pathlib import Path

import torch
import flashinfer

root = Path(__file__).resolve().parents[1]
out = root / '.port/flashinfer-comparison'
report = dict(version=importlib.metadata.version('flashinfer-python'), module=flashinfer.__file__, torch=torch.__version__, cuda=torch.version.cuda, cases=[], passed=False)
def save():
    (out / 'gpu-probe.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
try:
    assert report['version'] == '0.6.17+g2ea3c24b5dac'
    assert Path(flashinfer.__file__).is_relative_to(root / '.venv/flashinfer0617')
    torch.manual_seed(20261028)
    with torch.inference_mode():
        for rows in (1, 4, 128, 512):
            x = torch.randn(rows, 1024, device='cuda', dtype=torch.bfloat16)
            weight = torch.randn(1024, device='cuda', dtype=x.dtype)
            expected = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6) * weight.float()).to(x.dtype)
            actual = flashinfer.rmsnorm(x, weight, eps=1e-6)
            torch.testing.assert_close(actual, expected, rtol=0.016, atol=0.016)
            residual = torch.randn_like(x)
            total = x.float() + residual.float()
            expected = (total * torch.rsqrt(total.square().mean(-1, keepdim=True) + 1e-6) * weight.float()).to(x.dtype)
            flashinfer.fused_add_rmsnorm(x, residual, weight, eps=1e-6)
            torch.testing.assert_close(x, expected, rtol=0.016, atol=0.016)
            torch.testing.assert_close(residual, total.to(x.dtype), rtol=0, atol=0)
            activation = torch.randn(rows, 6144, device='cuda', dtype=x.dtype)
            gate, value = activation.float().chunk(2, dim=-1)
            expected = (torch.nn.functional.silu(gate) * value).to(x.dtype)
            actual = flashinfer.silu_and_mul(activation)
            torch.testing.assert_close(actual, expected, rtol=0.016, atol=0.016)
            report['cases'].append(dict(rows=rows, rmsnorm=True, fused_add_rmsnorm=True, silu_and_mul=True))
            save()
            print(f'PASS rows={rows} RMSNorm/fused-add-RMSNorm/SwiGLU', flush=True)
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            flashinfer.rmsnorm(x, weight, eps=1e-6)
            flashinfer.silu_and_mul(activation)
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(graph, stream=stream):
            norm_output = flashinfer.rmsnorm(x, weight, eps=1e-6)
            silu_output = flashinfer.silu_and_mul(activation)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(silu_output, expected, rtol=0.016, atol=0.016)
        report['cuda_graph_passed'] = True
    from tokenspeed.runtime.entrypoints.engine import Engine
    report['engine_import'] = True
    files = []
    for path in Path(os.environ['FLASHINFER_CACHE_DIR']).rglob('*.dll'):
        raw = path.read_bytes()
        offset = struct.unpack_from('<I', raw, 0x3c)[0]
        assert raw[offset:offset+4] == b'PE\0\0'
        machine = struct.unpack_from('<H', raw, offset + 4)[0]
        assert machine == 0xaa64, path
        files.append(dict(path=str(path), machine=hex(machine), sha256=hashlib.sha256(raw).hexdigest()))
    report['native_files'] = files
    assert len(files) >= 2
    report['passed'] = True
except Exception as exc:
    report['error'] = str(exc)
    report['traceback'] = traceback.format_exc()
    save()
    raise
save()
