import json
from pathlib import Path

import flashinfer
import torch
import tokenspeed.runtime.execution.factory
from tokenspeed.runtime.models.qwen3 import Qwen3ForCausalLM
from tokenspeed_kernel.ops.activation import silu_and_mul
from tokenspeed_kernel.ops.layernorm import rmsnorm

root = Path(__file__).resolve().parents[1]
x = torch.randn(4, 1024, device='cuda', dtype=torch.bfloat16)
weight = torch.ones(1024, device='cuda', dtype=x.dtype)
result = rmsnorm(x, weight, 1e-6)
expected = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype)
torch.testing.assert_close(result, expected, rtol=0.016, atol=0.016)
activation = torch.randn(4, 6144, device='cuda', dtype=x.dtype)
result = silu_and_mul(activation)
gate, value = activation.float().chunk(2, dim=-1)
torch.testing.assert_close(result, (torch.nn.functional.silu(gate) * value).to(x.dtype), rtol=0.016, atol=0.016)
(root / '.port/flashinfer-comparison/dense-environment.json').write_text(json.dumps(dict(passed=True, flashinfer=flashinfer.__version__, norm=rmsnorm.__code__.co_filename, activation=silu_and_mul.__code__.co_filename), indent=2) + '\n', encoding='utf-8')
print('PASS dense model imports and TokenSpeed facade numerical tests')
