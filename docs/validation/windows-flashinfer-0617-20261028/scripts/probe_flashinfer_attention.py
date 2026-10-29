import json
import traceback
from pathlib import Path

import torch
from tokenspeed_kernel.ops.attention.mha import mha_decode_with_kvcache

out = Path(__file__).resolve().parent / 'flashinfer-comparison/attention-probe.json'
report = dict(solution='flashinfer', shape=dict(batch=1, q_heads=16, kv_heads=8, head_dim=128, page_size=64, kv_length=128), passed=False)
try:
    torch.manual_seed(20261028)
    q = torch.randn(1, 16, 128, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(2, 64, 8, 128, device='cuda', dtype=q.dtype)
    v = torch.randn_like(k)
    table = torch.tensor([[0, 1]], device='cuda', dtype=torch.int32)
    lengths = torch.tensor([128], device='cuda', dtype=torch.int32)
    actual = mha_decode_with_kvcache(q, k, v, table, lengths, max_seqlen_k=128, max_seqlen_q=1, solution='flashinfer')
    query = q.transpose(0, 1).unsqueeze(0).float()
    key = k.reshape(128, 8, 128).transpose(0, 1).repeat_interleave(2, dim=0).unsqueeze(0).float()
    value = v.reshape(128, 8, 128).transpose(0, 1).repeat_interleave(2, dim=0).unsqueeze(0).float()
    expected = torch.nn.functional.scaled_dot_product_attention(query, key, value).squeeze(0).transpose(0, 1).to(q.dtype)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    torch.cuda.synchronize()
    report['passed'] = True
except Exception as exc:
    report['error'] = str(exc)
    report['traceback'] = traceback.format_exc()
    raise
finally:
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
