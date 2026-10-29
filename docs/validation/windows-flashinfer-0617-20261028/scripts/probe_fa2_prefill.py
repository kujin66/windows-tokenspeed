import argparse
import json
import traceback
from pathlib import Path

import torch
from tokenspeed_kernel.ops.attention.mha import mha_extend_with_kvcache

parser = argparse.ArgumentParser()
parser.add_argument('--report', type=Path, default=Path(__file__).resolve().parent / 'flashinfer-comparison/fa2-prefill-probe.json')
out = parser.parse_args().report
report = dict(cases=[], passed=False)
try:
    torch.manual_seed(20261028)
    with torch.inference_mode():
        for q_lengths, kv_lengths in (((128,), (128,)), ((32, 64, 128, 1), (96, 128, 192, 33)), ((512,), (512,))):
            counts = [(n + 63) // 64 for n in kv_lengths]
            num_pages = sum(counts)
            page_ids = torch.randperm(num_pages)
            table = torch.full((len(counts), max(counts)), -1, dtype=torch.int32)
            start = 0
            for row, count in enumerate(counts):
                table[row, :count] = page_ids[start:start+count]
                start += count
            q = torch.randn(sum(q_lengths), 16, 128, device='cuda', dtype=torch.bfloat16)
            k = torch.randn(num_pages, 64, 8, 128, device='cuda', dtype=q.dtype)
            v = torch.randn_like(k)
            qptr = torch.tensor([0, *torch.tensor(q_lengths).cumsum(0).tolist()], device='cuda', dtype=torch.int32)
            kptr = torch.tensor([0, *torch.tensor(kv_lengths).cumsum(0).tolist()], device='cuda', dtype=torch.int32)
            lengths = torch.tensor(kv_lengths, device='cuda', dtype=torch.int32)
            actual = mha_extend_with_kvcache(q, qptr, kptr, k, v, table.to('cuda'), lengths,
                                             max_seqlen_q=max(q_lengths), max_seqlen_k=max(kv_lengths),
                                             is_causal=True, solution='flashinfer')
            expected_rows = []
            offset = 0
            for row, (qlen, kvlen) in enumerate(zip(q_lengths, kv_lengths, strict=True)):
                ids = table[row, :counts[row]].long().to('cuda')
                key = k.index_select(0, ids).reshape(-1, 8, 128)[:kvlen].transpose(0, 1).repeat_interleave(2, 0).unsqueeze(0).float()
                value = v.index_select(0, ids).reshape(-1, 8, 128)[:kvlen].transpose(0, 1).repeat_interleave(2, 0).unsqueeze(0).float()
                query = q[offset:offset+qlen].transpose(0, 1).unsqueeze(0).float()
                mask = torch.arange(kvlen, device='cuda')[None, :] <= (kvlen - qlen + torch.arange(qlen, device='cuda')[:, None])
                expected_rows.append(torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=mask).squeeze(0).transpose(0, 1))
                offset += qlen
            torch.testing.assert_close(actual, torch.cat(expected_rows).to(q.dtype), rtol=0.02, atol=0.02)
            report['cases'].append(dict(q_lengths=q_lengths, kv_lengths=kv_lengths, passed=True))
            print(f'PASS FA2 q={q_lengths} kv={kv_lengths}', flush=True)
    report['passed'] = True
except Exception as exc:
    report['error'] = str(exc)
    report['traceback'] = traceback.format_exc()
    raise
finally:
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
