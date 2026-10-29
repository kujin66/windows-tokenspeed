# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Probe actual PyTorch SDPA backend support outside benchmark timing."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch.nn.attention import SDPBackend, sdpa_kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    torch.manual_seed(20261029)
    torch.set_num_threads(1)
    report = dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), flash_compiled=torch.backends.cuda.is_flash_attention_available(), enabled=dict(flash=torch.backends.cuda.flash_sdp_enabled(), memory_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(), cudnn=torch.backends.cuda.cudnn_sdp_enabled(), math=torch.backends.cuda.math_sdp_enabled(), math_reduction=torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()), cases=[], throughput_samples=False)
    with torch.inference_mode():
        for label, queries, keys, kv_heads, masked, causal in (
            ("dynamic-prefill", 128, 128, 8, False, True),
            ("dynamic-decode", 1, 192, 8, False, False),
            ("static-decode-704", 1, 704, 16, True, False),
            ("static-decode-256", 1, 256, 16, True, False),
        ):
            q = torch.randn(1, 16, queries, 128, dtype=torch.bfloat16, device="cuda")
            k = torch.randn(1, kv_heads, keys, 128, dtype=q.dtype, device="cuda")
            v = torch.randn_like(k)
            mask = (torch.arange(keys, device="cuda") < 192).reshape(1, 1, 1, keys) if masked else None
            kwargs = dict(attn_mask=mask, dropout_p=0.0, is_causal=causal, enable_gqa=kv_heads != 16)
            with sdpa_kernel(SDPBackend.MATH):
                reference = functional.scaled_dot_product_attention(q.float(), k.float(), v.float(), **kwargs)
            case = dict(name=label, q_shape=list(q.shape), kv_shape=list(k.shape), mask=masked, backends={})
            for name, backend in (("flash", SDPBackend.FLASH_ATTENTION), ("efficient", SDPBackend.EFFICIENT_ATTENTION), ("cudnn", SDPBackend.CUDNN_ATTENTION), ("math", SDPBackend.MATH), ("math_reduced", SDPBackend.MATH)):
                previous_math = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
                try:
                    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(name == "math_reduced")
                    with sdpa_kernel(backend):
                        actual = functional.scaled_dot_product_attention(q, k, v, **kwargs)
                    torch.cuda.synchronize()
                    torch.testing.assert_close(actual.float(), reference, rtol=0.03, atol=0.03)
                    case["backends"][name] = dict(passed=True, fp32_max_abs_error=(actual.float() - reference).abs().max().item())
                except Exception as error:
                    case["backends"][name] = dict(passed=False, error=f"{type(error).__name__}: {error}")
                finally:
                    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(previous_math)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                functional.scaled_dot_product_attention(q, k, v, **kwargs)
                torch.cuda.synchronize()
            case["auto_cpu_operator_names"] = [event.key for event in profile.key_averages() if "scaled_dot_product" in event.key]
            report["cases"].append(case)
    imported = [name for name in sys.modules if name.startswith("tokenspeed")]
    if imported:
        raise ValueError(f"Unexpected TokenSpeed import: {imported}")
    report["tokenspeed_modules_loaded"] = imported
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}))


if __name__ == "__main__":
    main()
