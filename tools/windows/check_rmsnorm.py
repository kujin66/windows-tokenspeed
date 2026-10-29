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

import json
import os
from pathlib import Path

import tokenspeed_kernel
import torch

root = Path(__file__).resolve().parents[2]
handles = [os.add_dll_directory(str(Path(torch.__file__).parent / "lib"))]
import tvm_ffi

handles.append(os.add_dll_directory(str(Path(tvm_ffi.__file__).parent / "lib")))
handles.append(
    os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/bin")
)
library = (
    Path(tokenspeed_kernel.__file__).parent
    / "thirdparty/cuda/objs/rmsnorm_fused_parallel/rmsnorm_fused_parallel.so"
)
module = tvm_ffi.load_module(str(library))
torch.manual_seed(42)
cases = []
for dtype in (torch.float16, torch.bfloat16):
    for batch in (1, 7, 32):
        for dims in ((128, 256), (1024, 2048)):
            for pdl in (False, True):
                inputs = [
                    torch.randn(batch, dim, device="cuda", dtype=dtype) for dim in dims
                ]
                weights = [torch.randn(dim, device="cuda", dtype=dtype) for dim in dims]
                outputs = [torch.empty_like(x) for x in inputs]
                module.rmsnorm_fused_parallel(
                    inputs[0],
                    weights[0],
                    outputs[0],
                    inputs[1],
                    weights[1],
                    outputs[1],
                    1e-6,
                    pdl,
                )
                torch.cuda.synchronize()
                for x, w, actual in zip(inputs, weights, outputs):
                    reference = (
                        (
                            x.float()
                            * torch.rsqrt(
                                x.float().square().mean(-1, keepdim=True) + 1e-6
                            )
                        )
                        * w.float()
                    ).to(dtype)
                    torch.testing.assert_close(
                        actual,
                        reference,
                        atol=2e-3 if dtype == torch.float16 else 2e-2,
                        rtol=2e-3 if dtype == torch.float16 else 2e-2,
                    )
                cases.append(
                    {
                        "dtype": str(dtype),
                        "batch": batch,
                        "dims": dims,
                        "pdl": pdl,
                        "passed": True,
                    }
                )
report = {
    "library": str(library),
    "device": torch.cuda.get_device_name(),
    "capability": torch.cuda.get_device_capability(),
    "cuda": torch.version.cuda,
    "build_identity": json.loads(
        (library.parent / "build-identity.json").read_text(encoding="utf-8")
    ),
    "cases": cases,
}
(root / ".port/rmsnorm-results.json").write_text(json.dumps(report, indent=2))
print(f"{len(cases)} RMSNorm GPU cases passed on {report['device']}")
