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

"""Optional finite-shape Windows CuTe router artifacts, without a compiler import."""

import torch
from tokenspeed_kernel.cute_dsl_aot import configured_router_aot
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement, pdl_enabled
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


def cute_dsl_aot_router_supported(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether an opted-in artifact matches this BF16 call with PDL disabled."""
    catalog = configured_router_aot()
    return catalog is not None and not pdl_enabled() and catalog.supports(a, b)


@register_kernel(
    "gemm",
    "router_projection",
    name="cute_dsl_aot_router",
    solution="cute_dsl_aot",
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(12, 1),
        max_arch_version=ArchVersion(12, 1),
    ),
    signatures={
        format_signature(
            hidden_states=dense_tensor_format(torch.bfloat16),
            weight=dense_tensor_format(torch.bfloat16),
        )
    },
    traits={
        "m": frozenset({1, 2, 4}),
        "n": frozenset({896}),
        "k": frozenset({7168}),
        "aot_supported": frozenset({True}),
        "out_dtype": frozenset({torch.float32}),
    },
    priority=Priority.SPECIALIZED,
)
def cute_dsl_aot_router(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return FP32 A @ weight.T through the configured exported router kernel.

    Args:
        hidden_states: Contiguous CUDA BF16 [M,K] input, with an exported M/N/K.
        weight: Contiguous CUDA BF16 [N,K] weights on the same device.
        out: Optional contiguous FP32 [M,N] destination, disjoint from inputs.
    """
    if not cute_dsl_aot_router_supported(hidden_states, weight):
        raise ValueError(
            "AOT router requires matching configured artifacts and disabled PDL"
        )
    catalog = configured_router_aot()
    if out is None:
        out = torch.empty(
            (hidden_states.shape[0], weight.shape[0]),
            device=hidden_states.device,
            dtype=torch.float32,
        )
    return catalog.run(hidden_states, weight, out=out)
