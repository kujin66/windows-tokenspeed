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

"""compressed-tensors NVFP4 linears with packed weights and E4M3 block scales.

Checkpoint global scales are divisors. Keep each fused projection's own
weight and activation global scales rather than merging unequal scalars.
"""

from collections.abc import Callable

import torch
from tokenspeed_kernel.ops.gemm import nvfp4_linear

from tokenspeed.runtime.layers.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from tokenspeed.runtime.layers.quantization.compressed_tensors.schemes.compressed_tensors_scheme import (
    CompressedTensorsScheme,
)


class CompressedTensorsW4A4Nvfp4(CompressedTensorsScheme):
    def __init__(self):
        self._partition_ranges: list[tuple[int, int]] = []

    @classmethod
    def get_min_capability(cls) -> int:
        return 120

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size: int,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        output_size: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
    ) -> None:
        if input_size_per_partition % 16:
            raise ValueError("NVFP4 K must be divisible by 16")
        n, k = sum(output_partition_sizes), input_size_per_partition
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = k
        layer.output_size_per_partition = n
        layer.register_parameter(
            "weight_packed",
            ModelWeightParameter(
                data=torch.empty(n, k // 2, dtype=torch.uint8),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )
        layer.register_parameter(
            "weight_scale",
            GroupQuantScaleParameter(
                data=torch.empty(n, k // 16, dtype=torch.float8_e4m3fn),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )
        for name in ("weight_global_scale", "input_global_scale"):
            layer.register_parameter(
                name,
                PerTensorScaleParameter(
                    data=torch.full(
                        (len(output_partition_sizes),),
                        float("nan"),
                        dtype=torch.float32,
                    ),
                    weight_loader=weight_loader,
                ),
            )
        start = 0
        self._partition_ranges = []
        for width in output_partition_sizes:
            self._partition_ranges.append((start, start + width))
            start += width

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        for scale in (layer.weight_global_scale, layer.input_global_scale):
            if not (torch.isfinite(scale) & (scale > 0)).all():
                raise ValueError(
                    "NVFP4 global scales must be loaded, finite and positive"
                )

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None
    ):
        shape = x.shape[:-1]
        x = x.reshape(-1, layer.input_size_per_partition).contiguous()
        outputs = []
        for index, (start, end) in enumerate(self._partition_ranges):
            outputs.append(
                nvfp4_linear(
                    x,
                    layer.weight_packed[start:end],
                    layer.weight_scale[start:end],
                    layer.input_global_scale[index : index + 1],
                    layer.weight_global_scale[index : index + 1],
                    bias=None if bias is None else bias[start:end],
                    solution=None,
                )
            )
        return torch.cat(outputs, dim=-1).reshape(
            *shape, layer.output_size_per_partition
        )
