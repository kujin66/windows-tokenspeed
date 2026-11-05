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

import importlib
from collections.abc import MutableMapping
from enum import Enum

from tokenspeed.runtime.layers.quantization.base_config import (  # noqa: F401
    QuantizationConfig,
    QuantizeMethodBase,
)

_CONFIG_PATHS = {
    "fp8": "fp8:Fp8Config",
    "mxfp8": "fp8:Mxfp8Config",
    "mxfp8_nvfp4": "mxfp8_nvfp4:Mxfp8Nvfp4Config",
    "w8a8_fp8": "w8a8_fp8:W8A8Fp8Config",
    "compressed-tensors": "compressed_tensors.compressed_tensors:CompressedTensorsConfig",
    "nvfp4": "nvfp4:Nvfp4Config",
    "mxfp4": "mxfp4:Mxfp4Config",
    "modelopt_mixed": "modelopt_mixed:ModelOptMixedConfig",
}


class _LazyConfigRegistry(MutableMapping):
    """Resolve only selected quantizers, keeping dense startup vendor-neutral."""

    def __init__(self):
        self._entries = dict(_CONFIG_PATHS)

    def __getitem__(self, key):
        entry = self._entries[key]
        if isinstance(entry, str):
            module, name = entry.split(":", 1)
            entry = getattr(importlib.import_module(f"{__name__}.{module}"), name)
            self._entries[key] = entry
        return entry

    def __setitem__(self, key, value):
        self._entries[key] = value

    def __delitem__(self, key):
        del self._entries[key]

    def __iter__(self):
        return iter(self._entries)

    def __len__(self):
        return len(self._entries)

    def __contains__(self, key):
        return key in self._entries


def __getattr__(name):
    for key, path in _CONFIG_PATHS.items():
        if path.split(":", 1)[1] == name:
            return QUANTIZATION_METHODS[key]
    raise AttributeError(name)


BASE_QUANTIZATION_METHODS = _LazyConfigRegistry()

QUANTIZATION_METHODS = BASE_QUANTIZATION_METHODS


def get_quantization_config(quantization: str) -> type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(
            f"Invalid quantization method: {quantization}. "
            f"Available methods: {list(QUANTIZATION_METHODS.keys())}"
        )
    return QUANTIZATION_METHODS[quantization]


class FusedMoeWeightScaleSupported(Enum):
    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"
    BLOCK = "block"
