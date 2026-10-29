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

"""Model registry helpers for resolving runtime model entry classes."""

import ast
import importlib
import importlib.util
import pkgutil
from collections.abc import Set
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import torch.nn as nn

from tokenspeed.runtime.plugins.registry import (
    registered_architectures,
    registered_model,
)
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


@dataclass
class _ModelRegistry:
    # Keyed by model_arch
    models: dict[str, type[nn.Module] | str] = field(default_factory=dict)

    def get_supported_archs(self) -> Set[str]:
        return self.models.keys() | registered_architectures()

    def _raise_for_unsupported(self, architectures: list[str]):
        all_supported_archs = self.get_supported_archs()

        if any(arch in all_supported_archs for arch in architectures):
            raise ValueError(
                f"Model architectures {architectures} failed "
                "to be inspected. Please check the logs for more details."
            )

        raise ValueError(
            f"Model architectures {architectures} are not supported for now. "
            f"Supported architectures: {all_supported_archs}"
        )

    def _try_load_model_cls(self, model_arch: str) -> type[nn.Module] | None:
        registered = registered_model(model_arch)
        in_tree = self.models.get(model_arch)
        if isinstance(in_tree, str) and (registered is None or not registered.override):
            module_name, class_name = in_tree.split(":", 1)
            in_tree = getattr(importlib.import_module(module_name), class_name)
            self.models[model_arch] = in_tree
        if registered is None:
            return in_tree
        if in_tree is not None and not registered.override:
            raise ValueError(
                f"Plugin model {registered.cls.__name__} registers architecture "
                f"{model_arch!r}, which the in-tree {model_arch} already "
                "serves; register it with override=True to replace it"
            )
        return registered.cls

    def _normalize_archs(
        self,
        architectures: str | list[str],
    ) -> list[str]:
        if isinstance(architectures, str):
            architectures = [architectures]
        if not architectures:
            logger.warning("No model architectures are specified")

        return architectures

    def resolve_model_cls(
        self,
        architectures: str | list[str],
    ) -> tuple[type[nn.Module], str]:
        architectures = self._normalize_archs(architectures)

        for arch in architectures:
            model_cls = self._try_load_model_cls(arch)
            if model_cls is not None:
                return (model_cls, arch)

        return self._raise_for_unsupported(architectures)


@lru_cache
def import_model_classes():
    """Discover entrypoints without importing unrelated models and their backends."""
    model_arch_name_to_cls = {}
    package_name = "tokenspeed.runtime.models"
    package = importlib.import_module(package_name)

    for _, name, _ in pkgutil.iter_modules(package.__path__, package_name + "."):
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None or not spec.origin.endswith(".py"):
            continue
        tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
        for statement in tree.body:
            if not isinstance(statement, ast.Assign) or not any(
                isinstance(target, ast.Name) and target.id == "EntryClass"
                for target in statement.targets
            ):
                continue
            value = statement.value
            entries = (
                value.elts if isinstance(value, (ast.List, ast.Tuple)) else [value]
            )
            for entry in entries:
                if not isinstance(entry, ast.Name):
                    raise ValueError(f"Model entrypoint in {name} must name a class")
                if entry.id in model_arch_name_to_cls:
                    raise ValueError(f"Duplicated model implementation for {entry.id}")
                model_arch_name_to_cls[entry.id] = f"{name}:{entry.id}"

    return model_arch_name_to_cls


ModelRegistry = _ModelRegistry(import_model_classes())
