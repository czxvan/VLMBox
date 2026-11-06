"""Access common language-model components in Transformers model wrappers."""

from __future__ import annotations

from collections.abc import Sequence
from operator import attrgetter
from typing import Protocol, runtime_checkable

from torch import nn


class AdapterError(RuntimeError):
    """Raised when a model's internal language architecture cannot be resolved."""


@runtime_checkable
class ModelAdapter(Protocol):
    """Decoder components indexed from zero; layer outputs precede final model normalization."""

    @property
    def layers(self) -> Sequence[nn.Module]: ...

    def layer(self, index: int) -> nn.Module: ...

    def attention(self, index: int) -> nn.Module: ...

    def mlp(self, index: int) -> nn.Module: ...


class TransformersAdapter:
    """Resolve decoder layers and common submodules from a Transformers model.

    This adapter deliberately supports only a small set of common module layouts.
    It does not guess by recursively searching the whole multimodal model, which
    could accidentally select the vision encoder instead of the language decoder.
    """

    _LANGUAGE_ROOTS = (
        "language_model",
        "model.language_model",
        "text_model",
        "model.text_model",
        "model",
        "",
    )
    _LAYER_PATHS = (
        "layers",
        "h",
        "decoder.layers",
        "model.layers",
        "transformer.h",
    )
    _ATTENTION_NAMES = ("self_attn", "attention", "attn", "self_attention")
    _MLP_NAMES = ("mlp", "feed_forward", "ffn")

    def __init__(self, model: nn.Module) -> None:
        roots: list[nn.Module] = []
        for path in self._LANGUAGE_ROOTS:
            try:
                root = attrgetter(path)(model) if path else model
            except AttributeError:
                continue
            if isinstance(root, nn.Module) and all(root is not existing for existing in roots):
                roots.append(root)
        # Prefer native decoder accessors before inspecting supported fallback paths.
        for use_accessor in (True, False):
            for root in roots:
                candidate = root
                if use_accessor:
                    getter = getattr(root, "get_decoder", None)
                    if not callable(getter):
                        continue
                    try:
                        candidate = getter()
                    except (AttributeError, NotImplementedError, TypeError):
                        continue
                for path in self._LAYER_PATHS:
                    try:
                        layers = attrgetter(path)(candidate)
                    except AttributeError:
                        continue
                    if (
                        isinstance(layers, (nn.ModuleList, list, tuple))
                        and layers
                        and all(isinstance(layer, nn.Module) for layer in layers)
                    ):
                        self._layers = layers
                        return
        model_type = getattr(getattr(model, "config", None), "model_type", type(model).__name__)
        raise AdapterError(
            f"Unable to locate decoder layers for model type {model_type!r}. "
            "The model may require a dedicated ModelAdapter."
        )

    @property
    def layers(self) -> Sequence[nn.Module]:
        return self._layers

    def layer(self, index: int) -> nn.Module:
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("layer index must be an integer")
        if index < 0 or index >= len(self._layers):
            raise IndexError(
                f"decoder layer index {index} is out of range for {len(self._layers)} layers"
            )
        return self._layers[index]

    def attention(self, index: int) -> nn.Module:
        return component_module(
            self.layer(index), self._ATTENTION_NAMES, f"attention in decoder layer {index}"
        )

    def mlp(self, index: int) -> nn.Module:
        return component_module(self.layer(index), self._MLP_NAMES, f"MLP in decoder layer {index}")


def component_module(layer: nn.Module, names: tuple[str, ...], label: str) -> nn.Module:
    """Resolve a named decoder component with an actionable layout error."""
    for name in names:
        module = getattr(layer, name, None)
        if isinstance(module, nn.Module):
            return module
    available = [name for name, _child in layer.named_children()]
    raise AdapterError(f"Could not resolve {label}. Available child modules: {available}")


__all__ = ["AdapterError", "ModelAdapter", "TransformersAdapter"]
