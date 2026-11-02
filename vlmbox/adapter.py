"""Access common language-model components in Transformers model wrappers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from torch import nn


class AdapterError(RuntimeError):
    """Raised when a model's internal language architecture cannot be resolved."""


@runtime_checkable
class ModelAdapter(Protocol):
    """Minimal interface used by VLMBox to access decoder components."""

    @property
    def layers(self) -> Sequence[nn.Module]: ...

    def layer(self, index: int) -> nn.Module: ...

    def attention(self, index: int) -> nn.Module: ...

    def mlp(self, index: int) -> nn.Module: ...

    @property
    def final_norm(self) -> nn.Module | None: ...

    @property
    def lm_head(self) -> nn.Module | None: ...


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
    _NORM_PATHS = (
        "norm",
        "model.norm",
        "final_layer_norm",
        "model.final_layer_norm",
        "final_layernorm",
        "model.final_layernorm",
        "ln_f",
        "model.ln_f",
    )

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._decoder, self._layers = self._resolve_layers()

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
        return self._component(index, self._ATTENTION_NAMES, "attention")

    def mlp(self, index: int) -> nn.Module:
        return self._component(index, self._MLP_NAMES, "MLP")

    @property
    def final_norm(self) -> nn.Module | None:
        for path in self._NORM_PATHS:
            module = self._get_path(self._decoder, path)
            if isinstance(module, nn.Module):
                return module
        return None

    @property
    def lm_head(self) -> nn.Module | None:
        for root in (self.model, self._decoder):
            module = self._find_module(root, ("lm_head", "output_projection"))
            if module is not None:
                return module
        return None

    def _component(self, index: int, names: tuple[str, ...], label: str) -> nn.Module:
        layer = self.layer(index)
        module = self._find_module(layer, names)
        if module is None:
            available = [name for name, child in layer.named_children()]
            raise AdapterError(
                f"Could not resolve {label} in decoder layer {index}. "
                f"Available child modules: {available}"
            )
        return module

    def _resolve_layers(self) -> tuple[nn.Module, Sequence[nn.Module]]:
        # Prefer the public Transformers decoder accessor when it is implemented.
        for root in self._candidate_roots():
            getter = getattr(root, "get_decoder", None)
            if callable(getter):
                try:
                    decoder = getter()
                except (AttributeError, NotImplementedError, TypeError):
                    continue
                layers = self._layers_from(decoder)
                if layers is not None:
                    return decoder, layers

        for root in self._candidate_roots():
            layers = self._layers_from(root)
            if layers is not None:
                return root, layers

        config = getattr(self.model, "config", None)
        model_type = getattr(config, "model_type", type(self.model).__name__)
        raise AdapterError(
            f"Unable to locate decoder layers for model type {model_type!r}. "
            "The model may require a dedicated ModelAdapter."
        )

    def _candidate_roots(self) -> list[nn.Module]:
        roots: list[nn.Module] = []
        seen: set[int] = set()
        for path in self._LANGUAGE_ROOTS:
            root = self._get_path(self.model, path)
            if isinstance(root, nn.Module) and id(root) not in seen:
                roots.append(root)
                seen.add(id(root))
        return roots

    def _layers_from(self, root: nn.Module) -> Sequence[nn.Module] | None:
        for path in self._LAYER_PATHS:
            value = self._get_path(root, path)
            if self._is_layer_sequence(value):
                return value
        return None

    @staticmethod
    def _is_layer_sequence(value: object) -> bool:
        return (
            isinstance(value, (nn.ModuleList, list, tuple))
            and len(value) > 0
            and all(isinstance(item, nn.Module) for item in value)
        )

    @staticmethod
    def _get_path(root: object, path: str) -> object | None:
        current = root
        if not path:
            return current
        for part in path.split("."):
            current = getattr(current, part, None)
            if current is None:
                return None
        return current

    @staticmethod
    def _find_module(root: nn.Module, names: tuple[str, ...]) -> nn.Module | None:
        for name in names:
            module = getattr(root, name, None)
            if isinstance(module, nn.Module):
                return module
        return None
