"""Capture and intervene on decoder activations with temporary forward hooks."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import Literal

import torch
from torch import Tensor, nn

from vlmbox.adapter import ModelAdapter

Component = Literal["block", "attention", "mlp"]
PositionSelector = Literal["all", "last"] | int | Sequence[int]
InterventionOperation = Literal["add", "replace", "project_out", "zero"]
InterventionScope = Literal["first_forward", "every_forward"]


class ActivationError(ValueError):
    """Raised when an activation target or intervention is invalid."""


def _normalize_layers(layers: int | Sequence[int], count: int) -> tuple[int, ...]:
    if isinstance(layers, bool):
        raise TypeError("layers must be an integer or a sequence of integers")
    if isinstance(layers, int):
        selected = (layers,)
    elif isinstance(layers, Sequence) and not isinstance(layers, (str, bytes)):
        selected = tuple(layers)
    else:
        raise TypeError("layers must be an integer or a sequence of integers")

    if not selected:
        raise ValueError("layers must contain at least one decoder layer")
    if any(not isinstance(index, int) or isinstance(index, bool) for index in selected):
        raise TypeError("every layer index must be an integer")
    if len(set(selected)) != len(selected):
        raise ValueError("layers must not contain duplicate indices")

    for index in selected:
        if index < 0 or index >= count:
            raise IndexError(f"decoder layer index {index} is out of range for {count} layers")
    return selected


def _normalize_positions(positions: PositionSelector) -> str | tuple[int, ...]:
    if isinstance(positions, str):
        if positions not in ("all", "last"):
            raise ValueError("positions must be 'all', 'last', or integer indices")
        return positions
    if isinstance(positions, bool):
        raise TypeError("positions must be 'all', 'last', or integer indices")
    if isinstance(positions, int):
        return (positions,)
    if isinstance(positions, Sequence):
        selected = tuple(positions)
        if not selected:
            raise ValueError("positions must contain at least one index")
        if any(not isinstance(index, int) or isinstance(index, bool) for index in selected):
            raise TypeError("every position index must be an integer")
        if len(set(selected)) != len(selected):
            raise ValueError("positions must not contain duplicate indices")
        return selected
    raise TypeError("positions must be 'all', 'last', or integer indices")


def _resolve_modules(
    adapter: ModelAdapter,
    component: Component,
    layers: int | Sequence[int],
) -> tuple[tuple[int, nn.Module], ...]:
    if component not in ("block", "attention", "mlp"):
        raise ValueError("component must be 'block', 'attention', or 'mlp'")

    indices = _normalize_layers(layers, len(adapter.layers))
    resolver = {
        "block": adapter.layer,
        "attention": adapter.attention,
        "mlp": adapter.mlp,
    }[component]
    return tuple((index, resolver(index)) for index in indices)


def _position_indices(tensor: Tensor, positions: str | tuple[int, ...]) -> Tensor | None:
    if tensor.ndim == 0:
        raise ActivationError("activation output must have at least one dimension")
    if positions == "all":
        return None
    if positions == "last":
        if tensor.ndim == 1:
            return None
        return torch.tensor([tensor.shape[-2] - 1], device=tensor.device, dtype=torch.long)

    if tensor.ndim == 1:
        if all(index in (0, -1) for index in positions):
            if len(positions) > 1:
                raise ActivationError(
                    "position indices resolve to duplicates for a single-token output"
                )
            return None
        raise ActivationError(
            "explicit token positions require an activation with a sequence dimension"
        )

    sequence_length = tensor.shape[-2]
    normalized = tuple(index if index >= 0 else sequence_length + index for index in positions)
    if any(index < 0 or index >= sequence_length for index in normalized):
        raise IndexError(
            f"token position is out of range for activation sequence length {sequence_length}"
        )
    if len(set(normalized)) != len(normalized):
        raise ActivationError("position indices resolve to duplicate tokens")
    return torch.tensor(normalized, device=tensor.device, dtype=torch.long)


def _select_positions(tensor: Tensor, positions: str | tuple[int, ...]) -> Tensor:
    indices = _position_indices(tensor, positions)
    if indices is None:
        return tensor
    return tensor.index_select(-2, indices)


def _first_tensor(output: object) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
        return output[0]
    raise ActivationError(
        "target module must return a tensor or a tuple/list starting with a tensor"
    )


def _replace_first_tensor(output: object, transform) -> object:
    if isinstance(output, Tensor):
        return transform(output)
    if isinstance(output, tuple):
        if not output or not isinstance(output[0], Tensor):
            raise ActivationError(
                "target module must return a tensor or a tuple/list starting with a tensor"
            )
        updated = transform(output[0])
        if hasattr(output, "_fields"):
            return type(output)(updated, *output[1:])
        return (updated, *output[1:])
    if isinstance(output, list):
        if not output or not isinstance(output[0], Tensor):
            raise ActivationError(
                "target module must return a tensor or a tuple/list starting with a tensor"
            )
        updated = list(output)
        updated[0] = transform(output[0])
        return updated
    raise ActivationError(
        "target module must return a tensor or a tuple/list starting with a tensor"
    )


def _move_transformers_output_capture_hooks_after(module: nn.Module) -> None:
    """Keep Transformers' persistent output-capture hooks after active edits."""
    hooks = module._forward_hooks
    capture_hooks = [
        (key, hook)
        for key, hook in hooks.items()
        if getattr(hook, "__module__", None) == "transformers.utils.output_capturing"
        and getattr(hook, "__name__", None) == "output_capturing_hook"
    ]
    for key, _hook in capture_hooks:
        hooks.pop(key)
    for key, hook in capture_hooks:
        hooks[key] = hook


class ActivationCapture:
    """Capture the first output of selected decoder modules in a context."""

    def __init__(
        self,
        adapter: ModelAdapter,
        component: Component,
        layers: int | Sequence[int],
        positions: PositionSelector = "last",
    ) -> None:
        self._targets = _resolve_modules(adapter, component, layers)
        self._positions = _normalize_positions(positions)
        self.values: dict[int, Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._entered = False

    def __enter__(self) -> ActivationCapture:
        if self._entered:
            raise RuntimeError("an activation capture context cannot be entered more than once")
        self._entered = True
        try:
            for index, module in self._targets:
                self._handles.append(module.register_forward_hook(self._make_hook(index)))
        except Exception:
            self._remove_hooks()
            raise
        return self

    def _make_hook(self, index: int):
        captured = False

        def hook(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> object:
            nonlocal captured
            if not captured:
                tensor = _first_tensor(output)
                self.values[index] = _select_positions(tensor, self._positions).clone()
                captured = True
            return output

        return hook

    def __getitem__(self, layer: int) -> Tensor:
        return self.values[layer]

    def _remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self._remove_hooks()
        return False


class ActivationIntervention:
    """Apply a temporary activation edit to one decoder module.

    Add computes ``h + scale * value``; replace broadcasts ``value`` over the
    selected activations; project_out removes the component along ``value``;
    zero sets the selected activations to zero. Normal nested-hook ordering is
    preserved. Previously installed Transformers output-capture hooks are moved
    behind the intervention so native traces observe the edited activation.
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        component: Component,
        layer: int,
        operation: InterventionOperation,
        value: Tensor | None = None,
        scale: float = 1.0,
        positions: PositionSelector = "last",
        scope: InterventionScope = "every_forward",
    ) -> None:
        self._targets = _resolve_modules(adapter, component, layer)
        self._positions = _normalize_positions(positions)
        if operation not in ("add", "replace", "project_out", "zero"):
            raise ValueError("operation must be 'add', 'replace', 'project_out', or 'zero'")
        if scope not in ("first_forward", "every_forward"):
            raise ValueError("scope must be 'first_forward' or 'every_forward'")
        if isinstance(scale, bool) or not isinstance(scale, (int, float)):
            raise TypeError("scale must be a finite number")
        if not isfinite(float(scale)):
            raise ValueError("scale must be a finite number")
        if operation in ("replace", "zero") and float(scale) != 1.0:
            raise ValueError("scale is only supported by 'add' and 'project_out' operations")
        if operation == "zero":
            if value is not None:
                raise ValueError("value must be omitted for the 'zero' operation")
        elif not isinstance(value, Tensor):
            raise TypeError(f"value must be a torch.Tensor for the '{operation}' operation")

        self._operation = operation
        self._value = value
        self._scale = float(scale)
        self._scope = scope
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._entered = False

    def __enter__(self) -> ActivationIntervention:
        if self._entered:
            raise RuntimeError(
                "an activation intervention context cannot be entered more than once"
            )
        self._entered = True
        try:
            for _index, module in self._targets:
                self._handles.append(module.register_forward_hook(self._make_hook()))
                _move_transformers_output_capture_hooks_after(module)
        except Exception:
            self._remove_hooks()
            raise
        return self

    def _make_hook(self):
        applied = False

        def hook(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> object:
            nonlocal applied
            if self._scope == "first_forward" and applied:
                return output
            updated = _replace_first_tensor(
                output,
                lambda tensor: _apply_intervention(
                    tensor,
                    self._positions,
                    self._operation,
                    self._value,
                    self._scale,
                ),
            )
            applied = True
            return updated

        return hook

    def _remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self._remove_hooks()
        return False


def _apply_intervention(
    tensor: Tensor,
    positions: str | tuple[int, ...],
    operation: InterventionOperation,
    value: Tensor | None,
    scale: float,
) -> Tensor:
    if not tensor.is_floating_point():
        raise ActivationError("activation interventions require floating-point tensors")

    indices = _position_indices(tensor, positions)
    selected = tensor if indices is None else tensor.index_select(-2, indices)

    if operation == "zero":
        updated = torch.zeros_like(selected)
    else:
        assert value is not None
        direction = value.to(device=selected.device, dtype=selected.dtype)
        try:
            direction = torch.broadcast_to(direction, selected.shape)
        except RuntimeError as exc:
            raise ActivationError(
                f"intervention value with shape {tuple(value.shape)} cannot broadcast "
                f"to selected activation shape {tuple(selected.shape)}"
            ) from exc
        if not torch.isfinite(direction).all().item():
            raise ActivationError("intervention value must contain only finite values")

        if operation == "add":
            updated = selected + scale * direction
        elif operation == "replace":
            updated = direction
        else:
            norm_squared = direction.square().sum(dim=-1, keepdim=True)
            if torch.any(norm_squared == 0).item():
                raise ActivationError("project_out direction must be non-zero")
            coefficient = (selected * direction).sum(dim=-1, keepdim=True) / norm_squared
            updated = selected - scale * coefficient * direction

    if indices is None:
        return updated
    return tensor.index_copy(-2, indices, updated)
