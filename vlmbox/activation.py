"""Capture and intervene on decoder activations with temporary forward hooks."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import ExitStack
from functools import partial
from math import isfinite
from typing import Literal

import torch
from torch import Tensor, nn

from vlmbox.adapter import ModelAdapter
from vlmbox.utils import project

Component = Literal["block", "attention", "mlp"]
PositionSelector = Literal["all", "last"] | int | Sequence[int]
InterventionOperation = Literal["add", "replace", "project_out", "zero"]
InterventionScope = Literal["first_forward", "every_forward"]


class ActivationError(ValueError):
    """Raised when an activation target or intervention is invalid."""


def layer_indices(layers: int | Sequence[int], count: int) -> tuple[int, ...]:
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


def normalize_positions(positions: PositionSelector) -> str | tuple[int, ...]:
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


def target_modules(
    adapter: ModelAdapter,
    component: Component,
    layers: int | Sequence[int],
) -> tuple[tuple[int, nn.Module], ...]:
    if component not in ("block", "attention", "mlp"):
        raise ValueError("component must be 'block', 'attention', or 'mlp'")

    indices = layer_indices(layers, len(adapter.layers))
    resolver = {
        "block": adapter.layer,
        "attention": adapter.attention,
        "mlp": adapter.mlp,
    }[component]
    return tuple((index, resolver(index)) for index in indices)


def token_indices(tensor: Tensor, positions: str | tuple[int, ...]) -> Tensor | None:
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


def activation_tensor(output: object) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
        return output[0]
    raise ActivationError(
        "target module must return a tensor or a tuple/list starting with a tensor"
    )


class ActivationCapture:
    """Capture the first output of selected decoder modules in a context."""

    def __init__(
        self,
        adapter: ModelAdapter,
        component: Component,
        layers: int | Sequence[int],
        positions: PositionSelector = "last",
    ) -> None:
        self._targets = target_modules(adapter, component, layers)
        self._positions = normalize_positions(positions)
        self.values: dict[int, Tensor] = {}
        self._stack = ExitStack()
        self._entered = False

    def __enter__(self) -> ActivationCapture:
        if self._entered:
            raise RuntimeError("an activation capture context cannot be entered more than once")
        self._entered = True

        def record(index: int, _module, _inputs, output):
            if index not in self.values:
                tensor = activation_tensor(output)
                indices = token_indices(tensor, self._positions)
                selected = tensor if indices is None else tensor.index_select(-2, indices)
                self.values[index] = selected.clone()

        with ExitStack() as stack:
            for index, module in self._targets:
                handle = module.register_forward_hook(partial(record, index))
                stack.callback(handle.remove)
            self._stack = stack.pop_all()
        return self

    def __getitem__(self, layer: int) -> Tensor:
        return self.values[layer]

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return self._stack.__exit__(exc_type, exc_value, traceback)


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
        self._targets = target_modules(adapter, component, layer)
        self._positions = normalize_positions(positions)
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
        self._stack = ExitStack()
        self._entered = False

    def __enter__(self) -> ActivationIntervention:
        if self._entered:
            raise RuntimeError(
                "an activation intervention context cannot be entered more than once"
            )
        self._entered = True
        applied = False

        def edit(_module, _inputs, output):
            nonlocal applied
            if self._scope == "first_forward" and applied:
                return output
            updated = edit_activation(
                activation_tensor(output),
                self._positions,
                self._operation,
                self._value,
                self._scale,
            )
            applied = True
            if isinstance(output, Tensor):
                return updated
            if isinstance(output, list):
                return [updated, *output[1:]]
            if hasattr(output, "_fields"):
                return type(output)(updated, *output[1:])
            return (updated, *output[1:])

        module = self._targets[0][1]
        with ExitStack() as stack:
            stack.callback(module.register_forward_hook(edit).remove)
            # Native Transformers traces must observe the edited output. Keep all
            # other hooks in registration order, including nested user captures.
            hooks = module._forward_hooks
            for key, hook in list(hooks.items()):
                if (
                    getattr(hook, "__module__", None) == "transformers.utils.output_capturing"
                    and getattr(hook, "__name__", None) == "output_capturing_hook"
                ):
                    hooks[key] = hooks.pop(key)
            self._stack = stack.pop_all()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return self._stack.__exit__(exc_type, exc_value, traceback)


def edit_activation(
    tensor: Tensor,
    positions: str | tuple[int, ...],
    operation: InterventionOperation,
    value: Tensor | None,
    scale: float,
) -> Tensor:
    if not tensor.is_floating_point():
        raise ActivationError("activation interventions require floating-point tensors")

    indices = token_indices(tensor, positions)
    selected = tensor if indices is None else tensor.index_select(-2, indices)

    if operation == "zero":
        updated = torch.zeros_like(selected)
    else:
        assert value is not None
        dtype = (
            torch.promote_types(selected.dtype, value.dtype)
            if operation == "project_out"
            else selected.dtype
        )
        direction = value.to(device=selected.device, dtype=dtype)
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
            try:
                projection = project(selected, direction)
            except ValueError as exc:
                raise ActivationError(
                    f"project_out requires a non-zero finite direction: {exc}"
                ) from exc
            updated = (selected.to(projection.dtype) - scale * projection).to(selected.dtype)

    if indices is None:
        return updated
    return tensor.index_copy(-2, indices, updated)


__all__ = [
    "ActivationCapture",
    "ActivationError",
    "ActivationIntervention",
    "Component",
    "PositionSelector",
    "InterventionOperation",
    "InterventionScope",
]
