"""Universal, differentiable image parameters for VLM optimization."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from numbers import Real
from typing import Literal

import torch
from torch import Tensor, nn

ImageSize = tuple[int, int]
PatchPosition = Literal["top_left", "top_right", "bottom_left", "bottom_right", "center"]


def _positive_size(name: str, value: tuple[int, int]) -> ImageSize:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must be a tuple of (height, width)")
    if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        raise TypeError(f"{name} must contain only integers")
    if any(item <= 0 for item in value):
        raise ValueError(f"{name} dimensions must be positive")
    return value


def _floating_dtype(dtype: torch.dtype) -> torch.dtype:
    if not isinstance(dtype, torch.dtype) or not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("dtype must be a floating-point torch.dtype")
    return dtype


def _validate_batch(images: Tensor, canvas_size: ImageSize) -> None:
    if not isinstance(images, Tensor):
        raise TypeError("images must be a torch.Tensor")
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape [batch, 3, height, width]")
    if not images.is_floating_point():
        raise TypeError("images must use a floating-point dtype")
    if tuple(images.shape[-2:]) != canvas_size:
        raise ValueError(
            "image canvas "
            f"{tuple(images.shape[-2:])} does not match configured canvas {canvas_size}"
        )


class UniversalImageParameter(nn.Module, ABC):
    """Base class for one image parameter shared by every training sample."""

    canvas_size: ImageSize

    @abstractmethod
    def apply(self, images: Tensor) -> Tensor:
        """Apply the same trainable image parameter to every image in a batch."""

    @abstractmethod
    def project_(self) -> None:
        """Project trainable values back into their feasible set in-place."""


class UniversalPerturbation(UniversalImageParameter):
    """A single shared, full-canvas L-infinity-bounded RGB perturbation.

    ``canvas_size`` is ``(height, width)``. The trainable ``delta`` is shared
    across the entire dataset and is constrained to ``[-epsilon, epsilon]``.
    """

    def __init__(
        self,
        canvas_size: ImageSize,
        epsilon: float,
        *,
        init: Literal["zeros", "random"] = "zeros",
        seed: int = 2026,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.canvas_size = _positive_size("canvas_size", canvas_size)
        dtype = _floating_dtype(dtype)
        if isinstance(epsilon, bool) or not isinstance(epsilon, Real):
            raise TypeError("epsilon must be a real number")
        epsilon = float(epsilon)
        if not math.isfinite(epsilon) or not 0.0 < epsilon <= 1.0:
            raise ValueError("epsilon must be finite and in the interval (0, 1]")
        if init not in {"zeros", "random"}:
            raise ValueError("init must be 'zeros' or 'random'")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer")
        if not 0 <= seed <= 2**63 - 1:
            raise ValueError("seed must be between 0 and 2**63 - 1")
        self.epsilon = epsilon

        shape = (3, *self.canvas_size)
        if init == "zeros":
            initial = torch.zeros(shape, dtype=dtype)
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            initial = torch.empty(shape, dtype=dtype).uniform_(
                -epsilon, epsilon, generator=generator
            )
        self.delta = nn.Parameter(initial.to(device=device, dtype=dtype))

    @property
    def parameter(self) -> nn.Parameter:
        """The single shared perturbation tensor, shaped ``[3, H, W]``."""
        return self.delta

    def apply(self, images: Tensor) -> Tensor:
        _validate_batch(images, self.canvas_size)
        delta = self.delta.to(device=images.device, dtype=images.dtype)
        return (images + delta.unsqueeze(0)).clamp(0.0, 1.0)

    def project_(self) -> None:
        with torch.no_grad():
            self.delta.clamp_(-self.epsilon, self.epsilon)


class UniversalPatch(UniversalImageParameter):
    """A single shared RGB patch placed at one fixed location on every image.

    ``canvas_size`` and ``size`` use ``(height, width)`` order. Explicit
    ``position`` coordinates use ``(x, y)`` order.
    """

    def __init__(
        self,
        canvas_size: ImageSize,
        size: ImageSize,
        *,
        position: PatchPosition | tuple[int, int] = "top_left",
        init: Literal["zeros", "random"] = "random",
        seed: int = 2026,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.canvas_size = _positive_size("canvas_size", canvas_size)
        self.size = _positive_size("size", size)
        dtype = _floating_dtype(dtype)
        patch_height, patch_width = self.size
        canvas_height, canvas_width = self.canvas_size
        if patch_height > canvas_height or patch_width > canvas_width:
            raise ValueError("patch size must fit inside canvas_size")
        if init not in {"zeros", "random"}:
            raise ValueError("init must be 'zeros' or 'random'")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer")
        if not 0 <= seed <= 2**63 - 1:
            raise ValueError("seed must be between 0 and 2**63 - 1")
        self.position = self._resolve_position(position)

        shape = (3, patch_height, patch_width)
        if init == "zeros":
            initial = torch.zeros(shape, dtype=dtype)
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            initial = torch.rand(shape, dtype=dtype, generator=generator)
        self.patch = nn.Parameter(initial.to(device=device, dtype=dtype))

    @property
    def parameter(self) -> nn.Parameter:
        """The single shared patch tensor, shaped ``[3, patch_height, patch_width]``."""
        return self.patch

    def _resolve_position(self, position: PatchPosition | tuple[int, int]) -> tuple[int, int]:
        canvas_height, canvas_width = self.canvas_size
        patch_height, patch_width = self.size
        if isinstance(position, str):
            positions: dict[str, tuple[int, int]] = {
                "top_left": (0, 0),
                "top_right": (canvas_width - patch_width, 0),
                "bottom_left": (0, canvas_height - patch_height),
                "bottom_right": (canvas_width - patch_width, canvas_height - patch_height),
                "center": (
                    (canvas_width - patch_width) // 2,
                    (canvas_height - patch_height) // 2,
                ),
            }
            if position not in positions:
                raise ValueError(f"unknown patch position {position!r}")
            return positions[position]
        if (
            not isinstance(position, tuple)
            or len(position) != 2
            or any(not isinstance(item, int) or isinstance(item, bool) for item in position)
        ):
            raise TypeError("position must be a supported name or an (x, y) integer tuple")
        x, y = position
        if x < 0 or y < 0 or x + patch_width > canvas_width or y + patch_height > canvas_height:
            raise ValueError("patch position places the patch outside canvas_size")
        return position

    def apply(self, images: Tensor) -> Tensor:
        _validate_batch(images, self.canvas_size)
        x, y = self.position
        patch_height, patch_width = self.size
        patch = self.patch.to(device=images.device, dtype=images.dtype)
        attacked = images.clone()
        attacked[:, :, y : y + patch_height, x : x + patch_width] = patch
        return attacked

    def project_(self) -> None:
        with torch.no_grad():
            self.patch.clamp_(0.0, 1.0)


__all__ = ["UniversalImageParameter", "UniversalPatch", "UniversalPerturbation"]
