"""Universal differentiable image parameters for fixed or native-size hosts."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from numbers import Real
from typing import Literal

import torch
from torch import Tensor, nn

from vlmbox.image import anchored_bbox

ImageSize = tuple[int, int]
PatchPosition = Literal["top_left", "top_right", "bottom_left", "bottom_right", "center"]


def _positive_size(name: str, value: ImageSize) -> ImageSize:
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


def _validate_seed(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if not 0 <= seed <= 2**63 - 1:
        raise ValueError("seed must be between 0 and 2**63 - 1")


def _validate_epsilon(epsilon: float) -> float:
    if isinstance(epsilon, bool) or not isinstance(epsilon, Real):
        raise TypeError("epsilon must be a real number")
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be finite and in the interval (0, 1]")
    return epsilon


def _validate_position(position: PatchPosition | tuple[int, int]) -> None:
    if isinstance(position, str):
        if position not in {"top_left", "top_right", "bottom_left", "bottom_right", "center"}:
            raise ValueError(f"unknown patch position {position!r}")
    elif (
        not isinstance(position, tuple)
        or len(position) != 2
        or any(not isinstance(v, int) or isinstance(v, bool) for v in position)
    ):
        raise TypeError("position must be an anchor or (x, y) integer tuple")


def _as_image_batch(images: Tensor, canvas_size: ImageSize | None) -> tuple[Tensor, bool]:
    if not isinstance(images, Tensor):
        raise TypeError("images must be a torch.Tensor")
    single = images.ndim == 3
    batch = images.unsqueeze(0) if single else images
    if batch.ndim != 4 or batch.shape[1] != 3:
        raise ValueError("images must be CHW or BCHW RGB tensors")
    if not batch.is_floating_point():
        raise TypeError("images must use a floating-point dtype")
    if batch.shape[-2] <= 0 or batch.shape[-1] <= 0:
        raise ValueError("images must be non-empty")
    if canvas_size is not None and tuple(batch.shape[-2:]) != canvas_size:
        raise ValueError(
            f"image canvas {tuple(batch.shape[-2:])} does not match configured canvas {canvas_size}"
        )
    return batch, single


def _initial(
    shape: tuple[int, ...],
    *,
    init: Literal["zeros", "random"],
    seed: int,
    dtype: torch.dtype,
    epsilon: float | None = None,
) -> Tensor:
    if init not in {"zeros", "random"}:
        raise ValueError("init must be 'zeros' or 'random'")
    _validate_seed(seed)
    if init == "zeros":
        return torch.zeros(shape, dtype=dtype)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if epsilon is not None:
        return torch.empty(shape, dtype=dtype).uniform_(-epsilon, epsilon, generator=generator)
    return torch.rand(shape, dtype=dtype, generator=generator)


class UniversalImageParameter(nn.Module, ABC):
    """One trainable parameter reused across all host images."""

    canvas_size: ImageSize | None

    @abstractmethod
    def apply(self, images: Tensor) -> Tensor:
        """Apply a shared parameter to CHW/BCHW RGB Tensor images."""

    @abstractmethod
    def project_(self) -> None:
        """Project parameter in-place to its feasible set."""


class UniversalPerturbation(UniversalImageParameter):
    """Fixed-canvas full-image L-infinity-bounded perturbation (existing API)."""

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
        self.epsilon = _validate_epsilon(epsilon)
        dtype = _floating_dtype(dtype)
        initial = _initial(
            (3, *self.canvas_size),
            init=init,
            seed=seed,
            dtype=dtype,
            epsilon=self.epsilon,
        )
        self.delta = nn.Parameter(initial.to(device=device, dtype=dtype))

    @property
    def parameter(self) -> nn.Parameter:
        return self.delta

    def apply(self, images: Tensor) -> Tensor:
        batch, single = _as_image_batch(images, self.canvas_size)
        delta = self.delta.to(device=batch.device, dtype=batch.dtype)
        result = (batch + delta.unsqueeze(0)).clamp(0.0, 1.0)
        return result[0] if single else result

    def project_(self) -> None:
        with torch.no_grad():
            self.delta.clamp_(-self.epsilon, self.epsilon)


class UniversalPatch(UniversalImageParameter):
    """Shared RGB overlay patch anchored to each host's image coordinates.

    Sizes use (height, width), position tuples use (x, y). A None canvas_size
    enables the same fixed-size patch on hosts of different native resolutions.
    Passing a canvas_size preserves the original fixed-canvas API.
    """

    def __init__(
        self,
        canvas_size: ImageSize | None,
        size: ImageSize,
        *,
        position: PatchPosition | tuple[int, int] = "top_left",
        init: Literal["zeros", "random"] = "random",
        seed: int = 2026,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.canvas_size = (
            _positive_size("canvas_size", canvas_size) if canvas_size is not None else None
        )
        self.size = _positive_size("size", size)
        dtype = _floating_dtype(dtype)
        _validate_position(position)
        if self.canvas_size is not None:
            self.position: PatchPosition | tuple[int, int] = self._bounds(
                self.canvas_size, position
            )[:2]
        else:
            self.position = position
        initial = _initial((3, *self.size), init=init, seed=seed, dtype=dtype)
        self.patch = nn.Parameter(initial.to(device=device, dtype=dtype))

    def _bounds(
        self, host_size: ImageSize, position: PatchPosition | tuple[int, int]
    ) -> tuple[int, int, int, int]:
        h, w = host_size
        ph, pw = self.size
        return anchored_bbox((w, h), (pw, ph), position)

    @property
    def parameter(self) -> nn.Parameter:
        return self.patch

    def apply(self, images: Tensor) -> Tensor:
        batch, single = _as_image_batch(images, self.canvas_size)
        x0, y0, x1, y1 = self._bounds(tuple(batch.shape[-2:]), self.position)
        patch = self.patch.to(device=batch.device, dtype=batch.dtype)
        result = batch.clone()
        result[:, :, y0:y1, x0:x1] = patch
        return result[0] if single else result

    def project_(self) -> None:
        with torch.no_grad():
            self.patch.clamp_(0.0, 1.0)


class UniversalLocalPerturbation(UniversalImageParameter):
    """One L-infinity-bounded local delta shared across native-size host images."""

    canvas_size: None = None

    def __init__(
        self,
        size: ImageSize,
        epsilon: float,
        *,
        position: PatchPosition | tuple[int, int] = "bottom_right",
        init: Literal["zeros", "random"] = "zeros",
        seed: int = 2026,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.size = _positive_size("size", size)
        self.epsilon = _validate_epsilon(epsilon)
        dtype = _floating_dtype(dtype)
        _validate_position(position)
        self.position = position
        initial = _initial((3, *self.size), init=init, seed=seed, dtype=dtype, epsilon=self.epsilon)
        self.delta = nn.Parameter(initial.to(device=device, dtype=dtype))

    @property
    def parameter(self) -> nn.Parameter:
        return self.delta

    def apply(self, images: Tensor) -> Tensor:
        batch, single = _as_image_batch(images, None)
        h, w = batch.shape[-2:]
        ph, pw = self.size
        x0, y0, x1, y1 = anchored_bbox((w, h), (pw, ph), self.position)
        delta = self.delta.to(device=batch.device, dtype=batch.dtype)
        output = batch.clone()
        output[:, :, y0:y1, x0:x1] = (batch[:, :, y0:y1, x0:x1] + delta).clamp(0.0, 1.0)
        return output[0] if single else output

    def project_(self) -> None:
        with torch.no_grad():
            self.delta.clamp_(-self.epsilon, self.epsilon)


__all__ = [
    "UniversalImageParameter",
    "UniversalPatch",
    "UniversalPerturbation",
    "UniversalLocalPerturbation",
]
