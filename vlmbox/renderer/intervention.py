"""Trainable visual interventions in native host-image coordinates."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn

Anchor = Literal["top_left", "top_right", "bottom_left", "bottom_right", "center"] | tuple[int, int]
InterventionType = Literal["overlay_patch", "local_additive"]


def anchored_bbox(
    image_size: tuple[int, int],
    region_size: tuple[int, int],
    anchor: Anchor = "bottom_right",
) -> tuple[int, int, int, int]:
    """Resolve a half-open host-local bbox using PIL (width, height) dimensions."""
    for name, size in (("image_size", image_size), ("region_size", region_size)):
        if not isinstance(size, tuple) or len(size) != 2:
            raise TypeError(f"{name} must be a tuple of two integers")
        if any(not isinstance(v, int) or isinstance(v, bool) for v in size):
            raise TypeError(f"{name} must contain only integers")
        if min(size) <= 0:
            raise ValueError("image and region dimensions must be positive")
    width, height = image_size
    region_width, region_height = region_size
    if region_width > width or region_height > height:
        raise ValueError("region must fit inside host image")
    if isinstance(anchor, str):
        positions = {
            "top_left": (0, 0),
            "top_right": (width - region_width, 0),
            "bottom_left": (0, height - region_height),
            "bottom_right": (width - region_width, height - region_height),
            "center": ((width - region_width) // 2, (height - region_height) // 2),
        }
        if anchor not in positions:
            raise ValueError(f"unknown anchor {anchor!r}")
        x, y = positions[anchor]
    elif (
        isinstance(anchor, tuple)
        and len(anchor) == 2
        and all(isinstance(v, int) and not isinstance(v, bool) for v in anchor)
    ):
        x, y = anchor
    else:
        raise TypeError("anchor must be a named anchor or (x, y) integer tuple")
    if x < 0 or y < 0 or x + region_width > width or y + region_height > height:
        raise ValueError("anchored region is outside host image; must fit inside")
    return x, y, x + region_width, y + region_height


class Intervention(nn.Module):
    """A shared RGB patch or bounded local additive delta.

    Both modes use one parameter, the same host geometry and standard PyTorch
    module operations. Only application and projection bounds differ.
    """

    def __init__(
        self,
        kind: InterventionType,
        size: tuple[int, int],
        *,
        anchor: Anchor = "bottom_right",
        epsilon: float | None = None,
        init: Literal["zeros", "random"] = "zeros",
        seed: int = 2026,
    ) -> None:
        super().__init__()
        if kind not in ("overlay_patch", "local_additive"):
            raise ValueError("kind must be overlay_patch or local_additive")
        # Validate size and anchor without requiring a particular host image.
        if not isinstance(size, tuple) or len(size) != 2:
            raise TypeError("size must be a tuple of (height, width)")
        if any(not isinstance(v, int) or isinstance(v, bool) for v in size):
            raise TypeError("size must contain only integers")
        if min(size) <= 0:
            raise ValueError("size dimensions must be positive")
        if isinstance(anchor, str):
            anchored_bbox((size[1], size[0]), (size[1], size[0]), anchor)
        elif (
            not isinstance(anchor, tuple)
            or len(anchor) != 2
            or any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in anchor)
        ):
            raise ValueError("anchor must be a named anchor or non-negative (x, y) integers")
        if kind == "local_additive":
            if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
                raise TypeError("epsilon must be a real number")
            if not math.isfinite(epsilon) or not 0 < epsilon <= 1:
                raise ValueError("epsilon must be finite and in (0, 1]")
            epsilon = float(epsilon)
        elif epsilon is not None:
            raise ValueError("overlay_patch does not accept epsilon")
        if init not in ("zeros", "random"):
            raise ValueError("init must be 'zeros' or 'random'")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer")
        if not 0 <= seed <= 2**63 - 1:
            raise ValueError("seed must be between 0 and 2**63 - 1")
        self.kind, self.size, self.anchor, self.epsilon = kind, size, anchor, epsilon
        initial = torch.zeros((3, *size), dtype=torch.float32)
        if init == "random":
            generator = torch.Generator(device="cpu").manual_seed(seed)
            lower, upper = self.bounds
            initial.uniform_(lower, upper, generator=generator)
        self.parameter = nn.Parameter(initial)

    @property
    def bounds(self) -> tuple[float, float]:
        return (0.0, 1.0) if self.epsilon is None else (-self.epsilon, self.epsilon)

    @property
    def spec(self) -> dict[str, object]:
        """Geometry and bounds saved with weights, independent of initialization."""
        return {
            "type": self.kind,
            "size": self.size,
            "anchor": self.anchor,
            "epsilon": self.epsilon,
        }

    def bbox(self, image_size: tuple[int, int]) -> tuple[int, int, int, int]:
        height, width = self.size
        return anchored_bbox(image_size, (width, height), self.anchor)

    def forward(self, images: Tensor) -> Tensor:
        if not isinstance(images, Tensor) or not images.is_floating_point():
            raise TypeError("images must be floating-point tensors")
        if images.ndim not in (3, 4) or images.shape[-3] != 3:
            raise ValueError("images must be CHW or BCHW RGB tensors")
        height, width = images.shape[-2:]
        x0, y0, x1, y1 = self.bbox((width, height))
        parameter = self.parameter.to(device=images.device, dtype=images.dtype)
        region = (..., slice(y0, y1), slice(x0, x1))
        result = images.clone()
        result[region] = (
            parameter if self.kind == "overlay_patch" else (images[region] + parameter).clamp(0, 1)
        )
        return result

    def project_(self) -> None:
        with torch.no_grad():
            self.parameter.clamp_(*self.bounds)
