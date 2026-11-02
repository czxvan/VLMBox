"""Training-free direction extraction and geometry for hidden-state tensors."""

from __future__ import annotations

from math import isfinite

import torch
from torch import Tensor


def _require_floating(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if value.ndim == 0:
        raise ValueError(f"{name} must have at least one dimension")


def _require_same_device(first_name: str, first: Tensor, second_name: str, second: Tensor) -> None:
    if first.device != second.device:
        raise ValueError(
            f"{first_name} and {second_name} must be on the same device; "
            f"got {first.device} and {second.device}"
        )


def _compute_dtype(*values: Tensor) -> torch.dtype:
    dtype = values[0].dtype
    for value in values[1:]:
        dtype = torch.promote_types(dtype, value.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _normalize_dim(dim: int, ndim: int) -> int:
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise TypeError("dim must be an integer")
    if not -ndim <= dim < ndim:
        raise IndexError(f"dimension {dim} is out of range for a {ndim}-dimensional tensor")
    return dim % ndim


def _validate_eps(eps: float) -> float:
    if isinstance(eps, bool) or not isinstance(eps, (int, float)):
        raise TypeError("eps must be a positive finite number")
    if not isfinite(float(eps)) or eps <= 0:
        raise ValueError("eps must be a positive finite number")
    return float(eps)


def _validate_nonzero_norm(norm: Tensor, eps: float, name: str) -> None:
    if not torch.isfinite(norm).all().item():
        raise ValueError(f"{name} must contain only finite values")
    if torch.any(norm <= eps).item():
        raise ValueError(f"{name} must have a norm greater than eps={eps}")


def mean_difference(
    positive: Tensor,
    negative: Tensor,
    dim: int = 0,
) -> Tensor:
    """Return mean(positive) - mean(negative) along the sample dimension.

    The tensors must have equal rank and matching shapes outside dim. Their
    sample counts along dim may differ. The result keeps all non-sample
    dimensions and uses at least float32 accumulation for fp16/bf16.
    The direction points from the negative condition toward the positive one.
    """
    _require_floating("positive", positive)
    _require_floating("negative", negative)
    _require_same_device("positive", positive, "negative", negative)
    if positive.ndim != negative.ndim:
        raise ValueError("positive and negative must have the same number of dimensions")

    dim = _normalize_dim(dim, positive.ndim)
    if positive.shape[dim] == 0 or negative.shape[dim] == 0:
        raise ValueError("positive and negative must contain at least one sample")
    for axis, (positive_size, negative_size) in enumerate(
        zip(positive.shape, negative.shape, strict=True)
    ):
        if axis != dim and positive_size != negative_size:
            raise ValueError(
                "positive and negative must have matching non-sample dimensions; "
                f"dimension {axis} has sizes {positive_size} and {negative_size}"
            )

    dtype = _compute_dtype(positive, negative)
    return positive.to(dtype=dtype).mean(dim=dim) - negative.to(dtype=dtype).mean(dim=dim)


def normalize_direction(direction: Tensor, dim: int = -1, eps: float = 1e-12) -> Tensor:
    """Normalize a direction along dim; reject zero or non-finite vectors."""
    _require_floating("direction", direction)
    dim = _normalize_dim(dim, direction.ndim)
    eps = _validate_eps(eps)
    value = direction.to(dtype=_compute_dtype(direction))
    norm = torch.linalg.vector_norm(value, dim=dim, keepdim=True)
    _validate_nonzero_norm(norm, eps, "direction")
    return value / norm


def _broadcast_pair(hidden: Tensor, direction: Tensor) -> tuple[Tensor, Tensor]:
    _require_floating("hidden", hidden)
    _require_floating("direction", direction)
    _require_same_device("hidden", hidden, "direction", direction)
    try:
        hidden, direction = torch.broadcast_tensors(hidden, direction)
    except RuntimeError as exc:
        raise ValueError(
            f"hidden shape {tuple(hidden.shape)} and direction shape "
            f"{tuple(direction.shape)} are not broadcastable"
        ) from exc
    dtype = _compute_dtype(hidden, direction)
    return hidden.to(dtype=dtype), direction.to(dtype=dtype)


def projection_coefficient(
    hidden: Tensor,
    direction: Tensor,
    dim: int = -1,
    eps: float = 1e-12,
    keepdim: bool = False,
) -> Tensor:
    """Return dot(hidden, direction) / dot(direction, direction).

    Inputs follow standard PyTorch broadcasting rules. The feature dimension
    is reduced; set keepdim=True to retain it as a singleton dimension.
    """
    if not isinstance(keepdim, bool):
        raise TypeError("keepdim must be a boolean")
    hidden, direction = _broadcast_pair(hidden, direction)
    dim = _normalize_dim(dim, hidden.ndim)
    eps = _validate_eps(eps)
    norm_squared = direction.square().sum(dim=dim, keepdim=True)
    _validate_nonzero_norm(torch.sqrt(norm_squared), eps, "direction")
    coefficient = (hidden * direction).sum(dim=dim, keepdim=True) / norm_squared
    return coefficient if keepdim else coefficient.squeeze(dim)


def projection_length(
    hidden: Tensor,
    direction: Tensor,
    dim: int = -1,
    eps: float = 1e-12,
) -> Tensor:
    """Return the signed scalar projection of hidden onto a unit direction."""
    hidden, direction = _broadcast_pair(hidden, direction)
    dim = _normalize_dim(dim, hidden.ndim)
    unit_direction = normalize_direction(direction, dim=dim, eps=eps)
    return (hidden * unit_direction).sum(dim=dim)


def project(
    hidden: Tensor,
    direction: Tensor,
    dim: int = -1,
    eps: float = 1e-12,
) -> Tensor:
    """Return the vector projection of hidden onto direction."""
    hidden, direction = _broadcast_pair(hidden, direction)
    dim = _normalize_dim(dim, hidden.ndim)
    coefficient = projection_coefficient(
        hidden, direction, dim=dim, eps=eps, keepdim=True
    )
    return coefficient * direction


def cosine_similarity(
    first: Tensor,
    second: Tensor,
    dim: int = -1,
    eps: float = 1e-12,
) -> Tensor:
    """Compute cosine similarity with broadcasting and zero-vector checks."""
    first, second = _broadcast_pair(first, second)
    dim = _normalize_dim(dim, first.ndim)
    eps = _validate_eps(eps)
    first_norm = torch.linalg.vector_norm(first, dim=dim)
    second_norm = torch.linalg.vector_norm(second, dim=dim)
    _validate_nonzero_norm(first_norm, eps, "first")
    _validate_nonzero_norm(second_norm, eps, "second")
    numerator = (first * second).sum(dim=dim)
    return (numerator / (first_norm * second_norm)).clamp(-1.0, 1.0)


def angle(
    first: Tensor,
    second: Tensor,
    dim: int = -1,
    degrees: bool = False,
    eps: float = 1e-12,
) -> Tensor:
    """Compute the angle between vectors in radians or degrees."""
    if not isinstance(degrees, bool):
        raise TypeError("degrees must be a boolean")
    similarity = cosine_similarity(first, second, dim=dim, eps=eps)
    result = torch.acos(similarity)
    return torch.rad2deg(result) if degrees else result


__all__ = [
    "angle",
    "cosine_similarity",
    "mean_difference",
    "normalize_direction",
    "project",
    "projection_coefficient",
    "projection_length",
]
