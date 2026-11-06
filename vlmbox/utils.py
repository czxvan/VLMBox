"""Small numerical and reproducibility utilities for VLM experiments."""

from __future__ import annotations

import os
import random
from math import isfinite
from numbers import Integral

import numpy as np
import torch
from torch import Tensor


def check_floating(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if value.ndim == 0:
        raise ValueError(f"{name} must have at least one dimension")


def check_same_device(first_name: str, first: Tensor, second_name: str, second: Tensor) -> None:
    if first.device != second.device:
        raise ValueError(
            f"{first_name} and {second_name} must be on the same device; "
            f"got {first.device} and {second.device}"
        )


def compute_dtype(*values: Tensor) -> torch.dtype:
    dtype = values[0].dtype
    for value in values[1:]:
        dtype = torch.promote_types(dtype, value.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def normalize_dim(dim: int, ndim: int) -> int:
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise TypeError("dim must be an integer")
    if not -ndim <= dim < ndim:
        raise IndexError(f"dimension {dim} is out of range for a {ndim}-dimensional tensor")
    return dim % ndim


def validate_eps(eps: float) -> float:
    if isinstance(eps, bool) or not isinstance(eps, (int, float)):
        raise TypeError("eps must be a positive finite number")
    if not isfinite(float(eps)) or eps <= 0:
        raise ValueError("eps must be a positive finite number")
    return float(eps)


def check_nonzero_norm(norm: Tensor, eps: float, name: str) -> None:
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
    check_floating("positive", positive)
    check_floating("negative", negative)
    check_same_device("positive", positive, "negative", negative)
    if positive.ndim != negative.ndim:
        raise ValueError("positive and negative must have the same number of dimensions")

    dim = normalize_dim(dim, positive.ndim)
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

    dtype = compute_dtype(positive, negative)
    return positive.to(dtype=dtype).mean(dim=dim) - negative.to(dtype=dtype).mean(dim=dim)


def normalize_direction(direction: Tensor, dim: int = -1, eps: float = 1e-12) -> Tensor:
    """Normalize a direction along dim; reject zero or non-finite vectors."""
    check_floating("direction", direction)
    dim = normalize_dim(dim, direction.ndim)
    eps = validate_eps(eps)
    value = direction.to(dtype=compute_dtype(direction))
    norm = torch.linalg.vector_norm(value, dim=dim, keepdim=True)
    check_nonzero_norm(norm, eps, "direction")
    return value / norm


def broadcast_pair(hidden: Tensor, direction: Tensor) -> tuple[Tensor, Tensor]:
    check_floating("hidden", hidden)
    check_floating("direction", direction)
    check_same_device("hidden", hidden, "direction", direction)
    try:
        hidden, direction = torch.broadcast_tensors(hidden, direction)
    except RuntimeError as exc:
        raise ValueError(
            f"hidden shape {tuple(hidden.shape)} and direction shape "
            f"{tuple(direction.shape)} are not broadcastable"
        ) from exc
    dtype = compute_dtype(hidden, direction)
    return hidden.to(dtype=dtype), direction.to(dtype=dtype)


def projection_length(
    hidden: Tensor,
    direction: Tensor,
    dim: int = -1,
    eps: float = 1e-12,
) -> Tensor:
    """Return the signed scalar projection of hidden onto a unit direction."""
    hidden, direction = broadcast_pair(hidden, direction)
    dim = normalize_dim(dim, hidden.ndim)
    unit_direction = normalize_direction(direction, dim=dim, eps=eps)
    return (hidden * unit_direction).sum(dim=dim)


def project(
    hidden: Tensor,
    direction: Tensor,
    dim: int = -1,
    eps: float = 1e-12,
) -> Tensor:
    """Return the vector projection of hidden onto direction."""
    hidden, direction = broadcast_pair(hidden, direction)
    dim = normalize_dim(dim, hidden.ndim)
    eps = validate_eps(eps)
    norm_squared = direction.square().sum(dim=dim, keepdim=True)
    check_nonzero_norm(torch.sqrt(norm_squared), eps, "direction")
    coefficient = (hidden * direction).sum(dim=dim, keepdim=True) / norm_squared
    return coefficient * direction


def cosine_similarity(
    first: Tensor,
    second: Tensor,
    dim: int = -1,
    eps: float = 1e-12,
) -> Tensor:
    """Compute cosine similarity with broadcasting and zero-vector checks."""
    first, second = broadcast_pair(first, second)
    dim = normalize_dim(dim, first.ndim)
    eps = validate_eps(eps)
    first_norm = torch.linalg.vector_norm(first, dim=dim)
    second_norm = torch.linalg.vector_norm(second, dim=dim)
    check_nonzero_norm(first_norm, eps, "first")
    check_nonzero_norm(second_norm, eps, "second")
    numerator = (first * second).sum(dim=dim)
    return (numerator / (first_norm * second_norm)).clamp(-1.0, 1.0)


def set_seed(
    seed: int,
    *,
    deterministic: bool = False,
    warn_only: bool = False,
) -> None:
    """Seed Python, NumPy, and PyTorch, optionally enabling deterministic algorithms.

    Seeds must fit NumPy's legacy global RNG range (0 through 2**32 - 1).
    torch.manual_seed seeds PyTorch CPU and CUDA generators.

    By default, only RNG seeds are changed. When deterministic=True, PyTorch
    deterministic algorithms are enabled, cuDNN benchmarking is disabled, and
    cuDNN deterministic mode is enabled. If CUDA is available, an unset
    CUBLAS_WORKSPACE_CONFIG is set to :4096:8 before CUDA initialization.
    Existing non-empty values are preserved. If CUDA has already initialized and
    the variable is unset, a RuntimeError is raised rather than claiming that the
    configuration took effect.

    warn_only applies only when deterministic mode is enabled. TF32 settings
    and PYTHONHASHSEED are intentionally left unchanged.
    """
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer")
    normalized_seed = int(seed)
    if not 0 <= normalized_seed <= 2**32 - 1:
        raise ValueError("seed must be between 0 and 2**32 - 1, inclusive")
    if not isinstance(deterministic, bool):
        raise TypeError("deterministic must be a bool")
    if not isinstance(warn_only, bool):
        raise TypeError("warn_only must be a bool")
    if warn_only and not deterministic:
        raise ValueError("warn_only can only be used when deterministic=True")

    # Configure cuBLAS before torch.manual_seed(), which also seeds CUDA generators.
    if deterministic and torch.cuda.is_available():
        current = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if current == "":
            raise ValueError("CUBLAS_WORKSPACE_CONFIG must not be empty")
        if current is None:
            if torch.cuda.is_initialized():
                raise RuntimeError(
                    "deterministic CUDA execution requires CUBLAS_WORKSPACE_CONFIG to be set "
                    "before CUDA initialization. Restart the Python process, then call set_seed() "
                    "before creating CUDA tensors or loading a model."
                )
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(normalized_seed)
    np.random.seed(normalized_seed)
    torch.manual_seed(normalized_seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


__all__ = [
    "mean_difference",
    "normalize_direction",
    "project",
    "projection_length",
    "cosine_similarity",
    "set_seed",
]
