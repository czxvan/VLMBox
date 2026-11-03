"""Random seed and deterministic execution utilities."""

from __future__ import annotations

import os
import random
from numbers import Integral

import numpy as np
import torch

_DEFAULT_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_MAX_SEED = 2**32 - 1


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer")
    normalized = int(seed)
    if not 0 <= normalized <= _MAX_SEED:
        raise ValueError(f"seed must be between 0 and {_MAX_SEED}, inclusive")
    return normalized


def _configure_cublas_workspace() -> None:
    """Configure cuBLAS before CUDA initialization when deterministic mode needs it."""
    if not torch.cuda.is_available():
        return

    current = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if current:
        return
    if "CUBLAS_WORKSPACE_CONFIG" in os.environ:
        raise ValueError("CUBLAS_WORKSPACE_CONFIG must not be empty")

    if torch.cuda.is_initialized():
        raise RuntimeError(
            "deterministic CUDA execution requires CUBLAS_WORKSPACE_CONFIG to be set "
            "before CUDA initialization. Restart the Python process, then call set_seed() "
            "before creating CUDA tensors or loading a model."
        )

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = _DEFAULT_CUBLAS_WORKSPACE_CONFIG


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
    normalized_seed = _validate_seed(seed)
    if not isinstance(deterministic, bool):
        raise TypeError("deterministic must be a bool")
    if not isinstance(warn_only, bool):
        raise TypeError("warn_only must be a bool")
    if warn_only and not deterministic:
        raise ValueError("warn_only can only be used when deterministic=True")

    # Configure cuBLAS before torch.manual_seed(), which also seeds CUDA generators.
    if deterministic:
        _configure_cublas_workspace()

    random.seed(normalized_seed)
    np.random.seed(normalized_seed)
    torch.manual_seed(normalized_seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


__all__ = ["set_seed"]
