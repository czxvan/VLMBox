from __future__ import annotations

import os
import random

import numpy as np
import pytest
import torch

import vlmbox
from vlmbox.repro import set_seed


def test_set_seed_is_exported_from_package_root() -> None:
    assert vlmbox.set_seed is set_seed


def test_set_seed_reproduces_python_numpy_and_torch_rngs() -> None:
    set_seed(42)
    python_values = [random.random() for _ in range(3)]
    numpy_values = np.random.random(3)
    torch_values = torch.rand(3)

    set_seed(42)

    assert [random.random() for _ in range(3)] == python_values
    np.testing.assert_array_equal(np.random.random(3), numpy_values)
    torch.testing.assert_close(torch.rand(3), torch_values, rtol=0, atol=0)


@pytest.mark.parametrize("seed", [True, False, 1.5, "42", None])
def test_set_seed_rejects_non_integer_seed(seed) -> None:
    with pytest.raises(TypeError, match="seed must be an integer"):
        set_seed(seed)


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_set_seed_rejects_seed_outside_numpy_range(seed: int) -> None:
    with pytest.raises(ValueError, match="between 0 and"):
        set_seed(seed)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"deterministic": 1}, TypeError, "deterministic must be a bool"),
        ({"warn_only": 1}, TypeError, "warn_only must be a bool"),
        ({"warn_only": True}, ValueError, "only be used when deterministic=True"),
    ],
)
def test_set_seed_validates_determinism_options(kwargs, error, message) -> None:
    with pytest.raises(error, match=message):
        set_seed(42, **kwargs)


def test_default_mode_only_sets_rngs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(
        torch,
        "use_deterministic_algorithms",
        lambda *_args, **_kwargs: pytest.fail("deterministic algorithms must remain untouched"),
    )
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark

    set_seed(42)

    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
    assert torch.backends.cudnn.deterministic is cudnn_deterministic
    assert torch.backends.cudnn.benchmark is cudnn_benchmark


def test_deterministic_mode_configures_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[bool, bool]] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch,
        "use_deterministic_algorithms",
        lambda enabled, *, warn_only: calls.append((enabled, warn_only)),
    )
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)

    set_seed(42, deterministic=True, warn_only=True)

    assert calls == [(True, True)]
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_deterministic_mode_sets_cublas_config_before_cuda_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch, "manual_seed", lambda _seed: None)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda *_args, **_kwargs: None)

    set_seed(42, deterministic=True)

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_deterministic_mode_preserves_existing_cublas_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch, "manual_seed", lambda _seed: None)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda *_args, **_kwargs: None)

    set_seed(42, deterministic=True)

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


def test_deterministic_mode_fails_if_cuda_initialized_without_cublas_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch,
        "manual_seed",
        lambda _seed: pytest.fail("RNG state must not change after preflight failure"),
    )

    with pytest.raises(RuntimeError, match="before CUDA initialization"):
        set_seed(42, deterministic=True)

    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ


def test_deterministic_mode_does_not_set_cublas_config_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda *_args, **_kwargs: None)

    set_seed(42, deterministic=True)

    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
