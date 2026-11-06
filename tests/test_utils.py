from __future__ import annotations

import math

import pytest
import torch

from vlmbox.utils import (
    cosine_similarity,
    mean_difference,
    normalize_direction,
    project,
    projection_length,
)


def test_mean_difference_supports_unequal_sample_counts() -> None:
    positive = torch.tensor([[2.0, 4.0], [4.0, 6.0], [6.0, 8.0]])
    negative = torch.tensor([[1.0, 1.0], [3.0, 3.0]])

    result = mean_difference(positive, negative)

    assert torch.allclose(result, torch.tensor([2.0, 4.0]))


def test_mean_difference_equals_mean_of_pairwise_differences_when_aligned() -> None:
    positive = torch.tensor([[3.0, 2.0], [5.0, 8.0], [7.0, 4.0]])
    negative = torch.tensor([[1.0, 1.0], [2.0, 4.0], [3.0, 2.0]])

    result = mean_difference(positive, negative)

    assert torch.allclose(result, (positive - negative).mean(dim=0))


def test_mean_difference_reduces_arbitrary_sample_dimension() -> None:
    positive = torch.tensor(
        [
            [[2.0, 4.0], [4.0, 6.0]],
            [[4.0, 6.0], [6.0, 8.0]],
        ]
    )
    negative = torch.zeros_like(positive)

    result = mean_difference(positive, negative, dim=1)

    assert result.shape == (2, 2)
    assert torch.equal(result, positive.mean(dim=1))


def test_mean_difference_rejects_mismatched_non_sample_dimensions() -> None:
    with pytest.raises(ValueError, match="non-sample"):
        mean_difference(torch.ones(2, 3), torch.ones(4, 5))


def test_direction_extraction_rejects_empty_samples() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        mean_difference(torch.empty(0, 3), torch.ones(2, 3))


def test_normalize_direction_returns_unit_vectors() -> None:
    directions = torch.tensor([[3.0, 4.0], [5.0, 12.0]])

    result = normalize_direction(directions)

    assert torch.allclose(torch.linalg.vector_norm(result, dim=-1), torch.ones(2))
    assert torch.allclose(result[0], torch.tensor([0.6, 0.8]))


def test_normalize_direction_uses_float32_for_half_precision() -> None:
    direction = torch.tensor([3.0, 4.0], dtype=torch.float16)

    result = normalize_direction(direction)

    assert result.dtype == torch.float32
    assert torch.allclose(result, torch.tensor([0.6, 0.8]))


def test_normalize_direction_rejects_zero_and_non_finite_values() -> None:
    with pytest.raises(ValueError, match="norm greater"):
        normalize_direction(torch.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        normalize_direction(torch.tensor([float("nan"), 1.0]))


def test_vector_projection() -> None:
    hidden = torch.tensor([[3.0, 4.0], [1.0, 2.0]])
    direction = torch.tensor([1.0, 0.0])

    component = project(hidden, direction)

    assert torch.equal(component, torch.tensor([[3.0, 0.0], [1.0, 0.0]]))


def test_projection_length_is_signed_and_uses_unit_direction() -> None:
    hidden = torch.tensor([[3.0, 4.0], [-3.0, 4.0]])
    direction = torch.tensor([2.0, 0.0])

    result = projection_length(hidden, direction)

    assert torch.equal(result, torch.tensor([3.0, -3.0]))


def test_projection_length_is_invariant_to_positive_direction_scaling() -> None:
    hidden = torch.tensor([[3.0, 4.0], [-3.0, 4.0]])
    direction = torch.tensor([2.0, -1.0])

    original = projection_length(hidden, direction)
    scaled = projection_length(hidden, direction * 7.5)

    assert torch.allclose(original, scaled)


def test_projection_residual_is_orthogonal_to_direction() -> None:
    hidden = torch.tensor([[3.0, 4.0, -2.0], [1.0, -5.0, 7.0]])
    direction = torch.tensor([2.0, -3.0, 1.0])

    residual = hidden - project(hidden, direction)
    inner_product = (residual * direction).sum(dim=-1)

    assert torch.allclose(inner_product, torch.zeros_like(inner_product), atol=1e-6)


def test_projection_supports_broadcasting() -> None:
    hidden = torch.tensor([[[3.0, 4.0], [1.0, 2.0]], [[5.0, 6.0], [7.0, 8.0]]])
    direction = torch.tensor([1.0, 0.0])

    component = project(hidden, direction)

    assert component.shape == hidden.shape
    assert torch.equal(component[..., 0], hidden[..., 0])
    assert torch.count_nonzero(component[..., 1]) == 0


def test_projection_is_differentiable() -> None:
    hidden = torch.tensor([3.0, 4.0], requires_grad=True)
    component = project(hidden, torch.tensor([1.0, 0.0]))
    component.sum().backward()

    assert hidden.grad is not None
    assert torch.equal(hidden.grad, torch.tensor([1.0, 0.0]))


def test_projection_rejects_zero_direction() -> None:
    with pytest.raises(ValueError, match="norm greater"):
        project(torch.ones(3), torch.zeros(3))


def test_cosine_similarity() -> None:
    first = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    second = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    similarity = cosine_similarity(first, second)

    assert torch.allclose(similarity, torch.tensor([0.0, 1 / math.sqrt(2)]))


def test_cosine_supports_broadcasting() -> None:
    first = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    second = torch.tensor([1.0, 0.0])

    assert torch.allclose(cosine_similarity(first, second), torch.tensor([[1.0, 0.0]]))


def test_cosine_rejects_zero_vectors() -> None:
    with pytest.raises(ValueError, match="norm greater"):
        cosine_similarity(torch.zeros(3), torch.ones(3))


def test_direction_extraction_preserves_autograd() -> None:
    positive = torch.tensor([[2.0, 4.0], [4.0, 6.0]], requires_grad=True)
    negative = torch.tensor([[1.0, 2.0], [2.0, 3.0]], requires_grad=True)

    result = mean_difference(positive, negative)
    result.sum().backward()

    assert positive.grad is not None
    assert negative.grad is not None


def test_geometry_promotes_half_precision_to_float32() -> None:
    hidden = torch.tensor([3.0, 4.0], dtype=torch.float16)
    direction = torch.tensor([1.0, 0.0], dtype=torch.float16)

    assert project(hidden, direction).dtype == torch.float32
    assert cosine_similarity(hidden, direction).dtype == torch.float32


def test_invalid_dimensions_and_eps_are_rejected() -> None:
    with pytest.raises(IndexError, match="out of range"):
        mean_difference(torch.ones(2, 3), torch.ones(2, 3), dim=2)
    with pytest.raises(ValueError, match="positive finite"):
        normalize_direction(torch.ones(3), eps=0)
    with pytest.raises(TypeError, match="dim"):
        cosine_similarity(torch.ones(3), torch.ones(3), dim=True)
