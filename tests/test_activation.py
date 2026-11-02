from __future__ import annotations

import pytest
import torch
from torch import nn

from vlmbox.activation import ActivationError
from vlmbox.adapter import TransformersAdapter
from vlmbox.model import VLM


class FakeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Identity()
        self.mlp = nn.Identity()

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, dict[str, str]]:
        hidden = self.self_attn(hidden)
        hidden = self.mlp(hidden)
        return hidden, {"cache": "preserved"}


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.layers = nn.ModuleList([FakeBlock(), FakeBlock()])

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.language_model.layers:
            hidden = layer(hidden)[0]
        return hidden


def make_vlm() -> VLM:
    model = FakeModel()
    return VLM(
        model_id="Org/Model",
        model_path="/model-cache/fake-model",
        model=model,
        processor=object(),
        adapter=TransformersAdapter(model),
    )


def sample() -> torch.Tensor:
    return torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])


def test_capture_selected_layers_and_last_token_preserves_gradient() -> None:
    vlm = make_vlm()
    hidden = sample().requires_grad_()

    with vlm.capture("block", layers=[0, 1], positions="last") as cache:
        vlm.model(hidden)

    assert cache[0].shape == (1, 1, 2)
    assert torch.equal(cache[0], hidden[:, -1:, :])
    assert torch.equal(cache[1], hidden[:, -1:, :])
    cache[1].sum().backward()
    assert hidden.grad is not None
    assert torch.equal(hidden.grad[:, -1, :], torch.ones_like(hidden.grad[:, -1, :]))


def test_capture_supports_attention_and_mlp_components() -> None:
    vlm = make_vlm()
    hidden = sample()

    with vlm.capture("attention", layers=0, positions="all") as attention:
        vlm.model(hidden)
    with vlm.capture("mlp", layers=1, positions=[0]) as mlp:
        vlm.model(hidden)

    assert torch.equal(attention[0], hidden)
    assert torch.equal(mlp[1], hidden[:, :1, :])


@pytest.mark.parametrize(
    ("operation", "value", "scale", "expected"),
    [
        ("add", [1.0, 0.0], 2.0, [5.0, 4.0]),
        ("replace", [9.0, 8.0], 1.0, [9.0, 8.0]),
        ("project_out", [1.0, 0.0], 1.0, [0.0, 4.0]),
        ("zero", None, 1.0, [0.0, 0.0]),
    ],
)
def test_intervention_operations(operation, value, scale, expected) -> None:
    vlm = make_vlm()
    intervention_value = None if value is None else torch.tensor(value)

    with vlm.intervene(
        "block",
        layer=0,
        operation=operation,
        value=intervention_value,
        scale=scale,
        positions="last",
    ):
        output = vlm.model(sample())

    assert torch.equal(output[:, 0, :], sample()[:, 0, :])
    assert torch.equal(output[:, 1, :], torch.tensor([expected]))


def test_intervention_can_target_attention_or_mlp() -> None:
    vlm = make_vlm()
    value = torch.tensor([2.0, 0.0])

    with vlm.intervene("attention", 0, "add", value=value, positions="all"):
        attention_output = vlm.model(sample())
    with vlm.intervene("mlp", 1, "add", value=value, positions="last"):
        mlp_output = vlm.model(sample())

    assert torch.equal(attention_output, sample() + value)
    assert torch.equal(mlp_output[:, 0, :], sample()[:, 0, :])
    assert torch.equal(mlp_output[:, 1, :], sample()[:, 1, :] + value)


def test_first_forward_scope_is_one_shot_per_target_module() -> None:
    vlm = make_vlm()
    value = torch.tensor([1.0, 0.0])

    with vlm.intervene(
        "block", 0, "add", value=value, positions="last", scope="first_forward"
    ):
        first = vlm.model(sample())
        second = vlm.model(sample())

    assert torch.equal(first[:, -1, :], sample()[:, -1, :] + value)
    assert torch.equal(second, sample())


def test_every_forward_scope_is_persistent() -> None:
    vlm = make_vlm()
    value = torch.tensor([1.0, 0.0])

    with vlm.intervene(
        "block", 0, "add", value=value, positions="last", scope="every_forward"
    ):
        first = vlm.model(sample())
        second = vlm.model(sample())

    expected = sample()[:, -1, :] + value
    assert torch.equal(first[:, -1, :], expected)
    assert torch.equal(second[:, -1, :], expected)


def test_intervention_preserves_auxiliary_tuple_outputs() -> None:
    vlm = make_vlm()
    layer = vlm.model.language_model.layers[0]
    auxiliary = {"cache": "preserved"}

    with vlm.intervene("block", 0, "zero", positions="all"):
        output = layer(sample())

    assert output[1] == auxiliary
    assert torch.count_nonzero(output[0]) == 0


def test_intervention_is_differentiable() -> None:
    vlm = make_vlm()
    hidden = sample().requires_grad_()
    direction = torch.tensor([1.0, 0.0])

    with vlm.intervene("block", 0, "project_out", value=direction, positions="all"):
        output = vlm.model(hidden)

    output.sum().backward()
    assert hidden.grad is not None
    assert torch.allclose(hidden.grad[..., 0], torch.zeros_like(hidden.grad[..., 0]))
    assert torch.allclose(hidden.grad[..., 1], torch.ones_like(hidden.grad[..., 1]))


def test_hooks_are_removed_when_context_exits_with_exception() -> None:
    vlm = make_vlm()
    value = torch.tensor([1.0, 0.0])

    with pytest.raises(RuntimeError, match="test failure"):
        with vlm.intervene("block", 0, "add", value=value):
            raise RuntimeError("test failure")

    assert torch.equal(vlm.model(sample()), sample())


def test_capture_hooks_are_removed_after_context() -> None:
    vlm = make_vlm()

    with vlm.capture("block", layers=0) as cache:
        vlm.model(sample())
    assert 0 in cache.values

    with vlm.capture("block", layers=0) as next_cache:
        vlm.model(sample())
    assert 0 in next_cache.values


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"component": "vision"}, ValueError),
        ({"layers": []}, ValueError),
        ({"layers": True}, TypeError),
        ({"layers": 9}, IndexError),
        ({"positions": "visual"}, ValueError),
    ],
)
def test_invalid_capture_configuration_is_rejected(kwargs, error) -> None:
    vlm = make_vlm()
    component = kwargs.pop("component", "block")
    layers = kwargs.pop("layers", 0)
    positions = kwargs.pop("positions", "last")

    with pytest.raises(error):
        vlm.capture(component, layers, positions)


def test_nested_hook_order_controls_capture_before_or_after_intervention() -> None:
    vlm = make_vlm()
    direction = torch.tensor([2.0, 0.0])

    with vlm.capture("block", layers=0, positions="all") as before:
        with vlm.intervene("block", 0, "add", value=direction, positions="all"):
            output = vlm.model(sample())

    with vlm.intervene("block", 0, "add", value=direction, positions="all"):
        with vlm.capture("block", layers=0, positions="all") as after:
            vlm.model(sample())

    assert torch.equal(before[0], sample())
    assert torch.equal(output, sample() + direction)
    assert torch.equal(after[0], sample() + direction)


def test_capture_hooks_are_removed_when_context_exits_with_exception() -> None:
    vlm = make_vlm()

    with pytest.raises(RuntimeError, match="capture failure"):
        with vlm.capture("block", layers=0):
            raise RuntimeError("capture failure")

    assert torch.equal(vlm.model(sample()), sample())


def test_project_out_rejects_zero_direction() -> None:
    vlm = make_vlm()

    with pytest.raises(ActivationError, match="non-zero"):
        with vlm.intervene("block", 0, "project_out", value=torch.zeros(2)):
            vlm.model(sample())


def test_intervention_rejects_non_finite_value() -> None:
    vlm = make_vlm()

    with pytest.raises(ActivationError, match="finite"):
        with vlm.intervene("block", 0, "add", value=torch.tensor([float("nan"), 0.0])):
            vlm.model(sample())


def test_intervention_rejects_non_broadcastable_value() -> None:
    vlm = make_vlm()

    with pytest.raises(ActivationError, match="cannot broadcast"):
        with vlm.intervene("block", 0, "add", value=torch.ones(3)):
            vlm.model(sample())
