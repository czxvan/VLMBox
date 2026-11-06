from __future__ import annotations

import pytest
from torch import nn

from vlmbox.adapter import AdapterError, TransformersAdapter


class FakeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Linear(4, 4)
        self.mlp = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 4))


class FakeLanguageModel(nn.Module):
    def __init__(self, count: int = 3) -> None:
        super().__init__()
        self.layers = nn.ModuleList(FakeLayer() for _ in range(count))
        self.norm = nn.LayerNorm(4)


class FakeMultimodalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = FakeLanguageModel()
        self.vision_model = nn.Module()
        self.vision_model.blocks = nn.ModuleList([FakeLayer()])
        self.lm_head = nn.Linear(4, 16, bias=False)


class NestedTextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.text_model = FakeLanguageModel(2)
        self.lm_head = nn.Linear(4, 16)


def test_resolves_language_decoder_without_selecting_vision_layers() -> None:
    model = FakeMultimodalModel()
    adapter = TransformersAdapter(model)

    assert len(adapter.layers) == 3
    assert adapter.layer(0) is model.language_model.layers[0]
    assert adapter.attention(1) is model.language_model.layers[1].self_attn
    assert adapter.mlp(2) is model.language_model.layers[2].mlp


def test_resolves_nested_text_model() -> None:
    model = NestedTextModel()
    adapter = TransformersAdapter(model)

    assert len(adapter.layers) == 2
    assert adapter.layer(1) is model.model.text_model.layers[1]


class NestedDecoderWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = FakeLanguageModel(2)


class NestedDecoderModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = NestedDecoderWrapper()
        self.lm_head = nn.Linear(4, 16)


def test_resolves_layers_inside_nested_decoder() -> None:
    model = NestedDecoderModel()
    adapter = TransformersAdapter(model)

    assert len(adapter.layers) == 2
    assert adapter.layer(1) is model.language_model.model.layers[1]


def test_layer_index_is_zero_based_and_checked() -> None:
    adapter = TransformersAdapter(FakeMultimodalModel())

    with pytest.raises(IndexError, match="out of range"):
        adapter.layer(3)
    with pytest.raises(IndexError, match="out of range"):
        adapter.layer(-1)
    with pytest.raises(TypeError, match="integer"):
        adapter.layer(True)  # type: ignore[arg-type]


def test_missing_component_has_actionable_error() -> None:
    model = FakeMultimodalModel()
    del model.language_model.layers[0].mlp
    adapter = TransformersAdapter(model)

    with pytest.raises(AdapterError, match="Could not resolve MLP"):
        adapter.mlp(0)


class DecoderAccessorModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = FakeLanguageModel(4)

    def get_decoder(self) -> nn.Module:
        return self.decoder


def test_prefers_transformers_get_decoder_accessor() -> None:
    model = DecoderAccessorModel()
    adapter = TransformersAdapter(model)

    assert len(adapter.layers) == 4
    assert adapter.layer(0) is model.decoder.layers[0]


def test_unsupported_model_fails_instead_of_guessing() -> None:
    with pytest.raises(AdapterError, match="Unable to locate decoder layers"):
        TransformersAdapter(nn.Sequential(nn.Linear(4, 4)))
