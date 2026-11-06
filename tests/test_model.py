from __future__ import annotations

import json
import sys
import types

import pytest
import torch
from torch import nn

from vlmbox.adapter import TransformersAdapter
from vlmbox.model import VLM


class FakeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Linear(2, 2)
        self.mlp = nn.Linear(2, 2)


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.layers = nn.ModuleList([FakeLayer()])
        self.language_model.norm = nn.LayerNorm(2)
        self.lm_head = nn.Linear(2, 5)
        self.was_eval = False

    def eval(self):
        self.was_eval = True
        return super().eval()

    @property
    def device(self):
        return next(self.parameters()).device

    def generate(self, **kwargs):
        return kwargs


class FakeProcessor:
    calls = []

    def __init__(self) -> None:
        self.tokenizer = types.SimpleNamespace(padding_side="right")

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.calls.append((path, kwargs))
        return cls()

    def apply_chat_template(self, messages, **kwargs):
        return {"messages": messages, **kwargs}


class FakeAutoConfig:
    calls = []
    config = None

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.calls.append((path, kwargs))
        return cls.config


class FakeAutoModel:
    calls = []
    instance = FakeModel()

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.calls.append((path, kwargs))
        return cls.instance


def install_fake_dependencies(monkeypatch):
    downloaded = []
    download_options = []

    modelscope = types.ModuleType("modelscope")
    modelscope.__path__ = []
    hub = types.ModuleType("modelscope.hub")
    hub.__path__ = []
    snapshot_module = types.ModuleType("modelscope.hub.snapshot_download")

    def snapshot_download(model_id, **kwargs):
        downloaded.append(model_id)
        download_options.append(kwargs)
        return "/model-cache/fake-model"

    snapshot_module.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "modelscope", modelscope)
    monkeypatch.setitem(sys.modules, "modelscope.hub", hub)
    monkeypatch.setitem(sys.modules, "modelscope.hub.snapshot_download", snapshot_module)

    transformers = types.ModuleType("transformers")
    transformers.AutoConfig = FakeAutoConfig
    transformers.AutoProcessor = FakeProcessor

    class AutoModel(FakeAutoModel):
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            cls.calls.append((path, kwargs))
            cls.instance = FakeModel()
            return cls.instance

    transformers.AutoModelForMultimodalLM = AutoModel
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return downloaded, AutoModel, download_options


def test_from_pretrained_uses_modelscope_and_multimodal_transformers(monkeypatch) -> None:
    downloaded, auto_model, download_options = install_fake_dependencies(monkeypatch)
    FakeProcessor.calls.clear()
    auto_model.calls.clear()

    vlm = VLM.from_pretrained("Org/Model")

    assert downloaded == ["Org/Model"]
    assert download_options == [{"local_files_only": False}]
    assert vlm.model_id == "Org/Model"
    assert vlm.model_path == "/model-cache/fake-model"
    assert vlm.model.training is False
    assert vlm.processor.tokenizer.padding_side == "left"
    assert len(vlm.adapter.layers) == 1
    assert FakeProcessor.calls == [("/model-cache/fake-model", {"trust_remote_code": True})]
    assert auto_model.calls == [
        (
            "/model-cache/fake-model",
            {"dtype": "auto", "device_map": "auto", "trust_remote_code": True},
        )
    ]


def test_remote_code_is_enabled_for_processor_and_model(monkeypatch) -> None:
    _, auto_model, _ = install_fake_dependencies(monkeypatch)
    FakeProcessor.calls.clear()
    auto_model.calls.clear()

    VLM.from_pretrained("Org/CustomModel")

    assert FakeProcessor.calls[0][1]["trust_remote_code"] is True
    assert auto_model.calls[0][1]["trust_remote_code"] is True


def test_from_pretrained_corrects_omitted_internvl_tying_flag(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_type": "internvl",
                "text_config": {"model_type": "qwen3"},
            }
        ),
        encoding="utf-8",
    )
    _, auto_model, _ = install_fake_dependencies(monkeypatch)
    monkeypatch.setattr(
        sys.modules["modelscope.hub.snapshot_download"],
        "snapshot_download",
        lambda *_args, **_kwargs: str(tmp_path),
    )
    FakeAutoConfig.calls.clear()
    FakeAutoConfig.config = types.SimpleNamespace(
        tie_word_embeddings=True,
        text_config=types.SimpleNamespace(tie_word_embeddings=False),
    )
    auto_model.calls.clear()

    VLM.from_pretrained("OpenGVLab/InternVL3_5-2B-HF")

    assert FakeAutoConfig.calls == [
        (str(tmp_path), {"trust_remote_code": True}),
    ]
    loaded_config = auto_model.calls[0][1]["config"]
    assert loaded_config is FakeAutoConfig.config
    assert loaded_config.tie_word_embeddings is False
    assert loaded_config.text_config.tie_word_embeddings is False


@pytest.mark.parametrize("top_level_value", [True, False])
def test_from_pretrained_preserves_explicit_internvl_tying_flag(
    tmp_path, monkeypatch, top_level_value
) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "internvl",
                "tie_word_embeddings": top_level_value,
                "text_config": {"model_type": "qwen3"},
            }
        ),
        encoding="utf-8",
    )
    _, auto_model, _ = install_fake_dependencies(monkeypatch)
    monkeypatch.setattr(
        sys.modules["modelscope.hub.snapshot_download"],
        "snapshot_download",
        lambda *_args, **_kwargs: str(tmp_path),
    )
    FakeAutoConfig.calls.clear()
    auto_model.calls.clear()

    VLM.from_pretrained("OpenGVLab/InternVL3_5-2B-HF")

    assert FakeAutoConfig.calls == []
    assert "config" not in auto_model.calls[0][1]


def test_from_pretrained_can_require_a_local_cache(monkeypatch) -> None:
    downloaded, _auto_model, download_options = install_fake_dependencies(monkeypatch)

    vlm = VLM.from_pretrained("Org/Model", local_files_only=True)

    assert downloaded == ["Org/Model"]
    assert download_options == [{"local_files_only": True}]
    assert vlm.model_path == "/model-cache/fake-model"


@pytest.mark.parametrize("value", [1, None, "true"])
def test_from_pretrained_validates_local_files_only(value) -> None:
    with pytest.raises(TypeError, match="local_files_only must be a bool"):
        VLM.from_pretrained("Org/Model", local_files_only=value)


@pytest.mark.parametrize("model_id", ["", "   ", "/tmp/local-model"])
def test_invalid_model_id_is_rejected(monkeypatch, model_id) -> None:
    with pytest.raises(ValueError):
        VLM.from_pretrained(model_id)


def test_prepare_and_generate_delegate_to_processor_and_model() -> None:
    model = FakeModel()
    processor = FakeProcessor()
    vlm = VLM(
        model_id="Org/Model",
        model_path="/model-cache/fake-model",
        model=model,
        processor=processor,
        adapter=TransformersAdapter(model),
    )
    messages = [{"role": "user", "content": "hello"}]

    prepared = vlm.prepare(messages)
    assert prepared["messages"] == messages
    assert prepared["tokenize"] is True
    assert prepared["return_tensors"] == "pt"
    assert prepared["processor_kwargs"] == {"padding": True}
    assert processor.tokenizer.padding_side == "left"

    batch_messages = [
        [{"role": "user", "content": "short"}],
        [{"role": "user", "content": "a longer prompt"}],
    ]
    batch_prepared = vlm.prepare(batch_messages)
    assert batch_prepared["messages"] == batch_messages
    assert batch_prepared["processor_kwargs"] == {"padding": True}

    templated = vlm.prepare(messages, chat_template_kwargs={"enable_thinking": False})
    assert templated["enable_thinking"] is False

    result = vlm.generate({"input_ids": torch.tensor([[1, 2]])}, max_new_tokens=3)
    assert result["input_ids"].tolist() == [[1, 2]]
    assert result["max_new_tokens"] == 3


def test_prepare_differentiable_preserves_gradient_and_uses_processor_kwargs() -> None:
    model = FakeModel()
    processor = FakeProcessor()
    vlm = VLM(
        model_id="Org/Model",
        model_path="/model-cache/fake-model",
        model=model,
        processor=processor,
        adapter=TransformersAdapter(model),
    )
    image = torch.rand(3, 8, 8, requires_grad=True)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    prepared = vlm.prepare_differentiable(messages, chat_template_kwargs={"enable_thinking": False})
    assert prepared["processor_kwargs"] == {"padding": True, "do_rescale": False}
    assert prepared["enable_thinking"] is False
    prepared_image = prepared["messages"][0][0]["content"][0]["image"]
    prepared_image.square().mean().backward()

    assert image.grad is not None
    assert torch.count_nonzero(image.grad) > 0
    assert messages[0]["content"][0]["image"] is image


def test_wrap_native_objects_infers_decoder_and_preserves_model_state() -> None:
    model = FakeModel().train()
    model.language_model.norm.eval()
    model.lm_head.weight.requires_grad_(False)
    modes = [module.training for module in model.modules()]
    flags = [parameter.requires_grad for parameter in model.parameters()]
    processor = FakeProcessor()

    vlm = VLM(model, processor)

    assert vlm.model is model
    assert vlm.processor is processor
    assert vlm.adapter.layer(0) is model.language_model.layers[0]
    assert [module.training for module in model.modules()] == modes
    assert [parameter.requires_grad for parameter in model.parameters()] == flags
    assert vlm.input_device == next(model.parameters()).device
    assert processor.tokenizer.padding_side == "left"


def test_wrap_native_objects_accepts_a_custom_adapter() -> None:
    model = nn.Linear(2, 2)
    adapter = types.SimpleNamespace(layers=[model])
    vlm = VLM(model, FakeProcessor(), adapter=adapter)
    assert vlm.adapter is adapter
    assert vlm.model is model


def test_differentiable_preparation_preserves_open_assistant_prefix_options() -> None:
    vlm = VLM(FakeModel(), FakeProcessor())
    image = torch.rand(3, 8, 8, requires_grad=True)
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Let me"}]},
    ]
    prepared = vlm.prepare_differentiable(
        messages,
        add_generation_prompt=False,
        chat_template_kwargs={"continue_final_message": True, "enable_thinking": False},
    )
    assert prepared["add_generation_prompt"] is False
    assert prepared["continue_final_message"] is True
    assert prepared["enable_thinking"] is False
    assert prepared["messages"][0][-1] == messages[-1]
    prepared["messages"][0][0]["content"][0]["image"].sum().backward()
    torch.testing.assert_close(image.grad, torch.ones_like(image))
