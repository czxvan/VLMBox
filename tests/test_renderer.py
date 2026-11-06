from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image, ImageFont
from torch import nn

from vlmbox import Renderer
from vlmbox.trainer import LossFunction, Sample, Trainer


@pytest.fixture
def font_path() -> Path:
    return Path(ImageFont.truetype("DejaVuSans.ttf", 20).path)


def _write_config(
    path: Path,
    font_path: Path,
    *,
    intervention: str = "overlay_patch",
    extra: str = "",
    size: str = "[4, 5]",
) -> Path:
    path.write_text(
        f"""schema_version: 1
image_render:
  instruction:
    position: top
    font:
      path: {font_path.as_posix()}
      size: 20
    foreground: "#000000"
    background: "#FFFFFF"
    alignment: center
    margin: 2
    line_spacing: 4
  host:
    preserve_original_size: true
    color_space: RGB
  intervention:
    type: {intervention}
    size: {size}
    anchor: bottom_right
{extra}""",
        encoding="utf-8",
    )
    return path


def test_yaml_renderer_returns_pil_and_optional_metadata(tmp_path: Path, font_path: Path) -> None:
    renderer = Renderer.from_yaml(_write_config(tmp_path / "render.yaml", font_path))
    host = Image.new("RGB", (32, 24), (20, 40, 60))

    image = renderer.render(host, "Execute this instruction.")
    assert isinstance(image, Image.Image)
    assert image.width == host.width
    assert image.height > host.height

    image, metadata = renderer.render(host, "Execute this instruction.", return_metadata=True)
    host_bbox = tuple(metadata["host_bbox"])
    assert image.crop(host_bbox).size == host.size
    assert metadata["intervention_type"] == "overlay_patch"
    host_offset = host_bbox[1]
    assert metadata["intervention_bbox"] == [27, host_offset + 20, 32, host_offset + 24]
    assert metadata["config_sha256"]
    assert metadata["instruction_sha256"]
    assert len(metadata["intervention_sha256"]) == 64


def test_renderer_tensor_path_preserves_gradient_and_native_sizes(
    tmp_path: Path, font_path: Path
) -> None:
    renderer = Renderer.from_yaml(_write_config(tmp_path / "render.yaml", font_path, size="[2, 3]"))
    for height, width in [(8, 48), (12, 43)]:
        host = Image.new("RGB", (width, height), (50, 50, 50))
        rendered = renderer(host, "Do the task.")
        assert rendered.shape[0] == 3
        assert rendered.shape[1] > height
        assert rendered.shape[2] == width
        rendered.square().sum().backward()
    assert len(renderer._band_cache) == 2
    parameter = next(renderer.parameters())
    assert parameter.grad is not None
    assert torch.count_nonzero(parameter.grad) > 0


def test_yaml_local_additive_config_uses_same_geometry(tmp_path: Path, font_path: Path) -> None:
    config = _write_config(
        tmp_path / "local.yaml",
        font_path,
        intervention="local_additive",
        size="[2, 3]",
        extra="    epsilon: 0.1\n    init: zeros\n",
    )
    renderer = Renderer.from_yaml(config)
    with torch.no_grad():
        renderer.intervention.parameter.fill_(0.05)
    host = Image.new("RGB", (40, 8), (int(0.2 * 255),) * 3)
    rendered = renderer(host, "Do the task.")
    torch.testing.assert_close(
        rendered[:, -2:, -3:],
        torch.full((3, 2, 3), 0.25),
        atol=1 / 255,
        rtol=0,
    )


def test_yaml_renderer_can_disable_intervention(tmp_path: Path, font_path: Path) -> None:
    path = _write_config(tmp_path / "none.yaml", font_path, intervention="none")
    text = path.read_text(encoding="utf-8")
    text = text.replace("    size: [4, 5]\n    anchor: bottom_right\n", "")
    path.write_text(text, encoding="utf-8")
    renderer = Renderer.from_yaml(path)
    assert renderer.intervention is None
    assert tuple(renderer.parameters()) == ()
    with pytest.raises(ValueError, match="no intervention"):
        renderer.save_intervention(tmp_path / "none.pt")
    with pytest.raises(ValueError, match="no intervention"):
        renderer.load_intervention(tmp_path / "none.pt")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda text: text.replace(
                "preserve_original_size: true", "preserve_original_size: false"
            ),
            "preserve_original_size",
        ),
        (
            lambda text: text.replace("type: overlay_patch", "type: unsupported"),
            "intervention.type",
        ),
        (
            lambda text: text.replace("    line_spacing: 4", "    line_spacing: 4\n    typo: true"),
            "unknown keys",
        ),
    ],
)
def test_yaml_renderer_rejects_invalid_config(
    tmp_path: Path, font_path: Path, mutate, message: str
) -> None:
    path = _write_config(tmp_path / "invalid.yaml", font_path)
    path.write_text(mutate(path.read_text(encoding="utf-8")), encoding="utf-8")
    with pytest.raises((ValueError, TypeError), match=message):
        Renderer.from_yaml(path)


def test_yaml_renderer_resolves_relative_font_paths(tmp_path: Path, font_path: Path) -> None:
    fonts = tmp_path / "fonts"
    fonts.mkdir()
    relative_font = fonts / "test.ttf"
    relative_font.write_bytes(font_path.read_bytes())
    path = tmp_path / "config.yaml"
    _write_config(path, Path("fonts/test.ttf"))
    renderer = Renderer.from_yaml(path)
    assert renderer.style.font_path == relative_font.resolve()


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 1, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features).unsqueeze(1)


class _FakeVLM:
    def __init__(self) -> None:
        self.model = _FakeModel()

    def prepare_differentiable(self, conversations, *, add_generation_prompt=True):
        images = []
        for conversation in conversations:
            for message in conversation:
                for part in message.get("content", []):
                    if part.get("type") == "image":
                        images.append(part["image"])
        features = torch.stack([image.mean(dim=(-2, -1)) for image in images])
        return {"images": features}

    def forward(self, inputs, **_kwargs):
        return SimpleNamespace(logits=self.model(inputs["images"]))


def _loss(outputs, _inputs, _samples):
    return outputs.logits.square().mean()


class _CallableLoss:
    def __call__(self, outputs, inputs, samples):
        assert outputs.logits.shape[0] == inputs["images"].shape[0] == len(samples)
        return outputs.logits.square().mean()


@pytest.mark.parametrize("loss_function", [_loss, _CallableLoss()])
def test_image_trainer_optimizes_renderer_and_keeps_vlm_frozen(
    tmp_path: Path, font_path: Path, loss_function: LossFunction
) -> None:
    renderer = Renderer.from_yaml(_write_config(tmp_path / "train.yaml", font_path, size="[2, 2]"))
    vlm = _FakeVLM()
    vlm.model.train()
    vlm.model.projection.eval()
    parameter = next(renderer.parameters())
    before = parameter.detach().clone()
    trainer = Trainer(vlm=vlm, renderer=renderer, loss_function=loss_function, learning_rate=0.05)
    samples = [
        Sample(
            Image.new("RGB", (41, 8), (80, 80, 80)),
            "Answer the user prompt.",
            image_instruction="Follow the text shown in the image.",
        ),
        Sample(
            Image.new("RGB", (37, 10), (150, 150, 150)),
            "Answer the user prompt.",
            image_instruction="Perform the instruction visible above the image.",
        ),
    ]
    history = trainer.fit(samples, max_steps=2, batch_size=1, shuffle=False)

    assert len(history) == 2
    assert all(step.gradient_norm > 0 for step in history)
    assert not torch.equal(before, parameter.detach())
    assert parameter.grad is None
    assert vlm.model.training is True
    assert vlm.model.projection.training is False
    assert all(parameter.requires_grad for parameter in vlm.model.parameters())
    assert all(parameter.grad is None for parameter in vlm.model.parameters())


def test_image_trainer_rejects_renderer_without_trainable_intervention(
    tmp_path: Path, font_path: Path
) -> None:
    path = _write_config(tmp_path / "none.yaml", font_path, intervention="none")
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "    size: [4, 5]\n    anchor: bottom_right\n", ""
        ),
        encoding="utf-8",
    )
    renderer = Renderer.from_yaml(path)
    with pytest.raises(ValueError, match="no trainable intervention"):
        Trainer(_FakeVLM(), renderer, _loss)


def test_image_trainer_restores_model_state_after_failure(tmp_path: Path, font_path: Path) -> None:
    renderer = Renderer.from_yaml(_write_config(tmp_path / "train.yaml", font_path))
    vlm = _FakeVLM()
    vlm.model.train()
    vlm.model.projection.eval()
    vlm.model.projection.weight.requires_grad_(False)
    modes = [module.training for module in vlm.model.modules()]
    requires_grad = [parameter.requires_grad for parameter in vlm.model.parameters()]

    def failing_loss(*_args):
        raise RuntimeError("loss failed")

    trainer = Trainer(vlm, renderer, failing_loss)
    sample = Sample(Image.new("RGB", (40, 8)), "Answer.", image_instruction="Do the task.")
    with pytest.raises(RuntimeError, match="loss failed"):
        trainer.fit([sample], max_steps=1)
    assert [module.training for module in vlm.model.modules()] == modes
    assert [parameter.requires_grad for parameter in vlm.model.parameters()] == requires_grad
    assert all(parameter.grad is None for parameter in tuple(renderer.parameters()))


@pytest.mark.parametrize("kind", ["overlay_patch", "local_additive"])
def test_trained_intervention_round_trip(tmp_path: Path, font_path: Path, kind: str) -> None:
    config = _write_config(
        tmp_path / "render.yaml",
        font_path,
        intervention=kind,
        extra="    epsilon: 0.1\n" if kind == "local_additive" else "",
    )
    renderer = Renderer.from_yaml(config)
    sample = Sample(
        Image.new("RGB", (40, 8), (80, 80, 80)), "Answer.", image_instruction="Do the task."
    )
    _, before = renderer.render(sample.image, sample.image_instruction, return_metadata=True)
    vlm = _FakeVLM()
    with torch.no_grad():
        vlm.model.projection.weight.fill_(1)
    Trainer(vlm=vlm, renderer=renderer, loss_function=_loss, learning_rate=0.05).fit(
        [sample], max_steps=1
    )
    image, metadata = renderer.render(sample.image, sample.image_instruction, return_metadata=True)
    assert before["intervention_sha256"] != metadata["intervention_sha256"]
    assert before["config_sha256"] == metadata["config_sha256"]

    checkpoint = tmp_path / "intervention.pt"
    renderer.save_intervention(checkpoint)
    restored = Renderer.from_yaml(config)
    restored.load_intervention(checkpoint)
    actual, actual_metadata = restored.render(
        sample.image, sample.image_instruction, return_metadata=True
    )
    assert actual_metadata == metadata
    assert actual.tobytes() == image.tobytes()
    torch.testing.assert_close(
        restored(sample.image, sample.image_instruction),
        renderer(sample.image, sample.image_instruction),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("invalid", ["format", "spec", "shape", "dtype", "nan", "bounds"])
def test_loading_invalid_intervention_keeps_existing_weights(
    tmp_path: Path, font_path: Path, invalid: str
) -> None:
    renderer = Renderer.from_yaml(_write_config(tmp_path / "render.yaml", font_path))
    path = tmp_path / "weights.pt"
    renderer.save_intervention(path)
    state = torch.load(path, weights_only=True)
    if invalid == "format":
        state["format"] = "unknown"
    elif invalid == "spec":
        state["spec"]["anchor"] = "top_left"
    elif invalid == "shape":
        state["parameter"] = torch.zeros(3, 1, 1)
    elif invalid == "dtype":
        state["parameter"] = state["parameter"].long()
    else:
        state["parameter"].fill_(float("nan") if invalid == "nan" else 2.0)
    torch.save(state, path)
    before = next(renderer.parameters()).detach().clone()
    with pytest.raises(ValueError):
        renderer.load_intervention(path)
    torch.testing.assert_close(next(renderer.parameters()), before, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["none", "overlay_patch", "local_additive"])
@pytest.mark.parametrize("position", ["top", "bottom"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_renderer_module_matches_pil_and_preserves_gradients(
    tmp_path: Path, font_path: Path, kind: str, position: str, dtype: torch.dtype
) -> None:
    import numpy as np
    import yaml

    config_path = _write_config(tmp_path / "render.yaml", font_path)
    config = yaml.safe_load(config_path.read_text())
    config["image_render"]["instruction"]["position"] = position
    config["image_render"]["intervention"] = (
        {"type": "none"}
        if kind == "none"
        else {
            "type": kind,
            "size": [2, 3],
            "anchor": "bottom_right",
            "init": "zeros",
            **({"epsilon": 0.2} if kind == "local_additive" else {}),
        }
    )
    config_path.write_text(yaml.safe_dump(config))
    renderer = Renderer.from_yaml(config_path).to(dtype=dtype)
    host = Image.new("RGB", (48, 13), (20, 40, 60))
    tensor = renderer(host, "Do the task.")
    image, metadata = renderer.render(host, "Do the task.", return_metadata=True)
    quantized = tensor.detach().cpu().mul(255).round().to(torch.uint8).permute(1, 2, 0)
    assert np.array_equal(quantized.numpy(), np.asarray(image))
    # Host placement and band contents must agree for both composition orders.
    x0, y0, x1, y1 = metadata["host_bbox"]
    assert (x1 - x0, y1 - y0) == host.size
    visited = []
    assert renderer.apply(visited.append) is renderer
    assert visited[-1] is renderer
    if kind != "none":
        parameter = next(renderer.parameters())
        assert tensor.dtype == parameter.dtype == dtype
        tensor.square().sum().backward()
        assert parameter.grad is not None
        with torch.no_grad():
            parameter.fill_(10)
        renderer.project_()
        expected_bound = 1.0 if kind == "overlay_patch" else 0.2
        assert float(parameter.detach().max()) == pytest.approx(
            torch.tensor(expected_bound, dtype=dtype).item()
        )
