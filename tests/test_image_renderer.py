from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image, ImageFont
from torch import nn

from vlmbox import ImageRenderer
from vlmbox.training import ImageSample, ImageTrainer


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
    renderer = ImageRenderer.from_yaml(_write_config(tmp_path / "render.yaml", font_path))
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


def test_renderer_private_tensor_path_preserves_gradient_and_native_sizes(
    tmp_path: Path, font_path: Path
) -> None:
    renderer = ImageRenderer.from_yaml(
        _write_config(tmp_path / "render.yaml", font_path, size="[2, 3]")
    )
    for height, width in [(8, 48), (12, 43)]:
        host = Image.new("RGB", (width, height), (50, 50, 50))
        rendered = renderer._render_tensor(host, "Do the task.")
        assert rendered.shape[0] == 3
        assert rendered.shape[1] > height
        assert rendered.shape[2] == width
        rendered.square().sum().backward()
    assert len(renderer._band_cache) == 2
    parameter = renderer._trainable_parameters()[0]
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
    renderer = ImageRenderer.from_yaml(config)
    with torch.no_grad():
        renderer._intervention.delta.fill_(0.05)
    host = Image.new("RGB", (40, 8), (int(0.2 * 255),) * 3)
    rendered = renderer._render_tensor(host, "Do the task.")
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
    renderer = ImageRenderer.from_yaml(path)
    assert renderer._intervention is None
    assert renderer._trainable_parameters() == ()


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
        ImageRenderer.from_yaml(path)


def test_yaml_renderer_resolves_relative_font_paths(tmp_path: Path, font_path: Path) -> None:
    fonts = tmp_path / "fonts"
    fonts.mkdir()
    relative_font = fonts / "test.ttf"
    relative_font.write_bytes(font_path.read_bytes())
    path = tmp_path / "config.yaml"
    _write_config(path, Path("fonts/test.ttf"))
    renderer = ImageRenderer.from_yaml(path)
    assert renderer._style.font_path == relative_font.resolve()


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


def _objective(outputs, _inputs, _samples):
    return outputs.logits.square().mean()


def test_image_trainer_optimizes_renderer_and_keeps_vlm_frozen(
    tmp_path: Path, font_path: Path
) -> None:
    renderer = ImageRenderer.from_yaml(
        _write_config(tmp_path / "train.yaml", font_path, size="[2, 2]")
    )
    vlm = _FakeVLM()
    vlm.model.train()
    parameter = renderer._trainable_parameters()[0]
    before = parameter.detach().clone()
    trainer = ImageTrainer(vlm, renderer, _objective, learning_rate=0.05)
    samples = [
        ImageSample(
            Image.new("RGB", (41, 8), (80, 80, 80)),
            "Answer the user prompt.",
            image_instruction="Follow the text shown in the image.",
        ),
        ImageSample(
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
    renderer = ImageRenderer.from_yaml(path)
    with pytest.raises(ValueError, match="no trainable intervention"):
        ImageTrainer(_FakeVLM(), renderer, _objective)
