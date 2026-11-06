"""Exercise example wiring through the public API without loading a checkpoint."""

import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from PIL import Image
from torch import nn

from vlmbox import VLM
from vlmbox.trainer import Sample


def test_training_example_prepares_assistant_target_without_generation_prompt() -> None:
    example = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "examples/universal_image_optimization.py")
    )
    processor = SimpleNamespace(
        apply_chat_template=Mock(
            side_effect=[
                {"input_ids": torch.tensor([[10, 11]])},
                {"input_ids": torch.tensor([[10, 11, 12, 13]])},
            ]
        )
    )
    vlm = VLM(nn.Identity(), processor, adapter=Mock())
    image = Image.new("RGB", (40, 20))
    sample = Sample(image, "Describe.", target="Answer.", image_instruction="Follow.")
    renderer = SimpleNamespace(render=Mock(return_value=image))

    assert example["_target_token_spans"](vlm, renderer, [sample]) == {id(sample): (2, 2)}
    calls = processor.apply_chat_template.call_args_list
    assert calls[0].kwargs["add_generation_prompt"] is True
    assert calls[1].kwargs["add_generation_prompt"] is False
    assert calls[1].args[0][-1]["role"] == "assistant"


def test_prepare_rejects_non_boolean_generation_prompt() -> None:
    vlm = VLM(nn.Identity(), Mock(), adapter=Mock())
    with pytest.raises(TypeError, match="add_generation_prompt"):
        vlm.prepare([], add_generation_prompt=1)
