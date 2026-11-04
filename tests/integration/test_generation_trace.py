from __future__ import annotations

import gc
import os

import pytest
import torch
from PIL import Image

from vlmbox import VLM

MODEL_IDS = (
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "OpenGVLab/InternVL3_5-2B-HF",
    "HuggingFaceTB/SmolVLM-500M-Instruct",
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("VLMBOX_RUN_GPU_INTEGRATION") != "1",
        reason="set VLMBOX_RUN_GPU_INTEGRATION=1 to load cached VLM checkpoints",
    ),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


def _conversation(image: Image.Image) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image briefly."},
            ],
        }
    ]


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_generate_trace_selects_native_decoder_layers(model_id: str) -> None:
    vlm = VLM.from_pretrained(model_id, local_files_only=True)
    try:
        assert vlm.input_device.type == "cuda"
        layer_count = len(vlm.adapter.layers)
        selected_layers = tuple(sorted({0, layer_count // 2, layer_count - 1}))
        inputs = vlm.prepare(_conversation(Image.new("RGB", (224, 224), "white")))

        original_generate = vlm.generate
        recorded: dict[str, object] = {}

        def record_generate(generation_inputs, **generation_kwargs):
            recorded["output_hidden_states_arg"] = generation_kwargs["output_hidden_states"]
            output = original_generate(generation_inputs, **generation_kwargs)
            recorded["output"] = output
            return output

        vlm.generate = record_generate
        trace = vlm.generate_trace(
            inputs,
            capture_hidden_states=True,
            hidden_state_layers=selected_layers,
            decode_text=True,
            max_new_tokens=4,
            do_sample=False,
        )

        assert recorded["output_hidden_states_arg"] == selected_layers
        output = recorded["output"]
        raw_steps = output.hidden_states
        assert len(raw_steps) == len(trace.steps) > 0
        assert trace.texts is not None and len(trace.texts) == 1
        assert trace.steps[0].phase == "prefill"
        assert all(step.phase == "decode" for step in trace.steps[1:])

        for raw_step, trace_step in zip(raw_steps, trace.steps, strict=True):
            assert len(raw_step) in (layer_count, layer_count + 1)
            embedding_offset = len(raw_step) - layer_count
            assert set(trace_step.hidden_states) == set(selected_layers)
            for layer_index in range(layer_count):
                state = raw_step[layer_index + embedding_offset]
                if layer_index in selected_layers:
                    assert isinstance(state, torch.Tensor)
                    assert torch.equal(trace_step.hidden_states[layer_index], state)
                else:
                    assert state is None, (
                        f"{model_id} retained unselected decoder layer {layer_index}; "
                        "native selective output capture was not applied"
                    )

        prefill_length = trace.steps[0].hidden_states[selected_layers[0]].shape[1]
        assert prefill_length == inputs["input_ids"].shape[1]
        assert all(
            step.hidden_states[selected_layers[0]].shape[1] == 1
            for step in trace.steps[1:]
        )
    finally:
        del vlm
        gc.collect()
        torch.cuda.empty_cache()
