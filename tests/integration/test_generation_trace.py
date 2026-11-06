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


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_trace_and_capture_use_the_same_zero_based_raw_decoder_blocks(model_id: str) -> None:
    vlm = VLM.from_pretrained(model_id, local_files_only=True)
    handles = []
    try:
        assert vlm.input_device.type == "cuda"
        count = len(vlm.adapter.layers)
        selected = tuple(sorted({0, count // 2, count - 1}))
        observed = {layer: [] for layer in selected}

        def observer(layer):
            def hook(_module, _args, output):
                state = output if isinstance(output, torch.Tensor) else output[0]
                observed[layer].append(state.detach().clone())

            return hook

        for layer in selected:
            handles.append(vlm.adapter.layer(layer).register_forward_hook(observer(layer)))
        hook_counts = [len(layer._forward_hooks) for layer in vlm.adapter.layers]
        inputs = vlm.prepare(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": Image.new("RGB", (224, 224), "white")},
                        {"type": "text", "text": "Describe this image briefly."},
                    ],
                }
            ]
        )
        original_generate = vlm.generate
        recorded = {}

        def record_generate(generation_inputs, **kwargs):
            recorded["output_hidden_states"] = kwargs["output_hidden_states"]
            return original_generate(generation_inputs, **kwargs)

        vlm.generate = record_generate
        with vlm.capture("block", layers=selected, positions="all") as prefill:
            trace = vlm.generate_trace(
                inputs,
                capture_hidden_states=True,
                hidden_state_layers=selected,
                decode_text=True,
                max_new_tokens=4,
                do_sample=False,
            )
        assert recorded["output_hidden_states"] is False
        assert len(trace.steps) > 0
        assert trace.texts is not None and len(trace.texts) == 1
        for index, step in enumerate(trace.steps):
            assert step.phase == ("prefill" if index == 0 else "decode")
            assert set(step.hidden_states) == set(selected)
            for layer in selected:
                torch.testing.assert_close(
                    step.hidden_states[layer], observed[layer][index], rtol=0, atol=0
                )
                expected_length = inputs["input_ids"].shape[1] if index == 0 else 1
                assert step.hidden_states[layer].shape[1] == expected_length
        for layer in selected:
            torch.testing.assert_close(
                trace.steps[0].hidden_states[layer], prefill[layer], rtol=0, atol=0
            )
        assert [len(layer._forward_hooks) for layer in vlm.adapter.layers] == hook_counts
    finally:
        for handle in handles:
            handle.remove()
        del vlm
        gc.collect()
        torch.cuda.empty_cache()
