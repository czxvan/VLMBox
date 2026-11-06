from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vlmbox.model import VLM
from vlmbox.model.adapter import TransformersAdapter
from vlmbox.model.generation import (
    GenerationStep,
    GenerationTrace,
    GenerationTraceError,
)


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.language_model = nn.Module()
        self.language_model.layers = nn.ModuleList([nn.Identity(), nn.Identity()])
        self.generation_config = SimpleNamespace(
            eos_token_id=2,
            max_new_tokens=None,
            max_length=20,
            num_beams=1,
        )
        self.output = None
        self.generation_kwargs = None

    def generate(self, **kwargs):
        self.generation_kwargs = kwargs
        for step in getattr(self.output, "block_outputs", ()):
            for layer, state in zip(self.language_model.layers, step, strict=False):
                layer(state)
        return self.output


class FakeProcessor:
    def batch_decode(self, token_ids, skip_special_tokens=True):
        assert skip_special_tokens is True
        return [f"sequence-{index}" for index in range(token_ids.shape[0])]


def make_vlm(output=None) -> tuple[VLM, FakeModel]:
    model = FakeModel()
    model.output = output
    vlm = VLM(
        model_id="Org/Model",
        model_path="/model-cache/fake-model",
        model=model,
        processor=FakeProcessor(),
        adapter=TransformersAdapter(model),
    )
    return vlm, model


def make_output(sequences: torch.Tensor | None = None):
    if sequences is None:
        sequences = torch.tensor([[10, 11, 2, 0], [20, 21, 4, 5]])
    batch_size = sequences.shape[0]
    logits = (
        torch.full((batch_size, 8), 0.1),
        torch.full((batch_size, 8), 0.2),
    )
    prefill = tuple(torch.full((batch_size, 2, 4), float(index)) for index in range(3))
    decode = tuple(torch.full((batch_size, 1, 4), float(index + 10)) for index in range(3))
    return SimpleNamespace(
        sequences=sequences,
        logits=logits,
        hidden_states=(prefill, decode),
        block_outputs=(prefill[1:], decode[1:]),
    )


def test_generate_trace_captures_selected_layers_and_raw_logits() -> None:
    output = make_output()
    vlm, model = make_vlm(output)
    inputs = {"input_ids": torch.tensor([[10, 11], [20, 21]])}

    trace = vlm.generate_trace(
        inputs,
        capture_logits=True,
        capture_hidden_states=True,
        hidden_state_layers=[1],
        decode_text=True,
        max_new_tokens=2,
    )

    assert model.generation_kwargs["return_dict_in_generate"] is True
    assert model.generation_kwargs["output_logits"] is True
    assert model.generation_kwargs["output_hidden_states"] is False
    assert model.generation_kwargs["max_new_tokens"] == 2
    assert torch.equal(trace.prompt_token_ids, inputs["input_ids"])
    assert torch.equal(trace.generated_token_ids, torch.tensor([[2, 0], [4, 5]]))
    assert trace.finish_reasons == ("eos", "length")
    assert trace.texts == ("sequence-0", "sequence-1")

    assert len(trace.steps) == 2
    assert trace.steps[0].phase == "prefill"
    assert trace.steps[1].phase == "decode"
    assert torch.equal(trace.steps[0].token_ids, torch.tensor([2, 4]))
    assert torch.equal(trace.steps[1].token_ids, torch.tensor([0, 5]))
    assert torch.equal(trace.steps[0].logits, output.logits[0])
    assert set(trace.steps[0].hidden_states) == {1}
    assert trace.steps[0].hidden_states[1].shape == (2, 2, 4)
    assert trace.steps[1].hidden_states[1].shape == (2, 1, 4)


def _assert_canonical_trace(vlm: VLM, final_norm: nn.Module, *, max_new_tokens: int) -> None:
    selected = (0, len(vlm.adapter.layers) - 1)
    observed = {layer: [] for layer in selected}
    normalized = []
    handles = []

    def observer(layer):
        def hook(_module, _args, output):
            state = output if isinstance(output, torch.Tensor) else output[0]
            observed[layer].append(state.detach().clone())

        return hook

    for layer in selected:
        handles.append(vlm.adapter.layer(layer).register_forward_hook(observer(layer)))
    handles.append(
        final_norm.register_forward_hook(
            lambda _m, _i, output: normalized.append(output.detach().clone())
        )
    )
    before = [len(layer._forward_hooks) for layer in vlm.adapter.layers]
    try:
        inputs = {"input_ids": torch.tensor([[1, 5, 6]])}
        with vlm.capture("block", layers=selected, positions="all") as prefill:
            trace = vlm.generate_trace(
                inputs,
                capture_hidden_states=True,
                hidden_state_layers=selected,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        assert len(trace.steps) > 0
        for index, step in enumerate(trace.steps):
            assert set(step.hidden_states) == set(selected)
            for layer in selected:
                torch.testing.assert_close(
                    step.hidden_states[layer], observed[layer][index], rtol=0, atol=0
                )
        for layer in selected:
            torch.testing.assert_close(
                trace.steps[0].hidden_states[layer], prefill[layer], rtol=0, atol=0
            )
        assert not torch.allclose(trace.steps[0].hidden_states[selected[-1]], normalized[0])
        assert [len(layer._forward_hooks) for layer in vlm.adapter.layers] == before
    finally:
        for handle in handles:
            handle.remove()


def test_generate_trace_uses_raw_blocks_before_llama_final_norm() -> None:
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = LlamaForCausalLM(config).eval()
    processor = FakeProcessor()
    vlm = VLM(
        model_id="Org/TinyLlama",
        model_path="/model-cache/tiny-llama",
        model=model,
        processor=processor,
        adapter=TransformersAdapter(model),
    )

    _assert_canonical_trace(vlm, model.model.norm, max_new_tokens=3)


def test_generate_trace_uses_raw_blocks_before_qwen_final_norm() -> None:
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
        Qwen2_5_VLConfig,
        Qwen2_5_VLTextConfig,
        Qwen2_5_VLVisionConfig,
    )
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLForConditionalGeneration,
    )

    text_config = Qwen2_5_VLTextConfig(
        vocab_size=64,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    vision_config = Qwen2_5_VLVisionConfig(
        depth=2,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        window_size=4,
        out_hidden_size=512,
        fullatt_block_indexes=[1],
    )
    model = Qwen2_5_VLForConditionalGeneration(
        Qwen2_5_VLConfig(text_config=text_config, vision_config=vision_config)
    ).eval()
    vlm = VLM(
        model_id="Org/TinyQwen2.5-VL",
        model_path="/model-cache/tiny-qwen2.5-vl",
        model=model,
        processor=FakeProcessor(),
        adapter=TransformersAdapter(model),
    )

    _assert_canonical_trace(vlm, model.model.language_model.norm, max_new_tokens=2)


def test_generate_trace_disables_internal_capture_by_default() -> None:
    output = make_output()
    vlm, model = make_vlm(output)

    trace = vlm.generate_trace({"input_ids": torch.tensor([[10, 11], [20, 21]])})

    assert model.generation_kwargs["output_logits"] is False
    assert model.generation_kwargs["output_hidden_states"] is False
    assert trace.steps[0].logits is None
    assert dict(trace.steps[0].hidden_states) == {}
    assert trace.finish_reasons == ("eos", "unknown")


def test_generate_trace_captures_all_layers_when_explicitly_selected() -> None:
    vlm, _model = make_vlm(make_output())

    trace = vlm.generate_trace(
        {"input_ids": torch.tensor([[10, 11], [20, 21]])},
        capture_hidden_states=True,
        hidden_state_layers=[0, 1],
    )

    assert set(trace.steps[0].hidden_states) == {0, 1}


def test_trace_records_block_outputs_after_an_active_intervention() -> None:
    vlm, _model = make_vlm(make_output())
    with vlm.intervene("block", 1, "zero", positions="all"):
        trace = vlm.generate_trace(
            {"input_ids": torch.tensor([[10, 11], [20, 21]])},
            capture_hidden_states=True,
            hidden_state_layers=[1],
        )
    assert all(torch.count_nonzero(step.hidden_states[1]) == 0 for step in trace.steps)


def test_trace_removes_capture_hooks_when_generation_fails() -> None:
    vlm, model = make_vlm(make_output())
    before = [len(layer._forward_hooks) for layer in vlm.adapter.layers]

    def fail(**_kwargs):
        model.language_model.layers[0](torch.ones(2, 2, 4))
        raise RuntimeError("generation failed")

    model.generate = fail
    with pytest.raises(RuntimeError, match="generation failed"):
        vlm.generate_trace(
            {"input_ids": torch.tensor([[10, 11], [20, 21]])},
            capture_hidden_states=True,
            hidden_state_layers=[0, 1],
        )
    assert [len(layer._forward_hooks) for layer in vlm.adapter.layers] == before


def test_generate_trace_ignores_native_hidden_state_layout() -> None:
    output = make_output()
    output.hidden_states = ((None,),)
    vlm, _model = make_vlm(output)

    trace = vlm.generate_trace(
        {"input_ids": torch.tensor([[10, 11], [20, 21]])},
        capture_hidden_states=True,
        hidden_state_layers=0,
    )

    assert torch.equal(trace.steps[0].hidden_states[0], output.block_outputs[0][0])


def test_max_new_tokens_takes_precedence_over_max_length_for_finish_reason() -> None:
    output = make_output()
    vlm, _model = make_vlm(output)

    trace = vlm.generate_trace(
        {"input_ids": torch.tensor([[10, 11], [20, 21]])},
        max_new_tokens=4,
        max_length=3,
    )

    assert trace.finish_reasons == ("eos", "unknown")


def test_finish_reason_is_unknown_when_eos_id_is_also_padding_id() -> None:
    output = make_output()
    vlm, model = make_vlm(output)
    model.generation_config.pad_token_id = 2

    trace = vlm.generate_trace(
        {"input_ids": torch.tensor([[10, 11], [20, 21]])},
        max_new_tokens=2,
    )

    assert trace.finish_reasons == ("unknown", "length")


def test_finish_reason_is_unknown_when_eos_is_forced_at_length_limit() -> None:
    output = make_output()
    vlm, model = make_vlm(output)
    model.generation_config.forced_eos_token_id = 2

    trace = vlm.generate_trace(
        {"input_ids": torch.tensor([[10, 11], [20, 21]])},
        max_new_tokens=2,
    )

    assert trace.finish_reasons == ("unknown", "length")


def test_generate_trace_leaves_custom_stopping_reason_unknown() -> None:
    vlm, _model = make_vlm(make_output())

    trace = vlm.generate_trace(
        {"input_ids": torch.tensor([[10, 11], [20, 21]])},
        max_new_tokens=2,
        stopping_criteria=[object()],
    )

    assert trace.finish_reasons == ("eos", "unknown")


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"hidden_state_layers": [0]}, ValueError, "requires capture_hidden_states"),
        ({"capture_hidden_states": True}, ValueError, "explicit hidden_state_layers"),
        ({"capture_hidden_states": True, "hidden_state_layers": []}, ValueError, "at least one"),
        ({"capture_hidden_states": True, "hidden_state_layers": True}, TypeError, "integer"),
        ({"capture_hidden_states": True, "hidden_state_layers": [2]}, IndexError, "range"),
        ({"capture_hidden_states": True, "hidden_state_layers": [0, 0]}, ValueError, "duplicate"),
        ({"capture_logits": 1}, TypeError, "bool"),
        ({"decode_text": 1}, TypeError, "bool"),
    ],
)
def test_generate_trace_validates_capture_options(kwargs, error, message) -> None:
    vlm, model = make_vlm(make_output())

    with pytest.raises(error, match=message):
        vlm.generate_trace({"input_ids": torch.tensor([[10, 11], [20, 21]])}, **kwargs)

    assert model.generation_kwargs is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"return_dict_in_generate": False},
        {"output_logits": True},
        {"output_hidden_states": True},
    ],
)
def test_generate_trace_rejects_conflicting_transformers_output_options(kwargs) -> None:
    vlm, model = make_vlm(make_output())

    with pytest.raises(ValueError, match="controls"):
        vlm.generate_trace({"input_ids": torch.tensor([[10, 11], [20, 21]])}, **kwargs)

    assert model.generation_kwargs is None


def test_generate_trace_rejects_encoder_decoder_models() -> None:
    vlm, model = make_vlm(make_output())
    model.config = SimpleNamespace(is_encoder_decoder=True)

    with pytest.raises(GenerationTraceError, match="decoder-only"):
        vlm.generate_trace({"input_ids": torch.tensor([[10, 11]])})

    assert model.generation_kwargs is None


def test_generate_trace_requires_two_dimensional_input_ids() -> None:
    vlm, model = make_vlm(make_output())

    with pytest.raises(GenerationTraceError, match="input_ids"):
        vlm.generate_trace({"pixel_values": torch.ones(1, 3, 4, 4)})

    with pytest.raises(GenerationTraceError, match="input_ids"):
        vlm.generate_trace({"input_ids": torch.tensor([10, 11])})

    assert model.generation_kwargs is None


def test_generate_trace_requires_structured_generation_output() -> None:
    vlm, _model = make_vlm(torch.tensor([[10, 11, 12]]))

    with pytest.raises(GenerationTraceError, match="sequences"):
        vlm.generate_trace({"input_ids": torch.tensor([[10, 11]])})


def test_generate_trace_rejects_misaligned_logits_or_hidden_states() -> None:
    output = make_output()
    output.logits = output.logits[:1]
    vlm, _model = make_vlm(output)

    with pytest.raises(GenerationTraceError, match="one logits"):
        vlm.generate_trace(
            {"input_ids": torch.tensor([[10, 11], [20, 21]])},
            capture_logits=True,
        )

    output = make_output()
    output.block_outputs = output.block_outputs[:1]
    vlm, _model = make_vlm(output)

    with pytest.raises(GenerationTraceError, match="one block forward per token"):
        vlm.generate_trace(
            {"input_ids": torch.tensor([[10, 11], [20, 21]])},
            capture_hidden_states=True,
            hidden_state_layers=[0],
        )


def test_generate_trace_rejects_missing_decoder_block_forwards() -> None:
    output = make_output()
    output.block_outputs = tuple((step[0],) for step in output.block_outputs)
    vlm, _model = make_vlm(output)

    with pytest.raises(GenerationTraceError, match="decoder block 1"):
        vlm.generate_trace(
            {"input_ids": torch.tensor([[10, 11], [20, 21]])},
            capture_hidden_states=True,
            hidden_state_layers=[1],
        )


def test_generate_trace_rejects_beam_capture_before_generation() -> None:
    vlm, model = make_vlm(make_output())

    with pytest.raises(ValueError, match="num_beams=1"):
        vlm.generate_trace(
            {"input_ids": torch.tensor([[10, 11], [20, 21]])},
            capture_logits=True,
            num_beams=4,
        )

    assert model.generation_kwargs is None


def test_generate_trace_rejects_missing_text_decoder() -> None:
    vlm, model = make_vlm(make_output())
    vlm.processor = object()

    with pytest.raises(GenerationTraceError, match="batch_decode"):
        vlm.generate_trace(
            {"input_ids": torch.tensor([[10, 11], [20, 21]])},
            decode_text=True,
        )

    assert model.generation_kwargs["return_dict_in_generate"] is True


def test_generation_trace_is_structurally_immutable() -> None:
    step = GenerationStep(
        index=0,
        phase="prefill",
        token_ids=torch.tensor([3]),
        hidden_states={0: torch.ones(1, 2, 4)},
    )
    trace = GenerationTrace(
        prompt_length=2,
        sequences=torch.tensor([[1, 2, 3]]),
        steps=(step,),
        finish_reasons=("unknown",),
    )

    with pytest.raises(TypeError):
        trace.steps[0].hidden_states[1] = torch.zeros(1, 2, 4)

    assert trace.prompt_token_ids.tolist() == [[1, 2]]
    assert trace.generated_token_ids.tolist() == [[3]]


def test_generation_trace_rejects_inconsistent_step_order() -> None:
    step = GenerationStep(index=0, phase="decode", token_ids=torch.tensor([3]))

    with pytest.raises(ValueError, match="prefill-then-decode"):
        GenerationTrace(
            prompt_length=2,
            sequences=torch.tensor([[1, 2, 3]]),
            steps=(step,),
            finish_reasons=("unknown",),
        )
