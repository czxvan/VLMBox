from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vlmbox
from vlmbox.adapter import TransformersAdapter
from vlmbox.generation import (
    GenerationStep,
    GenerationTrace,
    GenerationTraceError,
)
from vlmbox.model import VLM


def test_generation_types_are_exported_from_package_root() -> None:
    assert vlmbox.GenerationStep is GenerationStep
    assert vlmbox.GenerationTrace is GenerationTrace
    assert vlmbox.GenerationTraceError is GenerationTraceError


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
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
    assert model.generation_kwargs["output_hidden_states"] is True
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


def test_generate_trace_supports_hidden_states_without_embedding_entry() -> None:
    output = make_output()
    output.hidden_states = tuple(step[1:] for step in output.hidden_states)
    vlm, _model = make_vlm(output)

    trace = vlm.generate_trace(
        {"input_ids": torch.tensor([[10, 11], [20, 21]])},
        capture_hidden_states=True,
        hidden_state_layers=0,
    )

    assert torch.equal(trace.steps[0].hidden_states[0], output.hidden_states[0][0])


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
    output.hidden_states = output.hidden_states[:1]
    vlm, _model = make_vlm(output)

    with pytest.raises(GenerationTraceError, match="one hidden-state"):
        vlm.generate_trace(
            {"input_ids": torch.tensor([[10, 11], [20, 21]])},
            capture_hidden_states=True,
            hidden_state_layers=[0],
        )


def test_generate_trace_rejects_unrecognized_hidden_state_layout() -> None:
    output = make_output()
    output.hidden_states = tuple((step[0],) for step in output.hidden_states)
    vlm, _model = make_vlm(output)

    with pytest.raises(GenerationTraceError, match="expected 2 decoder layers"):
        vlm.generate_trace(
            {"input_ids": torch.tensor([[10, 11], [20, 21]])},
            capture_hidden_states=True,
            hidden_state_layers=[0],
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
