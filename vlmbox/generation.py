"""Canonical decoder-block traces over native autoregressive generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import Tensor

from vlmbox.activation import activation_tensor, layer_indices

if TYPE_CHECKING:
    from vlmbox.model import VLM

GenerationPhase = Literal["prefill", "decode"]
FinishReason = Literal["eos", "length", "unknown"]


class GenerationTraceError(RuntimeError):
    """Raised when generation outputs cannot be represented as a consistent trace."""


@dataclass(frozen=True, slots=True, eq=False)
class GenerationStep:
    """One autoregressive forward pass and the token selected from its output.

    token_ids contains the next token selected for each sequence. Optional logits
    are raw, unprocessed next-token logits. hidden_states maps zero-based decoder
    block indices to direct block outputs before final normalization. Prefill states cover
    the prompt; decode states usually cover one token.
    """

    index: int
    phase: GenerationPhase
    token_ids: Tensor
    logits: Tensor | None = None
    hidden_states: Mapping[int, Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.index, int) or isinstance(self.index, bool) or self.index < 0:
            raise ValueError("step index must be a non-negative integer")
        if self.phase not in ("prefill", "decode"):
            raise ValueError("phase must be 'prefill' or 'decode'")
        if not isinstance(self.token_ids, Tensor) or self.token_ids.ndim != 1:
            raise TypeError("token_ids must be a one-dimensional torch.Tensor")
        if self.token_ids.dtype == torch.bool or self.token_ids.is_floating_point():
            raise TypeError("token_ids must use an integer tensor dtype")
        if self.token_ids.is_complex():
            raise TypeError("token_ids must use an integer tensor dtype")

        if self.logits is not None:
            if not isinstance(self.logits, Tensor) or self.logits.ndim < 2:
                raise TypeError("logits must be a tensor with batch and vocabulary dimensions")
            if self.logits.shape[0] != self.token_ids.shape[0]:
                raise ValueError("logits and token_ids must have the same batch size")

        if not isinstance(self.hidden_states, Mapping):
            raise TypeError("hidden_states must map decoder layer indices to tensors")
        normalized: dict[int, Tensor] = {}
        for layer, state in self.hidden_states.items():
            if not isinstance(layer, int) or isinstance(layer, bool) or layer < 0:
                raise ValueError("hidden-state layer indices must be non-negative integers")
            if not isinstance(state, Tensor) or state.ndim < 2:
                raise TypeError("each hidden state must be a tensor with a batch dimension")
            if state.shape[0] != self.token_ids.shape[0]:
                raise ValueError("hidden states and token_ids must have the same batch size")
            normalized[layer] = state

        object.__setattr__(self, "hidden_states", MappingProxyType(normalized))


@dataclass(frozen=True, slots=True, eq=False)
class GenerationTrace:
    """Structured generated sequences and optional per-step internal tensors.

    The dataclass structure and layer mapping are immutable. Tensor payloads are
    retained by reference and remain ordinary mutable PyTorch tensors.
    """

    prompt_length: int
    sequences: Tensor
    steps: tuple[GenerationStep, ...]
    finish_reasons: tuple[FinishReason, ...]
    texts: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.prompt_length, int)
            or isinstance(self.prompt_length, bool)
            or self.prompt_length < 0
        ):
            raise ValueError("prompt_length must be a non-negative integer")
        if not isinstance(self.sequences, Tensor) or self.sequences.ndim != 2:
            raise TypeError("sequences must be a two-dimensional torch.Tensor")
        if (
            self.sequences.dtype == torch.bool
            or self.sequences.is_floating_point()
            or self.sequences.is_complex()
        ):
            raise TypeError("sequences must use an integer tensor dtype")
        if self.prompt_length > self.sequences.shape[1]:
            raise ValueError("prompt_length cannot exceed the sequence length")

        steps = tuple(self.steps)
        generated_count = self.sequences.shape[1] - self.prompt_length
        if len(steps) != generated_count:
            raise ValueError("there must be exactly one trace step per generated token")
        batch_size = self.sequences.shape[0]
        for index, step in enumerate(steps):
            if not isinstance(step, GenerationStep):
                raise TypeError("steps must contain GenerationStep instances")
            expected_phase = "prefill" if index == 0 else "decode"
            if step.index != index or step.phase != expected_phase:
                raise ValueError("step indices and phases must follow prefill-then-decode order")
            if step.token_ids.shape[0] != batch_size:
                raise ValueError("every step must have the same batch size as sequences")

        finish_reasons = tuple(self.finish_reasons)
        if len(finish_reasons) != batch_size:
            raise ValueError("finish_reasons must contain one value per generated sequence")
        if any(reason not in ("eos", "length", "unknown") for reason in finish_reasons):
            raise ValueError("finish reasons must be 'eos', 'length', or 'unknown'")

        texts = None if self.texts is None else tuple(self.texts)
        if texts is not None:
            if len(texts) != batch_size or any(not isinstance(text, str) for text in texts):
                raise ValueError("texts must contain one string per generated sequence")

        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "finish_reasons", finish_reasons)
        object.__setattr__(self, "texts", texts)

    @property
    def prompt_token_ids(self) -> Tensor:
        """Return the prompt prefix from the model's returned sequences."""
        return self.sequences[:, : self.prompt_length]

    @property
    def generated_token_ids(self) -> Tensor:
        """Return generated tokens, including any EOS or padding tokens."""
        return self.sequences[:, self.prompt_length :]


def generation_option(name: str, kwargs: Mapping[str, Any], model: Any) -> Any:
    if name in kwargs:
        return kwargs[name]
    config = kwargs.get("generation_config")
    if config is None:
        config = getattr(model, "generation_config", None)
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def normalize_token_ids(value: Any) -> tuple[int, ...]:
    if value is None or isinstance(value, bool):
        return ()
    if isinstance(value, int):
        return (value,)
    if isinstance(value, Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    if any(not isinstance(token_id, int) or isinstance(token_id, bool) for token_id in value):
        return ()
    return tuple(value)


def infer_finish_reasons(
    sequences: Tensor,
    prompt_length: int,
    model: Any,
    generation_kwargs: Mapping[str, Any],
) -> tuple[FinishReason, ...]:
    """Infer EOS and explicit length-limit stops; leave other causes unknown."""
    generated = sequences[:, prompt_length:]
    eos_ids = normalize_token_ids(generation_option("eos_token_id", generation_kwargs, model))
    max_new_tokens = generation_option("max_new_tokens", generation_kwargs, model)
    max_length = generation_option("max_length", generation_kwargs, model)
    pad_ids = normalize_token_ids(generation_option("pad_token_id", generation_kwargs, model))
    forced_eos_ids = normalize_token_ids(
        generation_option("forced_eos_token_id", generation_kwargs, model)
    )
    ambiguous_eos_ids = set(eos_ids).intersection((*pad_ids, *forced_eos_ids))
    unambiguous_eos_ids = set(eos_ids).difference(ambiguous_eos_ids)
    has_custom_stop = (
        bool(generation_kwargs.get("stopping_criteria"))
        or bool(generation_option("stop_strings", generation_kwargs, model))
        or generation_option("max_time", generation_kwargs, model) is not None
        or generation_kwargs.get("custom_generate") is not None
    )

    if isinstance(max_new_tokens, int) and not isinstance(max_new_tokens, bool):
        reached_length = generated.shape[1] >= max_new_tokens
    else:
        reached_length = (
            isinstance(max_length, int)
            and not isinstance(max_length, bool)
            and sequences.shape[1] >= max_length
        )

    generated_rows = generated.detach().cpu().tolist()
    reasons: list[FinishReason] = []
    for row in generated_rows:
        if any(token_id in unambiguous_eos_ids for token_id in row):
            reasons.append("eos")
        elif any(token_id in ambiguous_eos_ids for token_id in row):
            reasons.append("unknown")
        elif reached_length and not has_custom_stop:
            reasons.append("length")
        else:
            reasons.append("unknown")
    return tuple(reasons)


def generate_trace(
    vlm: VLM,
    inputs: Mapping[str, Any],
    *,
    capture_logits: bool,
    capture_hidden_states: bool,
    hidden_state_layers: int | Sequence[int] | None,
    decode_text: bool,
    **generation_kwargs: Any,
) -> GenerationTrace:
    """Adapt native generation outputs without changing generation semantics."""
    if not isinstance(inputs, Mapping):
        raise TypeError("inputs must be a mapping returned by VLM.prepare() or the processor")
    if not isinstance(capture_logits, bool):
        raise TypeError("capture_logits must be a bool")
    if not isinstance(capture_hidden_states, bool):
        raise TypeError("capture_hidden_states must be a bool")
    if not isinstance(decode_text, bool):
        raise TypeError("decode_text must be a bool")
    if hidden_state_layers is not None and not capture_hidden_states:
        raise ValueError("hidden_state_layers requires capture_hidden_states=True")
    if capture_hidden_states and hidden_state_layers is None:
        raise ValueError(
            "capture_hidden_states=True requires an explicit hidden_state_layers selection"
        )

    controlled_options = {
        "return_dict_in_generate",
        "output_logits",
        "output_hidden_states",
    }
    conflicts = controlled_options.intersection(generation_kwargs)
    if conflicts:
        names = ", ".join(sorted(conflicts))
        raise ValueError(f"generate_trace controls these generation options: {names}")

    input_ids = inputs.get("input_ids")
    if not isinstance(input_ids, Tensor) or input_ids.ndim != 2:
        raise GenerationTraceError(
            "generate_trace requires two-dimensional input_ids to identify the prompt length"
        )
    prompt_length = input_ids.shape[1]
    model_config = getattr(vlm.model, "config", None)
    if getattr(model_config, "is_encoder_decoder", False):
        raise GenerationTraceError(
            "generate_trace currently supports decoder-only models, not encoder-decoder models"
        )

    selected_layers: tuple[int, ...] = ()
    if capture_hidden_states:
        selected_layers = layer_indices(
            hidden_state_layers,
            len(vlm.adapter.layers),
        )

    num_beams = generation_option("num_beams", generation_kwargs, vlm.model)
    if (capture_logits or capture_hidden_states) and num_beams not in (None, 1):
        raise ValueError(
            "capturing per-step logits or hidden states currently requires num_beams=1; "
            "beam outputs need beam-history alignment"
        )

    block_states: dict[int, list[Tensor]] = {layer: [] for layer in selected_layers}

    def record(layer, _module, _inputs, output):
        block_states[layer].append(activation_tensor(output).clone())

    with ExitStack() as stack:
        for layer in selected_layers:
            handle = vlm.adapter.layer(layer).register_forward_hook(partial(record, layer))
            stack.callback(handle.remove)
        output = vlm.generate(
            inputs,
            return_dict_in_generate=True,
            output_logits=capture_logits,
            output_hidden_states=False,
            **generation_kwargs,
        )
    sequences = getattr(output, "sequences", None)
    if not isinstance(sequences, Tensor) or sequences.ndim != 2:
        raise GenerationTraceError(
            "generation output must expose two-dimensional sequences; "
            "ensure the model supports return_dict_in_generate"
        )
    if prompt_length > sequences.shape[1]:
        raise GenerationTraceError(
            "returned sequence is shorter than the supplied prompt token length"
        )

    generated_count = sequences.shape[1] - prompt_length
    raw_logits = getattr(output, "logits", None)
    if capture_logits:
        if not isinstance(raw_logits, Sequence) or len(raw_logits) != generated_count:
            raise GenerationTraceError(
                "the model did not return one logits tensor per generated token"
            )

    for layer, states in block_states.items():
        if len(states) != generated_count:
            raise GenerationTraceError(
                f"decoder block {layer} returned {len(states)} outputs for {generated_count} "
                "generated tokens; trace capture requires one block forward per token"
            )

    steps: list[GenerationStep] = []
    for index in range(generated_count):
        logits = raw_logits[index] if capture_logits else None
        hidden_state_map = {layer: states[index] for layer, states in block_states.items()}

        steps.append(
            GenerationStep(
                index=index,
                phase="prefill" if index == 0 else "decode",
                token_ids=sequences[:, prompt_length + index],
                logits=logits,
                hidden_states=hidden_state_map,
            )
        )

    finish_reasons = infer_finish_reasons(
        sequences,
        prompt_length,
        vlm.model,
        generation_kwargs,
    )

    texts = None
    if decode_text:
        batch_decode = getattr(vlm.processor, "batch_decode", None)
        if not callable(batch_decode):
            raise GenerationTraceError("decode_text=True requires a processor with batch_decode()")
        decoded = batch_decode(
            sequences[:, prompt_length:],
            skip_special_tokens=True,
        )
        texts = (decoded,) if isinstance(decoded, str) else tuple(decoded)

    return GenerationTrace(
        prompt_length=prompt_length,
        sequences=sequences,
        steps=tuple(steps),
        finish_reasons=finish_reasons,
        texts=texts,
    )


__all__ = [
    "FinishReason",
    "GenerationPhase",
    "GenerationStep",
    "GenerationTrace",
    "GenerationTraceError",
]
