"""Architecture-independent data structures for autoregressive generation traces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

import torch
from torch import Tensor

GenerationPhase = Literal["prefill", "decode"]
FinishReason = Literal["eos", "length", "unknown"]


class GenerationTraceError(RuntimeError):
    """Raised when generation outputs cannot be represented as a consistent trace."""


@dataclass(frozen=True, slots=True, eq=False)
class GenerationStep:
    """One autoregressive forward pass and the token selected from its output.

    token_ids contains the next token selected for each sequence. Optional logits
    are raw, unprocessed next-token logits. hidden_states maps zero-based decoder
    layer indices to states produced during this forward pass. Prefill states cover
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


__all__ = [
    "FinishReason",
    "GenerationPhase",
    "GenerationStep",
    "GenerationTrace",
    "GenerationTraceError",
]
