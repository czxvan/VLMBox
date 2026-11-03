"""Unified Transformers runtime for ModelScope-hosted vision-language models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from torch import Tensor

from vlmbox.activation import (
    ActivationCapture,
    ActivationIntervention,
    Component,
    InterventionOperation,
    InterventionScope,
    PositionSelector,
)
from vlmbox.adapter import ModelAdapter, TransformersAdapter
from vlmbox.generation import (
    FinishReason,
    GenerationStep,
    GenerationTrace,
    GenerationTraceError,
)


def _normalize_generation_layers(
    layers: int | Sequence[int],
    count: int,
) -> tuple[int, ...]:
    if isinstance(layers, bool):
        raise TypeError("hidden_state_layers must be an integer or a sequence of integers")
    if isinstance(layers, int):
        selected = (layers,)
    elif isinstance(layers, Sequence) and not isinstance(layers, (str, bytes)):
        selected = tuple(layers)
    else:
        raise TypeError("hidden_state_layers must be an integer or a sequence of integers")

    if not selected:
        raise ValueError("hidden_state_layers must contain at least one layer")
    if any(not isinstance(index, int) or isinstance(index, bool) for index in selected):
        raise TypeError("every hidden-state layer index must be an integer")
    if len(set(selected)) != len(selected):
        raise ValueError("hidden_state_layers must not contain duplicate indices")
    if any(index < 0 or index >= count for index in selected):
        raise IndexError(f"hidden-state layer index must be in the range [0, {count})")
    return selected


def _generation_option(name: str, kwargs: Mapping[str, Any], model: Any) -> Any:
    if name in kwargs:
        return kwargs[name]
    config = kwargs.get("generation_config")
    if config is None:
        config = getattr(model, "generation_config", None)
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def _normalize_token_ids(value: Any) -> tuple[int, ...]:
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


def _infer_finish_reasons(
    sequences: Tensor,
    prompt_length: int,
    model: Any,
    generation_kwargs: Mapping[str, Any],
) -> tuple[FinishReason, ...]:
    """Infer EOS and explicit length-limit stops; leave other causes unknown."""
    generated = sequences[:, prompt_length:]
    eos_ids = _normalize_token_ids(
        _generation_option("eos_token_id", generation_kwargs, model)
    )
    max_new_tokens = _generation_option("max_new_tokens", generation_kwargs, model)
    max_length = _generation_option("max_length", generation_kwargs, model)
    pad_ids = _normalize_token_ids(
        _generation_option("pad_token_id", generation_kwargs, model)
    )
    forced_eos_ids = _normalize_token_ids(
        _generation_option("forced_eos_token_id", generation_kwargs, model)
    )
    ambiguous_eos_ids = set(eos_ids).intersection((*pad_ids, *forced_eos_ids))
    unambiguous_eos_ids = set(eos_ids).difference(ambiguous_eos_ids)
    has_custom_stop = (
        bool(generation_kwargs.get("stopping_criteria"))
        or bool(_generation_option("stop_strings", generation_kwargs, model))
        or _generation_option("max_time", generation_kwargs, model) is not None
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


class VLM:
    """A thin research-oriented wrapper around a Transformers multimodal model.

    Load models by ModelScope ID only. Trusted ModelScope repository code is
    enabled to support model-specific processors and architectures.
    """

    def __init__(
        self,
        model_id: str,
        model_path: str,
        model: Any,
        processor: Any,
        adapter: ModelAdapter,
    ) -> None:
        self.model_id = model_id
        self.model_path = model_path
        self.model = model
        self.processor = processor
        self.adapter = adapter

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        local_files_only: bool = False,
    ) -> VLM:
        """Resolve a ModelScope checkpoint and load it with Transformers.

        The public API intentionally accepts only a ModelScope model ID. Set
        ``local_files_only=True`` to require a complete local cache and prohibit
        ModelScope downloads. Device placement and dtype use fixed internal defaults.
        """
        if not isinstance(local_files_only, bool):
            raise TypeError("local_files_only must be a bool")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-empty ModelScope model ID")
        if Path(model_id).is_absolute() or Path(model_id).exists():
            raise ValueError(
                "VLM.from_pretrained() expects a ModelScope model ID, not a local path"
            )
        if "/" not in model_id.strip("/"):
            raise ValueError("ModelScope model IDs must use the 'namespace/model' form")

        try:
            from modelscope.hub.snapshot_download import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "ModelScope is required to download checkpoints. Install VLMBox dependencies."
            ) from exc

        try:
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Transformers 5.x with AutoModelForMultimodalLM is required."
            ) from exc

        model_path = str(
            snapshot_download(model_id.strip(), local_files_only=local_files_only)
        )
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        model = AutoModelForMultimodalLM.from_pretrained(
            model_path,
            dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )

        model.eval()
        adapter = TransformersAdapter(model)
        return cls(model_id.strip(), model_path, model, processor, adapter)

    def capture(
        self,
        component: Component,
        layers: int | Sequence[int],
        positions: PositionSelector = "last",
    ) -> ActivationCapture:
        """Capture the first output of selected decoder modules in a context.

        Captures are keyed by decoder layer index and record the first invocation
        of each selected module. With cached generation, this is normally prompt
        prefill. Nested hooks run in registration order: enter an intervention
        before capture to record the edited activation.
        """
        return ActivationCapture(self.adapter, component, layers, positions)

    def intervene(
        self,
        component: Component,
        layer: int,
        operation: InterventionOperation,
        value: Tensor | None = None,
        scale: float = 1.0,
        positions: PositionSelector = "last",
        scope: InterventionScope = "every_forward",
    ) -> ActivationIntervention:
        """Temporarily edit a decoder activation while a context is active.

        Add computes h + scale * value; replace substitutes value; project_out
        removes the projection onto value; zero clears the selected activations.
        Values must broadcast to the selected activation shape. ``first_forward``
        edits each target layer only on its first invocation; ``every_forward``
        reapplies the edit on every invocation, including autoregressive decoding.
        """
        return ActivationIntervention(
            self.adapter, component, layer, operation, value, scale, positions, scope
        )

    def prepare(self, messages: list[dict[str, Any]]) -> Any:
        """Apply the model's chat template and return tensor inputs."""
        return self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

    def generate(self, inputs: Mapping[str, Any], **generation_kwargs: Any) -> Any:
        """Run Transformers generation using prepared model inputs."""
        if not isinstance(inputs, Mapping):
            raise TypeError("inputs must be a mapping returned by VLM.prepare() or the processor")

        device = getattr(self.model, "device", None)
        if device is not None:
            if hasattr(inputs, "to"):
                inputs = inputs.to(device)
            else:
                import torch

                inputs = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in inputs.items()
                }
        return self.model.generate(**inputs, **generation_kwargs)

    def generate_trace(
        self,
        inputs: Mapping[str, Any],
        *,
        capture_logits: bool = False,
        capture_hidden_states: bool = False,
        hidden_state_layers: int | Sequence[int] | None = None,
        decode_text: bool = False,
        **generation_kwargs: Any,
    ) -> GenerationTrace:
        """Generate normally and return a structured, opt-in internal trace.

        Transformers still owns the generation loop and all decoding semantics.
        Raw logits and decoder hidden states are disabled by default because they
        can retain substantial GPU memory. hidden_state_layers uses zero-based
        decoder layer indices and must be selected explicitly when capturing states.

        A trace step records the forward pass that predicts its token_ids:
        step zero is prompt prefill, and later steps are autoregressive decode.
        The step's hidden states describe the prompt or preceding input token(s),
        not the token selected from that step's logits.

        Transformers does not expose a standard finish-reason field. VLMBox marks
        EOS or an explicit length limit only when observable; other stop causes
        remain unknown. This trace adapter currently supports decoder-only models;
        beam-search internal capture is rejected until beam histories can be aligned.
        """
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
        model_config = getattr(self.model, "config", None)
        if getattr(model_config, "is_encoder_decoder", False):
            raise GenerationTraceError(
                "generate_trace currently supports decoder-only models, not encoder-decoder models"
            )

        selected_layers: tuple[int, ...] = ()
        if capture_hidden_states:
            selected_layers = _normalize_generation_layers(
                hidden_state_layers,
                len(self.adapter.layers),
            )

        num_beams = _generation_option("num_beams", generation_kwargs, self.model)
        if (capture_logits or capture_hidden_states) and num_beams not in (None, 1):
            raise ValueError(
                "capturing per-step logits or hidden states currently requires num_beams=1; "
                "beam outputs need beam-history alignment"
            )

        output = self.generate(
            inputs,
            return_dict_in_generate=True,
            output_logits=capture_logits,
            output_hidden_states=capture_hidden_states,
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

        raw_hidden_states = getattr(output, "hidden_states", None)
        if capture_hidden_states:
            if (
                not isinstance(raw_hidden_states, Sequence)
                or len(raw_hidden_states) != generated_count
            ):
                raise GenerationTraceError(
                    "the model did not return one hidden-state collection per generated token"
                )

        steps: list[GenerationStep] = []
        for index in range(generated_count):
            logits = raw_logits[index] if capture_logits else None
            hidden_state_map: dict[int, Tensor] = {}
            if capture_hidden_states:
                layer_outputs = raw_hidden_states[index]
                if not isinstance(layer_outputs, Sequence):
                    raise GenerationTraceError(
                        "each hidden-state collection must be a sequence of layer tensors"
                    )
                layer_count = len(self.adapter.layers)
                if len(layer_outputs) == layer_count + 1:
                    offset = 1
                elif len(layer_outputs) == layer_count:
                    offset = 0
                else:
                    raise GenerationTraceError(
                        f"hidden-state collection has {len(layer_outputs)} entries; "
                        f"expected {layer_count} decoder layers, optionally preceded by embeddings"
                    )
                for layer in selected_layers:
                    state = layer_outputs[layer + offset]
                    if not isinstance(state, Tensor):
                        raise GenerationTraceError(
                            f"hidden state for decoder layer {layer} is not a tensor"
                        )
                    hidden_state_map[layer] = state

            steps.append(
                GenerationStep(
                    index=index,
                    phase="prefill" if index == 0 else "decode",
                    token_ids=sequences[:, prompt_length + index],
                    logits=logits,
                    hidden_states=hidden_state_map,
                )
            )

        finish_reasons = _infer_finish_reasons(
            sequences,
            prompt_length,
            self.model,
            generation_kwargs,
        )

        texts = None
        if decode_text:
            batch_decode = getattr(self.processor, "batch_decode", None)
            if not callable(batch_decode):
                raise GenerationTraceError(
                    "decode_text=True requires a processor with batch_decode()"
                )
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

    def __repr__(self) -> str:
        return f"VLM(model_id={self.model_id!r}, model_path={self.model_path!r})"
