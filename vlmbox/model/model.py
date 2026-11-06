"""Native model access, differentiable preparation, and decoder interventions."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .activation import (
    ActivationCapture,
    ActivationIntervention,
    Component,
    InterventionOperation,
    InterventionScope,
    PositionSelector,
)
from .adapter import ModelAdapter, TransformersAdapter
from .generation import GenerationTrace, generate_trace


def move_inputs(inputs: Mapping[str, Any], device: torch.device) -> Any:
    """Move native batch objects or tensor-valued mappings to the input device."""
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in inputs.items()
    }


def chat_inputs(
    processor: Any,
    messages: Any,
    add_generation_prompt: bool,
    chat_template_kwargs: Mapping[str, Any] | None,
    processor_kwargs: dict[str, Any],
) -> Any:
    if not isinstance(add_generation_prompt, bool):
        raise TypeError("add_generation_prompt must be a bool")
    if chat_template_kwargs is not None and not isinstance(chat_template_kwargs, Mapping):
        raise TypeError("chat_template_kwargs must be a mapping or None")
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs=processor_kwargs,
        **dict(chat_template_kwargs or {}),
    )


class VLM:
    """A thin research-oriented wrapper around a Transformers multimodal model.

    Wrap an existing native model and processor, or load a ModelScope checkpoint
    with from_pretrained(). The native objects remain available to research adapters.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        adapter: ModelAdapter | None = None,
        model_id: str | None = None,
        model_path: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.model_path = model_path
        self.model = model
        self.processor = processor
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            tokenizer = processor
        if hasattr(tokenizer, "padding_side"):
            tokenizer.padding_side = "left"
        self.adapter = TransformersAdapter(model) if adapter is None else adapter

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
            from transformers import AutoConfig, AutoModelForMultimodalLM, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Transformers 5.x with AutoModelForMultimodalLM is required."
            ) from exc

        model_path = str(snapshot_download(model_id.strip(), local_files_only=local_files_only))
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        model_kwargs = {
            "dtype": "auto",
            "device_map": "auto",
            "trust_remote_code": True,
        }
        # Some InternVL checkpoints omit the outer tying flag while their text
        # config is untied. Preserve explicit values, correct only that omission.
        config_path = Path(model_path) / "config.json"
        if config_path.is_file():
            raw_config = json.loads(config_path.read_text(encoding="utf-8"))
            if (
                raw_config.get("model_type") == "internvl"
                and "tie_word_embeddings" not in raw_config
            ):
                config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                if (
                    getattr(getattr(config, "text_config", None), "tie_word_embeddings", None)
                    is False
                ):
                    config.tie_word_embeddings = False
                    model_kwargs["config"] = config
        model = AutoModelForMultimodalLM.from_pretrained(model_path, **model_kwargs)

        model.eval()
        return cls(model, processor, model_id=model_id.strip(), model_path=model_path)

    @property
    def input_device(self) -> torch.device:
        """Return the device used for model inputs."""
        device = getattr(self.model, "device", None)
        if device is not None:
            return torch.device(device)
        try:
            return next(self.model.parameters()).device
        except StopIteration as exc:
            raise RuntimeError(
                "model has no parameters from which to infer an input device"
            ) from exc

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

    def prepare(
        self,
        messages: Sequence[Mapping[str, Any]] | Sequence[Sequence[Mapping[str, Any]]],
        *,
        add_generation_prompt: bool = True,
        chat_template_kwargs: Mapping[str, Any] | None = None,
    ) -> Any:
        """Prepare one conversation or a batch with the native chat template.

        Set add_generation_prompt=False for conversations containing an assistant target.
        ``chat_template_kwargs`` passes model-specific variables to the native
        Transformers chat template, for example ``{"enable_thinking": False}``.
        """
        return chat_inputs(
            self.processor, messages, add_generation_prompt, chat_template_kwargs, {"padding": True}
        )

    def prepare_differentiable(
        self,
        messages: Sequence[Mapping[str, Any]] | Sequence[Sequence[Mapping[str, Any]]],
        *,
        add_generation_prompt: bool = True,
        chat_template_kwargs: Mapping[str, Any] | None = None,
    ) -> Any:
        """Prepare Tensor images through the native processor without detaching gradients.

        Accepts one conversation or a batch of conversations. Every conversation
        must contain at least one ``{"type": "image", "image": tensor}`` item;
        image tensors must be RGB CHW floats in [0, 1]. Images are moved to CPU
        before preprocessing so configured resampling kernels (including
        LANCZOS) are preserved. The resulting BatchFeature is then moved to the
        model input device; both transfers remain differentiable.
        """
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)) or not messages:
            raise ValueError("messages must be a non-empty conversation or batch of conversations")

        first = messages[0]
        raw_conversations = [messages] if isinstance(first, Mapping) else messages
        conversations: list[list[dict[str, Any]]] = []
        for conversation_index, conversation in enumerate(raw_conversations):
            if (
                not isinstance(conversation, Sequence)
                or isinstance(conversation, (str, bytes))
                or not conversation
            ):
                raise ValueError(f"conversation {conversation_index} must be a non-empty sequence")
            copied_messages: list[dict[str, Any]] = []
            image_count = 0
            for message in conversation:
                if not isinstance(message, Mapping):
                    raise TypeError("every conversation message must be a mapping")
                copied_message = dict(message)
                content = message.get("content")
                if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
                    copied_parts: list[Any] = []
                    for part in content:
                        if isinstance(part, Mapping) and part.get("type") == "image":
                            image = part.get("image")
                            if not isinstance(image, Tensor):
                                raise TypeError(
                                    "differentiable image content must contain a torch.Tensor"
                                )
                            if image.ndim != 3 or image.shape[0] != 3:
                                raise ValueError(
                                    "differentiable images must have shape [3, height, width]"
                                )
                            if not image.is_floating_point():
                                raise TypeError(
                                    "differentiable images must use a floating-point dtype"
                                )
                            if image.shape[-2] <= 0 or image.shape[-1] <= 0:
                                raise ValueError("differentiable images must be non-empty")
                            cpu_image = image.to(device="cpu", dtype=torch.float32).contiguous()
                            detached = cpu_image.detach()
                            if not bool(torch.isfinite(detached).all().item()):
                                raise ValueError("differentiable images must contain finite values")
                            if (
                                float(detached.amin().item()) < 0.0
                                or float(detached.amax().item()) > 1.0
                            ):
                                raise ValueError("differentiable image values must be in [0, 1]")
                            copied_part = dict(part)
                            copied_part["image"] = cpu_image
                            copied_parts.append(copied_part)
                            image_count += 1
                        else:
                            copied_parts.append(part)
                    copied_message["content"] = copied_parts
                copied_messages.append(copied_message)
            if image_count == 0:
                raise ValueError(f"conversation {conversation_index} contains no Tensor image")
            conversations.append(copied_messages)

        inputs = chat_inputs(
            self.processor,
            conversations,
            add_generation_prompt,
            chat_template_kwargs,
            {"padding": True, "do_rescale": False},
        )
        return move_inputs(inputs, self.input_device)

    def forward(self, inputs: Mapping[str, Any], **forward_kwargs: Any) -> Any:
        """Run a normal autograd-enabled model forward pass on prepared inputs.

        Unlike ``generate()``, this method does not enter inference mode or detach
        image tensors, so gradients can flow back to a universal image parameter.
        """
        if not isinstance(inputs, Mapping):
            raise TypeError("inputs must be a mapping returned by a VLM preparation method")
        return self.model(**move_inputs(inputs, self.input_device), **forward_kwargs)

    def generate(self, inputs: Mapping[str, Any], **generation_kwargs: Any) -> Any:
        """Run Transformers generation using prepared model inputs."""
        if not isinstance(inputs, Mapping):
            raise TypeError("inputs must be a mapping returned by VLM.prepare() or the processor")

        return self.model.generate(**move_inputs(inputs, self.input_device), **generation_kwargs)

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

        Layers are zero-based decoder block indices, shared with capture() and
        intervene(). Hidden states are direct block outputs before final model
        normalization. Only selected blocks are recorded; native hidden-state
        tuple layouts do not affect the result.

        Each step records the forward pass predicting its token: prompt prefill
        first, then decode. Internal capture supports decoder-only generation
        with num_beams=1 and one block forward per generated token.
        """
        return generate_trace(
            self,
            inputs,
            capture_logits=capture_logits,
            capture_hidden_states=capture_hidden_states,
            hidden_state_layers=hidden_state_layers,
            decode_text=decode_text,
            **generation_kwargs,
        )

    def __repr__(self) -> str:
        return f"VLM(model_id={self.model_id!r}, model_path={self.model_path!r})"


__all__ = ["VLM"]
