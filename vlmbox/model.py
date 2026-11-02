"""Unified Transformers runtime for ModelScope-hosted vision-language models."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vlmbox.adapter import ModelAdapter, TransformersAdapter


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
    def from_pretrained(cls, model_id: str) -> VLM:
        """Download a ModelScope checkpoint and load it with Transformers.

        The public API intentionally accepts only a ModelScope model ID. Device
        placement and dtype use fixed internal defaults to keep the interface small.
        """
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

        model_path = str(snapshot_download(model_id.strip()))
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

    def __repr__(self) -> str:
        return f"VLM(model_id={self.model_id!r}, model_path={self.model_path!r})"
