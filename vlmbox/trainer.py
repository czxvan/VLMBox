"""Universal image-parameter optimization against a frozen vision-language model."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from PIL import Image
from torch import Tensor
from torch.optim import Optimizer

from vlmbox.model import VLM
from vlmbox.renderer import Renderer


@dataclass(frozen=True)
class Sample:
    """One native-size PIL host image and its conversation/in-image instruction."""

    image: Image.Image
    prompt: str
    system_prompt: str | None = None
    target: str | None = None
    image_instruction: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")
        if self.image.width <= 0 or self.image.height <= 0:
            raise ValueError("image must be non-empty")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if self.system_prompt is not None and (
            not isinstance(self.system_prompt, str) or not self.system_prompt.strip()
        ):
            raise ValueError("system_prompt must be a non-empty string when provided")
        if self.target is not None and (
            not isinstance(self.target, str) or not self.target.strip()
        ):
            raise ValueError("target must be a non-empty string when provided")
        if self.image_instruction is not None and (
            not isinstance(self.image_instruction, str) or not self.image_instruction.strip()
        ):
            raise ValueError("image_instruction must be a non-empty string when provided")


@dataclass(frozen=True)
class TrainStep:
    """Scalar diagnostics recorded after one universal image optimizer update."""

    step: int
    loss: float
    gradient_norm: float


class LossFunction(Protocol):
    """Compute a scalar minimization loss from one prepared model forward.

    Inputs and samples share the output batch order. Inputs contain processed
    tensors such as token IDs and attention masks; samples contain original
    prompts and targets. Functions and callable objects satisfy this protocol.
    """

    def __call__(
        self,
        outputs: Any,
        inputs: Mapping[str, Any],
        samples: Sequence[Sample],
        /,
    ) -> Tensor:
        """Return a finite scalar Tensor connected to the intervention parameters."""
        ...


class Trainer:
    """Optimize only a Renderer's intervention parameters against a frozen VLM.

    Host images remain PIL images in the dataset. Renderer creates the
    differentiable Tensor representation internally for each optimization step.
    VLM weights are frozen during optimization and their original requires_grad
    and training/evaluation states are restored afterwards.
    """

    def __init__(
        self,
        vlm: VLM,
        renderer: Renderer,
        loss_function: LossFunction,
        *,
        learning_rate: float = 1e-2,
        optimizer_cls: type[Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(renderer, Renderer):
            raise TypeError("renderer must be a Renderer loaded from YAML")
        if not callable(loss_function):
            raise TypeError("loss_function must be callable")
        if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
            raise TypeError("learning_rate must be a real number")
        learning_rate = float(learning_rate)
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not isinstance(optimizer_cls, type) or not issubclass(optimizer_cls, Optimizer):
            raise TypeError("optimizer_cls must be a torch.optim.Optimizer class")

        parameters = list(renderer.parameters())
        if not parameters:
            raise ValueError(
                "renderer has no trainable intervention; configure overlay_patch or local_additive"
            )
        options = dict(optimizer_kwargs or {})
        options.setdefault("lr", learning_rate)
        self.vlm = vlm
        self.renderer = renderer
        self.loss_function = loss_function
        self.optimizer = optimizer_cls(parameters, **options)

    def fit(
        self,
        samples: Sequence[Sample],
        *,
        max_steps: int,
        batch_size: int = 1,
        shuffle: bool = True,
        seed: int = 2026,
        forward_kwargs: Mapping[str, Any] | None = None,
    ) -> tuple[TrainStep, ...]:
        """Run exactly max_steps updates, cycling through the samples as needed."""
        if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)) or not samples:
            raise ValueError("samples must be a non-empty sequence of Sample objects")
        if any(not isinstance(sample, Sample) for sample in samples):
            raise TypeError("every sample must be a Sample")
        for name, value in (("max_steps", max_steps), ("batch_size", batch_size)):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be a bool")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer")
        if not 0 <= seed <= 2**63 - 1:
            raise ValueError("seed must be between 0 and 2**63 - 1")

        if any(sample.image_instruction is None for sample in samples):
            raise ValueError("every sample must provide image_instruction for Renderer")
        target_modes = {sample.target is not None for sample in samples}
        if len(target_modes) != 1:
            raise ValueError("all samples must either provide target text or omit it")
        add_generation_prompt = not next(iter(target_modes))
        parameters = tuple(self.renderer.parameters())
        device = parameters[0].device
        options = {"use_cache": False}
        options.update(dict(forward_kwargs or {}))

        generator = torch.Generator(device="cpu").manual_seed(seed)
        order = list(range(len(samples)))
        cursor = len(order)
        history: list[TrainStep] = []
        model_parameters = list(self.vlm.model.parameters())
        original_requires_grad = [parameter.requires_grad for parameter in model_parameters]
        original_training = [(module, module.training) for module in self.vlm.model.modules()]

        try:
            for parameter in model_parameters:
                parameter.requires_grad_(False)
            self.vlm.model.eval()
            for step in range(1, max_steps + 1):
                if cursor >= len(order):
                    order = (
                        torch.randperm(len(samples), generator=generator).tolist()
                        if shuffle
                        else list(range(len(samples)))
                    )
                    cursor = 0
                batch_indices = order[cursor : cursor + batch_size]
                cursor += len(batch_indices)
                batch = [samples[index] for index in batch_indices]

                conversations = []
                for sample in batch:
                    rendered_image = self.renderer(sample.image, sample.image_instruction)
                    messages = []
                    if sample.system_prompt is not None:
                        messages.append(
                            {
                                "role": "system",
                                "content": [{"type": "text", "text": sample.system_prompt}],
                            }
                        )
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": rendered_image},
                                {"type": "text", "text": sample.prompt},
                            ],
                        }
                    )
                    if sample.target is not None:
                        messages.append(
                            {
                                "role": "assistant",
                                "content": [{"type": "text", "text": sample.target}],
                            }
                        )
                    conversations.append(messages)
                inputs = self.vlm.prepare_differentiable(
                    conversations,
                    add_generation_prompt=add_generation_prompt,
                )
                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.vlm.forward(inputs, **options)
                loss_value = self.loss_function(outputs, inputs, batch)
                if not isinstance(loss_value, Tensor):
                    raise TypeError("loss_function must return a torch.Tensor")
                if loss_value.ndim != 0:
                    raise ValueError("loss_function must return a scalar Tensor")
                if not loss_value.requires_grad:
                    raise ValueError(
                        "loss_function must retain gradients to intervention parameters"
                    )
                if not bool(torch.isfinite(loss_value.detach()).item()):
                    raise FloatingPointError("loss_function returned a non-finite value")

                loss_value.backward()
                squared_norm = torch.zeros((), device=device, dtype=torch.float32)
                has_gradient = False
                for parameter in parameters:
                    if parameter.grad is not None:
                        has_gradient = True
                        squared_norm = squared_norm + parameter.grad.detach().float().square().sum()
                if not has_gradient:
                    raise RuntimeError(
                        "loss_function produced no gradient for the renderer intervention"
                    )
                gradient_norm = float(torch.sqrt(squared_norm).detach().cpu())
                if not math.isfinite(gradient_norm):
                    raise FloatingPointError("renderer intervention gradient is non-finite")

                self.optimizer.step()
                self.renderer.project_()
                self.optimizer.zero_grad(set_to_none=True)
                history.append(
                    TrainStep(
                        step=step,
                        loss=float(loss_value.detach().cpu()),
                        gradient_norm=gradient_norm,
                    )
                )
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            for parameter, requires_grad in zip(
                model_parameters, original_requires_grad, strict=True
            ):
                parameter.requires_grad_(requires_grad)
            for module, training in original_training:
                module.training = training

        return tuple(history)


__all__ = ["Sample", "Trainer", "TrainStep", "LossFunction"]
