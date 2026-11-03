"""Universal image optimization against a frozen vision-language model."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer

from vlmbox.model import VLM
from vlmbox.training.image import UniversalImageParameter


@dataclass(frozen=True)
class ImageSample:
    """One fixed RGB image and its text conversation for universal optimization."""

    image: Tensor
    prompt: str
    system_prompt: str | None = None
    target: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.image, Tensor):
            raise TypeError("image must be a torch.Tensor")
        if self.image.ndim != 3 or self.image.shape[0] != 3:
            raise ValueError("image must have shape [3, height, width]")
        if self.image.shape[-2] <= 0 or self.image.shape[-1] <= 0:
            raise ValueError("image must be non-empty")
        if not self.image.is_floating_point():
            raise TypeError("image must use a floating-point dtype")
        image = self.image.detach()
        if not bool(torch.isfinite(image).all().item()):
            raise ValueError("image must contain only finite values")
        if float(image.amin().item()) < 0.0 or float(image.amax().item()) > 1.0:
            raise ValueError("image values must be in [0, 1]")
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


@dataclass(frozen=True)
class TrainingStep:
    """Scalar diagnostics recorded after one universal image optimizer update."""

    step: int
    loss: float
    gradient_norm: float


Objective = Callable[[Any, Mapping[str, Any], Sequence[ImageSample]], Tensor]


class ImageTrainer:
    """Optimize one shared perturbation or patch over a fixed-resolution dataset.

    The objective receives ``(model_outputs, model_inputs, samples)`` and must
    return one differentiable scalar Tensor. The VLM is put in eval mode and its
    parameters are frozen during ``fit``; only the universal image parameter is
    updated. Native model preprocessing runs on CPU and remains in the autograd
    graph before processed tensors are transferred to the model device.
    """

    def __init__(
        self,
        vlm: VLM,
        image_parameter: UniversalImageParameter,
        objective: Objective,
        *,
        learning_rate: float = 1e-2,
        optimizer_cls: type[Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if not callable(objective):
            raise TypeError("objective must be callable")
        if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
            raise TypeError("learning_rate must be a real number")
        learning_rate = float(learning_rate)
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not isinstance(image_parameter, UniversalImageParameter):
            raise TypeError("image_parameter must be UniversalPerturbation or UniversalPatch")
        if not isinstance(optimizer_cls, type) or not issubclass(optimizer_cls, Optimizer):
            raise TypeError("optimizer_cls must be a torch.optim.Optimizer class")

        parameters = list(image_parameter.parameters())
        if not parameters:
            raise ValueError("image_parameter must expose trainable parameters")
        options = dict(optimizer_kwargs or {})
        options.setdefault("lr", learning_rate)
        self.vlm = vlm
        self.image_parameter = image_parameter
        self.objective = objective
        self.optimizer = optimizer_cls(parameters, **options)

    @staticmethod
    def _conversation(sample: ImageSample, image: Tensor) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if sample.system_prompt is not None:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": sample.system_prompt}]}
            )
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": sample.prompt},
                ],
            }
        )
        if sample.target is not None:
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": sample.target}]}
            )
        return messages

    def fit(
        self,
        samples: Sequence[ImageSample],
        *,
        max_steps: int,
        batch_size: int = 1,
        shuffle: bool = True,
        seed: int = 2026,
        forward_kwargs: Mapping[str, Any] | None = None,
    ) -> tuple[TrainingStep, ...]:
        """Run exactly ``max_steps`` updates, cycling through the samples as needed."""
        if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)) or not samples:
            raise ValueError("samples must be a non-empty sequence of ImageSample objects")
        if any(not isinstance(sample, ImageSample) for sample in samples):
            raise TypeError("every sample must be an ImageSample")
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

        canvas_sizes = {tuple(sample.image.shape[-2:]) for sample in samples}
        if len(canvas_sizes) != 1:
            raise ValueError("all training images must have the same canvas size")
        canvas_size = next(iter(canvas_sizes))
        if canvas_size != self.image_parameter.canvas_size:
            raise ValueError(
                f"dataset canvas {canvas_size} does not match image parameter canvas "
                f"{self.image_parameter.canvas_size}"
            )

        target_modes = {sample.target is not None for sample in samples}
        if len(target_modes) != 1:
            raise ValueError("all samples must either provide target text or omit it")
        add_generation_prompt = not next(iter(target_modes))
        device = next(self.image_parameter.parameters()).device
        options = {"use_cache": False}
        options.update(dict(forward_kwargs or {}))

        generator = torch.Generator(device="cpu").manual_seed(seed)
        order = list(range(len(samples)))
        cursor = len(order)
        history: list[TrainingStep] = []
        model_parameters = list(self.vlm.model.parameters())
        original_requires_grad = [parameter.requires_grad for parameter in model_parameters]
        original_training = self.vlm.model.training

        for parameter in model_parameters:
            parameter.requires_grad_(False)
        try:
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

                images = torch.stack(
                    [
                        sample.image.detach().to(device=device, dtype=torch.float32)
                        for sample in batch
                    ]
                )
                attacked = self.image_parameter.apply(images)
                conversations = [
                    self._conversation(sample, attacked[index])
                    for index, sample in enumerate(batch)
                ]
                inputs = self.vlm.prepare_differentiable(
                    conversations,
                    add_generation_prompt=add_generation_prompt,
                )
                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.vlm.forward(inputs, **options)
                loss = self.objective(outputs, inputs, batch)
                if not isinstance(loss, Tensor):
                    raise TypeError("objective must return a torch.Tensor")
                if loss.ndim != 0:
                    raise ValueError("objective must return a scalar Tensor")
                if not loss.requires_grad:
                    raise ValueError(
                        "objective must retain gradients to the universal image parameter"
                    )
                if not bool(torch.isfinite(loss.detach()).item()):
                    raise FloatingPointError("objective returned a non-finite loss")

                loss.backward()
                squared_norm = torch.zeros((), device=device, dtype=torch.float32)
                has_gradient = False
                for parameter in self.image_parameter.parameters():
                    if parameter.grad is not None:
                        has_gradient = True
                        squared_norm = squared_norm + parameter.grad.detach().float().square().sum()
                if not has_gradient:
                    raise RuntimeError(
                        "objective produced no gradient for the universal image parameter"
                    )
                gradient_norm = float(torch.sqrt(squared_norm).detach().cpu())
                if not math.isfinite(gradient_norm):
                    raise FloatingPointError("universal image gradient is non-finite")

                self.optimizer.step()
                self.image_parameter.project_()
                self.optimizer.zero_grad(set_to_none=True)
                history.append(
                    TrainingStep(
                        step=step,
                        loss=float(loss.detach().cpu()),
                        gradient_norm=gradient_norm,
                    )
                )
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            for parameter, requires_grad in zip(
                model_parameters, original_requires_grad, strict=True
            ):
                parameter.requires_grad_(requires_grad)
            self.vlm.model.train(original_training)

        return tuple(history)


__all__ = ["ImageSample", "ImageTrainer", "Objective", "TrainingStep"]
