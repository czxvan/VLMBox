from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import center_crop, resize

from vlmbox import VLM, ImageSample, ImageTrainer, UniversalPatch, UniversalPerturbation

DEFAULT_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"
DEFAULT_PROMPT = "Describe the image in one short sentence."


def _messages(image: torch.Tensor, prompt: str, target: str | None = None) -> list[dict]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
    return messages


def _load_samples(
    image_dir: Path, *, canvas_size: int, prompt: str, target: str, limit: int
) -> list[ImageSample]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in extensions)
    if limit:
        paths = paths[:limit]
    if not paths:
        raise ValueError(f"no supported images found in {image_dir}")

    samples = []
    for path in paths:
        image = read_image(str(path), mode=ImageReadMode.RGB).float().div_(255.0)
        image = resize(
            image, [canvas_size], interpolation=InterpolationMode.BICUBIC, antialias=True
        )
        image = center_crop(image, [canvas_size, canvas_size]).clamp_(0.0, 1.0).contiguous()
        samples.append(ImageSample(image=image, prompt=prompt, target=target))
    return samples


def _target_prefix_length(vlm: VLM, sample: ImageSample) -> tuple[int, int]:
    prompt_inputs = vlm.prepare_differentiable(_messages(sample.image, sample.prompt))
    target_inputs = vlm.prepare_differentiable(
        _messages(sample.image, sample.prompt, sample.target),
        add_generation_prompt=False,
    )
    prompt_ids = prompt_inputs["input_ids"][0].detach().cpu()
    target_ids = target_inputs["input_ids"][0].detach().cpu()
    prefix_length = prompt_ids.numel()
    if target_ids.numel() <= prefix_length or not torch.equal(
        target_ids[:prefix_length], prompt_ids
    ):
        raise RuntimeError("chat template did not preserve the prompt as an exact target prefix")
    return prefix_length, target_ids.numel() - prefix_length


def _target_sequence_objective(prefix_length: int):
    def objective(outputs, inputs, _samples) -> torch.Tensor:
        labels = inputs["input_ids"][:, prefix_length:]
        mask = inputs["attention_mask"][:, prefix_length:].bool()
        logits = outputs.logits[:, prefix_length - 1 : -1, :].float()
        if logits.shape[:2] != labels.shape or not bool(mask.any().item()):
            raise RuntimeError("assistant target token span is empty or misaligned")
        return F.cross_entropy(logits[mask], labels[mask])

    return objective


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize a universal image perturbation or patch with target-sequence CE."
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--parameterization", choices=("perturbation", "patch"), default="patch")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--canvas-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=48)
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--target", default="HACKED")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("universal_image_parameter.pt"))
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.steps, args.batch_size, args.canvas_size, args.patch_size) <= 0 or args.limit < 0:
        raise ValueError(
            "steps, batch-size, canvas-size and patch-size must be positive; limit >= 0"
        )
    if not args.image_dir.is_dir():
        raise NotADirectoryError(args.image_dir)

    torch.manual_seed(args.seed)
    samples = _load_samples(
        args.image_dir,
        canvas_size=args.canvas_size,
        prompt=args.prompt,
        target=args.target,
        limit=args.limit,
    )
    vlm = VLM.from_pretrained(args.model_id, local_files_only=args.local_files_only)
    vlm.model.eval()
    print(f"Model device: {vlm.input_device}; samples: {len(samples)}")
    if hasattr(vlm.processor.image_processor, "do_image_splitting"):
        print(f"Native do_image_splitting: {vlm.processor.image_processor.do_image_splitting}")

    canvas = (args.canvas_size, args.canvas_size)
    if args.parameterization == "perturbation":
        image_parameter = UniversalPerturbation(
            canvas, epsilon=args.epsilon, seed=args.seed, device="cpu"
        )
    else:
        patch_size = min(args.patch_size, args.canvas_size)
        image_parameter = UniversalPatch(
            canvas,
            (patch_size, patch_size),
            position="bottom_right",
            seed=args.seed,
            device="cpu",
        )

    prefix_length, target_length = _target_prefix_length(vlm, samples[0])
    print(f"Teacher-forced assistant target span: {target_length} tokens")
    trainer = ImageTrainer(
        vlm,
        image_parameter,
        _target_sequence_objective(prefix_length),
        learning_rate=args.learning_rate,
    )
    history = trainer.fit(samples, max_steps=args.steps, batch_size=args.batch_size, seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_id": args.model_id,
            "parameterization": args.parameterization,
            "canvas_size": canvas,
            "prompt": args.prompt,
            "target": args.target,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "state_dict": {
                key: value.detach().cpu() for key, value in image_parameter.state_dict().items()
            },
            "history": [asdict(step) for step in history],
        },
        args.output,
    )
    print(
        f"Finished {len(history)} steps; final loss={history[-1].loss:.6f}; "
        f"gradient_norm={history[-1].gradient_norm:.6f}"
    )
    print(f"Saved universal parameter to {args.output}")


if __name__ == "__main__":
    main()
