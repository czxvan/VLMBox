from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from vlmbox import VLM, ImageRenderer, ImageSample, ImageTrainer

DEFAULT_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"
DEFAULT_PROMPT = "Describe the image in one short sentence."
DEFAULT_IMAGE_INSTRUCTION = "Execute the instruction in the image."


def _messages(
    image: Image.Image,
    prompt: str,
    target: str | None = None,
    system_prompt: str | None = None,
) -> list[dict]:
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    )
    if target is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
    return messages


def _load_samples(
    image_dir: Path,
    *,
    prompt: str,
    image_instruction: str,
    target: str,
    limit: int,
) -> list[ImageSample]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in extensions)
    if limit:
        paths = paths[:limit]
    if not paths:
        raise ValueError(f"no supported images found in {image_dir}")

    samples = []
    for path in paths:
        with Image.open(path) as source:
            image = source.convert("RGB")
        samples.append(
            ImageSample(
                image=image,
                prompt=prompt,
                target=target,
                image_instruction=image_instruction,
            )
        )
    return samples


def _target_token_spans(
    vlm: VLM,
    renderer: ImageRenderer,
    samples: list[ImageSample],
) -> dict[int, tuple[int, int]]:
    """Return each sample's (unpadded prompt length, assistant target length)."""
    spans = {}
    for sample in samples:
        rendered = renderer.render(sample.image, sample.image_instruction)
        prompt_inputs = vlm.prepare(
            _messages(rendered, sample.prompt, system_prompt=sample.system_prompt)
        )
        target_inputs = vlm.prepare(
            _messages(
                rendered,
                sample.prompt,
                sample.target,
                system_prompt=sample.system_prompt,
            ),
            add_generation_prompt=False,
        )
        prompt_ids = prompt_inputs["input_ids"][0].detach().cpu()
        target_ids = target_inputs["input_ids"][0].detach().cpu()
        prefix_length = prompt_ids.numel()
        if target_ids.numel() <= prefix_length or not torch.equal(
            target_ids[:prefix_length], prompt_ids
        ):
            raise RuntimeError(
                "chat template did not preserve the prompt as an exact target prefix"
            )
        spans[id(sample)] = (prefix_length, target_ids.numel() - prefix_length)
    return spans


def _target_sequence_objective(target_spans: dict[int, tuple[int, int]]):
    def objective(outputs, inputs, samples) -> torch.Tensor:
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        total_loss = outputs.logits.new_zeros((), dtype=torch.float32)
        total_tokens = 0
        for index, sample in enumerate(samples):
            prefix_length, target_length = target_spans[id(sample)]
            valid_positions = attention_mask[index].nonzero(as_tuple=False).flatten()
            if valid_positions.numel() == 0:
                raise RuntimeError("conversation has no unmasked tokens")
            target_start = int(valid_positions[0].item()) + prefix_length
            target_end = target_start + target_length
            labels = input_ids[index, target_start:target_end]
            mask = attention_mask[index, target_start:target_end].bool()
            logits = outputs.logits[index, target_start - 1 : target_end - 1, :].float()
            if labels.numel() != target_length or logits.shape[0] != target_length:
                raise RuntimeError("assistant target token span is empty or misaligned")
            if not bool(mask.any().item()):
                raise RuntimeError("assistant target token span contains no valid tokens")
            total_loss = total_loss + F.cross_entropy(logits[mask], labels[mask], reduction="sum")
            total_tokens += int(mask.sum().item())
        if total_tokens == 0:
            raise RuntimeError("assistant target batch contains no valid tokens")
        return total_loss / total_tokens

    return objective


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize a YAML-configured universal image patch or perturbation."
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument(
        "--render-config",
        type=Path,
        default=Path(__file__).with_name("image_render.yaml"),
        help="YAML selecting patch or local additive perturbation settings.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--image-instruction", default=DEFAULT_IMAGE_INSTRUCTION)
    parser.add_argument("--target", default="HACKED")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("image_renderer_result.png"))
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.steps, args.batch_size) <= 0 or args.limit < 0:
        raise ValueError("steps and batch-size must be positive; limit >= 0")
    if not args.image_dir.is_dir():
        raise NotADirectoryError(args.image_dir)
    if not args.render_config.is_file():
        raise FileNotFoundError(args.render_config)

    torch.manual_seed(args.seed)
    samples = _load_samples(
        args.image_dir,
        prompt=args.prompt,
        image_instruction=args.image_instruction,
        target=args.target,
        limit=args.limit,
    )
    vlm = VLM.from_pretrained(args.model_id, local_files_only=args.local_files_only)
    vlm.model.eval()
    print(f"Model device: {vlm.input_device}; samples: {len(samples)}")

    renderer = ImageRenderer.from_yaml(args.render_config)
    target_spans = _target_token_spans(vlm, renderer, samples)
    print(
        "Teacher-forced assistant target spans: "
        + ", ".join(str(span[1]) for span in target_spans.values())
        + " tokens"
    )
    trainer = ImageTrainer(
        vlm,
        renderer,
        _target_sequence_objective(target_spans),
        learning_rate=args.learning_rate,
    )
    history = trainer.fit(
        samples,
        max_steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    preview, metadata = renderer.render(
        samples[0].image,
        samples[0].image_instruction,
        return_metadata=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    preview.save(args.output)
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "prompt": args.prompt,
                "image_instruction": args.image_instruction,
                "target": args.target,
                "learning_rate": args.learning_rate,
                "seed": args.seed,
                "training": [asdict(step) for step in history],
                "render": metadata,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Finished {len(history)} steps; final loss={history[-1].loss:.6f}; "
        f"gradient_norm={history[-1].gradient_norm:.6f}"
    )
    print(f"Saved rendered preview to {args.output}")
    print(f"Saved run metadata to {metadata_path}")


if __name__ == "__main__":
    main()
