"""Paired-prompt activation-steering probe built with public VLMBox APIs.

This is a small end-to-end research example, not a benchmark or a claim that
prompt-derived directions provide reliable control across inputs or models.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from vlmbox import VLM, set_seed
from vlmbox.utils import (
    cosine_similarity,
    mean_difference,
    normalize_direction,
    projection_length,
)


def _parse_scales(value: str) -> tuple[float, ...]:
    try:
        scales = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("scales must be comma-separated numbers") from exc
    if not scales or any(not math.isfinite(scale) for scale in scales):
        raise argparse.ArgumentTypeError("scales must contain finite numbers")
    if any(scale == 0 for scale in scales):
        raise argparse.ArgumentTypeError("scales must be non-zero")
    return scales


def _demo_image() -> Image.Image:
    image = Image.new("RGB", (384, 384), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((70, 90, 210, 230), fill=(220, 40, 40))
    draw.ellipse((230, 150, 330, 250), fill=(40, 80, 220))
    return image


def _message(image: Image.Image, prompt: str) -> list[dict[str, object]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _capture_last(vlm: VLM, image: Image.Image, prompt: str, layer: int) -> torch.Tensor:
    inputs = vlm.prepare(_message(image, prompt))
    with torch.inference_mode():
        with vlm.capture("block", layers=layer, positions="last") as cache:
            vlm.generate(inputs, max_new_tokens=1, do_sample=False)
    return cache[layer]


def _generate_text(vlm: VLM, inputs: object, max_new_tokens: int) -> str:
    sequences = vlm.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False)
    prompt_length = inputs["input_ids"].shape[1]  # type: ignore[index]
    generated = sequences[:, prompt_length:]
    return vlm.processor.batch_decode(generated, skip_special_tokens=True)[0]


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("This research example requires a CUDA GPU.")

    set_seed(args.seed)
    if args.image is None:
        image = _demo_image()
    else:
        with Image.open(args.image) as source:
            image = source.convert("RGB")

    vlm = VLM.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
    )
    layer = len(vlm.adapter.layers) // 2 if args.layer is None else args.layer
    if not 0 <= layer < len(vlm.adapter.layers):
        raise ValueError(f"layer must be in [0, {len(vlm.adapter.layers)}); received {layer}")

    positive_state = _capture_last(vlm, image, args.positive_prompt, layer)
    negative_state = _capture_last(vlm, image, args.negative_prompt, layer)
    direction = normalize_direction(mean_difference(positive_state, negative_state, dim=0))

    task_inputs = vlm.prepare(_message(image, args.task_prompt))
    with torch.inference_mode():
        baseline_text = _generate_text(vlm, task_inputs, args.max_new_tokens)

    interventions: list[dict[str, object]] = []
    for scale in args.scales:
        with torch.inference_mode():
            # Hook registration order lets the outer capture observe h and the
            # inner capture observe h + scale * direction.
            with vlm.capture("block", layers=layer, positions="last") as original:
                with vlm.intervene(
                    "block",
                    layer,
                    "add",
                    value=direction,
                    scale=scale,
                    positions="last",
                    scope=args.scope,
                ):
                    with vlm.capture("block", layers=layer, positions="last") as edited:
                        steered_text = _generate_text(vlm, task_inputs, args.max_new_tokens)

            delta = (edited[layer] - original[layer]).float()
            delta_norm = torch.linalg.vector_norm(delta, dim=-1)
            if torch.any(delta_norm > 0).item():
                measured_projection = projection_length(delta, direction)
                measured_cosine = cosine_similarity(delta, direction)
                projection_value = float(measured_projection.mean().cpu())
                cosine_value: float | None = float(measured_cosine.mean().cpu())
            else:
                projection_value = 0.0
                cosine_value = None

            interventions.append(
                {
                    "scale": scale,
                    "text": steered_text,
                    "prefill_activation_delta_norm": float(delta_norm.mean().cpu()),
                    "prefill_activation_delta_projection": projection_value,
                    "prefill_activation_delta_cosine": cosine_value,
                }
            )

    result: dict[str, object] = {
        "model_id": args.model_id,
        "device": str(next(vlm.model.parameters()).device),
        "dtype": str(next(vlm.model.parameters()).dtype),
        "decoder_layers": len(vlm.adapter.layers),
        "layer": layer,
        "seed": args.seed,
        "scope": args.scope,
        "activation_measurement_phase": "prompt_prefill",
        "max_new_tokens": args.max_new_tokens,
        "image": str(args.image) if args.image is not None else "synthetic-red-square-blue-circle",
        "positive_prompt": args.positive_prompt,
        "negative_prompt": args.negative_prompt,
        "task_prompt": args.task_prompt,
        "direction_norm": float(torch.linalg.vector_norm(direction, dim=-1).mean().cpu()),
        "baseline_text": baseline_text,
        "interventions": interventions,
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="ModelScope model ID (default: Qwen/Qwen2.5-VL-3B-Instruct).",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require the model to be present in the local ModelScope cache.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Optional input image; defaults to a synthetic image.",
    )
    parser.add_argument("--layer", type=int, help="Decoder layer; defaults to the middle layer.")
    parser.add_argument("--scales", type=_parse_scales, default=(0.5, 1.0, 2.0, 4.0))
    parser.add_argument(
        "--scope",
        choices=("first_forward", "every_forward"),
        default="every_forward",
    )
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--positive-prompt", default="Answer exactly RED.")
    parser.add_argument("--negative-prompt", default="Answer exactly BLUE.")
    parser.add_argument("--task-prompt", default="What color is the circle? Answer with one word.")
    parser.add_argument("--output", type=Path, help="Optional path for a JSON result.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
