"""VLMBox: a lightweight research toolbox for vision-language models."""

from vlmbox.generation import (
    FinishReason,
    GenerationPhase,
    GenerationStep,
    GenerationTrace,
    GenerationTraceError,
)
from vlmbox.image import anchored_bbox, crop_image, place_patch, resize_image, to_rgb
from vlmbox.image_render import ImageRenderer
from vlmbox.model import VLM
from vlmbox.representation import (
    angle,
    cosine_similarity,
    mean_difference,
    normalize_direction,
    project,
    projection_coefficient,
    projection_length,
)
from vlmbox.repro import set_seed
from vlmbox.training import ImageSample, ImageTrainer, TrainingStep

__all__ = [
    "VLM",
    "set_seed",
    "ImageRenderer",
    "ImageSample",
    "ImageTrainer",
    "TrainingStep",
    "FinishReason",
    "GenerationPhase",
    "GenerationStep",
    "GenerationTrace",
    "GenerationTraceError",
    "anchored_bbox",
    "crop_image",
    "place_patch",
    "resize_image",
    "to_rgb",
    "angle",
    "cosine_similarity",
    "mean_difference",
    "normalize_direction",
    "project",
    "projection_coefficient",
    "projection_length",
]

__version__ = "0.1.0"
