"""VLMBox: a lightweight research toolbox for vision-language models."""

from vlmbox.image import crop_image, place_patch, resize_image, to_rgb
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

__all__ = [
    "VLM",
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
