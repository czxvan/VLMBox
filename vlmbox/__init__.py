"""VLMBox: a lightweight research toolbox for vision-language models."""

from vlmbox.model import VLM
from vlmbox.representation import (
    angle,
    cosine_similarity,
    mean_difference,
    normalize_direction,
    paired_difference,
    project,
    projection_coefficient,
    projection_length,
)

__all__ = [
    "VLM",
    "angle",
    "cosine_similarity",
    "mean_difference",
    "normalize_direction",
    "paired_difference",
    "project",
    "projection_coefficient",
    "projection_length",
]
__version__ = "0.1.0"
