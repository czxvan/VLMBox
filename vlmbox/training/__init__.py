"""Universal image optimization APIs."""

from vlmbox.training.image import (
    UniversalImageParameter,
    UniversalPatch,
    UniversalPerturbation,
)
from vlmbox.training.trainer import ImageSample, ImageTrainer, TrainingStep

__all__ = [
    "ImageSample",
    "ImageTrainer",
    "TrainingStep",
    "UniversalImageParameter",
    "UniversalPatch",
    "UniversalPerturbation",
]
