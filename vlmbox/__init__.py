"""VLMBox: a lightweight research toolbox for vision-language models."""

from vlmbox.generation import (
    FinishReason,
    GenerationPhase,
    GenerationStep,
    GenerationTrace,
    GenerationTraceError,
)
from vlmbox.image import anchored_bbox, crop_image, place_patch, resize_image, to_rgb
from vlmbox.image_render import ImageRenderPipeline
from vlmbox.model import VLM
from vlmbox.render import (
    InstructionBand,
    InstructionStyle,
    RenderedVisualInput,
    compose_instruction_image,
    compose_instruction_tensor,
    render_instruction_band,
)
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
from vlmbox.training import (
    ImageSample,
    ImageTrainer,
    TrainingStep,
    UniversalImageParameter,
    UniversalLocalPerturbation,
    UniversalPatch,
    UniversalPerturbation,
)

__all__ = [
    "VLM",
    "set_seed",
    "ImageSample",
    "ImageTrainer",
    "TrainingStep",
    "UniversalImageParameter",
    "UniversalLocalPerturbation",
    "UniversalPatch",
    "UniversalPerturbation",
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
    "ImageRenderPipeline",
    "InstructionBand",
    "InstructionStyle",
    "RenderedVisualInput",
    "render_instruction_band",
    "compose_instruction_image",
    "compose_instruction_tensor",
    "angle",
    "cosine_similarity",
    "mean_difference",
    "normalize_direction",
    "project",
    "projection_coefficient",
    "projection_length",
]
__version__ = "0.1.0"
