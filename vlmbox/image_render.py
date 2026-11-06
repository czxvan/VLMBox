"""YAML-configured visual input rendering and intervention pipeline."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml
from PIL import Image
from torch import Tensor, nn

from vlmbox.image import anchored_bbox, to_rgb
from vlmbox.render import (
    BandPosition,
    InstructionStyle,
    RenderedVisualInput,
    compose_instruction_image,
    compose_instruction_tensor,
    render_instruction_band,
)
from vlmbox.training.image import UniversalLocalPerturbation, UniversalPatch

InterventionType = Literal["none", "overlay_patch", "local_additive"]
Anchor = str | tuple[int, int]


def _mapping(
    value: Any, name: str, *, required: set[str], optional: set[str] | None = None
) -> dict:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a YAML mapping")
    optional = optional or set()
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing:
        raise ValueError(f"{name} is missing required keys: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{name} contains unknown keys: {sorted(unknown)}")
    return value


def _integer(value: Any, name: str, *, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _rgb(value: Any, name: str) -> tuple[int, int, int]:
    if isinstance(value, str):
        text = value.removeprefix("#")
        if len(text) != 6:
            raise ValueError(f"{name} must be a #RRGGBB color or a three-item RGB list")
        try:
            return tuple(int(text[index : index + 2], 16) for index in (0, 2, 4))
        except ValueError as exc:
            raise ValueError(f"{name} must be a valid #RRGGBB color") from exc
    if isinstance(value, list) and len(value) == 3:
        channels = tuple(_integer(channel, f"{name} channel", minimum=0) for channel in value)
        if any(channel > 255 for channel in channels):
            raise ValueError(f"{name} channels must be <= 255")
        return channels
    raise ValueError(f"{name} must be a #RRGGBB color or a three-item RGB list")


def _anchor(value: Any) -> Anchor:
    valid = {"top_left", "top_right", "bottom_left", "bottom_right", "center"}
    if isinstance(value, str):
        if value not in valid:
            raise ValueError(f"anchor must be one of {sorted(valid)}")
        return value
    if isinstance(value, list) and len(value) == 2:
        x, y = value
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in (x, y)):
            raise ValueError("an explicit anchor must contain non-negative integer [x, y]")
        return x, y
    raise ValueError("anchor must be a named anchor or an explicit [x, y] coordinate")


def _size(value: Any) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("intervention.size must be [height, width] in absolute pixels")
    return (
        _integer(value[0], "intervention.size[0] (height)", minimum=1),
        _integer(value[1], "intervention.size[1] (width)", minimum=1),
    )


class ImageRenderPipeline(nn.Module):
    """Execute one complete image-rendering configuration loaded from YAML.

    The YAML file is the user-facing configuration surface. Trainable intervention
    parameters are registered as child modules, so this object can be optimized,
    moved between devices, checkpointed, and passed directly to ImageTrainer.
    """

    schema_version = 1
    canvas_size = None

    def __init__(
        self,
        *,
        style: InstructionStyle,
        instruction_position: BandPosition,
        intervention_type: InterventionType,
        intervention_size: tuple[int, int] | None,
        intervention_anchor: Anchor | None,
        epsilon: float | None,
        init: Literal["zeros", "random"],
        seed: int,
        config_sha256: str,
        config_source: Path,
    ) -> None:
        super().__init__()
        self.style = style
        self.instruction_position = instruction_position
        self.intervention_type = intervention_type
        self.intervention_size = intervention_size
        self.intervention_anchor = intervention_anchor
        self.epsilon = epsilon
        self.config_sha256 = config_sha256
        self.config_source = config_source
        # Training revisits the same sample instructions across optimization steps.
        # Cache bounded CPU band tensors to avoid rerunning Pillow/font layout each step.
        self._band_cache: OrderedDict[tuple[int, str, bool], Tensor] = OrderedDict()
        self._band_cache_bytes = 0

        if intervention_type == "overlay_patch":
            assert intervention_size is not None and intervention_anchor is not None
            self.intervention: UniversalPatch | UniversalLocalPerturbation | None = UniversalPatch(
                canvas_size=None,
                size=intervention_size,
                position=intervention_anchor,
                init=init,
                seed=seed,
            )
        elif intervention_type == "local_additive":
            assert (
                intervention_size is not None
                and intervention_anchor is not None
                and epsilon is not None
            )
            self.intervention = UniversalLocalPerturbation(
                size=intervention_size,
                epsilon=epsilon,
                position=intervention_anchor,
                init=init,
                seed=seed,
            )
        else:
            self.intervention = None

    @classmethod
    def from_yaml(
        cls, path: str | Path, *, device: torch.device | str | None = None
    ) -> ImageRenderPipeline:
        """Load and strictly validate a complete ImageRenderPipeline YAML config.

        Relative font paths are resolved relative to the YAML file's directory.
        Unknown keys and unsupported schema versions fail instead of being ignored.
        """
        config_path = Path(path).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"image render config does not exist: {config_path}")
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid YAML in {config_path}: {exc}") from exc

        root = _mapping(raw, "config", required={"schema_version", "image_render"})
        version = _integer(root["schema_version"], "schema_version", minimum=1)
        if version != cls.schema_version:
            raise ValueError(
                f"unsupported ImageRenderPipeline schema_version {version}; "
                f"supported version is {cls.schema_version}"
            )

        section = _mapping(
            root["image_render"],
            "image_render",
            required={"instruction", "host", "intervention"},
        )
        instruction = _mapping(
            section["instruction"],
            "image_render.instruction",
            required={
                "position",
                "font",
                "foreground",
                "background",
                "alignment",
                "margin",
                "line_spacing",
            },
        )
        font = _mapping(
            instruction["font"],
            "image_render.instruction.font",
            required={"path", "size"},
        )
        font_path_value = font["path"]
        if not isinstance(font_path_value, str) or not font_path_value.strip():
            raise ValueError("image_render.instruction.font.path must be a non-empty path")
        font_path = Path(font_path_value).expanduser()
        if not font_path.is_absolute():
            font_path = (config_path.parent / font_path).resolve()
        else:
            font_path = font_path.resolve()
        if not font_path.is_file():
            raise FileNotFoundError(f"instruction font file does not exist: {font_path}")

        position = instruction["position"]
        if not isinstance(position, str) or position not in {"top", "bottom"}:
            raise ValueError("image_render.instruction.position must be 'top' or 'bottom'")
        alignment = instruction["alignment"]
        if not isinstance(alignment, str) or alignment not in {"left", "center", "right"}:
            raise ValueError("image_render.instruction.alignment must be left, center, or right")
        style = InstructionStyle(
            font_path=font_path,
            font_size=_integer(font["size"], "font.size", minimum=1),
            margin=_integer(instruction["margin"], "instruction.margin", minimum=0),
            line_spacing=_integer(
                instruction["line_spacing"], "instruction.line_spacing", minimum=0
            ),
            foreground=_rgb(instruction["foreground"], "instruction.foreground"),
            background=_rgb(instruction["background"], "instruction.background"),
            alignment=alignment,
        )

        host = _mapping(
            section["host"],
            "image_render.host",
            required={"preserve_original_size", "color_space"},
        )
        if host["preserve_original_size"] is not True:
            raise ValueError("VIRP requires host.preserve_original_size: true")
        if host["color_space"] != "RGB":
            raise ValueError("VIRP currently requires host.color_space: RGB")

        intervention = _mapping(
            section["intervention"],
            "image_render.intervention",
            required={"type"},
            optional={"size", "anchor", "epsilon", "init", "seed"},
        )
        intervention_type = intervention["type"]
        if not isinstance(intervention_type, str) or intervention_type not in {
            "none",
            "overlay_patch",
            "local_additive",
        }:
            raise ValueError("intervention.type must be none, overlay_patch, or local_additive")

        intervention_size = None
        intervention_anchor = None
        epsilon = None
        init = intervention.get(
            "init", "random" if intervention_type == "overlay_patch" else "zeros"
        )
        seed = _integer(intervention.get("seed", 2026), "intervention.seed", minimum=0)
        if seed > 2**63 - 1:
            raise ValueError("intervention.seed must be <= 2**63 - 1")
        if intervention_type == "none":
            if set(intervention) != {"type"}:
                raise ValueError("intervention.type=none does not accept other intervention keys")
        else:
            required = {"type", "size", "anchor"}
            missing = required - intervention.keys()
            if missing:
                raise ValueError(f"intervention is missing required keys: {sorted(missing)}")
            intervention_size = _size(intervention["size"])
            intervention_anchor = _anchor(intervention["anchor"])
            if not isinstance(init, str) or init not in {"zeros", "random"}:
                raise ValueError("intervention.init must be zeros or random")
            if intervention_type == "local_additive":
                if "epsilon" not in intervention:
                    raise ValueError("local_additive requires intervention.epsilon")
                epsilon_value = intervention["epsilon"]
                if isinstance(epsilon_value, bool) or not isinstance(epsilon_value, (int, float)):
                    raise ValueError("intervention.epsilon must be a number")
                epsilon = float(epsilon_value)
                if not 0.0 < epsilon <= 1.0:
                    raise ValueError("intervention.epsilon must be in (0, 1]")
            elif "epsilon" in intervention:
                raise ValueError("overlay_patch does not accept epsilon")

        config_digest = hashlib.sha256(
            json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
                "utf-8"
            )
        ).hexdigest()
        pipeline = cls(
            style=style,
            instruction_position=position,
            intervention_type=intervention_type,
            intervention_size=intervention_size,
            intervention_anchor=intervention_anchor,
            epsilon=epsilon,
            init=init,
            seed=seed,
            config_sha256=config_digest,
            config_source=config_path,
        )
        return pipeline.to(device) if device is not None else pipeline

    def _intervention_bbox(self, host_size: tuple[int, int]) -> tuple[int, int, int, int] | None:
        if self.intervention_size is None or self.intervention_anchor is None:
            return None
        height, width = self.intervention_size
        return anchored_bbox(host_size, (width, height), self.intervention_anchor)

    def _apply_intervention(self, host: Tensor) -> Tensor:
        if self.intervention is None:
            return host
        return self.intervention.apply(host)

    def _instruction_band_tensor(
        self,
        width: int,
        instruction: str,
        *,
        draw_text: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        key = (width, instruction, draw_text)
        band_tensor = self._band_cache.get(key)
        if band_tensor is None:
            band = render_instruction_band(
                instruction,
                width=width,
                style=self.style,
                draw_text=draw_text,
            )
            band_array = np.asarray(band.image, dtype=np.uint8).copy()
            band_tensor = torch.from_numpy(band_array).permute(2, 0, 1).contiguous()
            self._band_cache[key] = band_tensor
            self._band_cache_bytes += band_tensor.numel()
            while len(self._band_cache) > 256 or self._band_cache_bytes > 128 * 1024 * 1024:
                _, evicted = self._band_cache.popitem(last=False)
                self._band_cache_bytes -= evicted.numel()
        else:
            self._band_cache.move_to_end(key)
        return band_tensor.to(device=device, dtype=dtype).div(255)

    def render_tensor(
        self,
        host: Tensor,
        instruction: str,
        *,
        draw_text: bool = True,
    ) -> Tensor:
        """Render an RGB CHW float host Tensor and preserve intervention gradients.

        Host tensors are expected to be RGB floats in [0, 1]. ImageSample performs
        this validation once when samples enter ImageTrainer, avoiding GPU syncs
        inside every optimization step.
        """
        if not isinstance(host, Tensor) or host.ndim != 3 or host.shape[0] != 3:
            raise ValueError("host must be a CHW RGB Tensor")
        if not host.is_floating_point():
            raise TypeError("host must use a floating-point dtype")
        if host.shape[-2] <= 0 or host.shape[-1] <= 0:
            raise ValueError("host must have non-empty spatial dimensions")

        modified = self._apply_intervention(host)
        band_tensor = self._instruction_band_tensor(
            width=host.shape[-1],
            instruction=instruction,
            draw_text=draw_text,
            device=host.device,
            dtype=host.dtype,
        )
        return compose_instruction_tensor(
            modified,
            band_tensor,
            position=self.instruction_position,
        )

    def render(
        self,
        host: Image.Image,
        instruction: str,
        *,
        draw_text: bool = True,
    ) -> RenderedVisualInput:
        """Render a PIL host image, applying the configured intervention if present."""
        rgb = to_rgb(host)
        if self.intervention is None:
            modified = rgb
        else:
            host_array = np.asarray(rgb, dtype=np.uint8).copy()
            host_tensor = torch.from_numpy(host_array).permute(2, 0, 1).to(torch.float32).div(255)
            modified_tensor = self._apply_intervention(host_tensor)
            modified_array = (
                modified_tensor.detach()
                .cpu()
                .clamp(0.0, 1.0)
                .mul(255)
                .round()
                .to(torch.uint8)
                .permute(1, 2, 0)
                .numpy()
            )
            modified = Image.fromarray(modified_array, mode="RGB")

        rendered = compose_instruction_image(
            modified,
            instruction,
            self.style,
            position=self.instruction_position,
            draw_text=draw_text,
        )
        local_bbox = self._intervention_bbox(rgb.size)
        canvas_bbox = None
        if local_bbox is not None:
            offset_x, offset_y = rendered.host_bbox[:2]
            canvas_bbox = (
                local_bbox[0] + offset_x,
                local_bbox[1] + offset_y,
                local_bbox[2] + offset_x,
                local_bbox[3] + offset_y,
            )
        return replace(
            rendered,
            intervention_bbox=canvas_bbox,
            intervention_type=self.intervention_type,
            config_sha256=self.config_sha256,
            instruction_sha256=hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
        )

    def project_(self) -> None:
        """Project the configured trainable intervention to its feasible set."""
        if self.intervention is not None:
            self.intervention.project_()

    def forward(self, host: Tensor, instruction: str, *, draw_text: bool = True) -> Tensor:
        return self.render_tensor(host, instruction, draw_text=draw_text)


__all__ = ["ImageRenderPipeline"]
