"""YAML-configured rendering and reusable image intervention weights."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torch import Tensor, nn

from .intervention import Intervention
from .layout import (
    BandPosition,
    InstructionStyle,
    compose_instruction_image,
    image_tensor,
    render_instruction_band,
    to_rgb,
)


def config_mapping(
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


def config_integer(value: Any, name: str, *, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def config_rgb(value: Any, name: str) -> tuple[int, int, int]:
    if isinstance(value, str):
        text = value.removeprefix("#")
        if len(text) != 6:
            raise ValueError(f"{name} must be a #RRGGBB color or a three-item RGB list")
        try:
            return tuple(int(text[index : index + 2], 16) for index in (0, 2, 4))
        except ValueError as exc:
            raise ValueError(f"{name} must be a valid #RRGGBB color") from exc
    if isinstance(value, list) and len(value) == 3:
        channels = tuple(config_integer(channel, f"{name} channel", minimum=0) for channel in value)
        if any(channel > 255 for channel in channels):
            raise ValueError(f"{name} channels must be <= 255")
        return channels
    raise ValueError(f"{name} must be a #RRGGBB color or a three-item RGB list")


class Renderer(nn.Module):
    """YAML-configured image renderer with an optional trainable intervention.

    Load from YAML, render PIL images, and save or load intervention weights.
    Calling the module returns a differentiable CHW Tensor. render() returns PIL.
    Standard parameters() and to() support custom research optimization loops.
    """

    schema_version = 1

    def __init__(
        self,
        *,
        style: InstructionStyle,
        instruction_position: BandPosition,
        intervention: Intervention | None,
        config_sha256: str,
    ) -> None:
        super().__init__()
        self.style = style
        self.instruction_position = instruction_position
        self.config_sha256 = config_sha256
        # Bound CPU typography tensors retained across optimization steps.
        self._band_cache: OrderedDict[tuple[int, str, bool], Tensor] = OrderedDict()
        self._band_cache_bytes = 0
        self.intervention = intervention

    @classmethod
    def from_yaml(cls, path: str | Path, *, device: torch.device | str | None = None) -> Renderer:
        """Load and strictly validate a complete Renderer YAML config.

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

        root = config_mapping(raw, "config", required={"schema_version", "image_render"})
        version = config_integer(root["schema_version"], "schema_version", minimum=1)
        if version != cls.schema_version:
            raise ValueError(
                f"unsupported Renderer schema_version {version}; "
                f"supported version is {cls.schema_version}"
            )
        section = config_mapping(
            root["image_render"],
            "image_render",
            required={"instruction", "host", "intervention"},
        )
        instruction = config_mapping(
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
        font = config_mapping(
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
            font_size=config_integer(font["size"], "font.size", minimum=1),
            margin=config_integer(instruction["margin"], "instruction.margin", minimum=0),
            line_spacing=config_integer(
                instruction["line_spacing"], "instruction.line_spacing", minimum=0
            ),
            foreground=config_rgb(instruction["foreground"], "instruction.foreground"),
            background=config_rgb(instruction["background"], "instruction.background"),
            alignment=alignment,
        )

        host = config_mapping(
            section["host"],
            "image_render.host",
            required={"preserve_original_size", "color_space"},
        )
        if host["preserve_original_size"] is not True:
            raise ValueError("VIRP requires host.preserve_original_size: true")
        if host["color_space"] != "RGB":
            raise ValueError("VIRP currently requires host.color_space: RGB")

        intervention = config_mapping(
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
        seed = config_integer(intervention.get("seed", 2026), "intervention.seed", minimum=0)
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
            size = intervention["size"]
            if not isinstance(size, list) or len(size) != 2:
                raise ValueError("intervention.size must be [height, width] in absolute pixels")
            intervention_size = tuple(
                config_integer(value, f"intervention.size[{index}]", minimum=1)
                for index, value in enumerate(size)
            )
            anchor = intervention["anchor"]
            intervention_anchor = tuple(anchor) if isinstance(anchor, list) else anchor
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
        renderer = cls(
            style=style,
            instruction_position=position,
            intervention=(
                None
                if intervention_type == "none"
                else Intervention(
                    intervention_type,
                    intervention_size,
                    anchor=intervention_anchor,
                    epsilon=epsilon,
                    init=init,
                    seed=seed,
                )
            ),
            config_sha256=config_digest,
        )
        return renderer.to(device) if device is not None else renderer

    def forward(
        self,
        host_image: Image.Image,
        instruction: str,
        *,
        draw_text: bool = True,
    ) -> Tensor:
        """Render a differentiable RGB CHW Tensor in [0, 1], retaining parameter gradients."""
        host = image_tensor(to_rgb(host_image))
        if self.intervention is not None:
            host = self.intervention(host.to(self.intervention.parameter))
        key = (host.shape[-1], instruction, draw_text)
        band = self._band_cache.get(key)
        if band is None:
            rendered_band = render_instruction_band(
                instruction, width=host.shape[-1], style=self.style, draw_text=draw_text
            )
            band = torch.from_numpy(np.asarray(rendered_band.image).copy()).permute(2, 0, 1)
            self._band_cache[key] = band
            self._band_cache_bytes += band.numel()
            while len(self._band_cache) > 256 or self._band_cache_bytes > 128 * 1024 * 1024:
                _, evicted = self._band_cache.popitem(last=False)
                self._band_cache_bytes -= evicted.numel()
        else:
            self._band_cache.move_to_end(key)
        band = band.to(host).div(255)
        parts = (band, host) if self.instruction_position == "top" else (host, band)
        return torch.cat(parts, dim=1)

    def render(
        self,
        host_image: Image.Image,
        instruction: str,
        *,
        draw_text: bool = True,
        return_metadata: bool = False,
    ) -> Image.Image | tuple[Image.Image, dict[str, object]]:
        """Render a PIL host image and return a PIL image.

        Metadata is returned as a plain dictionary only when explicitly requested.
        """
        if not isinstance(host_image, Image.Image):
            raise TypeError("host_image must be a PIL.Image.Image")
        if not isinstance(return_metadata, bool):
            raise TypeError("return_metadata must be a bool")

        rgb = to_rgb(host_image)
        if self.intervention is None:
            modified = rgb
        else:
            with torch.no_grad():
                modified_tensor = self.intervention(
                    image_tensor(rgb).to(self.intervention.parameter)
                )
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
        if not return_metadata:
            return rendered.image
        local_bbox = self.intervention.bbox(rgb.size) if self.intervention is not None else None
        canvas_bbox = None
        if local_bbox is not None:
            offset_x, offset_y = rendered.host_bbox[:2]
            canvas_bbox = (
                local_bbox[0] + offset_x,
                local_bbox[1] + offset_y,
                local_bbox[2] + offset_x,
                local_bbox[3] + offset_y,
            )
        metadata = rendered.metadata()
        metadata.update(
            intervention_bbox=list(canvas_bbox) if canvas_bbox is not None else None,
            intervention_type=self.intervention.kind if self.intervention is not None else "none",
            config_sha256=self.config_sha256,
            instruction_sha256=hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
        )
        if self.intervention is not None:
            parameter = self.intervention.parameter.detach().cpu().contiguous()
            metadata["intervention_sha256"] = hashlib.sha256(
                parameter.view(torch.uint8).numpy().tobytes()
            ).hexdigest()
        return rendered.image, metadata

    def save_intervention(self, path: str | Path) -> None:
        """Save intervention weights and geometry independently of hosts and typography."""
        if self.intervention is None:
            raise ValueError("renderer has no intervention to save or load")
        spec = self.intervention.spec
        torch.save(
            {
                "format": "vlmbox-intervention-1",
                "spec": spec,
                "parameter": self.intervention.parameter.detach().cpu(),
            },
            path,
        )

    def load_intervention(self, path: str | Path) -> None:
        """Restore compatible weights in place; reject invalid or out-of-bounds values."""
        if self.intervention is None:
            raise ValueError("renderer has no intervention to save or load")
        spec = self.intervention.spec
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or checkpoint.get("format") != "vlmbox-intervention-1":
            raise ValueError("unsupported intervention checkpoint format")
        if checkpoint.get("spec") != spec:
            raise ValueError(
                "checkpoint intervention type, size, anchor and epsilon must match YAML"
            )
        parameter = self.intervention.parameter
        value = checkpoint.get("parameter")
        if not isinstance(value, Tensor) or not value.is_floating_point():
            raise ValueError("checkpoint parameter must be a floating-point Tensor")
        if value.shape != parameter.shape:
            raise ValueError("checkpoint parameter shape does not match intervention")
        value = value.to(dtype=parameter.dtype)
        lower, upper = self.intervention.bounds
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError("checkpoint parameter must contain finite values")
        if bool(((value < lower) | (value > upper)).any().item()):
            raise ValueError("checkpoint parameter exceeds intervention bounds")
        with torch.no_grad():
            parameter.copy_(value)

    def project_(self) -> None:
        """Project optimized weights back into the configured intervention bounds."""
        if self.intervention is not None:
            self.intervention.project_()


__all__ = ["Renderer"]
