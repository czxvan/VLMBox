"""Deterministic image-embedded text rendering without resizing the host image."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from PIL import Image, ImageDraw, ImageFont

from .image import to_rgb

Box = tuple[int, int, int, int]
BandPosition = Literal["top", "bottom"]


@dataclass(frozen=True)
class InstructionStyle:
    """Generic text rendering settings; project-specific values belong in YAML."""

    font_path: str | Path
    font_size: int
    margin: int = 0
    line_spacing: int = 0
    foreground: tuple[int, int, int] = (0, 0, 0)
    background: tuple[int, int, int] = (255, 255, 255)
    alignment: Literal["left", "center", "right"] = "left"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.font_size, int)
            or isinstance(self.font_size, bool)
            or self.font_size <= 0
        ):
            raise ValueError("font_size must be a positive integer")
        if not isinstance(self.margin, int) or isinstance(self.margin, bool) or self.margin < 0:
            raise ValueError("margin must be a non-negative integer")
        if (
            not isinstance(self.line_spacing, int)
            or isinstance(self.line_spacing, bool)
            or self.line_spacing < 0
        ):
            raise ValueError("line_spacing must be a non-negative integer")
        if self.alignment not in {"left", "center", "right"}:
            raise ValueError("alignment must be left, center, or right")
        for name in ("foreground", "background"):
            color = getattr(self, name)
            if (
                not isinstance(color, tuple)
                or len(color) != 3
                or any(
                    not isinstance(c, int) or isinstance(c, bool) or not 0 <= c <= 255
                    for c in color
                )
            ):
                raise ValueError(f"{name} must be an RGB integer tuple")

    @property
    def spacing(self) -> int:
        return self.line_spacing


@dataclass(frozen=True)
class InstructionBand:
    image: Image.Image
    lines: tuple[str, ...]
    font_size: int
    font_sha256: str


@dataclass(frozen=True)
class RenderedVisualInput:
    """Composed image and half-open boxes in final canvas coordinates."""

    image: Image.Image
    instruction_bbox: Box
    host_bbox: Box
    lines: tuple[str, ...]
    font_size: int
    font_sha256: str
    position: BandPosition
    intervention_bbox: Box | None = None
    intervention_type: str | None = None
    config_sha256: str | None = None
    instruction_sha256: str | None = None

    def metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {
            "renderer": "virp-1",
            "canvas_size": list(self.image.size),
            "instruction_bbox": list(self.instruction_bbox),
            "host_bbox": list(self.host_bbox),
            "host_size": [
                self.host_bbox[2] - self.host_bbox[0],
                self.host_bbox[3] - self.host_bbox[1],
            ],
            "line_count": len(self.lines),
            "font_size": self.font_size,
            "font_sha256": self.font_sha256,
            "instruction_position": self.position,
        }
        if self.intervention_type is not None:
            metadata["intervention_type"] = self.intervention_type
            metadata["intervention_bbox"] = (
                list(self.intervention_bbox) if self.intervention_bbox is not None else None
            )
        if self.config_sha256 is not None:
            metadata["config_sha256"] = self.config_sha256
        if self.instruction_sha256 is not None:
            metadata["instruction_sha256"] = self.instruction_sha256
        return metadata


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> tuple[str, ...]:
    """Width-aware word wrapping; split overlong words by character without clipping."""
    lines: list[str] = []
    for paragraph in text.splitlines():
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = ""
        for word in words:
            # Allow tokens without spaces (URLs, identifiers) to wrap instead of overflowing.
            pieces: list[str] = []
            while word:
                start, end = 1, len(word)
                while start < end:
                    middle = (start + end + 1) // 2
                    if font.getlength(word[:middle]) <= max_width:
                        start = middle
                    else:
                        end = middle - 1
                if font.getlength(word[:start]) > max_width:
                    raise ValueError("a single glyph is wider than the available text region")
                pieces.append(word[:start])
                word = word[start:]
            for i, piece in enumerate(pieces):
                candidate = f"{current} {piece}" if current and i == 0 else piece
                if current and font.getlength(candidate) > max_width:
                    lines.append(current)
                    current = piece
                elif current and i > 0:
                    lines.append(current)
                    current = piece
                else:
                    current = candidate
        lines.append(current)
    return tuple(lines)


def render_instruction_band(
    instruction: str,
    width: int,
    style: InstructionStyle,
    *,
    draw_text: bool = True,
) -> InstructionBand:
    """Render a full-width text band with fixed typography and content-dependent height.

    Explicit newlines are preserved. Otherwise, whitespace-separated text wraps like
    ARGUS; an overlong individual token is further split by character. No font scaling,
    truncation, host-image resizing, or implicit font substitution occurs.
    """
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be nonempty text")
    if not isinstance(width, int) or isinstance(width, bool) or width <= 2 * style.margin:
        raise ValueError("width must exceed twice the horizontal margin")
    path = Path(style.font_path)
    if not path.is_file():
        raise FileNotFoundError(f"font file does not exist: {path}")
    font = ImageFont.truetype(str(path), size=style.font_size)
    lines = _wrap_text(instruction, font, width - 2 * style.margin)
    height = (
        2 * style.margin + len(lines) * style.font_size + max(0, len(lines) - 1) * style.spacing
    )
    band = Image.new("RGB", (width, height), style.background)
    if draw_text:
        draw = ImageDraw.Draw(band)
        y = style.margin
        for line in lines:
            text_width = font.getlength(line)
            if style.alignment == "left":
                x = style.margin
            elif style.alignment == "right":
                x = int(width - style.margin - text_width)
            else:
                x = int((width - text_width) // 2)
            draw.text((x, y), line, fill=style.foreground, font=font)
            y += style.font_size + style.spacing
    return InstructionBand(
        image=band,
        lines=lines,
        font_size=style.font_size,
        font_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def compose_instruction_image(
    host: Image.Image,
    instruction: str,
    style: InstructionStyle,
    *,
    position: BandPosition = "top",
    draw_text: bool = True,
) -> RenderedVisualInput:
    """Compose an instruction band with an untouched-size RGB host image."""
    if position not in ("top", "bottom"):
        raise ValueError("position must be top or bottom")
    rgb = to_rgb(host)
    if rgb.width <= 0 or rgb.height <= 0:
        raise ValueError("host image must be non-empty")
    band = render_instruction_band(instruction, rgb.width, style, draw_text=draw_text)
    h = band.image.height
    canvas = Image.new("RGB", (rgb.width, rgb.height + h), style.background)
    if position == "top":
        canvas.paste(band.image, (0, 0))
        canvas.paste(rgb, (0, h))
        text_bbox, host_bbox = (0, 0, rgb.width, h), (0, h, rgb.width, h + rgb.height)
    else:
        canvas.paste(rgb, (0, 0))
        canvas.paste(band.image, (0, rgb.height))
        text_bbox, host_bbox = (
            (0, rgb.height, rgb.width, rgb.height + h),
            (0, 0, rgb.width, rgb.height),
        )
    return RenderedVisualInput(
        canvas,
        text_bbox,
        host_bbox,
        band.lines,
        band.font_size,
        band.font_sha256,
        position,
    )


def compose_instruction_tensor(
    host: torch.Tensor, band: torch.Tensor, *, position: BandPosition = "top"
) -> torch.Tensor:
    """Differentiably concatenate CHW host and CHW instruction tensors."""
    if position not in ("top", "bottom"):
        raise ValueError("position must be top or bottom")
    if not isinstance(host, torch.Tensor) or not isinstance(band, torch.Tensor):
        raise TypeError("host and band must be tensors")
    if host.ndim != 3 or band.ndim != 3 or host.shape[0] != 3 or band.shape[0] != 3:
        raise ValueError("host and band must be CHW RGB tensors")
    if host.shape[-1] != band.shape[-1]:
        raise ValueError("host and band widths must match")
    if host.device != band.device:
        raise ValueError("host and band must share a device")
    if host.dtype != band.dtype or not host.is_floating_point():
        raise ValueError("host and band must have the same floating-point dtype")
    return torch.cat((band, host) if position == "top" else (host, band), dim=1)


__all__ = [
    "InstructionStyle",
    "InstructionBand",
    "RenderedVisualInput",
    "render_instruction_band",
    "compose_instruction_image",
    "compose_instruction_tensor",
]
