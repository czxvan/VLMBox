"""Architecture-independent image operations built on Pillow."""

from __future__ import annotations

from PIL import Image


def _require_image(name: str, image: Image.Image) -> None:
    if not isinstance(image, Image.Image):
        raise TypeError(f"{name} must be a PIL.Image.Image")


def _integer_tuple(name: str, value: tuple[int, ...], length: int) -> tuple[int, ...]:
    if not isinstance(value, tuple) or len(value) != length:
        raise TypeError(f"{name} must be a tuple of {length} integers")
    if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        raise TypeError(f"{name} must contain only integers")
    return value


def to_rgb(image: Image.Image) -> Image.Image:
    """Return an independent RGB copy of a PIL image.

    Pillow's standard RGB conversion is used; an existing alpha channel is
    discarded rather than composited onto a background.
    """
    _require_image("image", image)
    return image.copy() if image.mode == "RGB" else image.convert("RGB")


def resize_image(
    image: Image.Image,
    size: tuple[int, int],
    resample: Image.Resampling = Image.Resampling.LANCZOS,
) -> Image.Image:
    """Resize to (width, height) and return an RGB image.

    Aspect ratio is not preserved automatically. LANCZOS is used by default;
    pass another Pillow Image.Resampling value if needed.
    """
    _require_image("image", image)
    if image.width <= 0 or image.height <= 0:
        raise ValueError("source image must be non-empty")
    width, height = _integer_tuple("size", size, 2)
    if width <= 0 or height <= 0:
        raise ValueError("size dimensions must be positive")

    if isinstance(resample, bool):
        raise TypeError("resample must be a Pillow Image.Resampling value")
    try:
        resample = Image.Resampling(resample)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unsupported resampling filter: {resample!r}") from exc

    return to_rgb(image).resize((width, height), resample=resample)


def crop_image(
    image: Image.Image,
    box: tuple[int, int, int, int],
) -> Image.Image:
    """Extract an RGB crop using (left, top, right, bottom) half-open coordinates.

    Coordinates are measured from the top-left corner. The crop must be
    non-empty and fully inside the source image; implicit padding is rejected.
    """
    _require_image("image", image)
    left, top, right, bottom = _integer_tuple("box", box, 4)
    width, height = image.size
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError(
            f"box {box!r} must be non-empty and inside image bounds (0, 0, {width}, {height})"
        )
    return to_rgb(image).crop((left, top, right, bottom))


def place_patch(
    image: Image.Image,
    patch: Image.Image,
    position: tuple[int, int],
    mask: Image.Image | None = None,
) -> Image.Image:
    """Place an RGB patch at (left, top) on an RGB copy of image.

    The patch is pasted at its original size; resizing and clipping are not
    implicit. An optional same-size L or 1 mask controls blending: zero keeps
    the base pixel and 255 replaces it with the patch pixel. Neither input
    image is modified.
    """
    _require_image("image", image)
    _require_image("patch", patch)
    left, top = _integer_tuple("position", position, 2)

    base = to_rgb(image)
    patch_rgb = to_rgb(patch)
    base_width, base_height = base.size
    patch_width, patch_height = patch_rgb.size
    if base_width <= 0 or base_height <= 0:
        raise ValueError("base image must be non-empty")
    if patch_width <= 0 or patch_height <= 0:
        raise ValueError("patch must be non-empty")

    if left < 0 or top < 0:
        raise ValueError("position coordinates must be non-negative")
    if left + patch_width > base_width or top + patch_height > base_height:
        raise ValueError(
            f"patch at {position!r} with size {patch_rgb.size} exceeds "
            f"base image bounds {base.size}"
        )

    if mask is not None:
        _require_image("mask", mask)
        if mask.mode not in ("1", "L"):
            raise ValueError("mask must have mode '1' or 'L'")
        if mask.size != patch.size:
            raise ValueError(f"mask size {mask.size} must match patch size {patch.size}")
        mask = mask.convert("L")

    base.paste(patch_rgb, (left, top), mask)
    return base


def anchored_bbox(
    image_size: tuple[int, int],
    region_size: tuple[int, int],
    anchor: str | tuple[int, int] = "bottom_right",
) -> tuple[int, int, int, int]:
    """Resolve a half-open host-local bbox using PIL (width, height) dimensions.

    Corner/center anchors use no margins. Explicit positions use (x, y).
    """
    width, height = _integer_tuple("image_size", image_size, 2)
    region_width, region_height = _integer_tuple("region_size", region_size, 2)
    if min(width, height, region_width, region_height) <= 0:
        raise ValueError("image and region dimensions must be positive")
    if region_width > width or region_height > height:
        raise ValueError("region must fit inside host image")
    if isinstance(anchor, str):
        positions = {
            "top_left": (0, 0),
            "top_right": (width - region_width, 0),
            "bottom_left": (0, height - region_height),
            "bottom_right": (width - region_width, height - region_height),
            "center": ((width - region_width) // 2, (height - region_height) // 2),
        }
        if anchor not in positions:
            raise ValueError(f"unknown anchor {anchor!r}")
        x, y = positions[anchor]
    else:
        x, y = _integer_tuple("anchor", anchor, 2)
    if x < 0 or y < 0 or x + region_width > width or y + region_height > height:
        raise ValueError("anchored region is outside host image; must fit inside")
    return x, y, x + region_width, y + region_height


__all__ = ["anchored_bbox", "crop_image", "place_patch", "resize_image", "to_rgb"]
