from __future__ import annotations

import pytest
from PIL import Image

import vlmbox
from vlmbox.image import crop_image, place_patch, resize_image, to_rgb


def test_image_helpers_are_exported_from_package_root() -> None:
    assert vlmbox.crop_image is crop_image
    assert vlmbox.place_patch is place_patch
    assert vlmbox.resize_image is resize_image
    assert vlmbox.to_rgb is to_rgb


def test_to_rgb_converts_mode_and_returns_independent_copy() -> None:
    source = Image.new("RGBA", (3, 2), (10, 20, 30, 40))

    result = to_rgb(source)

    assert result.mode == "RGB"
    assert result.size == source.size
    assert result.getpixel((0, 0)) == (10, 20, 30)
    result.putpixel((0, 0), (255, 0, 0))
    assert source.getpixel((0, 0)) == (10, 20, 30, 40)


def test_to_rgb_copies_existing_rgb_image() -> None:
    source = Image.new("RGB", (2, 2), (1, 2, 3))

    result = to_rgb(source)
    result.putpixel((0, 0), (9, 9, 9))

    assert result is not source
    assert source.getpixel((0, 0)) == (1, 2, 3)


def test_to_rgb_rejects_non_pillow_input() -> None:
    with pytest.raises(TypeError, match="PIL.Image"):
        to_rgb("image.png")  # type: ignore[arg-type]


def test_resize_image_uses_width_height_order_and_returns_rgb() -> None:
    source = Image.new("L", (2, 3), 128)

    result = resize_image(source, (5, 4))

    assert result.mode == "RGB"
    assert result.size == (5, 4)
    assert source.mode == "L"
    assert source.size == (2, 3)


@pytest.mark.parametrize("size", [(0, 2), (2, 0), (-1, 2)])
def test_resize_image_rejects_non_positive_dimensions(size: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="positive"):
        resize_image(Image.new("RGB", (2, 2)), size)


@pytest.mark.parametrize("size", [(2,), (2, 3, 4), (True, 2)])
def test_resize_image_rejects_invalid_size(size: tuple[int, ...]) -> None:
    with pytest.raises(TypeError, match="size"):
        resize_image(Image.new("RGB", (2, 2)), size)  # type: ignore[arg-type]


def test_resize_image_rejects_invalid_resampling_filter() -> None:
    with pytest.raises(ValueError, match="resampling"):
        resize_image(
            Image.new("RGB", (2, 2)),
            (4, 4),
            resample=999,  # type: ignore[arg-type]
        )


def test_resize_image_rejects_empty_source() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        resize_image(Image.new("RGB", (0, 0)), (4, 4))


def test_crop_image_uses_half_open_coordinates() -> None:
    source = Image.new("RGB", (4, 3))
    source.putpixel((1, 1), (255, 0, 0))
    source.putpixel((2, 2), (0, 255, 0))

    result = crop_image(source, (1, 1, 3, 3))

    assert result.mode == "RGB"
    assert result.size == (2, 2)
    assert result.getpixel((0, 0)) == (255, 0, 0)
    assert result.getpixel((1, 1)) == (0, 255, 0)
    assert source.size == (4, 3)


@pytest.mark.parametrize(
    "box",
    [
        (-1, 0, 2, 2),
        (0, -1, 2, 2),
        (0, 0, 5, 2),
        (0, 0, 2, 4),
        (2, 0, 2, 2),
        (0, 2, 2, 2),
    ],
)
def test_crop_image_rejects_empty_or_out_of_bounds_boxes(
    box: tuple[int, int, int, int],
) -> None:
    with pytest.raises(ValueError, match="box"):
        crop_image(Image.new("RGB", (4, 3)), box)


def test_crop_image_rejects_non_integer_coordinates() -> None:
    with pytest.raises(TypeError, match="integers"):
        crop_image(
            Image.new("RGB", (4, 3)),
            (0, 0, 2, 2.5),  # type: ignore[arg-type]
        )


def test_place_patch_places_without_mutating_inputs() -> None:
    base = Image.new("RGB", (4, 3), (0, 0, 0))
    patch = Image.new("RGB", (2, 2), (255, 0, 0))

    result = place_patch(base, patch, (1, 1))

    assert result.mode == "RGB"
    assert result.size == base.size
    assert result.getpixel((1, 1)) == (255, 0, 0)
    assert result.getpixel((2, 2)) == (255, 0, 0)
    assert result.getpixel((0, 0)) == (0, 0, 0)
    assert base.getpixel((1, 1)) == (0, 0, 0)
    assert patch.getpixel((0, 0)) == (255, 0, 0)
    assert result is not base


def test_place_patch_blends_using_l_mask() -> None:
    base = Image.new("RGB", (3, 1), (0, 0, 0))
    patch = Image.new("RGB", (3, 1), (255, 255, 255))
    mask = Image.new("L", (3, 1))
    mask.putdata([0, 128, 255])

    result = place_patch(base, patch, (0, 0), mask=mask)

    assert result.getpixel((0, 0)) == (0, 0, 0)
    assert result.getpixel((1, 0)) == (128, 128, 128)
    assert result.getpixel((2, 0)) == (255, 255, 255)
    assert base.getpixel((1, 0)) == (0, 0, 0)


def test_place_patch_accepts_binary_mask() -> None:
    base = Image.new("RGB", (2, 1), (0, 0, 0))
    patch = Image.new("RGB", (2, 1), (255, 0, 0))
    mask = Image.new("1", (2, 1))
    mask.putdata([0, 1])

    result = place_patch(base, patch, (0, 0), mask=mask)

    assert result.getpixel((0, 0)) == (0, 0, 0)
    assert result.getpixel((1, 0)) == (255, 0, 0)


def test_place_patch_rejects_out_of_bounds_placement() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        place_patch(
            Image.new("RGB", (4, 3)),
            Image.new("RGB", (2, 2)),
            (3, 1),
        )


def test_place_patch_rejects_empty_base_or_patch() -> None:
    with pytest.raises(ValueError, match="base image"):
        place_patch(Image.new("RGB", (0, 0)), Image.new("RGB", (1, 1)), (0, 0))
    with pytest.raises(ValueError, match="patch"):
        place_patch(Image.new("RGB", (1, 1)), Image.new("RGB", (0, 0)), (0, 0))


@pytest.mark.parametrize("position", [(-1, 0), (0, -1), (True, 0)])
def test_place_patch_rejects_invalid_position(position: tuple[int, int]) -> None:
    with pytest.raises((TypeError, ValueError)):
        place_patch(
            Image.new("RGB", (4, 3)),
            Image.new("RGB", (1, 1)),
            position,
        )


def test_place_patch_rejects_mask_with_wrong_mode_or_size() -> None:
    base = Image.new("RGB", (4, 3))
    patch = Image.new("RGB", (2, 2))

    with pytest.raises(ValueError, match="mode"):
        place_patch(base, patch, (0, 0), mask=Image.new("RGB", (2, 2)))
    with pytest.raises(ValueError, match="match patch size"):
        place_patch(base, patch, (0, 0), mask=Image.new("L", (1, 1)))


def test_place_patch_rejects_non_image_patch() -> None:
    with pytest.raises(TypeError, match="PIL.Image"):
        place_patch(
            Image.new("RGB", (4, 3)),
            object(),  # type: ignore[arg-type]
            (0, 0),
        )
