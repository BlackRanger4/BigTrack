from __future__ import annotations

from typing import Sequence, Tuple

from BigTracker.types import Box, Point, Size


def box_to_center_size(box: Box) -> tuple[Point, Size]:
    """Convert x, y, width, height into center point and size."""

    x, y, width, height = box
    return (
        (float(x) + float(width) / 2.0, float(y) + float(height) / 2.0),
        (float(width), float(height)),
    )


def center_size_to_box(center: Point, size: Size) -> Box:
    """Convert center point and size into x, y, width, height."""

    return (
        float(center[0]) - float(size[0]) / 2.0,
        float(center[1]) - float(size[1]) / 2.0,
        float(size[0]),
        float(size[1]),
    )


def clip_box(box: Box, image_shape: Sequence[int], margin: float = 0.0) -> Box:
    """Clip a frame-coordinate box to image bounds while preserving positive size."""

    if len(image_shape) < 2:
        raise ValueError("image_shape must contain height and width")

    image_height = float(image_shape[0])
    image_width = float(image_shape[1])
    x, y, width, height = (float(value) for value in box)

    min_x = float(margin)
    min_y = float(margin)
    max_x = max(min_x, image_width - float(margin))
    max_y = max(min_y, image_height - float(margin))

    left = _clamp(x, min_x, max_x)
    top = _clamp(y, min_y, max_y)
    right = _clamp(x + max(width, 1.0), min_x, max_x)
    bottom = _clamp(y + max(height, 1.0), min_y, max_y)

    if right <= left:
        right = min(max_x, left + 1.0)
        left = max(min_x, right - 1.0)
    if bottom <= top:
        bottom = min(max_y, top + 1.0)
        top = max(min_y, bottom - 1.0)

    return (left, top, right - left, bottom - top)


def map_crop_box_back(
    pred_box_cxcywh: Box,
    crop_center: Point,
    search_size: float,
    resize_factor: float,
) -> Box:
    """Map crop-coordinate cx, cy, width, height back to frame xywh."""

    if resize_factor <= 0.0:
        raise ValueError("resize_factor must be positive")

    cx, cy, width, height = (float(value) for value in pred_box_cxcywh)
    half_side = 0.5 * float(search_size) / float(resize_factor)
    cx_real = cx + (float(crop_center[0]) - half_side)
    cy_real = cy + (float(crop_center[1]) - half_side)
    return (
        cx_real - 0.5 * width,
        cy_real - 0.5 * height,
        width,
        height,
    )


def _clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp one float into a closed interval."""

    return max(minimum, min(maximum, value))

