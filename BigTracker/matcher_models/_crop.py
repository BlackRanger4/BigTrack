from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from BigTracker.types import Box, Point, Size


@dataclass(frozen=True)
class CropResult:
    """Image crop with geometry needed to map predictions back."""

    image: Any
    resize_factor: float
    crop_box: Box
    is_clipped: bool
    attention_mask: Optional[Any] = None


def crop_centered(
    image: Any,
    center: Point,
    size: Size,
    output_size: Optional[int] = None,
    pad_value: Any = 0,
) -> CropResult:
    """Crop around a center point and pad when the crop crosses image bounds."""

    np = _require_numpy()
    array = np.asarray(image)
    if array.ndim < 2:
        raise ValueError("image must have at least height and width dimensions")

    crop_width = max(1, int(round(float(size[0]))))
    crop_height = max(1, int(round(float(size[1]))))
    left = int(round(float(center[0]) - crop_width / 2.0))
    top = int(round(float(center[1]) - crop_height / 2.0))
    return _crop_xywh(
        array,
        left=left,
        top=top,
        width=crop_width,
        height=crop_height,
        output_size=output_size,
        pad_value=pad_value,
    )


def sample_target(
    image: Any,
    target_box: Box,
    search_area_factor: float,
    output_size: Optional[int] = None,
    pad_value: Any = 0,
) -> CropResult:
    """Extract an OSTrack-style square crop around a target box."""

    x, y, width, height = (float(value) for value in target_box)
    crop_side = _target_crop_side(width, height, search_area_factor)
    center = (x + width / 2.0, y + height / 2.0)
    return crop_centered(
        image=image,
        center=center,
        size=(crop_side, crop_side),
        output_size=output_size,
        pad_value=pad_value,
    )


def _crop_xywh(
    image: Any,
    left: int,
    top: int,
    width: int,
    height: int,
    output_size: Optional[int],
    pad_value: Any,
) -> CropResult:
    """Crop a rectangle from an array and pad with a constant value."""

    np = _require_numpy()
    image_height, image_width = image.shape[:2]
    right = left + width
    bottom = top + height

    src_left = max(0, left)
    src_top = max(0, top)
    src_right = min(image_width, right)
    src_bottom = min(image_height, bottom)

    crop_shape = (height, width) + tuple(image.shape[2:])
    crop = np.empty(crop_shape, dtype=image.dtype)
    crop[...] = pad_value

    attention_mask = np.ones((height, width), dtype=bool)
    if src_right > src_left and src_bottom > src_top:
        dst_left = src_left - left
        dst_top = src_top - top
        dst_right = dst_left + (src_right - src_left)
        dst_bottom = dst_top + (src_bottom - src_top)
        crop[dst_top:dst_bottom, dst_left:dst_right, ...] = image[
            src_top:src_bottom,
            src_left:src_right,
            ...,
        ]
        attention_mask[dst_top:dst_bottom, dst_left:dst_right] = False

    clipped = src_left != left or src_top != top or src_right != right or src_bottom != bottom
    resize_factor = 1.0
    if output_size is not None:
        output_size = max(1, int(output_size))
        resize_factor = output_size / float(width)
        crop = _resize_square(crop, output_size)
        attention_mask = _resize_square(attention_mask.astype("uint8"), output_size).astype(bool)

    return CropResult(
        image=crop,
        resize_factor=resize_factor,
        crop_box=(float(left), float(top), float(width), float(height)),
        is_clipped=clipped,
        attention_mask=attention_mask,
    )


def _resize_square(image: Any, output_size: int) -> Any:
    """Resize an image to a square output."""

    cv2 = _require_cv2()
    return cv2.resize(image, (output_size, output_size))


def _target_crop_side(width: float, height: float, factor: float) -> float:
    """Return the square crop side used by modern one-stream trackers."""

    import math

    side = math.ceil(math.sqrt(max(width, 1.0) * max(height, 1.0)) * float(factor))
    if side < 1:
        raise ValueError("target box is too small for cropping")
    return float(side)


def _require_numpy() -> Any:
    """Import numpy only when crop utilities are used."""

    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("matcher crop utilities require numpy") from error
    return np


def _require_cv2() -> Any:
    """Import OpenCV only when resizing is used."""

    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("matcher crop resizing requires opencv-python") from error
    return cv2
