from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

from BigTracker.state import BigTrackState
from BigTracker.types import FrameLike, Point, Size


FrameShape = Tuple[float, float]


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(float(minimum), min(float(maximum), float(value)))


def clamp01(value: float) -> float:
    return clamp(value, 0.0, 1.0)


def frame_dt(state: BigTrackState, frame: FrameLike) -> float:
    previous_timestamp = state.output.timestamp
    if frame.timestamp > previous_timestamp:
        return max(1e-6, frame.timestamp - previous_timestamp)

    previous_idx = state.output.frame_idx
    if frame.idx > previous_idx:
        return float(frame.idx - previous_idx)

    return 1.0


def frame_shape_from_frame(frame: FrameLike) -> Optional[FrameShape]:
    image = getattr(frame, "image", None)
    shape = getattr(image, "shape", None)
    if shape is None or len(shape) < 2:
        return None

    height = float(shape[0])
    width = float(shape[1])
    if width <= 0.0 or height <= 0.0:
        return None
    return width, height


def frame_shape_from_metadata(metadata: Mapping[str, Any], key: str) -> Optional[FrameShape]:
    value = metadata.get(key)
    if value is None:
        return None
    if len(value) < 2:
        return None

    width = float(value[0])
    height = float(value[1])
    if width <= 0.0 or height <= 0.0:
        return None
    return width, height


def put_frame_shape(metadata: dict[str, Any], key: str, frame: FrameLike) -> None:
    shape = frame_shape_from_frame(frame)
    if shape is not None:
        metadata[key] = shape


def clamp_center_size(
    target_pos: Point,
    target_size: Size,
    *,
    frame_shape: Optional[FrameShape],
    min_size: Size,
    clamp_to_frame: bool,
) -> tuple[Point, Size]:
    size = clamp_size_to_frame(
        target_size,
        frame_shape=frame_shape,
        min_size=min_size,
        clamp_to_frame=clamp_to_frame,
    )
    center = clamp_center_to_frame(
        target_pos,
        size,
        frame_shape=frame_shape,
        clamp_to_frame=clamp_to_frame,
    )
    return center, size


def clamp_size_to_frame(
    target_size: Size,
    *,
    frame_shape: Optional[FrameShape],
    min_size: Size,
    clamp_to_frame: bool,
) -> Size:
    width = max(float(min_size[0]), float(target_size[0]))
    height = max(float(min_size[1]), float(target_size[1]))

    if not clamp_to_frame or frame_shape is None:
        return width, height

    frame_width, frame_height = frame_shape
    max_width = max(1.0, frame_width)
    max_height = max(1.0, frame_height)
    return (
        clamp(width, min(float(min_size[0]), max_width), max_width),
        clamp(height, min(float(min_size[1]), max_height), max_height),
    )


def clamp_center_to_frame(
    target_pos: Point,
    target_size: Size,
    *,
    frame_shape: Optional[FrameShape],
    clamp_to_frame: bool,
) -> Point:
    if not clamp_to_frame or frame_shape is None:
        return float(target_pos[0]), float(target_pos[1])

    frame_width, frame_height = frame_shape
    half_w = max(0.0, float(target_size[0]) / 2.0)
    half_h = max(0.0, float(target_size[1]) / 2.0)

    if half_w * 2.0 >= frame_width:
        min_x = max_x = frame_width / 2.0
    else:
        min_x = half_w
        max_x = frame_width - half_w

    if half_h * 2.0 >= frame_height:
        min_y = max_y = frame_height / 2.0
    else:
        min_y = half_h
        max_y = frame_height - half_h

    return (
        clamp(float(target_pos[0]), min_x, max_x),
        clamp(float(target_pos[1]), min_y, max_y),
    )


def clamp_pair_abs(values: Point, max_abs_value: Optional[float]) -> Point:
    if max_abs_value is None:
        return float(values[0]), float(values[1])
    limit = abs(float(max_abs_value))
    return (
        clamp(float(values[0]), -limit, limit),
        clamp(float(values[1]), -limit, limit),
    )


def clamp_size_pair_abs(values: Size, max_abs_value: Optional[float]) -> Size:
    if max_abs_value is None:
        return float(values[0]), float(values[1])
    limit = abs(float(max_abs_value))
    return (
        clamp(float(values[0]), -limit, limit),
        clamp(float(values[1]), -limit, limit),
    )


def clamp_pair_acceleration(
    previous_velocity: Point,
    desired_velocity: Point,
    *,
    max_acceleration: Optional[float],
    dt: float,
) -> Point:
    if max_acceleration is None:
        return float(desired_velocity[0]), float(desired_velocity[1])
    limit = abs(float(max_acceleration)) * max(float(dt), 1e-6)
    return (
        float(previous_velocity[0])
        + clamp(float(desired_velocity[0]) - float(previous_velocity[0]), -limit, limit),
        float(previous_velocity[1])
        + clamp(float(desired_velocity[1]) - float(previous_velocity[1]), -limit, limit),
    )


def clamp_size_acceleration(
    previous_velocity: Size,
    desired_velocity: Size,
    *,
    max_acceleration: Optional[float],
    dt: float,
) -> Size:
    if max_acceleration is None:
        return float(desired_velocity[0]), float(desired_velocity[1])
    limit = abs(float(max_acceleration)) * max(float(dt), 1e-6)
    return (
        float(previous_velocity[0])
        + clamp(float(desired_velocity[0]) - float(previous_velocity[0]), -limit, limit),
        float(previous_velocity[1])
        + clamp(float(desired_velocity[1]) - float(previous_velocity[1]), -limit, limit),
    )


def damp_pair(values: Point, damping: float) -> Point:
    damping = clamp(damping, 0.0, 1.0)
    return float(values[0]) * damping, float(values[1]) * damping


def damp_size_pair(values: Size, damping: float) -> Size:
    damping = clamp(damping, 0.0, 1.0)
    return float(values[0]) * damping, float(values[1]) * damping


def clamp_uncertainty(
    uncertainty: float,
    *,
    min_uncertainty: float,
    max_uncertainty: Optional[float],
) -> float:
    lower = max(0.0, float(min_uncertainty))
    if max_uncertainty is None:
        return max(lower, float(uncertainty))
    return clamp(float(uncertainty), lower, max(lower, float(max_uncertainty)))
