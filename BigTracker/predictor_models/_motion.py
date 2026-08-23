from __future__ import annotations

from typing import Any, Mapping, Optional

from BigTracker.types import FrameLike, Point


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(float(minimum), min(float(maximum), float(value)))


def clamp01(value: float) -> float:
    return clamp(value, 0.0, 1.0)


def frame_dt(metadata: Mapping[str, Any], frame: FrameLike, *, prefix: str) -> float:
    """Estimate dt from predictor-owned metadata and the current frame."""
    previous_timestamp = metadata.get(f"{prefix}_last_timestamp")
    if previous_timestamp is not None and frame.timestamp > float(previous_timestamp):
        return max(1e-6, frame.timestamp - float(previous_timestamp))

    previous_idx = metadata.get(f"{prefix}_last_frame_idx")
    if previous_idx is not None and frame.idx > int(previous_idx):
        return float(frame.idx - int(previous_idx))

    return 1.0


def clamp_pair_abs(values: Point, max_abs_value: Optional[float]) -> Point:
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


def damp_pair(values: Point, damping: float) -> Point:
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
