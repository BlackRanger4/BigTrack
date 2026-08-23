from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Optional

from BigTracker.types import Box, OutputStatus, Point, TrackerMode


ScoreBand = str


@dataclass(frozen=True)
class SearchCandidate:
    """BigTrack-owned matcher request candidate."""

    candidate_id: str
    search_center: Point
    prediction_confidence: float = 1.0
    motion_uncertainty: float = 0.0
    reason: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class BigTrackDecision:
    """Internal BigTrack policy decision for one frame."""

    accepted: bool
    accepted_box: Optional[Box]
    accepted_target_pos: Optional[Point]
    output_status: OutputStatus
    next_mode: TrackerMode
    confidence: float
    allow_template_update: bool
    reason: str
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class BigTrackCounters:
    """Lifecycle counters used by BigTrack policies."""

    age: int = 0
    uncertain_count: int = 0
    lost_count: int = 0
    recovery_count: int = 0


def clamp01(value: object, default: float = 0.0) -> float:
    """Return a finite score clamped into [0, 1]."""

    try:
        score = float(value)
    except (TypeError, ValueError):
        return _clamp_default(default)

    if not math.isfinite(score):
        return _clamp_default(default)
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def score_band(score: object, th_bad: float, th_good: float) -> ScoreBand:
    """Classify one normalized score as bad, weak, or good."""

    _validate_score_thresholds(th_bad, th_good)
    value = clamp01(score)
    if value >= th_good:
        return "good"
    if value >= th_bad:
        return "weak"
    return "bad"


def normalize_predictor_score(
    prediction_confidence: Optional[float],
    motion_uncertainty: Optional[float],
    uncertainty_scale: float = 1.0,
) -> float:
    """Convert predictor confidence and uncertainty into one [0, 1] score.

    A predictor may already provide a confidence. Motion uncertainty is applied
    as a conservative multiplier so higher uncertainty lowers the score.
    """

    confidence = 1.0 if prediction_confidence is None else clamp01(prediction_confidence)
    if motion_uncertainty is None:
        return confidence

    scale = max(float(uncertainty_scale), 1e-6)
    uncertainty = max(0.0, float(motion_uncertainty))
    uncertainty_score = 1.0 / (1.0 + uncertainty / scale)
    return clamp01(confidence * uncertainty_score)


def box_center_distance_ratio(predicted_box: Box, matched_box: Box) -> float:
    """Measure center distance normalized by predicted box diagonal."""

    predicted_cx, predicted_cy, predicted_w, predicted_h = _box_center_size(predicted_box)
    matched_cx, matched_cy, _, _ = _box_center_size(matched_box)
    diagonal = math.hypot(predicted_w, predicted_h)
    if diagonal <= 1e-6:
        return math.inf
    return math.hypot(matched_cx - predicted_cx, matched_cy - predicted_cy) / diagonal


def box_size_change_ratio(predicted_box: Box, matched_box: Box) -> float:
    """Measure size disagreement using symmetric log width/height change."""

    _, _, predicted_w, predicted_h = _box_center_size(predicted_box)
    _, _, matched_w, matched_h = _box_center_size(matched_box)
    if predicted_w <= 0.0 or predicted_h <= 0.0 or matched_w <= 0.0 or matched_h <= 0.0:
        return math.inf
    return abs(math.log(matched_w / predicted_w)) + abs(math.log(matched_h / predicted_h))


def boxes_agree(
    predicted_box: Box,
    matched_box: Box,
    max_center_error: float,
    max_size_error: float,
) -> bool:
    """Return true when matcher geometry is close enough to predictor geometry."""

    return (
        box_center_distance_ratio(predicted_box, matched_box) <= float(max_center_error)
        and box_size_change_ratio(predicted_box, matched_box) <= float(max_size_error)
    )


def _box_center_size(box: Box) -> tuple[float, float, float, float]:
    x, y, width, height = box
    width = float(width)
    height = float(height)
    return (
        float(x) + width / 2.0,
        float(y) + height / 2.0,
        width,
        height,
    )


def _validate_score_thresholds(th_bad: float, th_good: float) -> None:
    bad = clamp01(th_bad)
    good = clamp01(th_good)
    if bad != float(th_bad) or good != float(th_good) or bad > good:
        raise ValueError("Score thresholds must satisfy 0 <= th_bad <= th_good <= 1")


def _clamp_default(default: float) -> float:
    default_value = float(default)
    if not math.isfinite(default_value):
        return 0.0
    return max(0.0, min(1.0, default_value))
