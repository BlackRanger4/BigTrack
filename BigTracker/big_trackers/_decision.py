from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping, Optional, Sequence

from BigTracker.state import MatchEvidence, SearchCandidate
from BigTracker.types import Box


ScoreBand = str


@dataclass(frozen=True)
class MatchChoice:
    """Selected matcher result plus the candidate that produced it."""

    match: MatchEvidence
    candidate: Optional[SearchCandidate]
    acceptance_score: float
    metadata: Mapping[str, object]


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


def combine_evidence_score(
    match: MatchEvidence,
    *,
    match_weight: float = 0.60,
    appearance_weight: float = 0.15,
    localization_weight: float = 0.15,
    identity_weight: float = 0.05,
    scale_weight: float = 0.05,
    ambiguity_penalty: float = 0.35,
    occlusion_penalty: float = 0.35,
    clipped_penalty: float = 0.20,
) -> float:
    """Combine common matcher evidence fields into one conservative score."""

    positive = (
        float(match_weight) * clamp01(match.match_score)
        + float(appearance_weight) * clamp01(match.appearance_score)
        + float(localization_weight) * clamp01(match.localization_score)
        + float(identity_weight) * clamp01(match.identity_score)
        + float(scale_weight) * clamp01(match.scale_score)
    )
    penalty = (
        float(ambiguity_penalty) * clamp01(match.ambiguity_score)
        + float(occlusion_penalty) * clamp01(match.occlusion_score)
    )
    if match.is_clipped:
        penalty += float(clipped_penalty)
    return clamp01(positive - penalty)


def evidence_reject_reasons(
    match: MatchEvidence,
    *,
    min_match_score: Optional[float] = None,
    max_ambiguity_score: Optional[float] = None,
    min_scale_score: Optional[float] = None,
    max_occlusion_score: Optional[float] = None,
    allow_clipped: bool = True,
) -> tuple[str, ...]:
    """Return policy-neutral reason labels for evidence that fails thresholds."""

    reasons: list[str] = []
    if min_match_score is not None and clamp01(match.match_score) < clamp01(min_match_score):
        reasons.append("low_match_score")
    if max_ambiguity_score is not None and clamp01(match.ambiguity_score) > clamp01(max_ambiguity_score):
        reasons.append("high_ambiguity")
    if min_scale_score is not None and clamp01(match.scale_score) < clamp01(min_scale_score):
        reasons.append("bad_scale")
    if max_occlusion_score is not None and clamp01(match.occlusion_score) > clamp01(max_occlusion_score):
        reasons.append("high_occlusion")
    if match.is_clipped and not allow_clipped:
        reasons.append("clipped")
    return tuple(reasons)


def select_best_match(
    candidates: Sequence[SearchCandidate],
    matches: Sequence[MatchEvidence],
    *,
    score_fn: Callable[[MatchEvidence], float] = combine_evidence_score,
    candidate_prior_weight: float = 0.0,
) -> Optional[MatchChoice]:
    """Select the highest-scored match and attach its candidate metadata."""

    if not matches:
        return None

    candidates_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    best_choice: Optional[MatchChoice] = None
    best_sort_key: tuple[float, int] = (-math.inf, -1)

    for index, match in enumerate(matches):
        candidate = candidates_by_id.get(match.candidate_id)
        score = clamp01(score_fn(match))
        if candidate is not None and candidate_prior_weight:
            score = clamp01(score + float(candidate_prior_weight) * clamp01(candidate.prediction_confidence))

        sort_key = (score, index)
        if sort_key > best_sort_key:
            metadata = {
                "candidate_id": match.candidate_id,
                "match_metadata": dict(match.metadata),
                "candidate_metadata": dict(candidate.metadata) if candidate is not None else {},
            }
            best_choice = MatchChoice(
                match=match,
                candidate=candidate,
                acceptance_score=score,
                metadata=metadata,
            )
            best_sort_key = sort_key

    return best_choice


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
