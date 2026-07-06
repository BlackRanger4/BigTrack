from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from BigTracker.types import Box, OutputStatus, Point, Size, TrackerMode


@dataclass(frozen=True)
class TrackerPredictionState:
    """Motion-side state used by Predictor models."""

    target_pos: Point
    target_size: Size
    target_velocity: Point
    target_size_velocity: Size
    last_score: float
    uncertainty: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchCandidate:
    """Predicted place where Matcher should search for the target."""

    candidate_id: str
    search_center: Point
    predicted_target_size: Size
    prediction_confidence: float
    motion_uncertainty: float
    reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MatcherState:
    """Visual-side state owned by the Matcher domain."""

    init_template: Any
    best_templates: Sequence[Any] = field(default_factory=tuple)
    adaptive_template: Optional[Any] = None
    cached_features: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TemplateCandidate:
    """Model-specific template built from an approved frame region."""

    template: Any
    source_frame_idx: int
    source_box: Box
    quality_score: float
    identity_score: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MatchEvidence:
    """Visual evidence returned by Matcher; not an accept/reject decision."""

    candidate_id: str
    box: Box
    match_score: float
    identity_score: float
    appearance_score: float
    localization_score: float
    ambiguity_score: float
    scale_score: float
    occlusion_score: float
    is_clipped: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrackingOutput:
    """Small client-facing output for one frame."""

    box: Optional[Box]
    frame_idx: int
    timestamp: float
    status: OutputStatus
    confidence: float


@dataclass(frozen=True)
class BigTrackCounters:
    """Lifecycle counters owned by BigTrack decision logic."""

    age: int = 0
    lost_count: int = 0
    uncertain_count: int = 0
    recovery_count: int = 0


@dataclass(frozen=True)
class BigTrackState:
    """Full internal state for one object track."""

    prediction: TrackerPredictionState
    matcher: MatcherState
    output: TrackingOutput
    mode: TrackerMode
    counters: BigTrackCounters = field(default_factory=BigTrackCounters)
    last_seen_frame: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BigTrackDecision:
    """Internal post-process decision created by BigTrack."""

    accepted: bool
    accepted_box: Optional[Box]
    accepted_target_pos: Optional[Point]
    accepted_target_size: Optional[Size]
    output_status: OutputStatus
    next_mode: TrackerMode
    confidence: float
    allow_template_update: bool
    reason: str
