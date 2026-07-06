from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, Tuple


Box = Tuple[float, float, float, float]
Point = Tuple[float, float]
Size = Tuple[float, float]
Frame = Any
Template = Any
FeatureMap = Mapping[str, Any]


class TrackerMode(str, Enum):
    INIT = "INIT"
    TRACKING = "TRACKING"
    UNCERTAIN = "UNCERTAIN"
    OCCLUDED = "OCCLUDED"
    RECOVERY = "RECOVERY"
    LOST = "LOST"


class MatcherMode(str, Enum):
    NORMAL = "normal"
    UNCERTAIN = "uncertain"
    RECOVERY = "recovery"


class OutputStatus(str, Enum):
    ACTIVE = "ACTIVE"
    UNCERTAIN = "UNCERTAIN"
    OCCLUDED = "OCCLUDED"
    LOST = "LOST"


@dataclass(frozen=True)
class KinematicState:
    position: Point
    size: Size
    velocity: Point
    size_velocity: Size
    uncertainty_position: float
    uncertainty_size: float


@dataclass(frozen=True)
class LastResult:
    accepted_box: Optional[Box]
    predicted_box: Optional[Box]
    matched_box: Optional[Box]
    match_score: float
    identity_score: float
    appearance_score: float
    localization_score: float
    ambiguity_score: float


@dataclass(frozen=True)
class CandidateState:
    candidate_id: str
    predicted_box: Box
    search_region: Box
    prediction_confidence: float
    motion_uncertainty: float
    expected_scale_range: Tuple[float, float]
    priority: float
    reason: str


@dataclass(frozen=True)
class MatchResult:
    candidate_id: str
    box: Box
    match_score: float
    identity_score: float
    appearance_score: float
    localization_score: float
    ambiguity_score: float
    scale_score: float
    occlusion_hint: float
    search_region_id: str
    template_update_candidate: Optional[Template]
    debug_maps: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrackerDecision:
    accept_match: bool
    strong_accept: bool
    allow_template_update: bool
    mode: TrackerMode
    status: OutputStatus
    best_result: Optional[MatchResult]
    best_box: Optional[Box]
    confidence: float
    reason: str


@dataclass(frozen=True)
class TrackingOutput:
    status: OutputStatus
    box: Optional[Box]
    confidence: float
    identity_score: float
    reason: str


@dataclass(frozen=True)
class TrackState:
    track_id: str
    mode: TrackerMode
    age: int
    last_seen_frame: int
    lost_count: int
    uncertain_count: int
    recovery_count: int
    kinematic_state: KinematicState
    visual_memory: Any
    last_result: LastResult
    candidate_history: Sequence[CandidateState] = field(default_factory=tuple)
