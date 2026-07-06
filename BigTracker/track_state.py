from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, TYPE_CHECKING, Tuple

from BigTracker.common_types import Box, Point, Size


if TYPE_CHECKING:
    from BigTracker.visual_memory.visual_memory import VisualMemory


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


class AcceptanceLevel(str, Enum):
    STRONG = "STRONG"
    WEAK = "WEAK"
    REJECTED = "REJECTED"


class OutputBoxSource(str, Enum):
    MATCHED = "MATCHED"
    PREDICTED = "PREDICTED"
    LAST_ACCEPTED = "LAST_ACCEPTED"
    NONE = "NONE"


class MemoryUpdateAction(str, Enum):
    FREEZE = "FREEZE"
    COLLECT = "COLLECT"
    APPLY = "APPLY"


@dataclass(frozen=True)
class KinematicState:
    position: Point
    size: Size
    velocity: Point
    size_velocity: Size
    uncertainty_position: float
    uncertainty_size: float


@dataclass(frozen=True)
class MatchScores:
    match: float
    identity: float
    appearance: float
    localization: float
    scale: float
    ambiguity: float
    occlusion: float


@dataclass(frozen=True)
class AmbiguityEvidence:
    second_best_score: Optional[float] = None
    peak_ratio: Optional[float] = None
    competing_candidates: int = 0


@dataclass(frozen=True)
class ScaleEvidence:
    estimated_scale: float
    expected_range: Tuple[float, float]
    size: Size


@dataclass(frozen=True)
class OcclusionEvidence:
    visible_ratio: Optional[float] = None
    clipped_by_frame: bool = False
    score: float = 0.0


@dataclass(frozen=True)
class CandidateState:
    candidate_id: str
    predicted_box: Box
    search_region: Box
    prediction_confidence: float
    motion_uncertainty: float
    size_uncertainty: float
    expected_scale_range: Tuple[float, float]
    priority: float
    reason: str
    source: str = "prediction"


@dataclass(frozen=True)
class MatchEvidence:
    candidate_id: str
    box: Box
    scores: MatchScores
    search_region_id: str
    scale: ScaleEvidence
    ambiguity: AmbiguityEvidence = field(default_factory=AmbiguityEvidence)
    occlusion: OcclusionEvidence = field(default_factory=OcclusionEvidence)
    debug_maps: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LastResult:
    accepted_box: Optional[Box]
    predicted_box: Optional[Box]
    matched_box: Optional[Box]
    scores: Optional[MatchScores]
    output_status: OutputStatus
    output_box_source: OutputBoxSource
    reason: str = ""


@dataclass(frozen=True)
class LifecycleTransition:
    next_mode: TrackerMode
    status: OutputStatus
    lost_count: int
    uncertain_count: int
    recovery_count: int
    last_seen_frame: int


@dataclass(frozen=True)
class MemoryUpdatePlan:
    action: MemoryUpdateAction
    source_candidate_id: Optional[str] = None
    reason: str = ""


@dataclass(frozen=True)
class TrackerDecision:
    acceptance: AcceptanceLevel
    transition: LifecycleTransition
    selected_candidate: Optional[CandidateState]
    selected_match: Optional[MatchEvidence]
    output_box: Optional[Box]
    output_box_source: OutputBoxSource
    confidence: float
    identity_confidence: float
    memory_update: MemoryUpdatePlan
    reason: str


@dataclass(frozen=True)
class TrackingOutput:
    status: OutputStatus
    box: Optional[Box]
    confidence: float
    identity_score: float
    box_source: OutputBoxSource
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
    visual_memory: "VisualMemory"
    last_result: LastResult
    candidate_history: Sequence[CandidateState] = field(default_factory=tuple)
