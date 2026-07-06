from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, TYPE_CHECKING, Tuple

from BigTracker.common_types import Box, Point, Size


if TYPE_CHECKING:
    from BigTracker.visual_memory.visual_memory import VisualMemory


class TrackerMode(str, Enum):
    """Internal lifecycle mode for one track."""

    INIT = "INIT"
    TRACKING = "TRACKING"
    UNCERTAIN = "UNCERTAIN"
    OCCLUDED = "OCCLUDED"
    RECOVERY = "RECOVERY"
    LOST = "LOST"


class MatcherMode(str, Enum):
    """Matcher runtime mode chosen from tracker lifecycle state."""

    NORMAL = "normal"
    UNCERTAIN = "uncertain"
    RECOVERY = "recovery"


class OutputStatus(str, Enum):
    """Public status exposed to tracker users."""

    ACTIVE = "ACTIVE"
    UNCERTAIN = "UNCERTAIN"
    OCCLUDED = "OCCLUDED"
    LOST = "LOST"


class AcceptanceLevel(str, Enum):
    """Post-matcher confidence level for the selected visual evidence."""

    STRONG = "STRONG"
    WEAK = "WEAK"
    REJECTED = "REJECTED"


class OutputBoxSource(str, Enum):
    """Describes where the public output box came from."""

    MATCHED = "MATCHED"
    PREDICTED = "PREDICTED"
    LAST_ACCEPTED = "LAST_ACCEPTED"
    NONE = "NONE"


class MemoryUpdateAction(str, Enum):
    """Post-matcher intent for visual-memory updates."""

    FREEZE = "FREEZE"
    COLLECT = "COLLECT"
    APPLY = "APPLY"


@dataclass(frozen=True)
class KinematicState:
    """Motion and size estimate owned by the tracker, not the matcher."""

    position: Point
    size: Size
    velocity: Point
    size_velocity: Size
    uncertainty_position: float
    uncertainty_size: float


@dataclass(frozen=True)
class MatchScores:
    """Normalized visual scores returned by a matcher backend."""

    match: float
    identity: float
    appearance: float
    localization: float
    scale: float
    ambiguity: float
    occlusion: float


@dataclass(frozen=True)
class AmbiguityEvidence:
    """Evidence that the best match may be confused with another target."""

    second_best_score: Optional[float] = None
    peak_ratio: Optional[float] = None
    competing_candidates: int = 0


@dataclass(frozen=True)
class ScaleEvidence:
    """Scale estimate and expected scale bounds for one match."""

    estimated_scale: float
    expected_range: Tuple[float, float]
    size: Size


@dataclass(frozen=True)
class OcclusionEvidence:
    """Signals that the object may be clipped, hidden, or only partly visible."""

    visible_ratio: Optional[float] = None
    clipped_by_frame: bool = False
    score: float = 0.0


@dataclass(frozen=True)
class CandidateState:
    """One pre-matcher hypothesis describing where and how to search."""

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
    """Visual evidence returned by the matcher; never a lifecycle decision."""

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
    """Compact summary of the previous public/internal tracking result."""

    accepted_box: Optional[Box]
    predicted_box: Optional[Box]
    matched_box: Optional[Box]
    scores: Optional[MatchScores]
    output_status: OutputStatus
    output_box_source: OutputBoxSource
    reason: str = ""


@dataclass(frozen=True)
class LifecycleTransition:
    """Single source of truth for mode, status, counters, and last-seen frame."""

    next_mode: TrackerMode
    status: OutputStatus
    lost_count: int
    uncertain_count: int
    recovery_count: int
    last_seen_frame: int


@dataclass(frozen=True)
class MemoryUpdatePlan:
    """Post-matcher plan for whether visual memory stays frozen or changes."""

    action: MemoryUpdateAction
    source_candidate_id: Optional[str] = None
    reason: str = ""


@dataclass(frozen=True)
class TrackerDecision:
    """Complete post-matcher decision for state, output, and memory update."""

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
    """Public per-frame tracker output."""

    status: OutputStatus
    box: Optional[Box]
    confidence: float
    identity_score: float
    box_source: OutputBoxSource
    reason: str


@dataclass(frozen=True)
class TrackState:
    """Internal state for one tracked object."""

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
