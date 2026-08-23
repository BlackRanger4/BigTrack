from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from BigTracker.types.common import Box, FrameLike, OutputStatus, TrackerMode
from BigTracker.types.matcher import MatcherInitializeInput, MatcherState
from BigTracker.types.predictor import PredictorInitializeInput, TrackerPredictionState


@dataclass(frozen=True)
class BigTrackInitializeInput:
    """Input object for initializing BigTrack, Predictor, and Matcher."""

    frame: FrameLike
    box: Box
    initial_confidence: float = 1.0
    predictor: PredictorInitializeInput | None = None
    matcher: MatcherInitializeInput | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BigTrackInitializeOutput:
    """Output object returned after BigTrack initialization."""

    ok: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BigTrackState:
    """Internal state owned by one BigTrack instance."""

    predictor_state: TrackerPredictionState
    matcher_state: MatcherState
    mode: TrackerMode
    output: BigTrackUpdateOutput | None = None
    last_seen_frame: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BigTrackUpdateInput:
    """Input object for updating a BigTrack instance with one frame."""

    frame: FrameLike
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BigTrackUpdateOutput:
    """Public output returned after initialization or one BigTrack update."""

    ok: bool
    box: Box | None = None
    frame_idx: int | None = None
    timestamp: float | None = None
    status: OutputStatus | None = None
    confidence: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
