from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from BigTracker.types.common import Box, FrameLike, Point, Size

@dataclass(frozen=True)
class TrackerPredictionState:
    """Motion-side state used by Predictor models."""

    target_pos: Point
    target_velocity: Point
    uncertainty: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictorInitializeInput:
    """Predictor-specific initialization options."""

    predictor_state: TrackerPredictionState
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictorInitializeOutput:
    """Predictor-specific initialization result."""

    ok: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictorPredictInput:
    """Input object for predictor motion prediction."""

    frame: FrameLike
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictorPredictOutput:
    """Output object from predictor motion prediction."""

    predictor_state: TrackerPredictionState
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictorUpdateInput:
    """Input object for updating predictor state after a BigTrack decision."""

    accepted: bool
    predictor_state: TrackerPredictionState
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictorUpdateOutput:
    """Output object from predictor state update."""

    ok: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)
