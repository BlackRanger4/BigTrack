from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

from BigTracker.matcher import Matcher
from BigTracker.predictor import Predictor
from BigTracker.state import (
    BigTrackDecision,
    BigTrackState,
    MatchEvidence,
    SearchCandidate,
    TrackerPredictionState,
    TrackingOutput,
)
from BigTracker.types import Box, FrameLike, Point, Size


class BigTrack(ABC):
    """Main tracker orchestrator that owns Predictor, Matcher, and lifecycle decisions."""

    predictor: Predictor
    matcher: Matcher

    @abstractmethod
    def initialize(
        self,
        frame: FrameLike,
        box: Box,
        target_velocity: Optional[Point] = None,
        target_size_velocity: Optional[Size] = None,
        initial_confidence: float = 1.0,
    ) -> BigTrackState:
        """Initialize prediction state, matcher templates, mode, counters, and output."""
        ...

    @abstractmethod
    def update(self, frame: FrameLike) -> TrackingOutput:
        """Process one frame through prediction, matching, decision, and state update."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear internal state and latest output."""
        ...

    @abstractmethod
    def get_state(self) -> Optional[BigTrackState]:
        """Return internal state for debugging, checkpointing, or advanced users."""
        ...

    @abstractmethod
    def get_output(self) -> Optional[TrackingOutput]:
        """Return the latest small client-facing output."""
        ...

    @abstractmethod
    def decide(
        self,
        state: BigTrackState,
        prediction: TrackerPredictionState,
        candidates: Sequence[SearchCandidate],
        matches: Sequence[MatchEvidence],
    ) -> BigTrackDecision:
        """Accept, reject, recover, lose, and decide whether templates may update."""
        ...

    @abstractmethod
    def apply_decision(self, state: BigTrackState, decision: BigTrackDecision) -> BigTrackState:
        """Apply decision results to prediction state, matcher state, output, and mode."""
        ...
