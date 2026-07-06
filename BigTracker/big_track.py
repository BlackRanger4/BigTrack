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
    """Base API for tracker orchestrators.

    A BigTrack implementation owns one Predictor, one Matcher, one internal
    state, and the policy decisions that connect visual evidence to lifecycle
    state. Concrete implementations live in BigTracker/big_trackers.
    """

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
    def make_candidates(
        self,
        state: BigTrackState,
        prediction: TrackerPredictionState,
        frame: FrameLike,
    ) -> Sequence[SearchCandidate]:
        """Create search candidates from prediction and current tracker mode."""
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
    def apply_decision(
        self,
        state: BigTrackState,
        prediction: TrackerPredictionState,
        decision: BigTrackDecision,
        frame: FrameLike,
    ) -> BigTrackState:
        """Apply decision results to prediction state, matcher state, output, and mode."""
        ...
