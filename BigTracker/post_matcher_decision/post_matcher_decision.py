from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from BigTracker.track_state import (
    CandidateState,
    MatchEvidence,
    TrackState,
    TrackerDecision,
    TrackingOutput,
)


class PostMatcherDecision(ABC):
    """Turns visual evidence into lifecycle, output, and memory-update decisions."""

    @abstractmethod
    def decide(
        self,
        track_state: TrackState,
        candidates: Sequence[CandidateState],
        match_evidence: Sequence[MatchEvidence],
        frame_index: int,
    ) -> TrackerDecision:
        """Accept, reject, recover, or declare lost from matcher evidence."""
        ...

    @abstractmethod
    def build_output(self, track_state: TrackState, decision: TrackerDecision) -> TrackingOutput:
        """Convert internal state and decision into the public tracker output."""
        ...
