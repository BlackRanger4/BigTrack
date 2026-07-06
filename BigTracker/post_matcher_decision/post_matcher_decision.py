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
    @abstractmethod
    def decide(
        self,
        track_state: TrackState,
        candidates: Sequence[CandidateState],
        match_evidence: Sequence[MatchEvidence],
        frame_index: int,
    ) -> TrackerDecision:
        ...

    @abstractmethod
    def build_output(self, track_state: TrackState, decision: TrackerDecision) -> TrackingOutput:
        ...
