from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

from BigTracker.track_state import CandidateState, MatchResult, TrackState


@dataclass(frozen=True)
class RankedMatch:
    candidate: CandidateState
    result: MatchResult
    score: float
    reason: str


class MatchRanker(ABC):
    @abstractmethod
    def rank(
        self,
        track_state: TrackState,
        candidates: Sequence[CandidateState],
        match_results: Sequence[MatchResult],
    ) -> Sequence[RankedMatch]:
        ...

    @abstractmethod
    def select_best(self, ranked_matches: Sequence[RankedMatch]) -> Optional[RankedMatch]:
        ...
