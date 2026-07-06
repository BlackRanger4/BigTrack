from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

from BigTracker.track_state import CandidateState, MatchEvidence, TrackState


@dataclass(frozen=True)
class RankedMatch:
    """A candidate/evidence pair with a post-matcher ranking score."""

    candidate: CandidateState
    evidence: MatchEvidence
    score: float
    reason: str


class MatchRanker(ABC):
    """Ranks visual evidence using identity, ambiguity, motion, and scale consistency."""

    @abstractmethod
    def rank(
        self,
        track_state: TrackState,
        candidates: Sequence[CandidateState],
        match_evidence: Sequence[MatchEvidence],
    ) -> Sequence[RankedMatch]:
        """Score and order candidate/evidence pairs for decision making."""
        ...

    @abstractmethod
    def select_best(self, ranked_matches: Sequence[RankedMatch]) -> Optional[RankedMatch]:
        """Choose the best ranked match, or None when nothing is usable."""
        ...
