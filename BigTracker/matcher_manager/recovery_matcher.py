from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from BigTracker.matcher_manager.matcher_adapter import MatcherAdapter
from BigTracker.track_state import CandidateState, Frame, MatchResult
from BigTracker.visual_memory.visual_memory import VisualMemory


class RecoveryMatcher(ABC):
    @abstractmethod
    def recover(
        self,
        frame: Frame,
        candidates: Sequence[CandidateState],
        visual_memory: VisualMemory,
    ) -> Sequence[MatchResult]:
        ...

    @abstractmethod
    def verify_identity_first(
        self,
        result: MatchResult,
        visual_memory: VisualMemory,
    ) -> bool:
        ...

    @abstractmethod
    def get_adapter(self) -> MatcherAdapter:
        ...
