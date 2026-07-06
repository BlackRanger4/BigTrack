from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from BigTracker.track_state import CandidateState, Frame, MatchResult, MatcherMode
from BigTracker.visual_memory.visual_memory import VisualMemory


class MatcherManager(ABC):
    @abstractmethod
    def run(
        self,
        frame: Frame,
        candidates: Sequence[CandidateState],
        visual_memory: VisualMemory,
        mode: MatcherMode,
    ) -> Sequence[MatchResult]:
        ...

    @abstractmethod
    def supports_mode(self, mode: MatcherMode) -> bool:
        ...
