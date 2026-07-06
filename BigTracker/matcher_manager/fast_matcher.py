from __future__ import annotations

from abc import ABC, abstractmethod

from BigTracker.track_state import CandidateState, Frame, MatchResult, MatcherMode
from BigTracker.visual_memory.visual_memory import VisualMemory


class FastMatcher(ABC):
    @abstractmethod
    def match(
        self,
        frame: Frame,
        candidate: CandidateState,
        visual_memory: VisualMemory,
        mode: MatcherMode,
    ) -> MatchResult:
        ...

    @abstractmethod
    def warm_cache(self, visual_memory: VisualMemory) -> None:
        ...
