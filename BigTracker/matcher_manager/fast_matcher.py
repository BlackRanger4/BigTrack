from __future__ import annotations

from abc import ABC, abstractmethod

from BigTracker.matcher_manager.matcher_adapter import MatcherAdapter
from BigTracker.common_types import Frame
from BigTracker.track_state import CandidateState, MatchEvidence, MatcherMode
from BigTracker.visual_memory.visual_memory import VisualMemory


class FastMatcher(ABC):
    @abstractmethod
    def match(
        self,
        frame: Frame,
        candidate: CandidateState,
        visual_memory: VisualMemory,
        mode: MatcherMode,
    ) -> MatchEvidence:
        ...

    @abstractmethod
    def warm_cache(self, visual_memory: VisualMemory) -> None:
        ...

    @abstractmethod
    def get_adapter(self) -> MatcherAdapter:
        ...
