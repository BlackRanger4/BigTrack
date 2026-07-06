from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from BigTracker.matcher_manager.matcher_adapter import MatcherAdapter
from BigTracker.common_types import Frame
from BigTracker.track_state import CandidateState, MatchEvidence, MatcherMode
from BigTracker.visual_memory.visual_memory import VisualMemory


class MatcherManager(ABC):
    @abstractmethod
    def run(
        self,
        frame: Frame,
        candidates: Sequence[CandidateState],
        visual_memory: VisualMemory,
        mode: MatcherMode,
    ) -> Sequence[MatchEvidence]:
        ...

    @abstractmethod
    def supports_mode(self, mode: MatcherMode) -> bool:
        ...

    @abstractmethod
    def get_adapter(self, mode: MatcherMode) -> MatcherAdapter:
        ...
