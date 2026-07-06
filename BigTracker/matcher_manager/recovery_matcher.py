from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from BigTracker.matcher_manager.matcher_adapter import MatcherAdapter
from BigTracker.common_types import Frame
from BigTracker.track_state import CandidateState, MatchEvidence
from BigTracker.visual_memory.visual_memory import VisualMemory


class RecoveryMatcher(ABC):
    @abstractmethod
    def recover(
        self,
        frame: Frame,
        candidates: Sequence[CandidateState],
        visual_memory: VisualMemory,
    ) -> Sequence[MatchEvidence]:
        ...

    @abstractmethod
    def get_adapter(self) -> MatcherAdapter:
        ...
