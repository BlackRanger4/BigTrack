from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from BigTracker.matcher_manager.matcher_adapter import MatcherAdapter
from BigTracker.track_state import CandidateState, Frame, MatchResult, MatcherMode
from BigTracker.visual_memory.visual_memory import VisualMemory


class MultiTemplateMatcher(ABC):
    @abstractmethod
    def match_many(
        self,
        frame: Frame,
        candidates: Sequence[CandidateState],
        visual_memory: VisualMemory,
        mode: MatcherMode,
    ) -> Sequence[MatchResult]:
        ...

    @abstractmethod
    def score_template_agreement(
        self,
        result: MatchResult,
        visual_memory: VisualMemory,
    ) -> float:
        ...

    @abstractmethod
    def get_adapter(self) -> MatcherAdapter:
        ...
