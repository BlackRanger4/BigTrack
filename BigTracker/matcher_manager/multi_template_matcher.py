from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from BigTracker.matcher_manager.matcher_adapter import MatcherAdapter
from BigTracker.common_types import Frame
from BigTracker.track_state import CandidateState, MatchEvidence, MatcherMode
from BigTracker.visual_memory.visual_memory import VisualMemory


class MultiTemplateMatcher(ABC):
    """Matcher path that compares candidates against multiple memory templates."""

    @abstractmethod
    def match_many(
        self,
        frame: Frame,
        candidates: Sequence[CandidateState],
        visual_memory: VisualMemory,
        mode: MatcherMode,
    ) -> Sequence[MatchEvidence]:
        """Evaluate many candidates and return one evidence item per match."""
        ...

    @abstractmethod
    def score_template_agreement(
        self,
        evidence: MatchEvidence,
        visual_memory: VisualMemory,
    ) -> float:
        """Score agreement between evidence and the current visual memory."""
        ...

    @abstractmethod
    def get_adapter(self) -> MatcherAdapter:
        """Return the adapter used by this matcher implementation."""
        ...
