from __future__ import annotations

from abc import ABC, abstractmethod

from BigTracker.matcher_manager.matcher_adapter import MatcherAdapter
from BigTracker.common_types import Frame
from BigTracker.track_state import CandidateState, MatchEvidence, MatcherMode
from BigTracker.visual_memory.visual_memory import VisualMemory


class FastMatcher(ABC):
    """Low-latency matcher path for confident normal tracking."""

    @abstractmethod
    def match(
        self,
        frame: Frame,
        candidate: CandidateState,
        visual_memory: VisualMemory,
        mode: MatcherMode,
    ) -> MatchEvidence:
        """Evaluate one candidate and return visual evidence only."""
        ...

    @abstractmethod
    def warm_cache(self, visual_memory: VisualMemory) -> None:
        """Prepare matcher caches before repeated normal tracking calls."""
        ...

    @abstractmethod
    def get_adapter(self) -> MatcherAdapter:
        """Return the adapter used by this matcher implementation."""
        ...
