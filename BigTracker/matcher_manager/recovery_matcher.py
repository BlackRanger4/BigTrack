from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from BigTracker.matcher_manager.matcher_adapter import MatcherAdapter
from BigTracker.common_types import Frame
from BigTracker.track_state import CandidateState, MatchEvidence
from BigTracker.visual_memory.visual_memory import VisualMemory


class RecoveryMatcher(ABC):
    """Wider, identity-focused matcher path used during recovery."""

    @abstractmethod
    def recover(
        self,
        frame: Frame,
        candidates: Sequence[CandidateState],
        visual_memory: VisualMemory,
    ) -> Sequence[MatchEvidence]:
        """Search recovery candidates and return evidence without accepting identity."""
        ...

    @abstractmethod
    def get_adapter(self) -> MatcherAdapter:
        """Return the adapter used by this matcher implementation."""
        ...
