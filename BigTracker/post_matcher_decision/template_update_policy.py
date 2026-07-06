from __future__ import annotations

from abc import ABC, abstractmethod

from BigTracker.post_matcher_decision.template_update_adapter import TemplateUpdateAdapter
from BigTracker.common_types import Frame
from BigTracker.track_state import TrackState, TrackerDecision
from BigTracker.visual_memory.visual_memory import VisualMemory


class TemplateUpdatePolicy(ABC):
    """Owns the safety gate for all visual-memory updates."""

    @abstractmethod
    def can_update(
        self,
        track_state: TrackState,
        decision: TrackerDecision,
    ) -> bool:
        """Return whether the approved decision is safe for memory learning."""
        ...

    @abstractmethod
    def get_adapter(self) -> TemplateUpdateAdapter:
        """Return the adapter that extracts and builds backend memory objects."""
        ...

    @abstractmethod
    def apply_approved_update(
        self,
        frame: Frame,
        visual_memory: VisualMemory,
        decision: TrackerDecision,
        frame_index: int,
    ) -> VisualMemory:
        """Extract and apply a memory update after policy approval."""
        ...
