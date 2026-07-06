from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from BigTracker.track_state import MatchResult, Template, TrackState, TrackerDecision
from BigTracker.visual_memory.visual_memory import VisualMemory


class TemplateUpdatePolicy(ABC):
    @abstractmethod
    def can_update(
        self,
        track_state: TrackState,
        result: MatchResult,
        decision: TrackerDecision,
    ) -> bool:
        ...

    @abstractmethod
    def collect_candidate(self, result: MatchResult) -> Optional[Template]:
        ...

    @abstractmethod
    def apply_update(
        self,
        visual_memory: VisualMemory,
        template_candidate: Template,
        frame_index: int,
    ) -> VisualMemory:
        ...
