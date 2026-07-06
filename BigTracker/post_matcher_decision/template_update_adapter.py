from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional

from BigTracker.track_state import Frame, MatchResult, Template, TrackerDecision
from BigTracker.visual_memory.identity_anchor import IdentityAnchor
from BigTracker.visual_memory.short_term_template import ShortTermTemplate
from BigTracker.visual_memory.template_bank import TemplateBankEntry
from BigTracker.visual_memory.variation_state import VariationState
from BigTracker.visual_memory.visual_memory import VisualMemory


class TemplateUpdateAdapter(ABC):
    @abstractmethod
    def extract_template_candidate(
        self,
        frame: Frame,
        result: MatchResult,
        decision: TrackerDecision,
    ) -> Optional[Template]:
        ...

    @abstractmethod
    def build_short_term_template(
        self,
        template_candidate: Template,
        result: MatchResult,
        frame_index: int,
    ) -> ShortTermTemplate:
        ...

    @abstractmethod
    def build_template_bank_entry(
        self,
        template_candidate: Template,
        result: MatchResult,
        frame_index: int,
    ) -> TemplateBankEntry:
        ...

    @abstractmethod
    def build_variation_state(
        self,
        identity_anchor: IdentityAnchor,
        template_candidate: Template,
        result: MatchResult,
        frame_index: int,
    ) -> VariationState:
        ...

    @abstractmethod
    def refresh_cached_features(self, visual_memory: VisualMemory) -> Mapping[str, Any]:
        ...
