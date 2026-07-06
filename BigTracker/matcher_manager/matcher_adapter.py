from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple

from BigTracker.track_state import Box, CandidateState, Frame, MatchResult, MatcherMode, Template
from BigTracker.visual_memory.visual_memory import VisualMemory


@dataclass(frozen=True)
class MatcherTemplateBundle:
    identity_template: Template
    short_term_template: Optional[Template]
    bank_templates: Sequence[Template]
    cached_features: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MatcherSearchInput:
    candidate_id: str
    frame: Frame
    search_region: Box
    predicted_box: Box
    expected_scale_range: Tuple[float, float]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RawMatcherOutput:
    candidate_id: str
    raw_box: Any
    raw_scores: Mapping[str, float]
    raw_template_candidate: Optional[Any] = None
    raw_debug_maps: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class MatcherAdapter(ABC):
    @abstractmethod
    def prepare_templates(
        self,
        visual_memory: VisualMemory,
        mode: MatcherMode,
    ) -> MatcherTemplateBundle:
        ...

    @abstractmethod
    def prepare_search_input(
        self,
        frame: Frame,
        candidate: CandidateState,
        mode: MatcherMode,
    ) -> MatcherSearchInput:
        ...

    @abstractmethod
    def run_native_matcher(
        self,
        search_input: MatcherSearchInput,
        templates: MatcherTemplateBundle,
        mode: MatcherMode,
    ) -> RawMatcherOutput:
        ...

    @abstractmethod
    def decode_result(
        self,
        raw_output: RawMatcherOutput,
        candidate: CandidateState,
        mode: MatcherMode,
    ) -> MatchResult:
        ...

    @abstractmethod
    def refresh_cache(self, visual_memory: VisualMemory) -> Mapping[str, Any]:
        ...
