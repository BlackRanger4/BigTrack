from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple

from BigTracker.common_types import Box, Frame, Size, Template
from BigTracker.track_state import CandidateState, MatchEvidence, MatcherMode
from BigTracker.visual_memory.visual_memory import VisualMemory


@dataclass(frozen=True)
class MatcherTemplateBundle:
    identity_template: Template
    short_term_template: Optional[Template]
    bank_templates: Sequence[Template]
    cached_features: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CoordinateTransform:
    crop_region: Box
    model_input_size: Size
    padding: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MatcherSearchInput:
    candidate_id: str
    frame: Frame
    search_region: Box
    predicted_box: Box
    expected_scale_range: Tuple[float, float]
    model_input: Any
    coordinate_transform: CoordinateTransform
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RawMatcherOutput:
    candidate_id: str
    raw_box: Any
    raw_scores: Mapping[str, float]
    response_map: Optional[Any] = None
    second_best_score: Optional[float] = None
    template_scores: Mapping[str, float] = field(default_factory=dict)
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
    ) -> MatchEvidence:
        ...

    @abstractmethod
    def refresh_cache(self, visual_memory: VisualMemory) -> Mapping[str, Any]:
        ...
