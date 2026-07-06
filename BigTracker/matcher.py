from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from BigTracker.state import MatchEvidence, MatcherState, SearchCandidate, TemplateCandidate
from BigTracker.types import Box, FrameLike, Point, Size, TrackerMode


class MatcherModel(ABC):
    """Visual backend API for models such as MixFormer, OSTrack, or Siamese trackers."""

    @abstractmethod
    def build_template_crop(self, frame: FrameLike, target_pos: Point, target_size: Size) -> Any:
        """Build the model-specific template crop from frame and target geometry."""
        ...

    @abstractmethod
    def build_search_crop(self, frame: FrameLike, candidate: SearchCandidate) -> Any:
        """Build the model-specific search crop for one predicted candidate."""
        ...

    @abstractmethod
    def encode_template(self, template_crop: Any) -> Any:
        """Convert a template crop into the model-specific template object."""
        ...

    @abstractmethod
    def run_match(self, templates: Sequence[Any], search_crop: Any) -> Any:
        """Run the visual model on templates and one search crop."""
        ...

    @abstractmethod
    def decode_box(self, raw_output: Any, candidate: SearchCandidate) -> Box:
        """Decode raw model output into a frame-coordinate box."""
        ...

    @abstractmethod
    def score_identity(self, raw_output: Any) -> float:
        """Return identity confidence for the raw model output."""
        ...

    @abstractmethod
    def score_ambiguity(self, raw_output: Any) -> float:
        """Return how ambiguous the best visual match is."""
        ...


class Matcher(ABC):
    """Owns visual template extraction, template updates, and matching."""

    @abstractmethod
    def initialize_template(
        self,
        frame: FrameLike,
        target_pos: Point,
        target_size: Size,
    ) -> MatcherState:
        """Create initial matcher state from the first trusted target region."""
        ...

    @abstractmethod
    def extract_template(
        self,
        frame: FrameLike,
        target_pos: Point,
        target_size: Size,
        previous_state: MatcherState,
    ) -> TemplateCandidate:
        """Build a template candidate from a BigTrack-approved target region."""
        ...

    @abstractmethod
    def update_templates(
        self,
        state: MatcherState,
        template: TemplateCandidate,
    ) -> MatcherState:
        """Return matcher state after inserting an approved template candidate."""
        ...

    @abstractmethod
    def match(
        self,
        frame: FrameLike,
        matcher_state: MatcherState,
        candidate: SearchCandidate,
        mode: TrackerMode,
    ) -> MatchEvidence:
        """Run visual matching for one candidate and return evidence only."""
        ...
