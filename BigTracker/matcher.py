from __future__ import annotations

from abc import ABC, abstractmethod

from BigTracker.state import MatchEvidence, MatcherState, SearchCandidate, TemplateCandidate
from BigTracker.types import FrameLike, Point, Size, TrackerMode


class Matcher(ABC):
    """Base API for visual matching models.

    A matcher owns model-specific template extraction and visual matching. It
    returns evidence only; it must not decide lifecycle, recovery success, lost
    state, or whether template learning is safe.
    """

    @abstractmethod
    def initialize_template(
        self,
        frame: FrameLike,
        target_pos: Point,
        target_size: Size,
    ) -> MatcherState:
        """Create the first trusted identity template from an initialized target."""
        ...

    @abstractmethod
    def extract_template(
        self,
        frame: FrameLike,
        target_pos: Point,
        target_size: Size,
        previous_state: MatcherState,
    ) -> TemplateCandidate:
        """Build a model-specific template candidate from an approved target."""
        ...

    @abstractmethod
    def update_templates(
        self,
        state: MatcherState,
        template: TemplateCandidate,
    ) -> MatcherState:
        """Insert a BigTrack-approved template candidate into matcher state."""
        ...

    @abstractmethod
    def match(
        self,
        frame: FrameLike,
        matcher_state: MatcherState,
        candidate: SearchCandidate,
        mode: TrackerMode,
    ) -> MatchEvidence:
        """Search one candidate region and return visual evidence only."""
        ...


class MatcherModel(Matcher):
    """Base class for concrete matcher models in BigTracker/matcher_models."""
