from __future__ import annotations

from abc import ABC, abstractmethod

from BigTracker.types import (
    MatcherInitializeInput,
    MatcherInitializeOutput,
    MatcherMatchInput,
    MatcherMatchOutput,
    MatcherTemplateInput,
    MatcherTemplateOutput,
    MatcherUpdateInput,
    MatcherUpdateOutput,
)


class Matcher(ABC):
    """Base API for visual matching models.

    A matcher owns model-specific template extraction and visual matching. It
    returns evidence only; it must not decide lifecycle, recovery success, lost
    state, or whether template learning is safe.
    """

    @abstractmethod
    def initialize_template(self, request: MatcherInitializeInput) -> MatcherInitializeOutput:
        """Initialize or restore matcher template state from the initial frame and box."""
        ...

    @abstractmethod
    def extract_template(self, request: MatcherTemplateInput) -> MatcherTemplateOutput:
        """Extract a model-specific template from a BigTrack-approved frame box."""
        ...

    @abstractmethod
    def update_templates(self, request: MatcherUpdateInput) -> MatcherUpdateOutput:
        """Commit a BigTrack-approved template update into matcher-owned state."""
        ...

    @abstractmethod
    def match(self, request: MatcherMatchInput) -> MatcherMatchOutput:
        """Search one frame around each target position and return one best box per target."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear matcher runtime state without unloading reusable model resources."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release model/session resources held by the matcher."""
        ...

class MatcherModel(Matcher):
    """Base class for concrete matcher models in BigTracker/matcher_models."""
