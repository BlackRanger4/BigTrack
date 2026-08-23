from __future__ import annotations

from abc import ABC, abstractmethod

from BigTracker.matcher import Matcher
from BigTracker.predictor import Predictor
from BigTracker.types import (
    BigTrackInitializeInput,
    BigTrackInitializeOutput,
    BigTrackState,
    BigTrackUpdateInput,
    BigTrackUpdateOutput,
)


class BigTrack(ABC):
    """Base API for tracker orchestrators.

    A BigTrack implementation owns one Predictor, one Matcher, one internal
    state, and the policy decisions that connect visual evidence to lifecycle
    state. Concrete implementations live in BigTracker/big_trackers.
    """

    predictor: Predictor
    matcher: Matcher

    @abstractmethod
    def initialize(self, request: BigTrackInitializeInput) -> BigTrackInitializeOutput:
        """Initialize predictor state, matcher template state, and BigTrack runtime state."""
        ...

    @abstractmethod
    def initialize_from_state(self, request: BigTrackInitializeInput) -> BigTrackInitializeOutput:
        """Restore BigTrack using provided predictor and matcher state in the request."""
        ...

    @abstractmethod
    def update(self, request: BigTrackUpdateInput) -> BigTrackUpdateOutput:
        """Process one frame and return the public tracking output."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear runtime state while keeping constructed predictor and matcher objects reusable."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release resources owned directly by BigTrack and its child components."""
        ...

    @abstractmethod
    def get_state(self) -> BigTrackState:
        """Return the latest composed predictor, matcher, and BigTrack state."""
        ...
