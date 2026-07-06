from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

from BigTracker.common_types import Box, Frame
from BigTracker.track_state import TrackState, TrackingOutput


class BigTracker(ABC):
    """Public object-tracker interface used by applications."""

    @abstractmethod
    def initialize(self, frame: Frame, box: Box, track_id: Optional[str] = None) -> TrackState:
        """Create a new track from the first trusted box and initialize memory."""
        ...

    @abstractmethod
    def update(
        self,
        frame: Frame,
        frame_index: int,
        external_detections: Optional[Sequence[Box]] = None,
    ) -> TrackingOutput:
        """Process one frame and return the public tracking result."""
        ...

    @abstractmethod
    def get_state(self) -> Optional[TrackState]:
        """Return the current internal track state, or None before initialization."""
        ...

    @abstractmethod
    def get_history(self) -> Sequence[TrackingOutput]:
        """Return public outputs already produced by this tracker."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear state, memory, and output history."""
        ...
