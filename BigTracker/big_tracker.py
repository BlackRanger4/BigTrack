from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

from BigTracker.common_types import Box, Frame
from BigTracker.track_state import TrackState, TrackingOutput


class BigTracker(ABC):
    @abstractmethod
    def initialize(self, frame: Frame, box: Box, track_id: Optional[str] = None) -> TrackState:
        ...

    @abstractmethod
    def update(
        self,
        frame: Frame,
        frame_index: int,
        external_detections: Optional[Sequence[Box]] = None,
    ) -> TrackingOutput:
        ...

    @abstractmethod
    def get_state(self) -> Optional[TrackState]:
        ...

    @abstractmethod
    def get_history(self) -> Sequence[TrackingOutput]:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...
