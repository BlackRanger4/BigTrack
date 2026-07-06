from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

from BigTracker.track_state import Box, Frame, TrackState, TrackingOutput


class BigTracker(ABC):
    @abstractmethod
    def initialize(self, frame: Frame, box: Box, track_id: Optional[str] = None) -> TrackState:
        ...

    @abstractmethod
    def update(self, frame: Frame, frame_index: int) -> TrackingOutput:
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
