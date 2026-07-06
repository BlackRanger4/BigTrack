from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

from BigTracker.common_types import Box, Frame
from BigTracker.track_state import KinematicState, TrackState


class StatePredictor(ABC):
    @abstractmethod
    def predict_state(
        self,
        track_state: TrackState,
        frame: Frame,
        frame_index: int,
    ) -> KinematicState:
        ...

    @abstractmethod
    def predict_box(self, kinematic_state: KinematicState) -> Box:
        ...

    @abstractmethod
    def predict_uncertainty(
        self,
        track_state: TrackState,
        frame_shape: Optional[Tuple[int, int]] = None,
    ) -> Tuple[float, float]:
        ...
