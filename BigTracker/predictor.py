from __future__ import annotations

from abc import ABC, abstractmethod

from BigTracker.state import BigTrackState, TrackerPredictionState
from BigTracker.types import FrameLike, Point, Size


class Predictor(ABC):
    """Base API for anything that predicts and updates motion state."""

    @abstractmethod
    def predict(self, state: BigTrackState, frame: FrameLike) -> TrackerPredictionState:
        """Predict target center, target size, velocity, score, and uncertainty."""
        ...

    @abstractmethod
    def update_from_accept(
        self,
        state: BigTrackState,
        accepted_pos: Point,
        accepted_size: Size,
        score: float,
    ) -> TrackerPredictionState:
        """Update motion state after BigTrack accepts visual evidence."""
        ...

    @abstractmethod
    def update_from_reject(self, state: BigTrackState) -> TrackerPredictionState:
        """Update motion state after BigTrack rejects visual evidence."""
        ...


class PredictorModel(Predictor):
    """Base class for concrete predictor models in BigTracker/predictor_models."""
