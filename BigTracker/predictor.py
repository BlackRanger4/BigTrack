from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from BigTracker.state import BigTrackState, SearchCandidate, TrackerPredictionState
from BigTracker.types import FrameLike, Point, Size


class Predictor(ABC):
    """Base API for anything that predicts motion and creates search candidates."""

    @abstractmethod
    def predict(self, state: BigTrackState, frame: FrameLike) -> TrackerPredictionState:
        """Predict target center, target size, velocity, score, and uncertainty."""
        ...

    @abstractmethod
    def make_candidates(
        self,
        state: BigTrackState,
        prediction: TrackerPredictionState,
        frame: FrameLike,
    ) -> Sequence[SearchCandidate]:
        """Create candidate search centers for Matcher to evaluate."""
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
