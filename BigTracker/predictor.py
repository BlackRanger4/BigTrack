from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from BigTracker.state import BigTrackState, SearchCandidate, TrackerPredictionState
from BigTracker.types import FrameLike, Point, Size


class PredictorModel(ABC):
    """Motion model used by Predictor, such as Kalman or alpha-beta filtering."""

    @abstractmethod
    def predict_position(self, state: BigTrackState, frame: FrameLike) -> Point:
        """Predict the next target center from the current tracker state."""
        ...

    @abstractmethod
    def predict_size(self, state: BigTrackState, frame: FrameLike) -> Size:
        """Predict the next target size from the current tracker state."""
        ...

    @abstractmethod
    def predict_uncertainty(self, state: BigTrackState, frame: FrameLike) -> float:
        """Estimate motion uncertainty for candidate generation."""
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


class Predictor(ABC):
    """Creates prediction state and search candidates before matching."""

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
