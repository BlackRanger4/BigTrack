from __future__ import annotations

from abc import ABC, abstractmethod

from BigTracker.types import (
    PredictorInitializeInput,
    PredictorInitializeOutput,
    PredictorPredictInput,
    PredictorPredictOutput,
    PredictorUpdateInput,
    PredictorUpdateOutput,
)


class Predictor(ABC):
    """Base API for anything that predicts and updates motion state."""

    @abstractmethod
    def initialize(self, request: PredictorInitializeInput) -> PredictorInitializeOutput:
        """Initialize or restore predictor-owned motion state."""
        ...

    @abstractmethod
    def predict(self, request: PredictorPredictInput) -> PredictorPredictOutput:
        """Predict the next target position, velocity, and uncertainty."""
        ...

    @abstractmethod
    def update(self, request: PredictorUpdateInput) -> PredictorUpdateOutput:
        """Update predictor state after BigTrack accepts or rejects matcher output."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear predictor runtime state while keeping configuration reusable."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release resources held by the predictor."""
        ...


class PredictorModel(Predictor):
    """Base class for concrete predictor models in BigTracker/predictor_models."""
