from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from BigTracker.predictor import PredictorModel
from BigTracker.types import (
    PredictorInitializeInput,
    PredictorInitializeOutput,
    PredictorPredictInput,
    PredictorPredictOutput,
    PredictorUpdateInput,
    PredictorUpdateOutput,
    TrackerPredictionState,
)


@dataclass(frozen=True)
class MatcherTargetPredictorConfig:
    """Configuration for the no-motion matcher-target predictor.

    The empty configuration keeps this predictor selectable through the same
    component registry as the motion models.
    """


class MatcherTargetPredictorModel(PredictorModel):
    """Retain the latest target center accepted from the matcher.

    This baseline deliberately performs no extrapolation, filtering, timing,
    score handling, or uncertainty adjustment.  BigTrack supplies an accepted
    matcher center in ``PredictorUpdateInput.predictor_state.target_pos``;
    rejected updates retain the previously predicted (and therefore latest
    accepted) position.
    """

    def __init__(self, config: Optional[MatcherTargetPredictorConfig] = None) -> None:
        self.config = config or MatcherTargetPredictorConfig()
        self._state: TrackerPredictionState | None = None

    def initialize(self, request: PredictorInitializeInput) -> PredictorInitializeOutput:
        self._state = request.predictor_state
        return PredictorInitializeOutput(ok=True, metadata=request.metadata)

    def predict(self, request: PredictorPredictInput) -> PredictorPredictOutput:
        return PredictorPredictOutput(predictor_state=self._require_state(), metadata=request.metadata)

    def update(self, request: PredictorUpdateInput) -> PredictorUpdateOutput:
        self._state = request.predictor_state
        return PredictorUpdateOutput(ok=True, predictor_state=self._state, metadata=request.metadata)

    def reset(self) -> None:
        self._state = None

    def close(self) -> None:
        self.reset()

    def _require_state(self) -> TrackerPredictionState:
        if self._state is None:
            raise RuntimeError("MatcherTargetPredictorModel is not initialized.")
        return self._state
