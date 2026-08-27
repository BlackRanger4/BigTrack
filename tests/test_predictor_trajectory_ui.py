from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tools.predictor_trajectory_ui as trajectory_ui  # noqa: E402
from BigTracker.types import (  # noqa: E402
    PredictorInitializeOutput,
    PredictorPredictOutput,
    PredictorUpdateOutput,
    TrackerPredictionState,
)


class _ConstantVelocityPredictor:
    """Minimal predictor used to verify the UI's frame lifecycle."""

    def __init__(self) -> None:
        self._state: TrackerPredictionState | None = None

    def initialize(self, request):
        self._state = request.predictor_state
        return PredictorInitializeOutput(ok=True)

    def predict(self, request):
        if self._state is None:
            raise AssertionError("predictor was not initialized")
        previous_timestamp = self._state.metadata.get("last_timestamp")
        dt = 1.0 if previous_timestamp is None else request.frame.timestamp - previous_timestamp
        self._state = TrackerPredictionState(
            target_pos=(
                self._state.target_pos[0] + self._state.target_velocity[0] * dt,
                self._state.target_pos[1] + self._state.target_velocity[1] * dt,
            ),
            target_velocity=self._state.target_velocity,
            uncertainty=self._state.uncertainty,
            metadata={"last_timestamp": request.frame.timestamp},
        )
        return PredictorPredictOutput(predictor_state=self._state)

    def update(self, request):
        self._state = request.predictor_state
        return PredictorUpdateOutput(ok=True)


class PredictorTrajectoryUiTest(unittest.TestCase):
    def test_m1_cache_predicts_one_frame_from_the_labelled_source(self) -> None:
        trajectory = tuple(
            trajectory_ui.TrajectoryPoint(
                frame_idx=index,
                timestamp=float(index),
                pos=(float(index), 0.0),
                velocity=(1.0, 0.0),
                observation=(float(index), 0.0),
                observation_score=1.0,
            )
            for index in range(4)
        )
        original_predictors = trajectory_ui.PREDICTORS
        trajectory_ui.PREDICTORS = (
            trajectory_ui.PredictorSpec("constant_velocity", "#000000", _ConstantVelocityPredictor),
        )
        try:
            cache = trajectory_ui.precompute_predictions(trajectory, (1,))
        finally:
            trajectory_ui.PREDICTORS = original_predictors

        for target_index in range(1, len(trajectory)):
            row = cache[("constant_velocity", 1, target_index)]
            self.assertEqual(row.source_frame, target_index - 1)
            self.assertEqual(row.pos, trajectory[target_index].pos)


if __name__ == "__main__":
    unittest.main()
