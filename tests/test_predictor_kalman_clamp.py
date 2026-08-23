from __future__ import annotations

from dataclasses import dataclass
import unittest

from BigTracker.predictor_models.kalman import KalmanPredictorConfig, KalmanPredictorModel
from BigTracker.types import (
    PredictorInitializeInput,
    PredictorPredictInput,
    PredictorUpdateInput,
    TrackerPredictionState,
)


@dataclass(frozen=True)
class _Frame:
    image: object
    idx: int
    timestamp: float


class KalmanPredictorTest(unittest.TestCase):
    def test_predict_uses_timestamp_delta_and_constant_velocity(self) -> None:
        model = _initialized_model(pos=(10.0, 20.0), velocity=(3.0, -2.0))

        predicted = model.predict(PredictorPredictInput(frame=_frame(2, 2.0))).predictor_state

        self.assertEqual(predicted.target_pos, (16.0, 16.0))
        self.assertEqual(predicted.target_velocity, (3.0, -2.0))
        self.assertEqual(predicted.metadata["kalman_last_frame_idx"], 2)
        self.assertEqual(predicted.metadata["kalman_last_timestamp"], 2.0)

    def test_accept_updates_toward_measurement(self) -> None:
        model = _initialized_model(pos=(0.0, 0.0), velocity=(0.0, 0.0))
        predicted = model.predict(PredictorPredictInput(frame=_frame(1, 1.0))).predictor_state

        model.update(
            PredictorUpdateInput(
                accepted=True,
                predictor_state=TrackerPredictionState(
                    target_pos=(10.0, 0.0),
                    target_velocity=predicted.target_velocity,
                    uncertainty=predicted.uncertainty,
                    metadata=predicted.metadata,
                ),
                metadata={"score": 1.0},
            )
        )
        accepted = _current_state(model)

        self.assertGreater(accepted.target_pos[0], predicted.target_pos[0])
        self.assertLess(accepted.target_pos[0], 10.0)
        self.assertEqual(accepted.metadata["kalman_reject_count"], 0)

    def test_reject_increases_uncertainty_and_preserves_motion(self) -> None:
        model = _initialized_model(
            config=KalmanPredictorConfig(reject_uncertainty_growth=2.0),
            pos=(0.0, 0.0),
            velocity=(4.0, 0.0),
        )
        predicted = model.predict(PredictorPredictInput(frame=_frame(1, 1.0))).predictor_state

        model.update(PredictorUpdateInput(accepted=False, predictor_state=predicted))
        rejected = _current_state(model)

        self.assertEqual(rejected.target_pos, predicted.target_pos)
        self.assertEqual(rejected.target_velocity, predicted.target_velocity)
        self.assertGreater(rejected.uncertainty, predicted.uncertainty)
        self.assertEqual(rejected.metadata["kalman_reject_count"], 1)


def _frame(idx: int, timestamp: float) -> _Frame:
    return _Frame(image=None, idx=idx, timestamp=timestamp)


def _initialized_model(
    *,
    pos: tuple[float, float],
    velocity: tuple[float, float] = (0.0, 0.0),
    config: KalmanPredictorConfig | None = None,
) -> KalmanPredictorModel:
    model = KalmanPredictorModel(config)
    model.initialize(
        PredictorInitializeInput(
            predictor_state=TrackerPredictionState(
                target_pos=pos,
                target_velocity=velocity,
                uncertainty=0.0,
                metadata={
                    "kalman_last_frame_idx": 0,
                    "kalman_last_timestamp": 0.0,
                },
            )
        )
    )
    return model


def _current_state(model: KalmanPredictorModel) -> TrackerPredictionState:
    state = model._state
    if state is None:
        raise AssertionError("predictor state was not initialized")
    return state


if __name__ == "__main__":
    unittest.main()
