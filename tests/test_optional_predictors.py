from __future__ import annotations

from dataclasses import dataclass, replace
import unittest

from BigTracker.predictor_models.alpha_beta import (
    AlphaBetaPredictorConfig,
    AlphaBetaPredictorModel,
)
from BigTracker.predictor_models.history import HistoryPredictorConfig, HistoryPredictorModel
from BigTracker.predictor_models.kalman_accel import (
    ConstantAccelerationKalmanPredictorConfig,
    ConstantAccelerationKalmanPredictorModel,
)
from BigTracker.state import (
    BigTrackState,
    MatcherState,
    TrackerPredictionState,
    TrackingOutput,
)
from BigTracker.types import OutputStatus, TrackerMode


@dataclass(frozen=True)
class _Image:
    shape: tuple[int, int, int]


@dataclass(frozen=True)
class _Frame:
    image: object
    idx: int
    timestamp: float


class OptionalPredictorTest(unittest.TestCase):
    def test_alpha_beta_clamps_frame_velocity_and_acceleration(self) -> None:
        model = AlphaBetaPredictorModel(
            AlphaBetaPredictorConfig(
                beta_position=1.0,
                max_position_velocity=20.0,
                max_position_acceleration=5.0,
                max_size_velocity=10.0,
                max_size_acceleration=2.0,
                clamp_to_frame=True,
            )
        )
        state = _state(
            pos=(50.0, 50.0),
            size=(20.0, 20.0),
            velocity=(100.0, 0.0),
            size_velocity=(50.0, 0.0),
        )

        predicted = model.predict(state, _frame(1))
        accepted = model.update_from_accept(
            replace(state, prediction=predicted),
            accepted_pos=(200.0, 200.0),
            accepted_size=(200.0, 200.0),
            score=1.0,
        )

        self.assertEqual(predicted.target_velocity, (20.0, 0.0))
        self.assertLessEqual(accepted.target_velocity[0] - predicted.target_velocity[0], 5.0)
        self.assertEqual(accepted.target_size, (100.0, 100.0))
        self.assertEqual(accepted.target_pos, (50.0, 50.0))

    def test_history_predictor_uses_bounded_accepted_history(self) -> None:
        model = HistoryPredictorModel(
            HistoryPredictorConfig(
                history_length=2,
                velocity_window=2,
                velocity_smoothing=1.0,
                size_velocity_smoothing=1.0,
                max_position_velocity=20.0,
                max_size_velocity=20.0,
                clamp_to_frame=False,
            )
        )
        state0 = _state(pos=(0.0, 0.0), size=(20.0, 20.0))
        pred1 = model.predict(state0, _frame(1))
        accepted1 = model.update_from_accept(
            replace(state0, prediction=pred1),
            accepted_pos=(10.0, 0.0),
            accepted_size=(20.0, 20.0),
            score=1.0,
        )
        state1 = _state_from_prediction(accepted1, frame_idx=1, timestamp=1.0)
        pred2 = model.predict(state1, _frame(2))
        accepted2 = model.update_from_accept(
            replace(state1, prediction=pred2),
            accepted_pos=(20.0, 0.0),
            accepted_size=(22.0, 20.0),
            score=1.0,
        )
        state2 = _state_from_prediction(accepted2, frame_idx=2, timestamp=2.0)
        pred3 = model.predict(state2, _frame(3))

        self.assertEqual(accepted2.target_velocity, (10.0, 0.0))
        self.assertEqual(accepted2.target_size_velocity, (2.0, 0.0))
        self.assertEqual(pred3.target_pos, (30.0, 0.0))
        self.assertEqual(len(accepted2.metadata["history_predictor_history"]), 2)

    def test_constant_accel_kalman_predicts_with_clamped_acceleration(self) -> None:
        model = ConstantAccelerationKalmanPredictorModel(
            ConstantAccelerationKalmanPredictorConfig(
                max_position_acceleration=5.0,
                max_position_velocity=20.0,
                max_size_acceleration=2.0,
                max_size_velocity=10.0,
                clamp_to_frame=False,
            )
        )
        state = _state(
            pos=(50.0, 50.0),
            size=(20.0, 20.0),
            velocity=(10.0, 0.0),
            size_velocity=(0.0, 0.0),
            metadata={
                "constant_accel_kalman_acceleration": (100.0, 0.0),
                "constant_accel_kalman_size_acceleration": (100.0, 0.0),
            },
        )

        predicted = model.predict(state, _frame(1))

        self.assertEqual(predicted.target_pos, (62.5, 50.0))
        self.assertEqual(predicted.target_velocity, (15.0, 0.0))
        self.assertEqual(predicted.target_size, (21.0, 20.0))
        self.assertEqual(predicted.target_size_velocity, (2.0, 0.0))
        self.assertEqual(predicted.metadata["constant_accel_kalman_acceleration"], (5.0, 0.0))

    def test_constant_accel_reject_damps_velocity_and_acceleration(self) -> None:
        model = ConstantAccelerationKalmanPredictorModel(
            ConstantAccelerationKalmanPredictorConfig(
                reject_velocity_damping=0.5,
                reject_acceleration_damping=0.25,
                clamp_to_frame=False,
            )
        )
        state = _state(
            pos=(50.0, 50.0),
            size=(20.0, 20.0),
            velocity=(10.0, 0.0),
            metadata={"constant_accel_kalman_acceleration": (8.0, 0.0)},
        )
        predicted = model.predict(state, _frame(1))
        rejected = model.update_from_reject(replace(state, prediction=predicted))

        self.assertEqual(rejected.target_velocity, (9.0, 0.0))
        self.assertEqual(rejected.metadata["constant_accel_kalman_acceleration"], (2.0, 0.0))
        self.assertGreater(rejected.uncertainty, predicted.uncertainty)

    def test_predictors_share_basic_contract(self) -> None:
        predictors = (
            AlphaBetaPredictorModel(AlphaBetaPredictorConfig(clamp_to_frame=True)),
            HistoryPredictorModel(HistoryPredictorConfig(clamp_to_frame=True)),
            ConstantAccelerationKalmanPredictorModel(
                ConstantAccelerationKalmanPredictorConfig(clamp_to_frame=True)
            ),
        )
        for predictor in predictors:
            with self.subTest(predictor=type(predictor).__name__):
                state = _state(pos=(50.0, 50.0), size=(20.0, 20.0), velocity=(1.0, 0.0))
                predicted = predictor.predict(state, _frame(1))
                accepted = predictor.update_from_accept(
                    replace(state, prediction=predicted),
                    accepted_pos=(52.0, 50.0),
                    accepted_size=(21.0, 20.0),
                    score=0.8,
                )
                rejected = predictor.update_from_reject(replace(state, prediction=predicted))

                self.assertIsInstance(accepted, TrackerPredictionState)
                self.assertIsInstance(rejected, TrackerPredictionState)
                self.assertGreaterEqual(accepted.last_score, 0.0)
                self.assertLessEqual(accepted.last_score, 1.0)
                self.assertGreaterEqual(rejected.uncertainty, predicted.uncertainty)


def _frame(idx: int, image: object | None = None) -> _Frame:
    return _Frame(image=image or _Image((100, 100, 3)), idx=idx, timestamp=float(idx))


def _state(
    *,
    pos: tuple[float, float],
    size: tuple[float, float],
    velocity: tuple[float, float] = (0.0, 0.0),
    size_velocity: tuple[float, float] = (0.0, 0.0),
    metadata: dict | None = None,
) -> BigTrackState:
    prediction = TrackerPredictionState(
        target_pos=pos,
        target_size=size,
        target_velocity=velocity,
        target_size_velocity=size_velocity,
        last_score=1.0,
        uncertainty=0.0,
        metadata=metadata or {},
    )
    return _state_from_prediction(prediction, frame_idx=0, timestamp=0.0)


def _state_from_prediction(
    prediction: TrackerPredictionState,
    *,
    frame_idx: int,
    timestamp: float,
) -> BigTrackState:
    pos = prediction.target_pos
    size = prediction.target_size
    return BigTrackState(
        prediction=prediction,
        matcher=MatcherState(init_template=object()),
        output=TrackingOutput(
            box=(pos[0] - size[0] / 2.0, pos[1] - size[1] / 2.0, size[0], size[1]),
            frame_idx=frame_idx,
            timestamp=timestamp,
            status=OutputStatus.ACTIVE,
            confidence=prediction.last_score,
        ),
        mode=TrackerMode.TRACKING,
        last_seen_frame=frame_idx,
    )


if __name__ == "__main__":
    unittest.main()
