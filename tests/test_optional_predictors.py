from __future__ import annotations

from dataclasses import dataclass
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
from BigTracker.types import (
    PredictorInitializeInput,
    PredictorPredictInput,
    PredictorPredictOutput,
    PredictorUpdateInput,
    PredictorUpdateOutput,
    TrackerPredictionState,
)


@dataclass(frozen=True)
class _Frame:
    image: object
    idx: int
    timestamp: float


class OptionalPredictorTest(unittest.TestCase):
    def test_alpha_beta_clamps_velocity_and_acceleration(self) -> None:
        model = _initialized_alpha_beta(
            AlphaBetaPredictorConfig(
                beta_position=1.0,
                max_position_velocity=20.0,
                max_position_acceleration=5.0,
            ),
            pos=(50.0, 50.0),
            velocity=(100.0, 0.0),
        )

        predicted = model.predict(PredictorPredictInput(frame=_frame(1))).predictor_state
        accepted = _accept(model, (200.0, 200.0), score=1.0)

        self.assertEqual(predicted.target_velocity, (20.0, 0.0))
        self.assertLessEqual(accepted.target_velocity[0] - predicted.target_velocity[0], 5.0)

    def test_history_predictor_uses_bounded_accepted_history(self) -> None:
        model = _initialized_history(
            HistoryPredictorConfig(
                history_length=2,
                velocity_window=2,
                velocity_smoothing=1.0,
                max_position_velocity=20.0,
            ),
            pos=(0.0, 0.0),
        )

        model.predict(PredictorPredictInput(frame=_frame(1)))
        accepted1 = _accept(model, (10.0, 0.0), score=1.0)
        self.assertEqual(accepted1.target_velocity, (0.0, 0.0))

        model.predict(PredictorPredictInput(frame=_frame(2)))
        accepted2 = _accept(model, (20.0, 0.0), score=1.0)
        pred3 = model.predict(PredictorPredictInput(frame=_frame(3))).predictor_state

        self.assertEqual(accepted2.target_velocity, (10.0, 0.0))
        self.assertEqual(pred3.target_pos, (30.0, 0.0))
        self.assertEqual(len(accepted2.metadata["history_predictor_history"]), 2)

    def test_constant_accel_kalman_predicts_with_clamped_acceleration(self) -> None:
        model = _initialized_constant_accel(
            ConstantAccelerationKalmanPredictorConfig(
                max_position_acceleration=5.0,
                max_position_velocity=20.0,
            ),
            pos=(50.0, 50.0),
            velocity=(10.0, 0.0),
            metadata={"constant_accel_kalman_acceleration": (100.0, 0.0)},
        )

        predicted = model.predict(PredictorPredictInput(frame=_frame(1))).predictor_state

        self.assertEqual(predicted.target_pos, (62.5, 50.0))
        self.assertEqual(predicted.target_velocity, (15.0, 0.0))
        self.assertEqual(predicted.metadata["constant_accel_kalman_acceleration"], (5.0, 0.0))

    def test_constant_accel_reject_damps_velocity_and_acceleration(self) -> None:
        model = _initialized_constant_accel(
            ConstantAccelerationKalmanPredictorConfig(
                reject_velocity_damping=0.5,
                reject_acceleration_damping=0.25,
            ),
            pos=(50.0, 50.0),
            velocity=(10.0, 0.0),
            metadata={"constant_accel_kalman_acceleration": (8.0, 0.0)},
        )
        predicted = model.predict(PredictorPredictInput(frame=_frame(1))).predictor_state
        rejected = _reject(model, predicted)

        self.assertEqual(rejected.target_velocity, (9.0, 0.0))
        self.assertEqual(rejected.metadata["constant_accel_kalman_acceleration"], (2.0, 0.0))
        self.assertGreater(rejected.uncertainty, predicted.uncertainty)

    def test_predictors_share_basic_contract(self) -> None:
        predictors = (
            _initialized_alpha_beta(AlphaBetaPredictorConfig(), pos=(50.0, 50.0), velocity=(1.0, 0.0)),
            _initialized_history(HistoryPredictorConfig(), pos=(50.0, 50.0), velocity=(1.0, 0.0)),
            _initialized_constant_accel(
                ConstantAccelerationKalmanPredictorConfig(),
                pos=(50.0, 50.0),
                velocity=(1.0, 0.0),
            ),
        )
        for predictor in predictors:
            with self.subTest(predictor=type(predictor).__name__):
                predict_output = predictor.predict(PredictorPredictInput(frame=_frame(1)))
                accepted_output = predictor.update(
                    PredictorUpdateInput(
                        accepted=True,
                        predictor_state=TrackerPredictionState(
                            target_pos=(52.0, 50.0),
                            target_velocity=predict_output.predictor_state.target_velocity,
                            uncertainty=predict_output.predictor_state.uncertainty,
                            metadata=predict_output.predictor_state.metadata,
                        ),
                        metadata={"score": 0.8},
                    )
                )
                rejected_output = predictor.update(
                    PredictorUpdateInput(
                        accepted=False,
                        predictor_state=predict_output.predictor_state,
                    )
                )

                self.assertIsInstance(predict_output, PredictorPredictOutput)
                self.assertIsInstance(accepted_output, PredictorUpdateOutput)
                self.assertIsInstance(rejected_output, PredictorUpdateOutput)
                self.assertIsInstance(_current_state(predictor), TrackerPredictionState)


def _frame(idx: int) -> _Frame:
    return _Frame(image=None, idx=idx, timestamp=float(idx))


def _initialized_alpha_beta(
    config: AlphaBetaPredictorConfig,
    *,
    pos: tuple[float, float],
    velocity: tuple[float, float] = (0.0, 0.0),
) -> AlphaBetaPredictorModel:
    model = AlphaBetaPredictorModel(config)
    _initialize(model, pos=pos, velocity=velocity)
    return model


def _initialized_history(
    config: HistoryPredictorConfig,
    *,
    pos: tuple[float, float],
    velocity: tuple[float, float] = (0.0, 0.0),
) -> HistoryPredictorModel:
    model = HistoryPredictorModel(config)
    _initialize(model, pos=pos, velocity=velocity)
    return model


def _initialized_constant_accel(
    config: ConstantAccelerationKalmanPredictorConfig,
    *,
    pos: tuple[float, float],
    velocity: tuple[float, float] = (0.0, 0.0),
    metadata: dict | None = None,
) -> ConstantAccelerationKalmanPredictorModel:
    model = ConstantAccelerationKalmanPredictorModel(config)
    _initialize(model, pos=pos, velocity=velocity, metadata=metadata)
    return model


def _initialize(
    model,
    *,
    pos: tuple[float, float],
    velocity: tuple[float, float],
    metadata: dict | None = None,
) -> None:
    model.initialize(
        PredictorInitializeInput(
            predictor_state=TrackerPredictionState(
                target_pos=pos,
                target_velocity=velocity,
                uncertainty=0.0,
                metadata=metadata or {},
            )
        )
    )


def _accept(model, pos: tuple[float, float], *, score: float) -> TrackerPredictionState:
    previous = _current_state(model)
    model.update(
        PredictorUpdateInput(
            accepted=True,
            predictor_state=TrackerPredictionState(
                target_pos=pos,
                target_velocity=previous.target_velocity,
                uncertainty=previous.uncertainty,
                metadata=previous.metadata,
            ),
            metadata={"score": score},
        )
    )
    return _current_state(model)


def _reject(model, prediction: TrackerPredictionState) -> TrackerPredictionState:
    model.update(PredictorUpdateInput(accepted=False, predictor_state=prediction))
    return _current_state(model)


def _current_state(model) -> TrackerPredictionState:
    state = model._state
    if state is None:
        raise AssertionError("predictor state was not initialized")
    return state


if __name__ == "__main__":
    unittest.main()
