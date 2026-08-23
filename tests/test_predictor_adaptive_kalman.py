from __future__ import annotations

from dataclasses import dataclass
import unittest

from BigTracker.predictor_models.adaptive_kalman import (
    AdaptiveKalmanPredictorConfig,
    AdaptiveKalmanPredictorModel,
)
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


class AdaptiveKalmanPredictorTest(unittest.TestCase):
    def test_adaptive_measurement_noise_trusts_high_scores_more(self) -> None:
        config = AdaptiveKalmanPredictorConfig(
            adaptive_measurement_noise=True,
            min_measurement_noise_scale=0.25,
            max_measurement_noise_scale=3.0,
            uncertainty_accept_decay=1.0,
        )
        high_model = _initialized_model(config, pos=(0.0, 0.0))
        low_model = _initialized_model(config, pos=(0.0, 0.0))

        high_score = _accept(high_model, (10.0, 0.0), score=1.0)
        low_score = _accept(low_model, (10.0, 0.0), score=0.0)

        self.assertGreater(high_score.target_pos[0], low_score.target_pos[0])
        self.assertAlmostEqual(
            high_score.metadata["adaptive_kalman_measurement_noise_scale"],
            0.25,
        )
        self.assertAlmostEqual(
            low_score.metadata["adaptive_kalman_measurement_noise_scale"],
            3.0,
        )

    def test_predict_clamps_unreasonable_velocity(self) -> None:
        model = _initialized_model(
            AdaptiveKalmanPredictorConfig(max_position_velocity=10.0),
            pos=(0.0, 0.0),
            velocity=(500.0, -500.0),
        )

        predicted = model.predict(PredictorPredictInput(frame=_frame(1))).predictor_state

        self.assertEqual(predicted.target_pos, (10.0, -10.0))
        self.assertEqual(predicted.target_velocity, (10.0, -10.0))

    def test_reject_grows_uncertainty_damps_velocity_and_preserves_prediction(self) -> None:
        model = _initialized_model(
            AdaptiveKalmanPredictorConfig(
                reject_uncertainty_growth=2.0,
                reject_covariance_growth=2.0,
                reject_velocity_damping=0.5,
            ),
            pos=(100.0, 100.0),
            velocity=(10.0, 0.0),
        )
        predicted = model.predict(PredictorPredictInput(frame=_frame(1))).predictor_state

        rejected = _reject(model, predicted)

        self.assertEqual(rejected.target_pos, predicted.target_pos)
        self.assertEqual(rejected.target_velocity, (5.0, 0.0))
        self.assertGreater(rejected.uncertainty, predicted.uncertainty)
        self.assertEqual(rejected.metadata["adaptive_kalman_reject_count"], 1)
        self.assertEqual(rejected.metadata["adaptive_kalman_last_stage"], "reject")

    def test_clean_accept_decays_uncertainty_and_resets_reject_count(self) -> None:
        no_decay_model = _initialized_model(
            AdaptiveKalmanPredictorConfig(
                adaptive_measurement_noise=False,
                uncertainty_accept_decay=1.0,
            ),
            pos=(0.0, 0.0),
        )
        decay_model = _initialized_model(
            AdaptiveKalmanPredictorConfig(
                adaptive_measurement_noise=False,
                uncertainty_accept_decay=0.5,
            ),
            pos=(0.0, 0.0),
        )

        no_decay = _accept(no_decay_model, (5.0, 0.0), score=1.0)
        decay = _accept(decay_model, (5.0, 0.0), score=1.0)

        self.assertLess(decay.uncertainty, no_decay.uncertainty)
        self.assertEqual(decay.metadata["adaptive_kalman_reject_count"], 0)
        self.assertEqual(decay.metadata["adaptive_kalman_last_stage"], "accept")


def _frame(idx: int) -> _Frame:
    return _Frame(image=None, idx=idx, timestamp=float(idx))


def _initialized_model(
    config: AdaptiveKalmanPredictorConfig,
    *,
    pos: tuple[float, float],
    velocity: tuple[float, float] = (0.0, 0.0),
) -> AdaptiveKalmanPredictorModel:
    model = AdaptiveKalmanPredictorModel(config)
    model.initialize(
        PredictorInitializeInput(
            predictor_state=TrackerPredictionState(
                target_pos=pos,
                target_velocity=velocity,
                uncertainty=0.0,
            )
        )
    )
    return model


def _accept(
    model: AdaptiveKalmanPredictorModel,
    pos: tuple[float, float],
    *,
    score: float,
) -> TrackerPredictionState:
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


def _reject(
    model: AdaptiveKalmanPredictorModel,
    prediction: TrackerPredictionState,
) -> TrackerPredictionState:
    model.update(PredictorUpdateInput(accepted=False, predictor_state=prediction))
    return _current_state(model)


def _current_state(model: AdaptiveKalmanPredictorModel) -> TrackerPredictionState:
    state = model._state
    if state is None:
        raise AssertionError("predictor state was not initialized")
    return state


if __name__ == "__main__":
    unittest.main()
