from __future__ import annotations

from dataclasses import dataclass, replace
import unittest

from BigTracker.predictor_models.adaptive_kalman import (
    AdaptiveKalmanPredictorConfig,
    AdaptiveKalmanPredictorModel,
)
from BigTracker.state import (
    BigTrackState,
    MatcherState,
    TrackerPredictionState,
    TrackingOutput,
)
from BigTracker.types import OutputStatus, TrackerMode


@dataclass(frozen=True)
class _Frame:
    image: object
    idx: int
    timestamp: float


@dataclass(frozen=True)
class _Image:
    shape: tuple[int, int, int]


class AdaptiveKalmanPredictorTest(unittest.TestCase):
    def test_adaptive_measurement_noise_trusts_high_scores_more(self) -> None:
        model = AdaptiveKalmanPredictorModel(
            AdaptiveKalmanPredictorConfig(
                adaptive_measurement_noise=True,
                min_measurement_noise_scale=0.25,
                max_measurement_noise_scale=3.0,
                uncertainty_accept_decay=1.0,
                clamp_to_frame=False,
            )
        )
        state = _state(pos=(0.0, 0.0), size=(20.0, 20.0))

        high_score = model.update_from_accept(state, (10.0, 0.0), (20.0, 20.0), score=1.0)
        low_score = model.update_from_accept(state, (10.0, 0.0), (20.0, 20.0), score=0.0)

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
        model = AdaptiveKalmanPredictorModel(
            AdaptiveKalmanPredictorConfig(
                max_position_velocity=10.0,
                max_size_velocity=2.0,
                clamp_to_frame=False,
            )
        )
        state = _state(
            pos=(0.0, 0.0),
            size=(20.0, 20.0),
            velocity=(500.0, -500.0),
            size_velocity=(50.0, -50.0),
        )

        predicted = model.predict(state, _frame(1))

        self.assertEqual(predicted.target_pos, (10.0, -10.0))
        self.assertEqual(predicted.target_velocity, (10.0, -10.0))
        self.assertEqual(predicted.target_size, (22.0, 18.0))
        self.assertEqual(predicted.target_size_velocity, (2.0, -2.0))

    def test_predict_clamps_position_to_frame_boundaries(self) -> None:
        model = AdaptiveKalmanPredictorModel(
            AdaptiveKalmanPredictorConfig(
                max_position_velocity=20.0,
                clamp_to_frame=True,
            )
        )
        state = _state(
            pos=(95.0, 95.0),
            size=(20.0, 20.0),
            velocity=(20.0, 20.0),
        )

        predicted = model.predict(state, _frame(1, image=_Image((100, 100, 3))))

        self.assertEqual(predicted.target_pos, (90.0, 90.0))

    def test_reject_grows_uncertainty_damps_velocity_and_preserves_prediction(self) -> None:
        model = AdaptiveKalmanPredictorModel(
            AdaptiveKalmanPredictorConfig(
                reject_uncertainty_growth=2.0,
                reject_covariance_growth=2.0,
                reject_velocity_damping=0.5,
                clamp_to_frame=False,
            )
        )
        state = _state(pos=(100.0, 100.0), size=(20.0, 20.0), velocity=(10.0, 0.0))
        predicted = model.predict(state, _frame(1))
        predicted_state = replace(state, prediction=predicted)

        rejected = model.update_from_reject(predicted_state)

        self.assertEqual(rejected.target_pos, predicted.target_pos)
        self.assertEqual(rejected.target_velocity, (5.0, 0.0))
        self.assertGreater(rejected.uncertainty, predicted.uncertainty)
        self.assertEqual(rejected.metadata["adaptive_kalman_reject_count"], 1)
        self.assertEqual(rejected.metadata["adaptive_kalman_last_stage"], "reject")

    def test_clean_accept_decays_uncertainty_and_resets_reject_count(self) -> None:
        state = _state(pos=(0.0, 0.0), size=(20.0, 20.0))
        no_decay_model = AdaptiveKalmanPredictorModel(
            AdaptiveKalmanPredictorConfig(
                adaptive_measurement_noise=False,
                uncertainty_accept_decay=1.0,
                clamp_to_frame=False,
            )
        )
        decay_model = AdaptiveKalmanPredictorModel(
            AdaptiveKalmanPredictorConfig(
                adaptive_measurement_noise=False,
                uncertainty_accept_decay=0.5,
                clamp_to_frame=False,
            )
        )

        no_decay = no_decay_model.update_from_accept(state, (5.0, 0.0), (20.0, 20.0), score=1.0)
        decay = decay_model.update_from_accept(state, (5.0, 0.0), (20.0, 20.0), score=1.0)

        self.assertLess(decay.uncertainty, no_decay.uncertainty)
        self.assertEqual(decay.metadata["adaptive_kalman_reject_count"], 0)
        self.assertEqual(decay.metadata["adaptive_kalman_last_stage"], "accept")


def _frame(idx: int, image: object | None = None) -> _Frame:
    return _Frame(image=image, idx=idx, timestamp=float(idx))


def _state(
    *,
    pos: tuple[float, float],
    size: tuple[float, float],
    velocity: tuple[float, float] = (0.0, 0.0),
    size_velocity: tuple[float, float] = (0.0, 0.0),
) -> BigTrackState:
    prediction = TrackerPredictionState(
        target_pos=pos,
        target_size=size,
        target_velocity=velocity,
        target_size_velocity=size_velocity,
        last_score=1.0,
        uncertainty=0.0,
    )
    return BigTrackState(
        prediction=prediction,
        matcher=MatcherState(init_template=object()),
        output=TrackingOutput(
            box=(pos[0] - size[0] / 2.0, pos[1] - size[1] / 2.0, size[0], size[1]),
            frame_idx=0,
            timestamp=0.0,
            status=OutputStatus.ACTIVE,
            confidence=1.0,
        ),
        mode=TrackerMode.TRACKING,
        last_seen_frame=0,
    )


if __name__ == "__main__":
    unittest.main()
