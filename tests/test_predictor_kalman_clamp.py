from __future__ import annotations

from dataclasses import dataclass, replace
import unittest

from BigTracker.predictor_models.kalman import KalmanPredictorConfig, KalmanPredictorModel
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


class KalmanPredictorClampTest(unittest.TestCase):
    def test_predict_clamps_center_and_size_to_frame(self) -> None:
        model = KalmanPredictorModel(KalmanPredictorConfig(clamp_to_frame=True))
        state = _state(
            pos=(95.0, 95.0),
            size=(20.0, 20.0),
            velocity=(20.0, 20.0),
            size_velocity=(200.0, 200.0),
        )

        predicted = model.predict(state, _frame(1))

        self.assertEqual(predicted.target_size, (100.0, 100.0))
        self.assertEqual(predicted.target_pos, (50.0, 50.0))
        self.assertEqual(predicted.metadata["kalman_frame_shape"], (100.0, 100.0))

    def test_accept_clamps_using_last_prediction_frame_shape(self) -> None:
        model = KalmanPredictorModel(KalmanPredictorConfig(clamp_to_frame=True))
        state = _state(pos=(50.0, 50.0), size=(20.0, 20.0))
        predicted = model.predict(state, _frame(1))

        accepted = model.update_from_accept(
            replace(state, prediction=predicted),
            accepted_pos=(500.0, 500.0),
            accepted_size=(500.0, 500.0),
            score=1.0,
        )

        self.assertEqual(accepted.target_size, (100.0, 100.0))
        self.assertEqual(accepted.target_pos, (50.0, 50.0))

    def test_clamp_can_be_disabled(self) -> None:
        model = KalmanPredictorModel(KalmanPredictorConfig(clamp_to_frame=False))
        state = _state(pos=(95.0, 95.0), size=(20.0, 20.0), velocity=(20.0, 20.0))

        predicted = model.predict(state, _frame(1))

        self.assertEqual(predicted.target_pos, (115.0, 115.0))


def _frame(idx: int) -> _Frame:
    return _Frame(image=_Image((100, 100, 3)), idx=idx, timestamp=float(idx))


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
