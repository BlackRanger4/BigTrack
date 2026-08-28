from __future__ import annotations

from dataclasses import dataclass
import unittest

from BigTracker.predictor_models import MatcherTargetPredictorModel
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


class MatcherTargetPredictorTest(unittest.TestCase):
    def test_predict_retains_latest_accepted_matcher_target_without_motion(self) -> None:
        model = MatcherTargetPredictorModel()
        model.initialize(PredictorInitializeInput(predictor_state=_state((10.0, 20.0), velocity=(8.0, -3.0))))

        first_prediction = model.predict(PredictorPredictInput(frame=_frame(1))).predictor_state
        self.assertEqual(first_prediction.target_pos, (10.0, 20.0))

        matcher_target = _state((35.0, 45.0), velocity=(8.0, -3.0), uncertainty=2.0)
        accepted = model.update(
            PredictorUpdateInput(accepted=True, predictor_state=matcher_target, metadata={"score": 0.9})
        )
        self.assertEqual(accepted.predictor_state, matcher_target)
        self.assertEqual(model.predict(PredictorPredictInput(frame=_frame(2))).predictor_state, matcher_target)

    def test_reject_retains_latest_matcher_target_and_lifecycle_is_safe(self) -> None:
        model = MatcherTargetPredictorModel()
        latest_matcher_target = _state((35.0, 45.0))
        model.initialize(PredictorInitializeInput(predictor_state=latest_matcher_target))

        rejected = model.update(PredictorUpdateInput(accepted=False, predictor_state=latest_matcher_target))
        self.assertEqual(rejected.predictor_state, latest_matcher_target)
        model.reset()
        with self.assertRaisesRegex(RuntimeError, "not initialized"):
            model.predict(PredictorPredictInput(frame=_frame(3)))
        model.close()


def _frame(idx: int) -> _Frame:
    return _Frame(image=None, idx=idx, timestamp=float(idx))


def _state(
    pos: tuple[float, float],
    *,
    velocity: tuple[float, float] = (0.0, 0.0),
    uncertainty: float = 0.0,
) -> TrackerPredictionState:
    return TrackerPredictionState(target_pos=pos, target_velocity=velocity, uncertainty=uncertainty)


if __name__ == "__main__":
    unittest.main()
