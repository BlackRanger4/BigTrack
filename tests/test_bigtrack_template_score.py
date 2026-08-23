from __future__ import annotations

from dataclasses import dataclass, replace
import unittest

from BigTracker.big_trackers.simple import SimpleBigTrack
from BigTracker.types import (
    BigTrackInitializeInput,
    BigTrackUpdateInput,
    MatcherInitializeOutput,
    MatcherMatchOutput,
    MatcherState,
    MatcherTemplateOutput,
    MatcherUpdateOutput,
    PredictorInitializeOutput,
    PredictorPredictOutput,
    PredictorUpdateOutput,
)


@dataclass(frozen=True)
class _Frame:
    image: object
    idx: int
    timestamp: float


@dataclass(frozen=True)
class _Template:
    name: str


class _UpdatingSimpleBigTrack(SimpleBigTrack):
    def decide(self, state, prediction, candidates, bboxes, scores):
        decision = super().decide(state, prediction, candidates, bboxes, scores)
        return replace(decision, allow_template_update=True)


class _FakePredictor:
    def __init__(self) -> None:
        self.state = None

    def initialize(self, request):
        self.state = request.predictor_state
        return PredictorInitializeOutput(ok=True)

    def predict(self, request):
        return PredictorPredictOutput(predictor_state=self.state)

    def update(self, request):
        self.state = request.predictor_state
        return PredictorUpdateOutput(ok=True)

    def reset(self):
        self.state = None

    def close(self):
        self.reset()


class _FakeMatcher:
    def __init__(self) -> None:
        self._state = None
        self.updated_template = None
        self.updated_score = None

    def initialize_template(self, request):
        template = _Template("init")
        self._state = MatcherState(init_template=template, adaptive_template=template)
        return MatcherInitializeOutput(ok=True)

    def match(self, request):
        return MatcherMatchOutput(bboxes=[(10.0, 10.0, 20.0, 20.0)], scores=[0.42])

    def extract_template(self, request):
        return MatcherTemplateOutput(template=_Template("approved"), score=0.73)

    def update_templates(self, request):
        self.updated_template = request.template
        self.updated_score = request.score
        return MatcherUpdateOutput(ok=True)

    def reset(self):
        self._state = None

    def close(self):
        self.reset()


class BigTrackTemplateScoreTest(unittest.TestCase):
    def test_template_update_uses_matcher_template_score(self) -> None:
        matcher = _FakeMatcher()
        tracker = _UpdatingSimpleBigTrack(
            predictor=_FakePredictor(),
            matcher=matcher,
        )
        tracker.initialize(
            BigTrackInitializeInput(
                frame=_Frame(image=None, idx=0, timestamp=0.0),
                box=(10.0, 10.0, 20.0, 20.0),
            )
        )

        tracker.update(BigTrackUpdateInput(frame=_Frame(image=None, idx=1, timestamp=1.0)))

        self.assertIsNotNone(matcher.updated_template)
        self.assertEqual(matcher.updated_template.name, "approved")
        self.assertAlmostEqual(matcher.updated_score, 0.73)


if __name__ == "__main__":
    unittest.main()
