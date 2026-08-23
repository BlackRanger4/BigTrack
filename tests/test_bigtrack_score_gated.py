from __future__ import annotations

from dataclasses import dataclass, replace
import unittest

from BigTracker.big_trackers.score_gated import ScoreGatedBigTrack, ScoreGatedBigTrackConfig
from BigTracker.types import (
    BigTrackInitializeInput,
    BigTrackUpdateInput,
    MatcherInitializeOutput,
    MatcherMatchOutput,
    MatcherState,
    MatcherTemplateOutput,
    MatcherUpdateOutput,
    OutputStatus,
    PredictorInitializeOutput,
    PredictorPredictOutput,
    PredictorUpdateOutput,
    TrackerMode,
)


@dataclass(frozen=True)
class _Frame:
    image: object
    idx: int
    timestamp: float


@dataclass(frozen=True)
class _Template:
    name: str


class _FakePredictor:
    def __init__(self) -> None:
        self.state = None

    def initialize(self, request):
        self.state = request.predictor_state
        return PredictorInitializeOutput(ok=True)

    def predict(self, request):
        return PredictorPredictOutput(predictor_state=self.state)

    def update(self, request):
        if request.accepted:
            self.state = replace(request.predictor_state, uncertainty=0.0)
        else:
            self.state = replace(request.predictor_state, uncertainty=request.predictor_state.uncertainty + 1.0)
        return PredictorUpdateOutput(ok=True)

    def reset(self):
        self.state = None

    def close(self):
        self.reset()


class _FakeMatcher:
    def __init__(self, matches: list[tuple[tuple[float, float, float, float], float]]) -> None:
        self.matches = matches
        self._state = None
        self.template_updates: list[_Template] = []

    def initialize_template(self, request):
        if request.matcher_state is not None:
            self._state = request.matcher_state
            return MatcherInitializeOutput(ok=True)
        template = _Template("init")
        self._state = MatcherState(init_template=template, adaptive_template=template)
        return MatcherInitializeOutput(ok=True)

    def match(self, request):
        if not self.matches:
            raise AssertionError("Fake matcher has no queued match")
        box, score = self.matches.pop(0)
        return MatcherMatchOutput(bboxes=[box], scores=[score])

    def extract_template(self, request):
        template = _Template(f"template-{request.frame.idx}")
        return MatcherTemplateOutput(template=template, score=1.0)

    def update_templates(self, request):
        self.template_updates.append(request.template)
        self._state = replace(self._state, adaptive_template=request.template)
        return MatcherUpdateOutput(ok=True)

    def reset(self):
        self._state = None

    def close(self):
        self.reset()


class ScoreGatedBigTrackTest(unittest.TestCase):
    def test_good_match_accepts_and_updates_template_on_interval(self) -> None:
        matcher = _FakeMatcher(
            [
                ((10.0, 10.0, 20.0, 20.0), 0.9),
                ((10.0, 10.0, 20.0, 20.0), 0.9),
            ]
        )
        tracker = _tracker(matcher, template_update_interval=2)
        tracker.initialize(BigTrackInitializeInput(frame=_frame(0), box=(10.0, 10.0, 20.0, 20.0)))

        first = tracker.update(BigTrackUpdateInput(frame=_frame(1)))
        second = tracker.update(BigTrackUpdateInput(frame=_frame(2)))

        self.assertEqual(first.status, OutputStatus.ACTIVE)
        self.assertEqual(second.status, OutputStatus.ACTIVE)
        self.assertEqual(len(matcher.template_updates), 1)
        self.assertEqual(matcher.template_updates[0].name, "template-2")

    def test_weak_match_near_prediction_accepts_without_template_update(self) -> None:
        matcher = _FakeMatcher([((11.0, 10.0, 20.0, 20.0), 0.5)])
        tracker = _tracker(matcher, template_update_interval=1)
        tracker.initialize(BigTrackInitializeInput(frame=_frame(0), box=(10.0, 10.0, 20.0, 20.0)))

        output = tracker.update(BigTrackUpdateInput(frame=_frame(1)))
        state = tracker.get_state()

        self.assertEqual(output.status, OutputStatus.ACTIVE)
        self.assertEqual(output.box, (11.0, 10.0, 20.0, 20.0))
        self.assertEqual(state.mode, TrackerMode.TRACKING)
        self.assertEqual(matcher.template_updates, [])

    def test_weak_match_far_from_prediction_uses_predictor_and_becomes_uncertain(self) -> None:
        matcher = _FakeMatcher([((80.0, 80.0, 20.0, 20.0), 0.5)])
        tracker = _tracker(matcher)
        tracker.initialize(BigTrackInitializeInput(frame=_frame(0), box=(10.0, 10.0, 20.0, 20.0)))

        output = tracker.update(BigTrackUpdateInput(frame=_frame(1)))
        state = tracker.get_state()

        self.assertEqual(output.status, OutputStatus.UNCERTAIN)
        self.assertEqual(output.box, (10.0, 10.0, 20.0, 20.0))
        self.assertEqual(state.mode, TrackerMode.UNCERTAIN)
        self.assertEqual(state.metadata["score_gated_counters"].uncertain_count, 1)
        self.assertEqual(matcher.template_updates, [])

    def test_bad_match_uses_predictor_and_enters_occluded(self) -> None:
        matcher = _FakeMatcher([((10.0, 10.0, 20.0, 20.0), 0.1)])
        tracker = _tracker(matcher)
        tracker.initialize(BigTrackInitializeInput(frame=_frame(0), box=(10.0, 10.0, 20.0, 20.0)))

        output = tracker.update(BigTrackUpdateInput(frame=_frame(1)))
        state = tracker.get_state()

        self.assertEqual(output.status, OutputStatus.OCCLUDED)
        self.assertEqual(output.box, (10.0, 10.0, 20.0, 20.0))
        self.assertEqual(state.mode, TrackerMode.OCCLUDED)
        self.assertEqual(state.metadata["score_gated_counters"].lost_count, 1)
        self.assertEqual(matcher.template_updates, [])

    def test_repeated_rejects_enter_recovery_then_lost(self) -> None:
        matcher = _FakeMatcher(
            [
                ((10.0, 10.0, 20.0, 20.0), 0.1),
                ((10.0, 10.0, 20.0, 20.0), 0.1),
                ((10.0, 10.0, 20.0, 20.0), 0.1),
            ]
        )
        tracker = _tracker(matcher, recovery_after=2, lost_after=2)
        tracker.initialize(BigTrackInitializeInput(frame=_frame(0), box=(10.0, 10.0, 20.0, 20.0)))

        first = tracker.update(BigTrackUpdateInput(frame=_frame(1)))
        second = tracker.update(BigTrackUpdateInput(frame=_frame(2)))
        third = tracker.update(BigTrackUpdateInput(frame=_frame(3)))

        self.assertEqual(first.status, OutputStatus.OCCLUDED)
        self.assertEqual(second.status, OutputStatus.UNCERTAIN)
        self.assertEqual(tracker.get_state().mode, TrackerMode.LOST)
        self.assertEqual(third.status, OutputStatus.LOST)
        self.assertIsNone(third.box)

    def test_good_match_recovers_from_uncertain_mode(self) -> None:
        matcher = _FakeMatcher(
            [
                ((80.0, 80.0, 20.0, 20.0), 0.5),
                ((12.0, 10.0, 20.0, 20.0), 0.9),
            ]
        )
        tracker = _tracker(matcher, template_update_interval=10)
        tracker.initialize(BigTrackInitializeInput(frame=_frame(0), box=(10.0, 10.0, 20.0, 20.0)))

        tracker.update(BigTrackUpdateInput(frame=_frame(1)))
        output = tracker.update(BigTrackUpdateInput(frame=_frame(2)))
        state = tracker.get_state()

        self.assertEqual(output.status, OutputStatus.ACTIVE)
        self.assertEqual(state.mode, TrackerMode.TRACKING)
        self.assertEqual(state.metadata["score_gated_counters"].uncertain_count, 0)


def _tracker(
    matcher: _FakeMatcher,
    *,
    recovery_after: int = 3,
    lost_after: int = 10,
    template_update_interval: int = 5,
) -> ScoreGatedBigTrack:
    return ScoreGatedBigTrack(
        predictor=_FakePredictor(),
        matcher=matcher,
        config=ScoreGatedBigTrackConfig(
            th_good=0.7,
            th_bad=0.3,
            max_center_error=0.35,
            max_size_error=0.5,
            predictor_uncertainty_scale=10.0,
            recovery_after=recovery_after,
            lost_after=lost_after,
            template_update_interval=template_update_interval,
        ),
    )


def _frame(idx: int) -> _Frame:
    return _Frame(image=None, idx=idx, timestamp=float(idx))


if __name__ == "__main__":
    unittest.main()
