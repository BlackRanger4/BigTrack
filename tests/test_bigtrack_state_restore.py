from __future__ import annotations

from dataclasses import dataclass, replace
import unittest

from BigTracker.big_trackers.score_gated import ScoreGatedBigTrack, ScoreGatedBigTrackConfig
from BigTracker.big_trackers.simple import SimpleBigTrack
from BigTracker.types import (
    BigTrackInitializeInput,
    BigTrackState,
    BigTrackUpdateInput,
    MatcherInitializeInput,
    MatcherInitializeOutput,
    MatcherMatchOutput,
    MatcherState,
    MatcherTemplateOutput,
    MatcherUpdateOutput,
    OutputStatus,
    PredictorInitializeInput,
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
        self.state = request.predictor_state
        return PredictorUpdateOutput(ok=True)

    def reset(self):
        self.state = None

    def close(self):
        self.reset()


class _FakeMatcher:
    def __init__(self) -> None:
        self.initialize_count = 0
        self.seen_templates: list[str] = []
        self._state = None

    def initialize_template(self, request):
        if request.matcher_state is not None:
            self._state = request.matcher_state
            return MatcherInitializeOutput(ok=True)
        self.initialize_count += 1
        template = _Template(f"init-{self.initialize_count}")
        self._state = MatcherState(init_template=template, adaptive_template=template)
        return MatcherInitializeOutput(ok=True)

    def match(self, request):
        template = self._state.adaptive_template or self._state.init_template
        self.seen_templates.append(template.name)
        return MatcherMatchOutput(bboxes=[(12.0, 10.0, 20.0, 20.0)], scores=[0.9])

    def extract_template(self, request):
        return MatcherTemplateOutput(template=_Template(f"approved-{request.frame.idx}"), score=1.0)

    def update_templates(self, request):
        self._state = replace(self._state, adaptive_template=request.template)
        return MatcherUpdateOutput(ok=True)

    def reset(self):
        self._state = None

    def close(self):
        self.reset()


class BigTrackStateRestoreTest(unittest.TestCase):
    def test_simple_bigtrack_restores_state_without_reinitializing_matcher_template(self) -> None:
        matcher = _FakeMatcher()
        tracker = SimpleBigTrack(predictor=_FakePredictor(), matcher=matcher)
        tracker.initialize(BigTrackInitializeInput(frame=_frame(0), box=(10.0, 10.0, 20.0, 20.0)))
        initial_state = tracker.get_state()
        restored_matcher = replace(
            initial_state.matcher_state,
            adaptive_template=_Template("restored-adaptive"),
        )

        tracker.reset()
        result = tracker.initialize_from_state(
            BigTrackInitializeInput(
                frame=_frame(0),
                box=(10.0, 10.0, 20.0, 20.0),
                predictor=PredictorInitializeInput(predictor_state=initial_state.predictor_state),
                matcher=MatcherInitializeInput(
                    frame=_frame(0),
                    box=(10.0, 10.0, 20.0, 20.0),
                    matcher_state=restored_matcher,
                ),
                metadata={"saved": True},
            )
        )
        output = tracker.update(BigTrackUpdateInput(frame=_frame(1)))

        self.assertTrue(result.ok)
        self.assertEqual(matcher.initialize_count, 1)
        self.assertEqual(matcher.seen_templates, ["restored-adaptive"])
        self.assertEqual(output.status, OutputStatus.ACTIVE)
        self.assertEqual(tracker.get_state().metadata["saved"], True)

    def test_score_gated_bigtrack_restores_state_and_policy_counters(self) -> None:
        matcher = _FakeMatcher()
        tracker = ScoreGatedBigTrack(
            predictor=_FakePredictor(),
            matcher=matcher,
            config=ScoreGatedBigTrackConfig(
                th_good=0.7,
                th_bad=0.3,
                template_update_interval=100,
            ),
        )
        tracker.initialize(BigTrackInitializeInput(frame=_frame(0), box=(10.0, 10.0, 20.0, 20.0)))
        tracker.update(BigTrackUpdateInput(frame=_frame(1)))
        saved_state = tracker.get_state()
        saved_age = saved_state.metadata["score_gated_counters"].age

        tracker.reset()
        tracker.initialize_from_state(
            BigTrackInitializeInput(
                frame=_frame(1),
                box=saved_state.output.box,
                predictor=PredictorInitializeInput(predictor_state=saved_state.predictor_state),
                matcher=MatcherInitializeInput(
                    frame=_frame(1),
                    box=saved_state.output.box,
                    matcher_state=saved_state.matcher_state,
                ),
                metadata=saved_state.metadata,
            )
        )
        output = tracker.update(BigTrackUpdateInput(frame=_frame(2)))
        restored = tracker.get_state()

        self.assertIsInstance(restored, BigTrackState)
        self.assertEqual(matcher.initialize_count, 1)
        self.assertEqual(output.status, OutputStatus.ACTIVE)
        self.assertEqual(restored.mode, TrackerMode.TRACKING)
        self.assertEqual(restored.metadata["score_gated_counters"].age, saved_age + 1)

    def test_initialize_from_state_requires_predictor_and_matcher_inputs(self) -> None:
        tracker = SimpleBigTrack(predictor=_FakePredictor(), matcher=_FakeMatcher())

        with self.assertRaises(ValueError):
            tracker.initialize_from_state(
                BigTrackInitializeInput(frame=_frame(0), box=(10.0, 10.0, 20.0, 20.0))
            )


def _frame(idx: int) -> _Frame:
    return _Frame(image=None, idx=idx, timestamp=float(idx))


if __name__ == "__main__":
    unittest.main()
