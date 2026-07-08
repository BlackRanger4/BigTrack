from __future__ import annotations

from dataclasses import dataclass, replace
import unittest

from BigTracker.big_trackers.score_gated import ScoreGatedBigTrack, ScoreGatedBigTrackConfig
from BigTracker.state import MatchEvidence, MatcherState, SearchCandidate, TemplateCandidate
from BigTracker.types import OutputStatus, TrackerMode


@dataclass(frozen=True)
class _Frame:
    image: object
    idx: int
    timestamp: float


@dataclass(frozen=True)
class _Template:
    name: str


class _FakePredictor:
    def predict(self, state, frame):
        return state.prediction

    def update_from_accept(self, state, accepted_pos, accepted_size, score):
        return replace(
            state.prediction,
            target_pos=accepted_pos,
            target_size=accepted_size,
            last_score=score,
            uncertainty=0.0,
        )

    def update_from_reject(self, state):
        return replace(
            state.prediction,
            last_score=0.0,
            uncertainty=state.prediction.uncertainty + 1.0,
        )


class _FakeMatcher:
    def __init__(self, matches: list[MatchEvidence]) -> None:
        self.matches = matches
        self.template_updates: list[TemplateCandidate] = []

    def initialize_template(self, frame, target_pos, target_size):
        template = _Template("init")
        return MatcherState(init_template=template, adaptive_template=template)

    def match(self, frame, matcher_state, candidate: SearchCandidate, mode: TrackerMode):
        if not self.matches:
            raise AssertionError("Fake matcher has no queued match")
        match = self.matches.pop(0)
        return replace(match, candidate_id=candidate.candidate_id)

    def extract_template(self, frame, target_pos, target_size, previous_state):
        return TemplateCandidate(
            template=_Template(f"template-{frame.idx}"),
            source_frame_idx=frame.idx,
            source_box=(
                target_pos[0] - target_size[0] / 2.0,
                target_pos[1] - target_size[1] / 2.0,
                target_size[0],
                target_size[1],
            ),
            quality_score=1.0,
            identity_score=1.0,
        )

    def update_templates(self, state, template):
        self.template_updates.append(template)
        return replace(
            state,
            best_templates=tuple(state.best_templates) + (template.template,),
            adaptive_template=template.template,
        )


class ScoreGatedBigTrackTest(unittest.TestCase):
    def test_good_match_accepts_and_updates_template_on_interval(self) -> None:
        matcher = _FakeMatcher(
            [
                _match(score=0.9, box=(10.0, 10.0, 20.0, 20.0)),
                _match(score=0.9, box=(10.0, 10.0, 20.0, 20.0)),
            ]
        )
        tracker = _tracker(matcher, template_update_interval=2)
        tracker.initialize(_frame(0), (10.0, 10.0, 20.0, 20.0))

        first = tracker.update(_frame(1))
        second = tracker.update(_frame(2))

        self.assertEqual(first.status, OutputStatus.ACTIVE)
        self.assertEqual(second.status, OutputStatus.ACTIVE)
        self.assertEqual(len(matcher.template_updates), 1)
        self.assertEqual(matcher.template_updates[0].source_frame_idx, 2)
        self.assertAlmostEqual(matcher.template_updates[0].quality_score, 0.9)

    def test_weak_match_near_prediction_accepts_without_template_update(self) -> None:
        matcher = _FakeMatcher([_match(score=0.5, box=(11.0, 10.0, 20.0, 20.0))])
        tracker = _tracker(matcher, template_update_interval=1)
        tracker.initialize(_frame(0), (10.0, 10.0, 20.0, 20.0))

        output = tracker.update(_frame(1))
        state = tracker.get_state()

        self.assertEqual(output.status, OutputStatus.ACTIVE)
        self.assertEqual(output.box, (11.0, 10.0, 20.0, 20.0))
        self.assertEqual(state.mode, TrackerMode.TRACKING)
        self.assertEqual(matcher.template_updates, [])

    def test_weak_match_far_from_prediction_uses_predictor_and_becomes_uncertain(self) -> None:
        matcher = _FakeMatcher([_match(score=0.5, box=(80.0, 80.0, 20.0, 20.0))])
        tracker = _tracker(matcher)
        tracker.initialize(_frame(0), (10.0, 10.0, 20.0, 20.0))

        output = tracker.update(_frame(1))
        state = tracker.get_state()

        self.assertEqual(output.status, OutputStatus.UNCERTAIN)
        self.assertEqual(output.box, (10.0, 10.0, 20.0, 20.0))
        self.assertEqual(state.mode, TrackerMode.UNCERTAIN)
        self.assertEqual(state.counters.uncertain_count, 1)
        self.assertEqual(matcher.template_updates, [])

    def test_bad_match_uses_predictor_and_enters_occluded(self) -> None:
        matcher = _FakeMatcher([_match(score=0.1, box=(10.0, 10.0, 20.0, 20.0))])
        tracker = _tracker(matcher)
        tracker.initialize(_frame(0), (10.0, 10.0, 20.0, 20.0))

        output = tracker.update(_frame(1))
        state = tracker.get_state()

        self.assertEqual(output.status, OutputStatus.OCCLUDED)
        self.assertEqual(output.box, (10.0, 10.0, 20.0, 20.0))
        self.assertEqual(state.mode, TrackerMode.OCCLUDED)
        self.assertEqual(state.counters.lost_count, 1)
        self.assertEqual(matcher.template_updates, [])

    def test_repeated_rejects_enter_recovery_then_lost(self) -> None:
        matcher = _FakeMatcher(
            [
                _match(score=0.1, box=(10.0, 10.0, 20.0, 20.0)),
                _match(score=0.1, box=(10.0, 10.0, 20.0, 20.0)),
                _match(score=0.1, box=(10.0, 10.0, 20.0, 20.0)),
            ]
        )
        tracker = _tracker(matcher, recovery_after=2, lost_after=2)
        tracker.initialize(_frame(0), (10.0, 10.0, 20.0, 20.0))

        first = tracker.update(_frame(1))
        second = tracker.update(_frame(2))
        third = tracker.update(_frame(3))

        self.assertEqual(first.status, OutputStatus.OCCLUDED)
        self.assertEqual(second.status, OutputStatus.UNCERTAIN)
        self.assertEqual(tracker.get_state().mode, TrackerMode.LOST)
        self.assertEqual(third.status, OutputStatus.LOST)
        self.assertIsNone(third.box)

    def test_good_match_recovers_from_uncertain_mode(self) -> None:
        matcher = _FakeMatcher(
            [
                _match(score=0.5, box=(80.0, 80.0, 20.0, 20.0)),
                _match(score=0.9, box=(12.0, 10.0, 20.0, 20.0)),
            ]
        )
        tracker = _tracker(matcher, template_update_interval=10)
        tracker.initialize(_frame(0), (10.0, 10.0, 20.0, 20.0))

        tracker.update(_frame(1))
        output = tracker.update(_frame(2))
        state = tracker.get_state()

        self.assertEqual(output.status, OutputStatus.ACTIVE)
        self.assertEqual(state.mode, TrackerMode.TRACKING)
        self.assertEqual(state.counters.uncertain_count, 0)


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


def _match(score: float, box) -> MatchEvidence:
    return MatchEvidence(
        candidate_id="queued",
        box=box,
        match_score=score,
        identity_score=1.0,
        appearance_score=score,
        localization_score=score,
        ambiguity_score=0.0,
        scale_score=1.0,
        occlusion_score=0.0,
    )


if __name__ == "__main__":
    unittest.main()
