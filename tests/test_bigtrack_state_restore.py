from __future__ import annotations

from dataclasses import dataclass, replace
import unittest

from BigTracker.big_trackers.score_gated import ScoreGatedBigTrack, ScoreGatedBigTrackConfig
from BigTracker.big_trackers.simple import SimpleBigTrack
from BigTracker.state import (
    BigTrackState,
    MatchEvidence,
    MatcherState,
    SearchCandidate,
    TemplateCandidate,
)
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
            metadata={
                **dict(state.prediction.metadata),
                "predictor_accept_frame": state.output.frame_idx,
            },
        )

    def update_from_reject(self, state):
        return replace(
            state.prediction,
            uncertainty=state.prediction.uncertainty + 1.0,
        )


class _FakeMatcher:
    def __init__(self) -> None:
        self.initialize_count = 0
        self.seen_templates: list[str] = []

    def initialize_template(self, frame, target_pos, target_size):
        self.initialize_count += 1
        template = _Template(f"init-{self.initialize_count}")
        return MatcherState(init_template=template, adaptive_template=template)

    def match(self, frame, matcher_state, candidate: SearchCandidate, mode: TrackerMode):
        template = matcher_state.adaptive_template or matcher_state.init_template
        self.seen_templates.append(template.name)
        return MatchEvidence(
            candidate_id=candidate.candidate_id,
            box=(12.0, 10.0, 20.0, 20.0),
            match_score=0.9,
            identity_score=1.0,
            appearance_score=0.9,
            localization_score=0.9,
            ambiguity_score=0.0,
            scale_score=1.0,
            occlusion_score=0.0,
        )

    def extract_template(self, frame, target_pos, target_size, previous_state):
        return TemplateCandidate(
            template=_Template(f"approved-{frame.idx}"),
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
        return replace(
            state,
            best_templates=tuple(state.best_templates) + (template.template,),
            adaptive_template=template.template,
        )


class BigTrackStateRestoreTest(unittest.TestCase):
    def test_simple_bigtrack_restores_state_without_reinitializing_matcher(self) -> None:
        matcher = _FakeMatcher()
        tracker = SimpleBigTrack(predictor=_FakePredictor(), matcher=matcher)
        initial_state = tracker.initialize(_frame(0), (10.0, 10.0, 20.0, 20.0))
        restored_state = replace(
            initial_state,
            matcher=replace(
                initial_state.matcher,
                adaptive_template=_Template("restored-adaptive"),
            ),
            metadata={"saved": True},
        )

        tracker.reset()
        returned_state = tracker.initialize_from_state(restored_state)
        output = tracker.update(_frame(1))

        self.assertIs(returned_state, restored_state)
        self.assertEqual(matcher.initialize_count, 1)
        self.assertEqual(matcher.seen_templates, ["restored-adaptive"])
        self.assertEqual(output.status, OutputStatus.ACTIVE)
        self.assertEqual(tracker.get_state().metadata, {"saved": True})

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
        state = tracker.initialize(_frame(0), (10.0, 10.0, 20.0, 20.0))
        tracker.update(_frame(1))
        saved_state = tracker.get_state()

        tracker.reset()
        tracker.initialize_from_state(saved_state)
        output = tracker.update(_frame(2))
        restored = tracker.get_state()

        self.assertIsInstance(restored, BigTrackState)
        self.assertEqual(matcher.initialize_count, 1)
        self.assertEqual(output.status, OutputStatus.ACTIVE)
        self.assertEqual(restored.mode, TrackerMode.TRACKING)
        self.assertEqual(restored.counters.age, saved_state.counters.age + 1)

    def test_initialize_from_state_rejects_wrong_type(self) -> None:
        tracker = SimpleBigTrack(predictor=_FakePredictor(), matcher=_FakeMatcher())

        with self.assertRaises(TypeError):
            tracker.initialize_from_state(object())


def _frame(idx: int) -> _Frame:
    return _Frame(image=None, idx=idx, timestamp=float(idx))


if __name__ == "__main__":
    unittest.main()
