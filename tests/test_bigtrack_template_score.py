from __future__ import annotations

from dataclasses import dataclass, replace
import unittest

from BigTracker.big_trackers.simple import SimpleBigTrack
from BigTracker.state import MatchEvidence, MatcherState, SearchCandidate, TemplateCandidate
from BigTracker.types import TrackerMode


@dataclass(frozen=True)
class _Frame:
    image: object
    idx: int
    timestamp: float


@dataclass(frozen=True)
class _Template:
    name: str


class _UpdatingSimpleBigTrack(SimpleBigTrack):
    def decide(self, state, prediction, candidates, matches):
        decision = super().decide(state, prediction, candidates, matches)
        return replace(decision, allow_template_update=True)


class _FakePredictor:
    def predict(self, state, frame):
        return state.prediction

    def update_from_accept(self, state, accepted_pos, accepted_size, score):
        return replace(
            state.prediction,
            target_pos=accepted_pos,
            target_size=accepted_size,
            last_score=score,
        )

    def update_from_reject(self, state):
        return state.prediction


class _FakeMatcher:
    def __init__(self) -> None:
        self.updated_template = None

    def initialize_template(self, frame, target_pos, target_size):
        template = _Template("init")
        return MatcherState(init_template=template, adaptive_template=template)

    def match(self, frame, matcher_state, candidate: SearchCandidate, mode: TrackerMode):
        return MatchEvidence(
            candidate_id=candidate.candidate_id,
            box=(10.0, 10.0, 20.0, 20.0),
            match_score=0.42,
            identity_score=0.8,
            appearance_score=0.42,
            localization_score=0.9,
            ambiguity_score=0.1,
            scale_score=1.0,
            occlusion_score=0.0,
        )

    def extract_template(self, frame, target_pos, target_size, previous_state):
        return TemplateCandidate(
            template=_Template("approved"),
            source_frame_idx=frame.idx,
            source_box=(10.0, 10.0, 20.0, 20.0),
            quality_score=1.0,
            identity_score=0.5,
        )

    def update_templates(self, state, template):
        self.updated_template = template
        return state


class BigTrackTemplateScoreTest(unittest.TestCase):
    def test_template_candidate_uses_accepted_tracking_score(self) -> None:
        matcher = _FakeMatcher()
        tracker = _UpdatingSimpleBigTrack(
            predictor=_FakePredictor(),
            matcher=matcher,
        )
        tracker.initialize(_Frame(image=None, idx=0, timestamp=0.0), (10.0, 10.0, 20.0, 20.0))

        tracker.update(_Frame(image=None, idx=1, timestamp=1.0))

        self.assertIsNotNone(matcher.updated_template)
        self.assertAlmostEqual(matcher.updated_template.quality_score, 0.42)
        self.assertAlmostEqual(matcher.updated_template.identity_score, 0.5)


if __name__ == "__main__":
    unittest.main()
