from __future__ import annotations

from dataclasses import dataclass
import unittest

from BigTracker.big_trackers.simple import SimpleBigTrack
from BigTracker.matcher_models.nanotrack import (
    NanoTrackMatcherConfig,
    NanoTrackMatcherModel,
    NanoTrackTemplate,
)
from BigTracker.predictor_models.kalman import KalmanPredictorModel
from BigTracker.state import MatcherState, SearchCandidate
from BigTracker.types import TrackerMode


@dataclass(frozen=True)
class _Frame:
    image: object
    idx: int
    timestamp: float


class _FakeNanoTrackBackend:
    def __init__(self) -> None:
        self.eval_count = 0
        self.template_count = 0
        self.track_count = 0
        self.zf = None
        self.active_features = []

    def eval(self):
        self.eval_count += 1
        return self

    def template(self, z):
        self.template_count += 1
        self.zf = {"feature_id": self.template_count, "shape": tuple(z.shape)}
        return None

    def track(self, x):
        torch = _require_torch()
        self.track_count += 1
        self.active_features.append(self.zf)
        cls = torch.zeros((1, 2, 3, 3), dtype=torch.float32)
        cls[:, 1, 1, 1] = 8.0
        loc = torch.full((1, 4, 3, 3), 10.0, dtype=torch.float32)
        return {"cls": cls, "loc": loc}


class NanoTrackMatcherTest(unittest.TestCase):
    def setUp(self) -> None:
        np = _require_numpy()
        self.image = np.zeros((80, 90, 3), dtype=np.uint8)
        self.frame = _Frame(image=self.image, idx=0, timestamp=0.0)
        self.config = NanoTrackMatcherConfig(
            device="cpu",
            max_best_templates=2,
            exemplar_size=31,
            instance_size=63,
            output_size=3,
            point_stride=8,
            min_box_size=1.0,
        )

    def test_backend_factory_runs_in_init_not_initialize_template(self) -> None:
        backend = _FakeNanoTrackBackend()
        calls = []

        def factory(config):
            calls.append(config)
            return backend

        matcher = NanoTrackMatcherModel(config=self.config, backend_factory=factory)
        self.assertEqual(len(calls), 1)
        self.assertEqual(backend.eval_count, 1)
        self.assertEqual(backend.template_count, 0)

        matcher.initialize_template(self.frame, target_pos=(30.0, 30.0), target_size=(20.0, 20.0))
        self.assertEqual(len(calls), 1)
        self.assertEqual(backend.template_count, 1)

    def test_initialize_template_sets_history_shape(self) -> None:
        backend = _FakeNanoTrackBackend()
        matcher = NanoTrackMatcherModel(config=self.config, backend=backend)

        state = matcher.initialize_template(
            self.frame,
            target_pos=(30.0, 30.0),
            target_size=(20.0, 20.0),
        )

        self.assertIsInstance(state.init_template, NanoTrackTemplate)
        self.assertEqual(state.best_templates, ())
        self.assertIs(state.adaptive_template, state.init_template)

    def test_match_uses_candidate_and_returns_evidence(self) -> None:
        backend = _FakeNanoTrackBackend()
        matcher = NanoTrackMatcherModel(config=self.config, backend=backend)
        state = matcher.initialize_template(
            self.frame,
            target_pos=(30.0, 30.0),
            target_size=(20.0, 20.0),
        )
        candidate = SearchCandidate(
            candidate_id="candidate-1",
            search_center=(45.0, 40.0),
            predicted_target_size=(20.0, 20.0),
            prediction_confidence=0.8,
            motion_uncertainty=0.1,
            reason="test",
        )

        evidence = matcher.match(self.frame, state, candidate, TrackerMode.TRACKING)

        self.assertEqual(evidence.candidate_id, "candidate-1")
        self.assertGreater(evidence.match_score, 0.9)
        self.assertAlmostEqual(evidence.box[0] + evidence.box[2] / 2.0, 45.0, delta=2.0)
        self.assertAlmostEqual(evidence.box[1] + evidence.box[3] / 2.0, 40.0, delta=2.0)
        self.assertEqual(backend.track_count, 1)
        self.assertEqual(backend.active_features[0]["feature_id"], 1)

    def test_update_templates_preserves_init_and_updates_adaptive(self) -> None:
        backend = _FakeNanoTrackBackend()
        matcher = NanoTrackMatcherModel(config=self.config, backend=backend)
        state = matcher.initialize_template(
            self.frame,
            target_pos=(30.0, 30.0),
            target_size=(20.0, 20.0),
        )
        init_template = state.init_template

        candidate_1 = matcher.extract_template(
            self.frame,
            target_pos=(35.0, 35.0),
            target_size=(18.0, 18.0),
            previous_state=state,
        )
        state = matcher.update_templates(state, candidate_1)
        candidate_2 = matcher.extract_template(
            self.frame,
            target_pos=(40.0, 40.0),
            target_size=(16.0, 16.0),
            previous_state=state,
        )
        state = matcher.update_templates(state, candidate_2)

        self.assertIs(state.init_template, init_template)
        self.assertEqual(len(state.best_templates), 2)
        self.assertIs(state.adaptive_template, candidate_2.template)

    def test_simple_bigtrack_integration_with_fake_backend(self) -> None:
        backend = _FakeNanoTrackBackend()
        matcher = NanoTrackMatcherModel(config=self.config, backend=backend)
        tracker = SimpleBigTrack(
            predictor=KalmanPredictorModel(),
            matcher=matcher,
        )
        tracker.initialize(self.frame, (20.0, 20.0, 20.0, 20.0))

        output = tracker.update(_Frame(image=self.image, idx=1, timestamp=1.0))

        self.assertIsNotNone(output.box)
        self.assertGreater(output.confidence, 0.9)
        self.assertEqual(backend.track_count, 1)


def _require_numpy():
    try:
        import numpy as np
    except ImportError as error:
        raise unittest.SkipTest("numpy is required for NanoTrack matcher tests") from error
    return np


def _require_torch():
    try:
        import torch
    except ImportError as error:
        raise unittest.SkipTest("torch is required for NanoTrack matcher tests") from error
    return torch


if __name__ == "__main__":
    unittest.main()

