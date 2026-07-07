from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
import unittest

from BigTracker.big_trackers.simple import SimpleBigTrack
from BigTracker.matcher_models.ostrack import (
    OSTrackMatcherConfig,
    OSTrackMatcherModel,
    OSTrackTemplate,
)
from BigTracker.predictor_models.kalman import KalmanPredictorModel
from BigTracker.state import SearchCandidate
from BigTracker.types import TrackerMode


@dataclass(frozen=True)
class _Frame:
    image: object
    idx: int
    timestamp: float


class _FakeProcessed:
    def __init__(self, tensors):
        self.tensors = tensors


class _FakeOSTrackBackend:
    def __init__(self) -> None:
        np = _require_numpy()
        self.output_window = np.ones((1, 1, 3, 3), dtype=np.float32)
        self.eval_count = 0
        self.preprocess_count = 0
        self.forward_count = 0
        self.box_mask_count = 0
        self.forward_calls = []

    def eval(self):
        self.eval_count += 1
        return self

    def preprocess(self, image, attention_mask):
        np = _require_numpy()
        self.preprocess_count += 1
        return _FakeProcessed(np.asarray(image, dtype=np.float32))

    def build_box_mask(self, source_box, resize_factor):
        self.box_mask_count += 1
        return {"source_box": source_box, "resize_factor": resize_factor}

    def forward(self, template, search, ce_template_mask):
        np = _require_numpy()
        self.forward_count += 1
        self.forward_calls.append(
            {
                "template_shape": tuple(template.shape),
                "search_shape": tuple(search.shape),
                "box_mask": ce_template_mask,
            }
        )
        score_map = np.zeros((1, 1, 3, 3), dtype=np.float32)
        score_map[:, :, 1, 1] = 0.9
        return {
            "score_map": score_map,
            "size_map": np.zeros((1, 2, 3, 3), dtype=np.float32),
            "offset_map": np.zeros((1, 2, 3, 3), dtype=np.float32),
        }

    def cal_bbox(self, response, size_map, offset_map):
        np = _require_numpy()
        return np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)


class OSTrackMatcherTest(unittest.TestCase):
    def setUp(self) -> None:
        np = _require_numpy()
        self.image = np.zeros((80, 90, 3), dtype=np.uint8)
        self.frame = _Frame(image=self.image, idx=0, timestamp=0.0)
        self.config = OSTrackMatcherConfig(
            device="cpu",
            max_best_templates=2,
            template_factor=2.0,
            template_size=31,
            search_factor=2.0,
            search_size=63,
            backbone_stride=21,
            clip_margin=0.0,
        )

    def test_backend_factory_runs_in_init_not_initialize_template(self) -> None:
        backend = _FakeOSTrackBackend()
        calls = []

        def factory(config):
            calls.append(config)
            return backend

        matcher = OSTrackMatcherModel(config=self.config, backend_factory=factory)
        self.assertEqual(len(calls), 1)
        self.assertEqual(backend.eval_count, 1)
        self.assertEqual(backend.preprocess_count, 0)

        matcher.initialize_template(self.frame, target_pos=(30.0, 30.0), target_size=(20.0, 20.0))
        self.assertEqual(len(calls), 1)
        self.assertEqual(backend.preprocess_count, 1)

    def test_initialize_template_sets_history_shape(self) -> None:
        backend = _FakeOSTrackBackend()
        matcher = OSTrackMatcherModel(config=self.config, backend=backend)

        state = matcher.initialize_template(
            self.frame,
            target_pos=(30.0, 30.0),
            target_size=(20.0, 20.0),
        )

        self.assertIsInstance(state.init_template, OSTrackTemplate)
        self.assertEqual(state.best_templates, ())
        self.assertIs(state.adaptive_template, state.init_template)

    def test_match_uses_candidate_and_returns_evidence(self) -> None:
        backend = _FakeOSTrackBackend()
        matcher = OSTrackMatcherModel(config=self.config, backend=backend)
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
        self.assertAlmostEqual(evidence.match_score, 0.9, places=5)
        self.assertAlmostEqual(evidence.box[0] + evidence.box[2] / 2.0, 45.0, delta=1.0)
        self.assertAlmostEqual(evidence.box[1] + evidence.box[3] / 2.0, 40.0, delta=1.0)
        self.assertEqual(backend.forward_count, 1)
        self.assertEqual(backend.forward_calls[0]["search_shape"], (63, 63, 3))

    def test_update_templates_preserves_init_and_updates_adaptive(self) -> None:
        backend = _FakeOSTrackBackend()
        matcher = OSTrackMatcherModel(config=self.config, backend=backend)
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
        backend = _FakeOSTrackBackend()
        matcher = OSTrackMatcherModel(config=self.config, backend=backend)
        tracker = SimpleBigTrack(
            predictor=KalmanPredictorModel(),
            matcher=matcher,
        )
        tracker.initialize(self.frame, (20.0, 20.0, 20.0, 20.0))

        output = tracker.update(_Frame(image=self.image, idx=1, timestamp=1.0))

        self.assertIsNotNone(output.box)
        self.assertAlmostEqual(output.confidence, 0.9, places=5)
        self.assertEqual(backend.forward_count, 1)

    def test_real_checkpoint_smoke_when_enabled(self) -> None:
        if os.environ.get("BIGTRACK_RUN_OSTRACK_REAL_SMOKE") != "1":
            raise unittest.SkipTest(
                "set BIGTRACK_RUN_OSTRACK_REAL_SMOKE=1 to run OSTrack real checkpoint smoke"
            )

        np = _require_numpy()
        config_path = Path(r"ignores\Models\Ostrack\config\vitb_384_mae_ce_32x4_ep300.yaml")
        checkpoint_path = Path(r"ignores\Models\Ostrack\models\vitb_384_mae_ce_32x4_ep300\OSTrack_ep0300.pth.tar")
        if not config_path.exists() or not checkpoint_path.exists():
            raise unittest.SkipTest("OSTrack config/checkpoint assets are not present")
        loaded_lib = sys.modules.get("lib")
        loaded_lib_file = getattr(loaded_lib, "__file__", "") if loaded_lib is not None else ""
        if loaded_lib_file and "LiteTrack" in loaded_lib_file:
            raise unittest.SkipTest("LiteTrack top-level lib package is already loaded")

        image = np.zeros((240, 320, 3), dtype=np.uint8)
        frame = _Frame(image=image, idx=0, timestamp=0.0)
        matcher = OSTrackMatcherModel(
            OSTrackMatcherConfig(
                source_root=r"ignores\Trackers\OSTrack",
                config_path=str(config_path),
                checkpoint_path=str(checkpoint_path),
                device="cpu",
            )
        )
        state = matcher.initialize_template(
            frame,
            target_pos=(160.0, 120.0),
            target_size=(40.0, 40.0),
        )
        evidence = matcher.match(
            frame,
            state,
            SearchCandidate(
                candidate_id="real-smoke",
                search_center=(160.0, 120.0),
                predicted_target_size=(40.0, 40.0),
                prediction_confidence=1.0,
                motion_uncertainty=0.0,
                reason="real checkpoint smoke",
            ),
            TrackerMode.TRACKING,
        )

        self.assertEqual(matcher.config.search_size, 384)
        self.assertEqual(matcher.config.template_size, 192)
        self.assertGreaterEqual(evidence.match_score, 0.0)
        self.assertLessEqual(evidence.match_score, 1.0)


def _require_numpy():
    try:
        import numpy as np
    except ImportError as error:
        raise unittest.SkipTest("numpy is required for OSTrack matcher tests") from error
    return np


if __name__ == "__main__":
    unittest.main()
