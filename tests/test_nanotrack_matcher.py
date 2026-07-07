from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import unittest

from BigTracker.big_trackers.simple import SimpleBigTrack
from BigTracker.matcher_models.nanotrack import (
    NanoTrackMatcherConfig,
    NanoTrackMatcherModel,
    NanoTrackTemplate,
    _SplitNanoTrackOnnxBackend,
    _SplitNanoTrackTorchBackend,
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


class _FakeNanoTrackModel:
    def __init__(self) -> None:
        self.eval_count = 0
        self.template_count = 0
        self.track_count = 0
        self.zf = None
        self.track_features = []

    def eval(self):
        self.eval_count += 1
        return self

    def template(self, z):
        self.template_count += 1
        self.zf = {"from_template_model": True, "shape": tuple(z.shape)}
        return None

    def track(self, x):
        self.track_count += 1
        self.track_features.append(self.zf)
        return {"cls": x, "loc": x}


class _FakeOnnxInput:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class _FakeOnnxSession:
    def __init__(self, inputs, output):
        self._inputs = inputs
        self.output = output
        self.calls = []

    def get_inputs(self):
        return self._inputs

    def run(self, output_names, feed):
        self.calls.append(feed)
        return [self.output]


class _FakeOnnxHeadSession(_FakeOnnxSession):
    def __init__(self, inputs, cls_output, loc_output):
        super().__init__(inputs, cls_output)
        self.loc_output = loc_output

    def run(self, output_names, feed):
        self.calls.append(feed)
        return [self.output, self.loc_output]


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

    def test_split_torch_backend_keeps_template_and_search_models_separate(self) -> None:
        template_model = _FakeNanoTrackModel()
        search_model = _FakeNanoTrackModel()
        backend = _SplitNanoTrackTorchBackend(template_model, search_model)

        backend.eval()
        template_features = backend.template(_FakeTensor((1, 3, 31, 31)))
        outputs = backend.track(_FakeTensor((1, 3, 63, 63)))

        self.assertEqual(template_model.eval_count, 1)
        self.assertEqual(search_model.eval_count, 1)
        self.assertEqual(template_model.template_count, 1)
        self.assertEqual(search_model.template_count, 0)
        self.assertEqual(template_model.track_count, 0)
        self.assertEqual(search_model.track_count, 1)
        self.assertIs(search_model.track_features[0], template_features)
        self.assertEqual(outputs["cls"].shape, (1, 3, 63, 63))

    def test_split_onnx_backend_uses_separate_template_and_search_backbones(self) -> None:
        np = _require_numpy()
        template_output = np.ones((1, 96, 16, 16), dtype=np.float32)
        search_output = np.full((1, 96, 16, 16), 2.0, dtype=np.float32)
        cls_output = np.zeros((1, 2, 3, 3), dtype=np.float32)
        loc_output = np.zeros((1, 4, 3, 3), dtype=np.float32)
        template_session = _FakeOnnxSession([_FakeOnnxInput("input", [1, 3, 255, 255])], template_output)
        search_session = _FakeOnnxSession([_FakeOnnxInput("input", [1, 3, 255, 255])], search_output)
        head_session = _FakeOnnxHeadSession(
            [
                _FakeOnnxInput("input1", [1, 96, 8, 8]),
                _FakeOnnxInput("input2", [1, 96, 16, 16]),
            ],
            cls_output,
            loc_output,
        )
        backend = _SplitNanoTrackOnnxBackend(template_session, search_session, head_session)

        zf = backend.template(np.zeros((1, 3, 127, 127), dtype=np.float32))
        outputs = backend.track(np.zeros((1, 3, 255, 255), dtype=np.float32))

        self.assertEqual(template_session.calls[0]["input"].shape, (1, 3, 255, 255))
        self.assertEqual(search_session.calls[0]["input"].shape, (1, 3, 255, 255))
        self.assertEqual(zf.shape, (1, 96, 8, 8))
        self.assertIs(head_session.calls[0]["input1"], zf)
        self.assertIs(head_session.calls[0]["input2"], search_output)
        self.assertIs(outputs["cls"], cls_output)
        self.assertIs(outputs["loc"], loc_output)

    def test_real_checkpoint_smoke_when_enabled(self) -> None:
        if os.environ.get("BIGTRACK_RUN_NANOTRACK_REAL_SMOKE") != "1":
            raise unittest.SkipTest(
                "set BIGTRACK_RUN_NANOTRACK_REAL_SMOKE=1 to run NanoTrack real checkpoint smoke"
            )

        np = _require_numpy()
        config_path = Path(r"ignores\Models\nanotrack\config\configv3.yaml")
        checkpoint_path = Path(r"ignores\Models\nanotrack\pretrained\nanotrackv3.pth")
        if not config_path.exists() or not checkpoint_path.exists():
            raise unittest.SkipTest("NanoTrack config/checkpoint assets are not present")

        image = np.zeros((240, 320, 3), dtype=np.uint8)
        frame = _Frame(image=image, idx=0, timestamp=0.0)
        matcher = NanoTrackMatcherModel(
            NanoTrackMatcherConfig(
                backend="torch",
                source_root=r"ignores\Trackers\NanoTrack",
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

        self.assertEqual(matcher.config.output_size, 15)
        self.assertEqual(matcher.config.point_stride, 16)
        self.assertGreaterEqual(evidence.match_score, 0.0)
        self.assertLessEqual(evidence.match_score, 1.0)
        self.assertIsNot(matcher.backend.template_model, matcher.backend.search_model)


class _FakeTensor:
    def __init__(self, shape):
        self.shape = shape


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
