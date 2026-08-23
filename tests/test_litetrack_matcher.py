from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import unittest

from BigTracker.matcher_models.litetrack import (
    LiteTrackMatcherConfig,
    LiteTrackMatcherModel,
    LiteTrackTemplate,
)
from BigTracker.types import (
    MatcherInitializeInput,
    MatcherMatchInput,
    MatcherTemplateInput,
    MatcherUpdateInput,
)


@dataclass(frozen=True)
class _Frame:
    image: object
    idx: int
    timestamp: float


class _FakeProcessed:
    def __init__(self, tensors):
        self.tensors = tensors


class _FakeLiteTrackBackend:
    def __init__(self) -> None:
        np = _require_numpy()
        self.output_window = np.ones((1, 1, 3, 3), dtype=np.float32)
        self.eval_count = 0
        self.preprocess_count = 0
        self.encode_count = 0
        self.forward_count = 0
        self.forward_calls = []

    def eval(self):
        self.eval_count += 1
        return self

    def preprocess(self, image, attention_mask):
        np = _require_numpy()
        self.preprocess_count += 1
        return _FakeProcessed(np.asarray(image, dtype=np.float32))

    def encode_template(self, template, source_box, resize_factor):
        self.encode_count += 1
        return {
            "feature_id": self.encode_count,
            "template_shape": tuple(template.shape),
            "source_box": source_box,
            "resize_factor": resize_factor,
        }

    def forward(self, template_features, search):
        np = _require_numpy()
        self.forward_count += 1
        self.forward_calls.append(
            {
                "template_features": template_features,
                "search_shape": tuple(search.shape),
            }
        )
        score_map = np.zeros((1, 1, 3, 3), dtype=np.float32)
        score_map[:, :, 1, 1] = 0.85
        return {
            "score_map": score_map,
            "size_map": np.zeros((1, 2, 3, 3), dtype=np.float32),
            "offset_map": np.zeros((1, 2, 3, 3), dtype=np.float32),
        }

    def cal_bbox(self, response, size_map, offset_map):
        np = _require_numpy()
        return np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)


class LiteTrackMatcherTest(unittest.TestCase):
    def setUp(self) -> None:
        np = _require_numpy()
        self.image = np.zeros((80, 90, 3), dtype=np.uint8)
        self.frame = _Frame(image=self.image, idx=0, timestamp=0.0)
        self.config = LiteTrackMatcherConfig(
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
        backend = _FakeLiteTrackBackend()
        calls = []

        def factory(config):
            calls.append(config)
            return backend

        matcher = LiteTrackMatcherModel(config=self.config, backend_factory=factory)
        self.assertEqual(len(calls), 1)
        self.assertEqual(backend.eval_count, 1)
        self.assertEqual(backend.encode_count, 0)

        matcher.initialize_template(MatcherInitializeInput(frame=self.frame, box=(20.0, 20.0, 20.0, 20.0)))
        self.assertEqual(len(calls), 1)
        self.assertEqual(backend.encode_count, 1)

    def test_initialize_template_sets_history_shape(self) -> None:
        backend = _FakeLiteTrackBackend()
        matcher = LiteTrackMatcherModel(config=self.config, backend=backend)

        result = matcher.initialize_template(MatcherInitializeInput(frame=self.frame, box=(20.0, 20.0, 20.0, 20.0)))
        state = matcher._require_state()

        self.assertTrue(result.ok)
        self.assertIsInstance(state.init_template, LiteTrackTemplate)
        self.assertEqual(state.best_templates, ())
        self.assertIs(state.adaptive_template, state.init_template)
        self.assertEqual(state.init_template.template_features["feature_id"], 1)

    def test_match_uses_target_positions_and_stored_template_features(self) -> None:
        backend = _FakeLiteTrackBackend()
        matcher = LiteTrackMatcherModel(config=self.config, backend=backend)
        matcher.initialize_template(MatcherInitializeInput(frame=self.frame, box=(20.0, 20.0, 20.0, 20.0)))

        output = matcher.match(MatcherMatchInput(frame=self.frame, target_poses=[(45.0, 40.0), (50.0, 42.0)]))

        self.assertEqual(len(output.bboxes), 2)
        self.assertEqual(len(output.scores), 2)
        self.assertAlmostEqual(output.scores[0], 0.85, places=5)
        self.assertAlmostEqual(output.bboxes[0][0] + output.bboxes[0][2] / 2.0, 45.0, delta=1.0)
        self.assertAlmostEqual(output.bboxes[0][1] + output.bboxes[0][3] / 2.0, 40.0, delta=1.0)
        self.assertEqual(backend.forward_count, 2)
        self.assertEqual(backend.forward_calls[0]["template_features"]["feature_id"], 1)
        self.assertEqual(backend.forward_calls[0]["search_shape"], (63, 63, 3))

    def test_update_templates_preserves_init_and_updates_adaptive(self) -> None:
        backend = _FakeLiteTrackBackend()
        matcher = LiteTrackMatcherModel(config=self.config, backend=backend)
        matcher.initialize_template(MatcherInitializeInput(frame=self.frame, box=(20.0, 20.0, 20.0, 20.0)))
        init_template = matcher._require_state().init_template

        candidate_1 = matcher.extract_template(MatcherTemplateInput(frame=self.frame, box=(26.0, 26.0, 18.0, 18.0)))
        matcher.update_templates(MatcherUpdateInput(template=candidate_1.template, score=0.4))
        candidate_2 = matcher.extract_template(MatcherTemplateInput(frame=self.frame, box=(32.0, 32.0, 16.0, 16.0)))
        matcher.update_templates(MatcherUpdateInput(template=candidate_2.template, score=0.9))
        state = matcher._require_state()

        self.assertIs(state.init_template, init_template)
        self.assertEqual(len(state.best_templates), 2)
        self.assertIs(state.adaptive_template, candidate_2.template)
        self.assertEqual(candidate_2.template.template_features["feature_id"], 3)

    def test_real_checkpoint_smoke_when_enabled(self) -> None:
        if os.environ.get("BIGTRACK_RUN_LITETRACK_REAL_SMOKE") != "1":
            raise unittest.SkipTest("set BIGTRACK_RUN_LITETRACK_REAL_SMOKE=1 to run LiteTrack real checkpoint smoke")

        np = _require_numpy()
        config_path = Path(r"ignores\Models\litetrack\config\B8_cae_center_all_ep300.yaml")
        checkpoint_path = Path(r"ignores\Models\litetrack\B8_cae_center_all_ep300\LiteTrack_ep0300.pth.tar")
        if not config_path.exists() or not checkpoint_path.exists():
            raise unittest.SkipTest("LiteTrack config/checkpoint assets are not present")

        image = np.zeros((240, 320, 3), dtype=np.uint8)
        frame = _Frame(image=image, idx=0, timestamp=0.0)
        matcher = LiteTrackMatcherModel(
            LiteTrackMatcherConfig(
                config_path=str(config_path),
                checkpoint_path=str(checkpoint_path),
                device=_real_device(),
            )
        )
        init = matcher.initialize_template(MatcherInitializeInput(frame=frame, box=(140.0, 100.0, 40.0, 40.0)))
        output = matcher.match(MatcherMatchInput(frame=frame, target_poses=[(160.0, 120.0)]))

        self.assertTrue(init.ok)
        self.assertEqual(matcher.config.search_size, 256)
        self.assertEqual(matcher.config.template_size, 128)
        self.assertEqual(len(output.bboxes), 1)
        self.assertGreaterEqual(output.scores[0], 0.0)
        self.assertLessEqual(output.scores[0], 1.0)


def _real_device() -> str:
    if os.environ.get("BIGTRACK_REAL_DEVICE"):
        return os.environ["BIGTRACK_REAL_DEVICE"]
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _require_numpy():
    try:
        import numpy as np
    except ImportError as error:
        raise unittest.SkipTest("numpy is required for LiteTrack matcher tests") from error
    return np


if __name__ == "__main__":
    unittest.main()
