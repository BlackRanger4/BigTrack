from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import unittest

from BigTracker.matcher_models.mixformerv2 import (
    MixFormerV2MatcherConfig,
    MixFormerV2MatcherModel,
    MixFormerV2Template,
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


class _FakeMixFormerV2Backend:
    def __init__(self) -> None:
        self.eval_count = 0
        self.preprocess_count = 0
        self.forward_count = 0
        self.forward_calls = []

    def eval(self):
        self.eval_count += 1
        return self

    def preprocess(self, image):
        np = _require_numpy()
        self.preprocess_count += 1
        processed = np.asarray(image, dtype=np.float32).copy()
        processed[0, 0, 0] = float(self.preprocess_count)
        return processed

    def forward(self, template, online_template, search):
        np = _require_numpy()
        self.forward_count += 1
        self.forward_calls.append(
            {
                "template_shape": tuple(template.shape),
                "online_template_shape": tuple(online_template.shape),
                "search_shape": tuple(search.shape),
                "online_template_id": float(online_template[0, 0, 0]),
            }
        )
        return {
            "pred_boxes": np.array([[[0.5, 0.5, 0.5, 0.5]]], dtype=np.float32),
            "pred_scores": np.array([0.8], dtype=np.float32),
        }


class MixFormerV2MatcherTest(unittest.TestCase):
    def setUp(self) -> None:
        np = _require_numpy()
        self.image = np.zeros((80, 90, 3), dtype=np.uint8)
        self.image[30:50, 30:50, :] = 7
        self.frame = _Frame(image=self.image, idx=0, timestamp=0.0)
        self.config = MixFormerV2MatcherConfig(
            device="cpu",
            max_best_templates=2,
            template_factor=2.0,
            template_size=31,
            search_factor=2.0,
            search_size=63,
            pred_scores_are_logits=False,
            clip_margin=0.0,
        )

    def test_backend_factory_runs_in_init_not_initialize_template(self) -> None:
        backend = _FakeMixFormerV2Backend()
        calls = []

        def factory(config):
            calls.append(config)
            return backend

        matcher = MixFormerV2MatcherModel(config=self.config, backend_factory=factory)
        self.assertEqual(len(calls), 1)
        self.assertEqual(backend.eval_count, 1)
        self.assertEqual(backend.preprocess_count, 0)

        matcher.initialize_template(MatcherInitializeInput(frame=self.frame, box=(20.0, 20.0, 20.0, 20.0)))
        self.assertEqual(len(calls), 1)
        self.assertEqual(backend.preprocess_count, 1)

    def test_initialize_template_sets_history_shape(self) -> None:
        backend = _FakeMixFormerV2Backend()
        matcher = MixFormerV2MatcherModel(config=self.config, backend=backend)

        result = matcher.initialize_template(MatcherInitializeInput(frame=self.frame, box=(20.0, 20.0, 20.0, 20.0)))
        state = matcher._require_state()

        self.assertTrue(result.ok)
        self.assertIsInstance(state.init_template, MixFormerV2Template)
        self.assertEqual(state.best_templates, ())
        self.assertIs(state.adaptive_template, state.init_template)

    def test_match_uses_target_positions_and_does_not_mutate_template_queue(self) -> None:
        backend = _FakeMixFormerV2Backend()
        matcher = MixFormerV2MatcherModel(config=self.config, backend=backend)
        matcher.initialize_template(MatcherInitializeInput(frame=self.frame, box=(20.0, 20.0, 20.0, 20.0)))
        state = matcher._require_state()

        output = matcher.match(MatcherMatchInput(frame=self.frame, target_poses=[(45.0, 40.0), (50.0, 42.0)]))

        self.assertEqual(len(output.bboxes), 2)
        self.assertEqual(len(output.scores), 2)
        self.assertAlmostEqual(output.scores[0], 0.8, places=5)
        self.assertAlmostEqual(output.bboxes[0][0] + output.bboxes[0][2] / 2.0, 45.0, delta=1.0)
        self.assertAlmostEqual(output.bboxes[0][1] + output.bboxes[0][3] / 2.0, 40.0, delta=1.0)
        self.assertEqual(state.best_templates, ())
        self.assertIs(state.adaptive_template, state.init_template)
        self.assertEqual(backend.forward_count, 2)
        self.assertEqual(backend.forward_calls[0]["search_shape"], (63, 63, 3))

    def test_update_templates_preserves_init_and_updates_adaptive(self) -> None:
        backend = _FakeMixFormerV2Backend()
        matcher = MixFormerV2MatcherModel(config=self.config, backend=backend)
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

    def test_match_uses_adaptive_template_after_approved_update(self) -> None:
        np = _require_numpy()
        backend = _FakeMixFormerV2Backend()
        matcher = MixFormerV2MatcherModel(config=self.config, backend=backend)
        matcher.initialize_template(MatcherInitializeInput(frame=self.frame, box=(20.0, 20.0, 20.0, 20.0)))
        new_image = np.zeros((80, 90, 3), dtype=np.uint8)
        new_image[30:50, 30:50, :] = 13
        new_frame = _Frame(image=new_image, idx=1, timestamp=1.0)
        template = matcher.extract_template(MatcherTemplateInput(frame=new_frame, box=(32.0, 32.0, 16.0, 16.0)))
        matcher.update_templates(MatcherUpdateInput(template=template.template, score=0.9))

        matcher.match(MatcherMatchInput(frame=new_frame, target_poses=[(45.0, 40.0)]))

        self.assertGreater(backend.forward_calls[-1]["online_template_id"], 0.0)

    def test_real_checkpoint_smoke_when_enabled(self) -> None:
        if os.environ.get("BIGTRACK_RUN_MIXFORMERV2_REAL_SMOKE") != "1":
            raise unittest.SkipTest(
                "set BIGTRACK_RUN_MIXFORMERV2_REAL_SMOKE=1 to run MixFormerV2 real checkpoint smoke"
            )

        np = _require_numpy()
        config_path = Path(r"ignores\Models\mixformerv2\config\224_depth4_mlp1_score.yaml")
        checkpoint_path = Path(r"ignores\Models\mixformerv2\models\mixformerv2_small.pth.tar")
        if not config_path.exists() or not checkpoint_path.exists():
            raise unittest.SkipTest("MixFormerV2 config/checkpoint assets are not present")

        image = np.zeros((240, 320, 3), dtype=np.uint8)
        frame = _Frame(image=image, idx=0, timestamp=0.0)
        matcher = MixFormerV2MatcherModel(
            MixFormerV2MatcherConfig(
                config_path=str(config_path),
                checkpoint_path=str(checkpoint_path),
                device=_real_device(),
                variant="online",
            )
        )
        init = matcher.initialize_template(MatcherInitializeInput(frame=frame, box=(140.0, 100.0, 40.0, 40.0)))
        output = matcher.match(MatcherMatchInput(frame=frame, target_poses=[(160.0, 120.0)]))

        self.assertTrue(init.ok)
        self.assertEqual(matcher.config.search_size, 224)
        self.assertEqual(matcher.config.template_size, 112)
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
        raise unittest.SkipTest("numpy is required for MixFormerV2 matcher tests") from error
    return np


if __name__ == "__main__":
    unittest.main()
