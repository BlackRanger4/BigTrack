from __future__ import annotations

import unittest

from BigTracker.matcher_models._boxes import (
    box_to_center_size,
    center_size_to_box,
    clip_box,
    map_crop_box_back,
)
from BigTracker.matcher_models._crop import crop_centered, sample_target


class MatcherBoxUtilsTest(unittest.TestCase):
    def test_box_center_size_roundtrip(self) -> None:
        box = (10.0, 20.0, 30.0, 40.0)

        center, size = box_to_center_size(box)
        self.assertEqual(center, (25.0, 40.0))
        self.assertEqual(size, (30.0, 40.0))
        self.assertEqual(center_size_to_box(center, size), box)

    def test_clip_box_to_image_bounds(self) -> None:
        clipped = clip_box((-5.0, 3.0, 20.0, 15.0), (10, 12, 3))

        self.assertEqual(clipped, (0.0, 3.0, 12.0, 7.0))

    def test_map_crop_box_back(self) -> None:
        mapped = map_crop_box_back(
            pred_box_cxcywh=(50.0, 50.0, 20.0, 10.0),
            crop_center=(100.0, 80.0),
            search_size=100.0,
            resize_factor=1.0,
        )

        self.assertEqual(mapped, (90.0, 75.0, 20.0, 10.0))


class MatcherCropUtilsTest(unittest.TestCase):
    def test_crop_centered_pads_outside_image(self) -> None:
        np = _require_numpy()
        image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)

        crop = crop_centered(
            image=image,
            center=(1.0, 1.0),
            size=(4.0, 4.0),
            pad_value=255,
        )

        self.assertEqual(crop.image.shape, (4, 4, 3))
        self.assertEqual(crop.crop_box, (-1.0, -1.0, 4.0, 4.0))
        self.assertTrue(crop.is_clipped)
        self.assertTrue(bool(crop.attention_mask[0, 0]))
        self.assertFalse(bool(crop.attention_mask[1, 1]))

    def test_sample_target_resize_factor(self) -> None:
        _require_cv2()
        np = _require_numpy()
        image = np.zeros((20, 20, 3), dtype=np.uint8)

        crop = sample_target(
            image=image,
            target_box=(5.0, 5.0, 4.0, 4.0),
            search_area_factor=2.0,
            output_size=16,
        )

        self.assertEqual(crop.image.shape, (16, 16, 3))
        self.assertEqual(crop.resize_factor, 2.0)
        self.assertEqual(crop.crop_box, (3.0, 3.0, 8.0, 8.0))


def _require_numpy():
    try:
        import numpy as np
    except ImportError as error:
        raise unittest.SkipTest("numpy is required for crop utility tests") from error
    return np


def _require_cv2():
    try:
        import cv2
    except ImportError as error:
        raise unittest.SkipTest("opencv-python is required for resize crop tests") from error
    return cv2


if __name__ == "__main__":
    unittest.main()

