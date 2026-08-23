from __future__ import annotations

import unittest

from BigTracker.big_trackers._decision import (
    SearchCandidate,
    box_center_distance_ratio,
    box_size_change_ratio,
    boxes_agree,
    clamp01,
    normalize_predictor_score,
    score_band,
)


class BigTrackDecisionHelperTest(unittest.TestCase):
    def test_clamp01_handles_invalid_and_out_of_range_values(self) -> None:
        self.assertEqual(clamp01(-0.25), 0.0)
        self.assertEqual(clamp01(1.25), 1.0)
        self.assertEqual(clamp01("bad", default=0.4), 0.4)
        self.assertEqual(clamp01(float("nan"), default=0.3), 0.3)

    def test_score_band_uses_bad_and_good_thresholds(self) -> None:
        self.assertEqual(score_band(0.8, th_bad=0.3, th_good=0.7), "good")
        self.assertEqual(score_band(0.5, th_bad=0.3, th_good=0.7), "weak")
        self.assertEqual(score_band(0.2, th_bad=0.3, th_good=0.7), "bad")

    def test_normalize_predictor_score_penalizes_uncertainty(self) -> None:
        self.assertAlmostEqual(
            normalize_predictor_score(0.8, motion_uncertainty=1.0, uncertainty_scale=1.0),
            0.4,
        )
        self.assertAlmostEqual(normalize_predictor_score(None, motion_uncertainty=None), 1.0)

    def test_box_agreement_uses_center_and_size_errors(self) -> None:
        predicted = (10.0, 10.0, 20.0, 20.0)
        close = (12.0, 11.0, 21.0, 19.0)
        far = (80.0, 80.0, 20.0, 20.0)
        wrong_size = (10.0, 10.0, 45.0, 20.0)

        self.assertLess(box_center_distance_ratio(predicted, close), 0.2)
        self.assertLess(box_size_change_ratio(predicted, close), 0.11)
        self.assertTrue(boxes_agree(predicted, close, max_center_error=0.2, max_size_error=0.11))
        self.assertFalse(boxes_agree(predicted, far, max_center_error=0.2, max_size_error=0.11))
        self.assertFalse(boxes_agree(predicted, wrong_size, max_center_error=0.5, max_size_error=0.1))

    def test_search_candidate_is_bigtrack_local_context(self) -> None:
        candidate = SearchCandidate(
            candidate_id="predicted",
            search_center=(10.0, 10.0),
            prediction_confidence=0.8,
            motion_uncertainty=0.2,
            reason="unit_test",
        )

        self.assertEqual(candidate.search_center, (10.0, 10.0))
        self.assertEqual(candidate.reason, "unit_test")


if __name__ == "__main__":
    unittest.main()
