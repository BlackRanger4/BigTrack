from __future__ import annotations

import unittest

from BigTracker.big_trackers._decision import (
    box_center_distance_ratio,
    box_size_change_ratio,
    boxes_agree,
    clamp01,
    combine_evidence_score,
    evidence_reject_reasons,
    normalize_predictor_score,
    score_band,
    select_best_match,
)
from BigTracker.state import MatchEvidence, SearchCandidate


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

    def test_combine_evidence_penalizes_ambiguity_occlusion_and_clipping(self) -> None:
        clean = _match("a", match_score=0.9)
        ambiguous = _match("b", match_score=0.9, ambiguity_score=0.8)
        clipped = _match("c", match_score=0.9, is_clipped=True)

        clean_score = combine_evidence_score(clean)

        self.assertGreater(clean_score, combine_evidence_score(ambiguous))
        self.assertGreater(clean_score, combine_evidence_score(clipped))

    def test_reject_reasons_report_failed_thresholds(self) -> None:
        reasons = evidence_reject_reasons(
            _match(
                "a",
                match_score=0.2,
                ambiguity_score=0.8,
                scale_score=0.4,
                occlusion_score=0.7,
                is_clipped=True,
            ),
            min_match_score=0.5,
            max_ambiguity_score=0.3,
            min_scale_score=0.6,
            max_occlusion_score=0.5,
            allow_clipped=False,
        )

        self.assertEqual(
            reasons,
            ("low_match_score", "high_ambiguity", "bad_scale", "high_occlusion", "clipped"),
        )

    def test_select_best_match_preserves_candidate_context(self) -> None:
        candidates = (
            _candidate("weak", confidence=0.1),
            _candidate("strong", confidence=0.9),
        )
        matches = (
            _match("weak", match_score=0.4),
            _match("strong", match_score=0.8),
        )

        choice = select_best_match(candidates, matches)

        self.assertIsNotNone(choice)
        self.assertEqual(choice.match.candidate_id, "strong")
        self.assertEqual(choice.candidate.candidate_id, "strong")
        self.assertEqual(choice.metadata["candidate_metadata"]["source"], "test")


def _candidate(candidate_id: str, confidence: float) -> SearchCandidate:
    return SearchCandidate(
        candidate_id=candidate_id,
        search_center=(10.0, 10.0),
        predicted_target_size=(20.0, 20.0),
        prediction_confidence=confidence,
        motion_uncertainty=0.0,
        reason="unit_test",
        metadata={"source": "test"},
    )


def _match(
    candidate_id: str,
    *,
    match_score: float,
    ambiguity_score: float = 0.0,
    scale_score: float = 1.0,
    occlusion_score: float = 0.0,
    is_clipped: bool = False,
) -> MatchEvidence:
    return MatchEvidence(
        candidate_id=candidate_id,
        box=(10.0, 10.0, 20.0, 20.0),
        match_score=match_score,
        identity_score=1.0,
        appearance_score=match_score,
        localization_score=match_score,
        ambiguity_score=ambiguity_score,
        scale_score=scale_score,
        occlusion_score=occlusion_score,
        is_clipped=is_clipped,
        metadata={"match": candidate_id},
    )


if __name__ == "__main__":
    unittest.main()
