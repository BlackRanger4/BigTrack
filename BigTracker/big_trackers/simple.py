from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from BigTracker.big_trackers.base import BaseBigTrack
from BigTracker.state import (
    BigTrackDecision,
    BigTrackState,
    MatchEvidence,
    SearchCandidate,
    TrackerPredictionState,
    TrackingOutput,
)
from BigTracker.types import Box, FrameLike, OutputStatus, Point, Size, TrackerMode


class SimpleBigTrack(BaseBigTrack):
    """Minimal tracker policy that blindly trusts the matcher.

    This class is for first integration tests. It does not threshold scores,
    does not create recovery candidates, does not handle lost state, and does
    not update templates.
    """

    def make_candidates(
        self,
        state: BigTrackState,
        prediction: TrackerPredictionState,
        frame: FrameLike,
    ) -> Sequence[SearchCandidate]:
        """Create one search candidate at the predictor's target position."""

        return (
            SearchCandidate(
                candidate_id="predicted",
                search_center=prediction.target_pos,
                predicted_target_size=prediction.target_size,
                prediction_confidence=prediction.last_score,
                motion_uncertainty=prediction.uncertainty,
                reason="simple_predicted_position",
                metadata={
                    "tracker": "simple",
                    "frame_idx": frame.idx,
                    "previous_mode": state.mode.value,
                },
            ),
        )

    def decide(
        self,
        state: BigTrackState,
        prediction: TrackerPredictionState,
        candidates: Sequence[SearchCandidate],
        matches: Sequence[MatchEvidence],
    ) -> BigTrackDecision:
        """Accept the first matcher result without score thresholds."""

        if not matches:
            raise ValueError("SimpleBigTrack requires one matcher result")

        match = matches[0]
        accepted_pos, accepted_size = _box_to_center_size(match.box)
        return BigTrackDecision(
            accepted=True,
            accepted_box=match.box,
            accepted_target_pos=accepted_pos,
            accepted_target_size=accepted_size,
            output_status=OutputStatus.ACTIVE,
            next_mode=TrackerMode.TRACKING,
            confidence=match.match_score,
            allow_template_update=False,
            reason="simple_trust_matcher",
        )

    def apply_decision(
        self,
        state: BigTrackState,
        prediction: TrackerPredictionState,
        decision: BigTrackDecision,
        frame: FrameLike,
    ) -> BigTrackState:
        """Apply the accepted matcher box as the new tracker state."""

        if not decision.accepted:
            raise ValueError("SimpleBigTrack does not support rejected decisions")
        if decision.accepted_box is None:
            raise ValueError("SimpleBigTrack accepted decision requires a box")
        if decision.accepted_target_pos is None or decision.accepted_target_size is None:
            raise ValueError("SimpleBigTrack accepted decision requires target geometry")

        predicted_state = replace(state, prediction=prediction)
        next_prediction = self.predictor.update_from_accept(
            state=predicted_state,
            accepted_pos=decision.accepted_target_pos,
            accepted_size=decision.accepted_target_size,
            score=decision.confidence,
        )
        output = TrackingOutput(
            box=decision.accepted_box,
            frame_idx=frame.idx,
            timestamp=frame.timestamp,
            status=decision.output_status,
            confidence=decision.confidence,
        )
        counters = replace(
            state.counters,
            age=state.counters.age + 1,
            lost_count=0,
            uncertain_count=0,
            recovery_count=0,
        )
        return replace(
            state,
            prediction=next_prediction,
            output=output,
            mode=decision.next_mode,
            counters=counters,
            last_seen_frame=frame.idx,
        )


def _box_to_center_size(box: Box) -> tuple[Point, Size]:
    """Convert frame-coordinate x, y, width, height into center point and size."""

    x, y, width, height = box
    return (
        (float(x) + float(width) / 2.0, float(y) + float(height) / 2.0),
        (float(width), float(height)),
    )
