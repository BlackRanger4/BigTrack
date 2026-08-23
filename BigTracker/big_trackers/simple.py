from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from BigTracker.big_trackers._decision import BigTrackDecision, SearchCandidate
from BigTracker.big_trackers.base import BaseBigTrack
from BigTracker.types import (
    BigTrackState,
    BigTrackUpdateOutput,
    Box,
    FrameLike,
    OutputStatus,
    Point,
    PredictorUpdateInput,
    TrackerMode,
    TrackerPredictionState,
)


class SimpleBigTrack(BaseBigTrack):
    """Minimal tracker policy that blindly trusts the matcher."""

    def make_candidates(
        self,
        state: BigTrackState,
        prediction: TrackerPredictionState,
        frame: FrameLike,
    ) -> Sequence[SearchCandidate]:
        return (
            SearchCandidate(
                candidate_id="predicted",
                search_center=prediction.target_pos,
                prediction_confidence=1.0,
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
        bboxes: Sequence[Box],
        scores: Sequence[float],
    ) -> BigTrackDecision:
        if not bboxes or not scores:
            raise ValueError("SimpleBigTrack requires one matcher result")

        box = bboxes[0]
        score = scores[0]
        return BigTrackDecision(
            accepted=True,
            accepted_box=box,
            accepted_target_pos=_box_to_center(box),
            output_status=OutputStatus.ACTIVE,
            next_mode=TrackerMode.TRACKING,
            confidence=score,
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
        if not decision.accepted:
            raise ValueError("SimpleBigTrack does not support rejected decisions")
        if decision.accepted_box is None or decision.accepted_target_pos is None:
            raise ValueError("SimpleBigTrack accepted decision requires target geometry")

        next_prediction = replace(prediction, target_pos=decision.accepted_target_pos)
        self.predictor.update(
            PredictorUpdateInput(
                accepted=True,
                predictor_state=next_prediction,
                metadata={"score": decision.confidence},
            )
        )
        output = BigTrackUpdateOutput(
            ok=True,
            box=decision.accepted_box,
            frame_idx=frame.idx,
            timestamp=frame.timestamp,
            status=decision.output_status,
            confidence=decision.confidence,
        )
        metadata = {
            **dict(state.metadata),
            "age": int(state.metadata.get("age", 0)) + 1,
            "target_size": _box_size(decision.accepted_box),
        }
        return replace(
            state,
            predictor_state=next_prediction,
            output=output,
            mode=decision.next_mode,
            last_seen_frame=frame.idx,
            metadata=metadata,
        )


def _box_to_center(box: Box) -> Point:
    x, y, width, height = box
    return (float(x) + float(width) / 2.0, float(y) + float(height) / 2.0)


def _box_size(box: Box) -> tuple[float, float]:
    return (float(box[2]), float(box[3]))
