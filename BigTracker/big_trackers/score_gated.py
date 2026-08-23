from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from BigTracker.big_trackers._decision import (
    BigTrackCounters,
    BigTrackDecision,
    SearchCandidate,
    boxes_agree,
    clamp01,
    normalize_predictor_score,
    score_band,
)
from BigTracker.big_trackers.base import BaseBigTrack
from BigTracker.types import (
    BigTrackState,
    BigTrackUpdateOutput,
    Box,
    FrameLike,
    OutputStatus,
    Point,
    PredictorUpdateInput,
    Size,
    TrackerMode,
    TrackerPredictionState,
)


@dataclass(frozen=True)
class ScoreGatedBigTrackConfig:
    """Thresholds and lifecycle limits for one-candidate score-gated tracking."""

    th_good: float = 0.70
    th_bad: float = 0.30
    max_center_error: float = 0.35
    max_size_error: float = 0.50
    predictor_uncertainty_scale: float = 10.0
    recovery_after: int = 3
    lost_after: int = 10
    template_update_interval: int = 5
    template_allow_clipped: bool = False


class ScoreGatedBigTrack(BaseBigTrack):
    """One-candidate policy using matcher score and predictor agreement."""

    def __init__(self, predictor, matcher, config: ScoreGatedBigTrackConfig | None = None) -> None:
        super().__init__(predictor=predictor, matcher=matcher)
        self.config = config or ScoreGatedBigTrackConfig()

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
                prediction_confidence=normalize_predictor_score(
                    _last_score(prediction),
                    prediction.uncertainty,
                    self.config.predictor_uncertainty_scale,
                ),
                motion_uncertainty=prediction.uncertainty,
                reason="score_gated_predicted_position",
                metadata={
                    "tracker": "score_gated",
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
        target_size = _target_size(state)
        predicted_box = _center_size_to_box(prediction.target_pos, target_size)
        predictor_score = normalize_predictor_score(
            _last_score(prediction),
            prediction.uncertainty,
            self.config.predictor_uncertainty_scale,
        )
        current_frame_idx = _candidate_frame_idx(candidates, state.output.frame_idx if state.output else 0)

        best_index = _best_score_index(scores)
        if best_index is None or best_index >= len(bboxes):
            return self._reject_to_prediction(
                state=state,
                predicted_box=predicted_box,
                predictor_score=predictor_score,
                reason="no_match",
            )

        matched_box = bboxes[best_index]
        matcher_score = clamp01(scores[best_index])
        band = score_band(matcher_score, self.config.th_bad, self.config.th_good)

        if state.mode in (TrackerMode.RECOVERY, TrackerMode.LOST):
            return self._reject_to_prediction(
                state=state,
                predicted_box=predicted_box,
                predictor_score=min(predictor_score, matcher_score),
                reason=f"{band}_match_cannot_recover",
            )

        if band == "good":
            return BigTrackDecision(
                accepted=True,
                accepted_box=matched_box,
                accepted_target_pos=_box_to_center(matched_box),
                output_status=OutputStatus.ACTIVE,
                next_mode=TrackerMode.TRACKING,
                confidence=matcher_score,
                allow_template_update=self._allow_template_update(matcher_score, state, current_frame_idx),
                reason="good_match",
            )


        if band == "weak" and boxes_agree(
            predicted_box,
            matched_box,
            self.config.max_center_error,
            self.config.max_size_error,
        ):
            next_mode = TrackerMode.TRACKING if state.mode == TrackerMode.TRACKING else TrackerMode.UNCERTAIN
            output_status = OutputStatus.ACTIVE if next_mode == TrackerMode.TRACKING else OutputStatus.UNCERTAIN
            return BigTrackDecision(
                accepted=True,
                accepted_box=matched_box,
                accepted_target_pos=_box_to_center(matched_box),
                output_status=output_status,
                next_mode=next_mode,
                confidence=matcher_score,
                allow_template_update=False,
                reason="weak_match_agrees_with_prediction",
            )

        return self._reject_to_prediction(
            state=state,
            predicted_box=predicted_box,
            predictor_score=min(predictor_score, matcher_score),
            reason="bad_match_score" if band == "bad" else "weak_match_far_from_prediction",
            preferred_mode=TrackerMode.OCCLUDED if band == "bad" else TrackerMode.UNCERTAIN,
        )

    def apply_decision(
        self,
        state: BigTrackState,
        prediction: TrackerPredictionState,
        decision: BigTrackDecision,
        frame: FrameLike,
    ) -> BigTrackState:
        if decision.accepted:
            if decision.accepted_box is None or decision.accepted_target_pos is None:
                raise ValueError("Accepted decision requires target position and box")
            next_prediction = replace(
                prediction,
                target_pos=decision.accepted_target_pos,
                metadata={**dict(prediction.metadata), "last_score": decision.confidence},
            )
            self.predictor.update(
                PredictorUpdateInput(
                    accepted=True,
                    predictor_state=next_prediction,
                    metadata={"score": decision.confidence},
                )
            )
        else:
            next_prediction = replace(
                prediction,
                metadata={**dict(prediction.metadata), "last_score": decision.confidence},
            )
            self.predictor.update(
                PredictorUpdateInput(
                    accepted=False,
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
        counters = self._next_counters(state, decision)
        metadata = dict(state.metadata)
        metadata["age"] = counters.age
        metadata["score_gated_counters"] = counters
        metadata.setdefault("score_gated_initial_frame", state.output.frame_idx if state.output else frame.idx)
        metadata["score_gated_last_reason"] = decision.reason
        metadata["score_gated_last_template_update"] = bool(decision.allow_template_update)
        if decision.accepted_box is not None:
            metadata["target_size"] = _box_size(decision.accepted_box)
        if decision.allow_template_update:
            metadata["score_gated_last_template_update_frame"] = frame.idx

        return replace(
            state,
            predictor_state=next_prediction,
            output=output,
            mode=decision.next_mode,
            last_seen_frame=frame.idx if decision.accepted else state.last_seen_frame,
            metadata=metadata,
        )

    def _allow_template_update(
        self,
        matcher_score: float,
        state: BigTrackState,
        current_frame_idx: int,
    ) -> bool:
        if clamp01(matcher_score) < clamp01(self.config.th_good):
            return False

        interval = max(1, int(self.config.template_update_interval))
        output_frame = state.output.frame_idx if state.output else current_frame_idx
        last_update_frame = int(
            state.metadata.get(
                "score_gated_last_template_update_frame",
                state.metadata.get("score_gated_initial_frame", output_frame),
            )
        )
        return current_frame_idx - last_update_frame >= interval

    def _reject_to_prediction(
        self,
        *,
        state: BigTrackState,
        predicted_box: Box,
        predictor_score: float,
        reason: str,
        preferred_mode: TrackerMode | None = None,
    ) -> BigTrackDecision:
        next_mode = self._reject_mode(state, preferred_mode)
        output_status = _status_for_mode(next_mode)
        output_box = None if next_mode == TrackerMode.LOST else predicted_box
        confidence = 0.0 if next_mode == TrackerMode.LOST else clamp01(predictor_score)

        return BigTrackDecision(
            accepted=False,
            accepted_box=output_box,
            accepted_target_pos=None,
            output_status=output_status,
            next_mode=next_mode,
            confidence=confidence,
            allow_template_update=False,
            reason=reason,
        )

    def _reject_mode(self, state: BigTrackState, preferred_mode: TrackerMode | None) -> TrackerMode:
        counters = _counters(state)
        if state.mode == TrackerMode.LOST:
            return TrackerMode.LOST

        if state.mode == TrackerMode.RECOVERY:
            if counters.recovery_count + 1 >= max(1, int(self.config.lost_after)):
                return TrackerMode.LOST
            return TrackerMode.RECOVERY

        if counters.uncertain_count + 1 >= max(1, int(self.config.recovery_after)):
            return TrackerMode.RECOVERY
        return preferred_mode or TrackerMode.UNCERTAIN

    def _next_counters(self, state: BigTrackState, decision: BigTrackDecision) -> BigTrackCounters:
        counters = _counters(state)
        age = counters.age + 1

        if decision.next_mode == TrackerMode.TRACKING:
            return BigTrackCounters(age=age)
        if decision.next_mode == TrackerMode.UNCERTAIN:
            return BigTrackCounters(age=age, uncertain_count=counters.uncertain_count + 1)
        if decision.next_mode == TrackerMode.OCCLUDED:
            return BigTrackCounters(
                age=age,
                uncertain_count=counters.uncertain_count + 1,
                lost_count=counters.lost_count + 1,
            )
        if decision.next_mode == TrackerMode.RECOVERY:
            return BigTrackCounters(
                age=age,
                uncertain_count=counters.uncertain_count + 1,
                lost_count=counters.lost_count + 1,
                recovery_count=counters.recovery_count + 1,
            )
        return BigTrackCounters(
            age=age,
            uncertain_count=counters.uncertain_count,
            lost_count=counters.lost_count,
            recovery_count=counters.recovery_count + 1,
        )


def _box_to_center(box: Box) -> Point:
    x, y, width, height = box
    return (float(x) + float(width) / 2.0, float(y) + float(height) / 2.0)


def _box_size(box: Box) -> Size:
    return (float(box[2]), float(box[3]))


def _center_size_to_box(target_pos: Point, target_size: Size) -> Box:
    width, height = target_size
    return (
        float(target_pos[0]) - float(width) / 2.0,
        float(target_pos[1]) - float(height) / 2.0,
        float(width),
        float(height),
    )


def _target_size(state: BigTrackState) -> Size:
    size = state.metadata.get("target_size")
    if size is not None:
        return (float(size[0]), float(size[1]))
    if state.output and state.output.box is not None:
        return _box_size(state.output.box)
    return (1.0, 1.0)


def _last_score(prediction: TrackerPredictionState) -> float:
    return float(prediction.metadata.get("last_score", 1.0))


def _counters(state: BigTrackState) -> BigTrackCounters:
    counters = state.metadata.get("score_gated_counters")
    if isinstance(counters, BigTrackCounters):
        return counters
    return BigTrackCounters(age=int(state.metadata.get("age", 0)))


def _best_score_index(scores: Sequence[float]) -> int | None:
    if not scores:
        return None
    return max(range(len(scores)), key=lambda index: (clamp01(scores[index]), index))


def _candidate_frame_idx(candidates: Sequence[SearchCandidate], fallback: int) -> int:
    for candidate in candidates:
        frame_idx = candidate.metadata.get("frame_idx")
        if frame_idx is not None:
            return int(frame_idx)
    return int(fallback)


def _status_for_mode(mode: TrackerMode) -> OutputStatus:
    if mode == TrackerMode.LOST:
        return OutputStatus.LOST
    if mode == TrackerMode.OCCLUDED:
        return OutputStatus.OCCLUDED
    return OutputStatus.UNCERTAIN
