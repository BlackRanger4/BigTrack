from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from BigTracker.big_trackers._decision import (
    boxes_agree,
    clamp01,
    normalize_predictor_score,
    score_band,
    select_best_match,
)
from BigTracker.big_trackers.base import BaseBigTrack
from BigTracker.state import (
    BigTrackCounters,
    BigTrackDecision,
    BigTrackState,
    MatchEvidence,
    SearchCandidate,
    TrackerPredictionState,
    TrackingOutput,
)
from BigTracker.types import Box, FrameLike, OutputStatus, Point, Size, TrackerMode


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
        """Create one search candidate at the predictor's current target position."""

        return (
            SearchCandidate(
                candidate_id="predicted",
                search_center=prediction.target_pos,
                predicted_target_size=prediction.target_size,
                prediction_confidence=normalize_predictor_score(
                    prediction.last_score,
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
        matches: Sequence[MatchEvidence],
    ) -> BigTrackDecision:
        """Choose visual evidence only when score and geometry are acceptable."""

        predicted_box = _center_size_to_box(prediction.target_pos, prediction.target_size)
        predictor_score = normalize_predictor_score(
            prediction.last_score,
            prediction.uncertainty,
            self.config.predictor_uncertainty_scale,
        )
        current_frame_idx = _candidate_frame_idx(candidates, state.output.frame_idx)

        choice = select_best_match(candidates, matches, score_fn=lambda match: clamp01(match.match_score))
        if choice is None:
            return self._reject_to_prediction(
                state=state,
                predicted_box=predicted_box,
                predictor_score=predictor_score,
                reason="no_match",
            )

        match = choice.match
        matcher_score = clamp01(match.match_score)
        band = score_band(matcher_score, self.config.th_bad, self.config.th_good)

        if band == "good":
            accepted_pos, accepted_size = _box_to_center_size(match.box)
            return BigTrackDecision(
                accepted=True,
                accepted_box=match.box,
                accepted_target_pos=accepted_pos,
                accepted_target_size=accepted_size,
                output_status=OutputStatus.ACTIVE,
                next_mode=TrackerMode.TRACKING,
                confidence=matcher_score,
                allow_template_update=self._allow_template_update(state, match, current_frame_idx),
                reason="good_match",
            )

        if state.mode in (TrackerMode.RECOVERY, TrackerMode.LOST):
            return self._reject_to_prediction(
                state=state,
                predicted_box=predicted_box,
                predictor_score=min(predictor_score, matcher_score),
                reason=f"{band}_match_cannot_recover",
            )

        if band == "weak":
            agrees = boxes_agree(
                predicted_box,
                match.box,
                self.config.max_center_error,
                self.config.max_size_error,
            )
            if agrees:
                accepted_pos, accepted_size = _box_to_center_size(match.box)
                next_mode = TrackerMode.TRACKING if state.mode == TrackerMode.TRACKING else TrackerMode.UNCERTAIN
                output_status = OutputStatus.ACTIVE if next_mode == TrackerMode.TRACKING else OutputStatus.UNCERTAIN
                return BigTrackDecision(
                    accepted=True,
                    accepted_box=match.box,
                    accepted_target_pos=accepted_pos,
                    accepted_target_size=accepted_size,
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
                reason="weak_match_far_from_prediction",
                preferred_mode=TrackerMode.UNCERTAIN,
            )

        return self._reject_to_prediction(
            state=state,
            predicted_box=predicted_box,
            predictor_score=min(predictor_score, matcher_score),
            reason="bad_match_score",
            preferred_mode=TrackerMode.OCCLUDED,
        )

    def apply_decision(
        self,
        state: BigTrackState,
        prediction: TrackerPredictionState,
        decision: BigTrackDecision,
        frame: FrameLike,
    ) -> BigTrackState:
        """Apply accepted visual evidence or propagate the predictor on reject."""

        predicted_state = replace(state, prediction=prediction)
        if decision.accepted:
            if decision.accepted_target_pos is None or decision.accepted_target_size is None:
                raise ValueError("Accepted decision requires target position and size")
            next_prediction = self.predictor.update_from_accept(
                state=predicted_state,
                accepted_pos=decision.accepted_target_pos,
                accepted_size=decision.accepted_target_size,
                score=decision.confidence,
            )
        else:
            next_prediction = self.predictor.update_from_reject(predicted_state)

        output = TrackingOutput(
            box=decision.accepted_box,
            frame_idx=frame.idx,
            timestamp=frame.timestamp,
            status=decision.output_status,
            confidence=decision.confidence,
        )
        metadata = dict(state.metadata)
        metadata.setdefault("score_gated_initial_frame", state.output.frame_idx)
        metadata["score_gated_last_reason"] = decision.reason
        metadata["score_gated_last_template_update"] = bool(decision.allow_template_update)
        if decision.allow_template_update:
            metadata["score_gated_last_template_update_frame"] = frame.idx

        return replace(
            state,
            prediction=next_prediction,
            output=output,
            mode=decision.next_mode,
            counters=self._next_counters(state, decision),
            last_seen_frame=frame.idx if decision.accepted else state.last_seen_frame,
            metadata=metadata,
        )

    def _allow_template_update(
        self,
        state: BigTrackState,
        match: MatchEvidence,
        current_frame_idx: int,
    ) -> bool:
        """Allow template updates only on good matches at the configured interval."""

        if clamp01(match.match_score) < clamp01(self.config.th_good):
            return False
        if match.is_clipped and not self.config.template_allow_clipped:
            return False

        interval = max(1, int(self.config.template_update_interval))
        last_update_frame = int(
            state.metadata.get(
                "score_gated_last_template_update_frame",
                state.metadata.get("score_gated_initial_frame", state.output.frame_idx),
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
        """Build a rejected visual decision that emits the predictor box when possible."""

        next_mode = self._reject_mode(state, preferred_mode)
        output_status = _status_for_mode(next_mode)
        output_box = None if next_mode == TrackerMode.LOST else predicted_box
        confidence = 0.0 if next_mode == TrackerMode.LOST else clamp01(predictor_score)

        return BigTrackDecision(
            accepted=False,
            accepted_box=output_box,
            accepted_target_pos=None,
            accepted_target_size=None,
            output_status=output_status,
            next_mode=next_mode,
            confidence=confidence,
            allow_template_update=False,
            reason=reason,
        )

    def _reject_mode(self, state: BigTrackState, preferred_mode: TrackerMode | None) -> TrackerMode:
        if state.mode == TrackerMode.LOST:
            return TrackerMode.LOST

        if state.mode == TrackerMode.RECOVERY:
            next_recovery_count = state.counters.recovery_count + 1
            if next_recovery_count >= max(1, int(self.config.lost_after)):
                return TrackerMode.LOST
            return TrackerMode.RECOVERY

        next_uncertain_count = state.counters.uncertain_count + 1
        if next_uncertain_count >= max(1, int(self.config.recovery_after)):
            return TrackerMode.RECOVERY
        return preferred_mode or TrackerMode.UNCERTAIN

    def _next_counters(self, state: BigTrackState, decision: BigTrackDecision) -> BigTrackCounters:
        age = state.counters.age + 1

        if decision.next_mode == TrackerMode.TRACKING:
            return BigTrackCounters(age=age)
        if decision.next_mode == TrackerMode.UNCERTAIN:
            return BigTrackCounters(
                age=age,
                uncertain_count=state.counters.uncertain_count + 1,
                lost_count=0,
                recovery_count=0,
            )
        if decision.next_mode == TrackerMode.OCCLUDED:
            return BigTrackCounters(
                age=age,
                uncertain_count=state.counters.uncertain_count + 1,
                lost_count=state.counters.lost_count + 1,
                recovery_count=0,
            )
        if decision.next_mode == TrackerMode.RECOVERY:
            return BigTrackCounters(
                age=age,
                uncertain_count=state.counters.uncertain_count + 1,
                lost_count=state.counters.lost_count + 1,
                recovery_count=state.counters.recovery_count + 1,
            )
        return BigTrackCounters(
            age=age,
            uncertain_count=state.counters.uncertain_count,
            lost_count=state.counters.lost_count,
            recovery_count=state.counters.recovery_count + 1,
        )


def _box_to_center_size(box: Box) -> tuple[Point, Size]:
    x, y, width, height = box
    return (
        (float(x) + float(width) / 2.0, float(y) + float(height) / 2.0),
        (float(width), float(height)),
    )


def _center_size_to_box(target_pos: Point, target_size: Size) -> Box:
    width, height = target_size
    return (
        float(target_pos[0]) - float(width) / 2.0,
        float(target_pos[1]) - float(height) / 2.0,
        float(width),
        float(height),
    )


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
