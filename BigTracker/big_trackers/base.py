from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence

from BigTracker.big_track import BigTrack
from BigTracker.big_trackers._decision import BigTrackDecision, SearchCandidate
from BigTracker.matcher import Matcher
from BigTracker.predictor import Predictor
from BigTracker.types import (
    BigTrackInitializeInput,
    BigTrackInitializeOutput,
    BigTrackState,
    BigTrackUpdateInput,
    BigTrackUpdateOutput,
    Box,
    FrameLike,
    MatcherInitializeInput,
    MatcherMatchInput,
    MatcherTemplateInput,
    MatcherUpdateInput,
    OutputStatus,
    Point,
    PredictorInitializeInput,
    PredictorPredictInput,
    TrackerMode,
    TrackerPredictionState,
)


@dataclass(frozen=True)
class BigTrackDebugSnapshot:
    """Last BigTrack flow values exposed for visualization tools."""

    frame_idx: int
    timestamp: float
    predictor_target_pos: Point
    predictor_target_velocity: Point
    candidate_target_poses: tuple[Point, ...]
    matcher_bboxes: tuple[Box, ...]
    matcher_scores: tuple[float, ...]
    accepted_box: Optional[Box]
    accepted_target_pos: Optional[Point]
    decision_reason: str
    mode: TrackerMode
    output: BigTrackUpdateOutput


class BaseBigTrack(BigTrack):
    """Reusable BigTrack flow without candidate or lifecycle policy."""

    def __init__(self, predictor: Predictor, matcher: Matcher) -> None:
        self.predictor = predictor
        self.matcher = matcher
        self._state: Optional[BigTrackState] = None
        self._output: Optional[BigTrackUpdateOutput] = None
        self._last_debug: Optional[BigTrackDebugSnapshot] = None

    def initialize(self, request: BigTrackInitializeInput) -> BigTrackInitializeOutput:
        target_pos = _box_to_center(request.box)
        predictor_request = request.predictor or PredictorInitializeInput(
            predictor_state=TrackerPredictionState(
                target_pos=target_pos,
                target_velocity=(0.0, 0.0),
                uncertainty=0.0,
                metadata={
                    "frame_idx": request.frame.idx,
                    "timestamp": request.frame.timestamp,
                },
            )
        )
        self.predictor.initialize(predictor_request)

        matcher_request = request.matcher or MatcherInitializeInput(
            frame=request.frame,
            box=request.box,
        )
        self.matcher.initialize_template(matcher_request)
        matcher_state = matcher_request.matcher_state or getattr(self.matcher, "_state", None)
        if matcher_state is None:
            raise RuntimeError("Matcher did not expose initialized state")

        output = BigTrackUpdateOutput(
            ok=True,
            box=request.box,
            frame_idx=request.frame.idx,
            timestamp=request.frame.timestamp,
            status=OutputStatus.ACTIVE,
            confidence=float(request.initial_confidence),
        )
        self._state = BigTrackState(
            predictor_state=predictor_request.predictor_state,
            matcher_state=matcher_state,
            mode=TrackerMode.TRACKING,
            output=output,
            last_seen_frame=request.frame.idx,
            metadata={
                **dict(request.metadata),
                "age": 1,
                "target_size": _box_size(request.box),
            },
        )
        self._output = output
        return BigTrackInitializeOutput(ok=True)

    def initialize_from_state(self, request: BigTrackInitializeInput) -> BigTrackInitializeOutput:
        if request.predictor is None or request.matcher is None:
            raise ValueError("initialize_from_state requires predictor and matcher inputs")
        if request.matcher.matcher_state is None:
            raise ValueError("initialize_from_state requires matcher state")

        self.predictor.initialize(request.predictor)
        self.matcher.initialize_template(request.matcher)
        output = BigTrackUpdateOutput(
            ok=True,
            box=request.box,
            frame_idx=request.frame.idx,
            timestamp=request.frame.timestamp,
            status=OutputStatus.ACTIVE,
            confidence=float(request.initial_confidence),
        )
        self._state = BigTrackState(
            predictor_state=request.predictor.predictor_state,
            matcher_state=request.matcher.matcher_state,
            mode=TrackerMode.TRACKING,
            output=output,
            last_seen_frame=request.frame.idx,
            metadata={
                **dict(request.metadata),
                "age": 1,
                "target_size": _box_size(request.box),
            },
        )
        self._output = output
        return BigTrackInitializeOutput(ok=True, metadata={"restored": True})

    def update(self, request: BigTrackUpdateInput) -> BigTrackUpdateOutput:
        state = self._require_state()
        prediction = self.predictor.predict(
            PredictorPredictInput(frame=request.frame, metadata=request.metadata)
        ).predictor_state
        candidates = self.make_candidates(state, prediction, request.frame)
        match_output = self.matcher.match(
            MatcherMatchInput(
                frame=request.frame,
                target_poses=[candidate.search_center for candidate in candidates],
            )
        )
        decision = self.decide(state, prediction, candidates, match_output.bboxes, match_output.scores)
        next_state = self.apply_decision(state, prediction, decision, request.frame)

        if decision.allow_template_update:
            if decision.accepted_box is None:
                raise ValueError("Template update requires accepted box")
            template_output = self.matcher.extract_template(
                MatcherTemplateInput(frame=request.frame, box=decision.accepted_box)
            )
            self.matcher.update_templates(
                MatcherUpdateInput(template=template_output.template, score=template_output.score)
            )
            next_state = replace(
                next_state,
                matcher_state=getattr(self.matcher, "_state", next_state.matcher_state),
            )

        self._state = next_state
        self._output = next_state.output
        self._last_debug = BigTrackDebugSnapshot(
            frame_idx=request.frame.idx,
            timestamp=request.frame.timestamp,
            predictor_target_pos=prediction.target_pos,
            predictor_target_velocity=prediction.target_velocity,
            candidate_target_poses=tuple(candidate.search_center for candidate in candidates),
            matcher_bboxes=tuple(match_output.bboxes),
            matcher_scores=tuple(float(score) for score in match_output.scores),
            accepted_box=decision.accepted_box,
            accepted_target_pos=decision.accepted_target_pos,
            decision_reason=decision.reason,
            mode=decision.next_mode,
            output=next_state.output,
        )
        return next_state.output

    def reset(self) -> None:
        self._state = None
        self._output = None
        self._last_debug = None
        self.predictor.reset()
        self.matcher.reset()

    def close(self) -> None:
        self.reset()
        self.predictor.close()
        self.matcher.close()

    def get_state(self) -> BigTrackState:
        return self._require_state()

    def get_output(self) -> Optional[BigTrackUpdateOutput]:
        return self._output

    def get_debug_snapshot(self) -> Optional[BigTrackDebugSnapshot]:
        return self._last_debug

    def make_candidates(
        self,
        state: BigTrackState,
        prediction: TrackerPredictionState,
        frame: FrameLike,
    ) -> Sequence[SearchCandidate]:
        raise NotImplementedError

    def decide(
        self,
        state: BigTrackState,
        prediction: TrackerPredictionState,
        candidates: Sequence[SearchCandidate],
        bboxes: Sequence[Box],
        scores: Sequence[float],
    ) -> BigTrackDecision:
        raise NotImplementedError

    def apply_decision(
        self,
        state: BigTrackState,
        prediction: TrackerPredictionState,
        decision: BigTrackDecision,
        frame: FrameLike,
    ) -> BigTrackState:
        raise NotImplementedError

    def _require_state(self) -> BigTrackState:
        if self._state is None:
            raise RuntimeError("BigTrack must be initialized before update")
        return self._state


def _box_to_center(box: Box) -> Point:
    x, y, width, height = box
    return (float(x) + float(width) / 2.0, float(y) + float(height) / 2.0)


def _box_size(box: Box) -> tuple[float, float]:
    return (float(box[2]), float(box[3]))
