from __future__ import annotations

from dataclasses import replace
from typing import Optional

from BigTracker.big_track import BigTrack
from BigTracker.matcher import Matcher
from BigTracker.predictor import Predictor
from BigTracker.state import (
    BigTrackCounters,
    BigTrackState,
    TemplateCandidate,
    TrackerPredictionState,
    TrackingOutput,
)
from BigTracker.types import Box, FrameLike, OutputStatus, Point, Size, TrackerMode


class BaseBigTrack(BigTrack):
    """Reusable BigTrack flow without candidate or lifecycle policy.

    Subclasses get initialize/update/reset/getters for free and must implement
    `make_candidates`, `decide`, and `apply_decision`.
    """

    def __init__(self, predictor: Predictor, matcher: Matcher) -> None:
        """Create a tracker from one motion predictor and one visual matcher."""

        self.predictor = predictor
        self.matcher = matcher
        self._state: Optional[BigTrackState] = None
        self._output: Optional[TrackingOutput] = None

    def initialize(
        self,
        frame: FrameLike,
        box: Box,
        target_velocity: Optional[Point] = None,
        target_size_velocity: Optional[Size] = None,
        initial_confidence: float = 1.0,
    ) -> BigTrackState:
        """Initialize prediction state, matcher templates, mode, counters, and output."""

        target_pos, target_size = _box_to_center_size(box)
        prediction = TrackerPredictionState(
            target_pos=target_pos,
            target_size=target_size,
            target_velocity=target_velocity or (0.0, 0.0),
            target_size_velocity=target_size_velocity or (0.0, 0.0),
            last_score=float(initial_confidence),
            uncertainty=0.0,
        )
        matcher_state = self.matcher.initialize_template(
            frame=frame,
            target_pos=target_pos,
            target_size=target_size,
        )
        output = TrackingOutput(
            box=box,
            frame_idx=frame.idx,
            timestamp=frame.timestamp,
            status=OutputStatus.ACTIVE,
            confidence=float(initial_confidence),
        )
        self._state = BigTrackState(
            prediction=prediction,
            matcher=matcher_state,
            output=output,
            mode=TrackerMode.TRACKING,
            counters=BigTrackCounters(age=1),
            last_seen_frame=frame.idx,
        )
        self._output = output
        return self._state

    def update(self, frame: FrameLike) -> TrackingOutput:
        """Process one frame through prediction, matching, decision, and state update."""

        state = self._require_state()
        prediction = self.predictor.predict(state, frame)
        candidates = self.make_candidates(state, prediction, frame)
        matches = tuple(
            self.matcher.match(
                frame=frame,
                matcher_state=state.matcher,
                candidate=candidate,
                mode=state.mode,
            )
            for candidate in candidates
        )
        decision = self.decide(
            state=state,
            prediction=prediction,
            candidates=candidates,
            matches=matches,
        )
        next_state = self.apply_decision(
            state=state,
            prediction=prediction,
            decision=decision,
            frame=frame,
        )

        if decision.allow_template_update:
            if decision.accepted_target_pos is None or decision.accepted_target_size is None:
                raise ValueError("Template update requires accepted target position and size")
            template = self.matcher.extract_template(
                frame=frame,
                target_pos=decision.accepted_target_pos,
                target_size=decision.accepted_target_size,
                previous_state=next_state.matcher,
            )
            template = _score_template_candidate(template, decision.confidence)
            matcher_state = self.matcher.update_templates(next_state.matcher, template)
            next_state = replace(next_state, matcher=matcher_state)

        self._state = next_state
        self._output = next_state.output
        return next_state.output

    def reset(self) -> None:
        """Clear internal state and latest output."""

        self._state = None
        self._output = None

    def get_state(self) -> Optional[BigTrackState]:
        """Return internal state for debugging, checkpointing, or advanced users."""

        return self._state

    def get_output(self) -> Optional[TrackingOutput]:
        """Return the latest small client-facing output."""

        return self._output

    def _require_state(self) -> BigTrackState:
        """Return current state or fail clearly when update is called before initialize."""

        if self._state is None:
            raise RuntimeError("BigTrack must be initialized before update")
        return self._state


def _box_to_center_size(box: Box) -> tuple[Point, Size]:
    """Convert frame-coordinate x, y, width, height into center point and size."""

    x, y, width, height = box
    return (
        (float(x) + float(width) / 2.0, float(y) + float(height) / 2.0),
        (float(width), float(height)),
    )


def _score_template_candidate(template: TemplateCandidate, tracking_score: float) -> TemplateCandidate:
    """Attach accepted tracking confidence to the approved template candidate."""

    return replace(
        template,
        quality_score=max(0.0, min(1.0, float(tracking_score))),
    )
