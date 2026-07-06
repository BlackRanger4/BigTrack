from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

from BigTracker.track_state import Box, CandidateState, Frame, MatcherMode, TrackState


class PreMatcherPredictor(ABC):
    @abstractmethod
    def predict(
        self,
        track_state: TrackState,
        frame: Frame,
        frame_index: int,
        external_detections: Optional[Sequence[Box]] = None,
    ) -> Sequence[CandidateState]:
        ...

    @abstractmethod
    def choose_matcher_mode(self, track_state: TrackState) -> MatcherMode:
        ...
