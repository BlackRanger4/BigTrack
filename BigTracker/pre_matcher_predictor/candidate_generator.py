from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple

from BigTracker.track_state import Box, CandidateState, TrackerMode


class CandidateGenerator(ABC):
    @abstractmethod
    def generate_candidates(
        self,
        predicted_box: Box,
        search_region: Box,
        mode: TrackerMode,
        expected_scale_range: Tuple[float, float],
        external_detections: Optional[Sequence[Box]] = None,
    ) -> Sequence[CandidateState]:
        ...

    @abstractmethod
    def generate_recovery_candidates(
        self,
        predicted_box: Box,
        last_seen_box: Optional[Box],
        frame_shape: Tuple[int, int],
        external_detections: Optional[Sequence[Box]] = None,
    ) -> Sequence[CandidateState]:
        ...
