from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple

from BigTracker.common_types import Box
from BigTracker.track_state import CandidateState, TrackerMode


class CandidateGenerator(ABC):
    """Creates one or more matcher candidates from prediction and detections."""

    @abstractmethod
    def generate_candidates(
        self,
        predicted_box: Box,
        search_region: Box,
        mode: TrackerMode,
        prediction_confidence: float,
        motion_uncertainty: float,
        size_uncertainty: float,
        expected_scale_range: Tuple[float, float],
        external_detections: Optional[Sequence[Box]] = None,
    ) -> Sequence[CandidateState]:
        """Generate normal or uncertain-mode candidates for visual matching."""
        ...

    @abstractmethod
    def generate_recovery_candidates(
        self,
        predicted_box: Box,
        last_seen_box: Optional[Box],
        frame_shape: Tuple[int, int],
        external_detections: Optional[Sequence[Box]] = None,
    ) -> Sequence[CandidateState]:
        """Generate wider recovery candidates after repeated misses or occlusion."""
        ...
