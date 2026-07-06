from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

from BigTracker.common_types import Box
from BigTracker.track_state import TrackerMode


class SearchRegionBuilder(ABC):
    """Builds matcher search regions from predicted tracker state."""

    @abstractmethod
    def build_search_region(
        self,
        predicted_box: Box,
        mode: TrackerMode,
        motion_uncertainty: float,
    ) -> Box:
        """Create the frame-coordinate region where the matcher should search."""
        ...

    @abstractmethod
    def build_expected_scale_range(
        self,
        predicted_box: Box,
        size_uncertainty: float,
    ) -> Tuple[float, float]:
        """Return plausible min/max scale factors for this candidate."""
        ...

    @abstractmethod
    def clip_to_frame(self, region: Box, frame_shape: Tuple[int, int]) -> Box:
        """Clip a search region to valid frame bounds."""
        ...
