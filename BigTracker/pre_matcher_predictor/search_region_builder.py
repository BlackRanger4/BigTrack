from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

from BigTracker.common_types import Box
from BigTracker.track_state import TrackerMode


class SearchRegionBuilder(ABC):
    @abstractmethod
    def build_search_region(
        self,
        predicted_box: Box,
        mode: TrackerMode,
        motion_uncertainty: float,
    ) -> Box:
        ...

    @abstractmethod
    def build_expected_scale_range(
        self,
        predicted_box: Box,
        size_uncertainty: float,
    ) -> Tuple[float, float]:
        ...

    @abstractmethod
    def clip_to_frame(self, region: Box, frame_shape: Tuple[int, int]) -> Box:
        ...
