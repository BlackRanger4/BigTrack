from __future__ import annotations

from abc import ABC, abstractmethod

from BigTracker.track_state import TrackState, TrackerDecision


class StateUpdatePolicy(ABC):
    """Applies one post-matcher decision to immutable track state."""

    @abstractmethod
    def apply_decision(
        self,
        track_state: TrackState,
        decision: TrackerDecision,
        frame_index: int,
    ) -> TrackState:
        """Update kinematics, counters, mode, and last-result from the decision."""
        ...
