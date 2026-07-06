from __future__ import annotations

from abc import ABC, abstractmethod

from BigTracker.track_state import TrackState, TrackerDecision, TrackerMode


class ModeTransitionPolicy(ABC):
    @abstractmethod
    def next_mode(self, track_state: TrackState, decision: TrackerDecision) -> TrackerMode:
        ...

    @abstractmethod
    def update_counters(self, track_state: TrackState, decision: TrackerDecision) -> TrackState:
        ...
