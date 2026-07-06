from __future__ import annotations

from abc import ABC, abstractmethod

from BigTracker.track_state import TrackState, TrackerDecision, TrackingOutput


class LostTrackPolicy(ABC):
    @abstractmethod
    def should_declare_lost(self, track_state: TrackState, decision: TrackerDecision) -> bool:
        ...

    @abstractmethod
    def build_lost_output(self, track_state: TrackState, reason: str) -> TrackingOutput:
        ...
