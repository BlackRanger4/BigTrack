from __future__ import annotations

from abc import ABC, abstractmethod

from BigTracker.track_state import CandidateState, MatchResult, TrackState


class StateUpdatePolicy(ABC):
    @abstractmethod
    def update_on_strong_accept(
        self,
        track_state: TrackState,
        candidate: CandidateState,
        result: MatchResult,
    ) -> TrackState:
        ...

    @abstractmethod
    def update_on_weak_accept(
        self,
        track_state: TrackState,
        candidate: CandidateState,
        result: MatchResult,
    ) -> TrackState:
        ...

    @abstractmethod
    def update_on_reject(self, track_state: TrackState, candidate: CandidateState) -> TrackState:
        ...
