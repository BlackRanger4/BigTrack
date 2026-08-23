from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from BigTracker.types.common import Box, FrameLike
from BigTracker.types.matcher import MatcherInitializeInput
from BigTracker.types.predictor import PredictorInitializeInput


@dataclass(frozen=True)
class BigTrackInitializeInput:
    """Input object for initializing BigTrack, Predictor, and Matcher."""

    frame: FrameLike
    box: Box
    initial_confidence: float = 1.0
    predictor: PredictorInitializeInput | None = None
    matcher: MatcherInitializeInput | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BigTrackInitializeOutput:
    """Output object returned after BigTrack initialization."""

    ok: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BigTrackUpdateInput:
    """Input object for updating a BigTrack instance with one frame."""

    frame: FrameLike
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BigTrackUpdateOutput:
    """Output object returned after one BigTrack update."""

    ok: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)
