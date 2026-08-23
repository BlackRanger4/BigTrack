from __future__ import annotations

from enum import Enum
from typing import Any, Protocol, Tuple


# Frame-coordinate box: x, y, width, height.
Box = Tuple[float, float, float, float]

# Frame-coordinate center point: x, y.
Point = Tuple[float, float]

# Object size: width, height.
Size = Tuple[float, float]

# Backend-specific image object, usually a numpy array or tensor.
ImageLike = Any


class FrameLike(Protocol):
    """Protocol for frames accepted by BigTrack, Predictor, and Matcher."""

    image: ImageLike
    idx: int
    timestamp: float


class TrackerMode(str, Enum):
    """Internal lifecycle mode for one object track."""

    INIT = "INIT"
    TRACKING = "TRACKING"
    UNCERTAIN = "UNCERTAIN"
    OCCLUDED = "OCCLUDED"
    RECOVERY = "RECOVERY"
    LOST = "LOST"


class OutputStatus(str, Enum):
    """Small public status returned to client code."""

    ACTIVE = "ACTIVE"
    UNCERTAIN = "UNCERTAIN"
    OCCLUDED = "OCCLUDED"
    LOST = "LOST"
