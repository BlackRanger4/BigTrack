from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, List
from BigTracker.types.common import Box, FrameLike, Point


@dataclass(frozen=True)
class TemplateState:
    """State of a single template."""

    template: Any
    template_score: float = 1.0


@dataclass(frozen=True)
class MatcherState:
    """Visual-side state owned by the Matcher domain."""

    init_template: Any
    best_templates: Sequence[TemplateState] = field(default_factory=tuple)
    adaptive_template: Optional[Any] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MatcherInitializeInput:
    """Matcher-specific initialization options."""

    frame: FrameLike
    box: Box
    matcher_state: MatcherState | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MatcherInitializeOutput:
    """Matcher-specific initialization result."""

    ok: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MatcherTemplateInput:
    """Input object for matcher template extraction."""

    frame: FrameLike
    box: Box
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MatcherTemplateOutput:
    """Output object from matcher template extraction."""

    template: Any
    score: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MatcherUpdateInput:
    """Input object for updating matcher state with an approved template."""

    template: Any
    score: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MatcherUpdateOutput:
    """Output object from matcher state update."""

    ok: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MatcherMatchInput:
    """Input object for matcher visual search."""

    frame: FrameLike
    target_poses: List[Point]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MatcherMatchOutput:
    """Output object from matcher visual search."""

    bboxes: List[Box]
    scores: List[float]
    metadata: Mapping[str, Any] = field(default_factory=dict)
