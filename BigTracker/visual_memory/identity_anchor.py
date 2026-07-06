from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from BigTracker.common_types import Template


@dataclass(frozen=True)
class IdentityAnchor:
    """First trusted identity template; template update code must preserve it."""

    track_id: str
    template: Template
    created_frame: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
