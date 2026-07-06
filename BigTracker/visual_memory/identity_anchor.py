from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from BigTracker.track_state import Template


@dataclass(frozen=True)
class IdentityAnchor:
    track_id: str
    template: Template
    created_frame: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
