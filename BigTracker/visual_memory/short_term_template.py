from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from BigTracker.common_types import Template


@dataclass(frozen=True)
class ShortTermTemplate:
    template: Template
    source_frame: int
    quality_score: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
