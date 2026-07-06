from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class VariationState:
    """Tracks accepted appearance change relative to the identity anchor."""

    anchor_difference: Any
    source_frame: int
    confidence: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
