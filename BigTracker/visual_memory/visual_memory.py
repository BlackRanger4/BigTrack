from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from BigTracker.visual_memory.identity_anchor import IdentityAnchor
from BigTracker.visual_memory.short_term_template import ShortTermTemplate
from BigTracker.visual_memory.template_bank import TemplateBank
from BigTracker.visual_memory.variation_state import VariationState


@dataclass(frozen=True)
class VisualMemory:
    identity_anchor: IdentityAnchor
    short_term_template: Optional[ShortTermTemplate]
    template_bank: TemplateBank
    variation_state: Optional[VariationState]
    cached_features: Mapping[str, Any] = field(default_factory=dict)
