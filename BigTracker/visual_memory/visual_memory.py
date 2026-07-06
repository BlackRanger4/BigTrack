from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from BigTracker.visual_memory.identity_anchor import IdentityAnchor
from BigTracker.visual_memory.short_term_template import ShortTermTemplate
from BigTracker.visual_memory.template_bank import TemplateBank
from BigTracker.visual_memory.variation_state import VariationState


@dataclass(frozen=True)
class VisualMemory:
    """All visual identity memory read by matchers and updated only by policy."""

    identity_anchor: IdentityAnchor
    short_term_template: Optional[ShortTermTemplate]
    template_bank: TemplateBank
    variation_state: Optional[VariationState]
    cached_features: Mapping[str, Any] = field(default_factory=dict)

    def with_template_update(
        self,
        *,
        short_term_template: Optional[ShortTermTemplate] = None,
        template_bank: Optional[TemplateBank] = None,
        variation_state: Optional[VariationState] = None,
        cached_features: Optional[Mapping[str, Any]] = None,
    ) -> "VisualMemory":
        """Return updated memory while preserving the immutable identity anchor."""
        return VisualMemory(
            identity_anchor=self.identity_anchor,
            short_term_template=(
                self.short_term_template if short_term_template is None else short_term_template
            ),
            template_bank=self.template_bank if template_bank is None else template_bank,
            variation_state=self.variation_state if variation_state is None else variation_state,
            cached_features=self.cached_features if cached_features is None else cached_features,
        )
