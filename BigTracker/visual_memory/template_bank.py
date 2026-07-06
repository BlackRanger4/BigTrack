from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from BigTracker.common_types import Template


@dataclass(frozen=True)
class TemplateBankEntry:
    template_id: str
    template: Template
    source_frame: int
    quality_score: float
    diversity_score: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TemplateBank:
    items: Sequence[TemplateBankEntry] = field(default_factory=tuple)
    max_size: int = 8

    def entries(self) -> Sequence[TemplateBankEntry]:
        return self.items

    def add(self, entry: TemplateBankEntry) -> "TemplateBank":
        deduplicated = tuple(item for item in self.items if item.template_id != entry.template_id)
        ordered = (entry, *deduplicated)
        return TemplateBank(items=ordered[: self.max_size], max_size=self.max_size)

    def remove(self, template_id: str) -> "TemplateBank":
        return TemplateBank(
            items=tuple(item for item in self.items if item.template_id != template_id),
            max_size=self.max_size,
        )

    def clear(self) -> "TemplateBank":
        return TemplateBank(max_size=self.max_size)
