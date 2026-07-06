from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from BigTracker.track_state import Template


@dataclass(frozen=True)
class TemplateBankEntry:
    template_id: str
    template: Template
    source_frame: int
    quality_score: float
    diversity_score: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


class TemplateBank(ABC):
    @abstractmethod
    def entries(self) -> Sequence[TemplateBankEntry]:
        ...

    @abstractmethod
    def add(self, entry: TemplateBankEntry) -> "TemplateBank":
        ...

    @abstractmethod
    def remove(self, template_id: str) -> "TemplateBank":
        ...

    @abstractmethod
    def clear(self) -> "TemplateBank":
        ...
