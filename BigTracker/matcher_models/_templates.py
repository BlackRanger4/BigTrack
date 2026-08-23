from __future__ import annotations

from dataclasses import replace
from typing import Any

from BigTracker.types import MatcherState, TemplateState


def update_template_bank(
    state: MatcherState,
    template: Any,
    score: float,
    max_templates: int,
) -> MatcherState:
    """Append one approved template and select the best template by score and age."""

    template_state = TemplateState(template=template, template_score=_clamp01(score))
    best_templates = tuple(state.best_templates) + (template_state,)
    max_templates = max(0, int(max_templates))
    if max_templates == 0:
        best_templates = ()
    elif len(best_templates) > max_templates:
        best_templates = best_templates[-max_templates:]

    adaptive_template = _best_template(best_templates).template if best_templates else state.init_template
    return replace(
        state,
        best_templates=best_templates,
        adaptive_template=adaptive_template,
    )


def _best_template(templates: tuple[TemplateState, ...]) -> TemplateState:
    _, best_template_state = max(
        enumerate(templates),
        key=lambda item: (item[1].template_score, item[0]),
    )
    return best_template_state


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
