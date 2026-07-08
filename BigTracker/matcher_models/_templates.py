from __future__ import annotations

from dataclasses import replace
from typing import Any

from BigTracker.state import MatcherState, TemplateCandidate


def update_template_bank(
    state: MatcherState,
    template: TemplateCandidate,
    max_templates: int,
) -> MatcherState:
    """Append one approved template and select the best-scored active template."""

    scored_template = _with_template_score(
        template.template,
        _candidate_score(template),
    )
    best_templates = tuple(state.best_templates) + (scored_template,)
    max_templates = max(0, int(max_templates))
    if max_templates == 0:
        best_templates = ()
    elif len(best_templates) > max_templates:
        best_templates = best_templates[-max_templates:]

    adaptive_template = _best_template(best_templates) if best_templates else state.init_template
    return replace(
        state,
        best_templates=best_templates,
        adaptive_template=adaptive_template,
    )


def _candidate_score(template: TemplateCandidate) -> float:
    quality_score = _clamp01(template.quality_score)
    identity_score = _clamp01(template.identity_score)
    return quality_score * identity_score


def _with_template_score(template: Any, score: float) -> Any:
    if hasattr(template, "template_score"):
        if abs(float(getattr(template, "template_score")) - float(score)) <= 1e-12:
            return template
        return replace(template, template_score=float(score))
    return template


def _best_template(templates: tuple[Any, ...]) -> Any:
    _, best_template = max(
        enumerate(templates),
        key=lambda item: (_template_score(item[1]), item[0]),
    )
    return best_template


def _template_score(template: Any) -> float:
    return float(getattr(template, "template_score", 0.0))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
