from __future__ import annotations

from dataclasses import dataclass
import unittest

from BigTracker.matcher_models._templates import update_template_bank
from BigTracker.state import MatcherState, TemplateCandidate


@dataclass(frozen=True)
class _Template:
    name: str
    template_score: float = 1.0


class TemplateBankTest(unittest.TestCase):
    def test_adaptive_template_is_best_score_in_bounded_window(self) -> None:
        init_template = _Template("init")
        state = MatcherState(init_template=init_template, adaptive_template=init_template)

        state = update_template_bank(
            state,
            _candidate("high-old", quality_score=0.9),
            max_templates=2,
        )
        state = update_template_bank(
            state,
            _candidate("low-new", quality_score=0.2),
            max_templates=2,
        )

        self.assertEqual([template.name for template in state.best_templates], ["high-old", "low-new"])
        self.assertEqual(state.adaptive_template.name, "high-old")
        self.assertAlmostEqual(state.adaptive_template.template_score, 0.9)

        state = update_template_bank(
            state,
            _candidate("middle-newest", quality_score=0.4),
            max_templates=2,
        )

        self.assertEqual([template.name for template in state.best_templates], ["low-new", "middle-newest"])
        self.assertEqual(state.adaptive_template.name, "middle-newest")
        self.assertAlmostEqual(state.adaptive_template.template_score, 0.4)

    def test_newest_template_wins_score_tie(self) -> None:
        init_template = _Template("init")
        state = MatcherState(init_template=init_template, adaptive_template=init_template)

        state = update_template_bank(state, _candidate("first", quality_score=0.7), max_templates=3)
        state = update_template_bank(state, _candidate("second", quality_score=0.7), max_templates=3)

        self.assertEqual(state.adaptive_template.name, "second")

    def test_zero_sized_bank_returns_to_init_template(self) -> None:
        init_template = _Template("init")
        state = MatcherState(init_template=init_template, adaptive_template=init_template)

        state = update_template_bank(state, _candidate("ignored", quality_score=1.0), max_templates=0)

        self.assertEqual(state.best_templates, ())
        self.assertIs(state.adaptive_template, init_template)


def _candidate(name: str, quality_score: float, identity_score: float = 1.0) -> TemplateCandidate:
    return TemplateCandidate(
        template=_Template(name),
        source_frame_idx=0,
        source_box=(0.0, 0.0, 1.0, 1.0),
        quality_score=quality_score,
        identity_score=identity_score,
    )


if __name__ == "__main__":
    unittest.main()
