from __future__ import annotations

from dataclasses import dataclass
import unittest

from BigTracker.matcher_models._templates import update_template_bank
from BigTracker.types.matcher import MatcherState


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
            _template("high-old"),
            score=0.9,
            max_templates=2,
        )
        state = update_template_bank(
            state,
            _template("low-new"),
            score=0.2,
            max_templates=2,
        )

        self.assertEqual([entry.template.name for entry in state.best_templates], ["high-old", "low-new"])
        self.assertEqual(state.adaptive_template.name, "high-old")

        state = update_template_bank(
            state,
            _template("middle-newest"),
            score=0.4,
            max_templates=2,
        )

        self.assertEqual([entry.template.name for entry in state.best_templates], ["low-new", "middle-newest"])
        self.assertEqual(state.adaptive_template.name, "middle-newest")

    def test_newest_template_wins_score_tie(self) -> None:
        init_template = _Template("init")
        state = MatcherState(init_template=init_template, adaptive_template=init_template)

        state = update_template_bank(state, _template("first"), score=0.7, max_templates=3)
        state = update_template_bank(state, _template("second"), score=0.7, max_templates=3)

        self.assertEqual(state.adaptive_template.name, "second")

    def test_zero_sized_bank_returns_to_init_template(self) -> None:
        init_template = _Template("init")
        state = MatcherState(init_template=init_template, adaptive_template=init_template)

        state = update_template_bank(state, _template("ignored"), score=1.0, max_templates=0)

        self.assertEqual(state.best_templates, ())
        self.assertIs(state.adaptive_template, init_template)


def _template(name: str) -> _Template:
    return _Template(name)


if __name__ == "__main__":
    unittest.main()
