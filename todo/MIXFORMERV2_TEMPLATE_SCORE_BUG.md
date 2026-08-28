# TODO: Preserve the real MixFormerV2 confidence when updating templates

## Problem

`MixFormerV2MatcherModel.extract_template()` currently returns a fixed
`MatcherTemplateOutput.score = 1.0` for every extracted template. The shared
template bank therefore treats every approved template as equally perfect and,
when scores tie, selects the newest one as the adaptive template.

This loses the actual MixFormerV2 `pred_scores` confidence that was used by
`ScoreGatedBigTrack` to accept the match. It differs from upstream online
MixFormerV2, which retains the highest-confidence template candidate observed
within an update interval.

Relevant code:

- `BigTracker/matcher_models/mixformerv2.py`: `extract_template()` returns
  `score=1.0`.
- `BigTracker/matcher_models/_templates.py`: ties select the later/newer
  template.
- `BigTracker/big_trackers/score_gated.py`: only forwards the extracted
  template score to the template bank; it does not pass the accepted match
  confidence into `extract_template()`.

## Impact

On a difficult background, a frame can barely pass the good-match threshold
yet replace a better adaptive template solely because it is newer. This can
cause appearance drift and make small-object tracking less stable. The current
bank's five entries do not fix this because all stored scores are `1.0`.

## Intended behavior

Store a meaningful visual confidence with each extracted MixFormerV2 template.
At minimum, the bank must receive the accepted matcher confidence rather than a
constant. Preferably, collect the best eligible candidate during an update
window, matching the upstream online tracker's max-score behavior.

## Implementation questions

- Extend the matcher template request/output contract, or template-update
  request, so `ScoreGatedBigTrack` can pass the accepted `decision.confidence`
  to the matcher/template bank.
- Ensure template confidence remains tied to the exact frame and accepted box
  used to extract the template.
- Decide whether the fixed initial template remains unscored/protected (it
  should).
- Consider a separate minimum template-update threshold; it should not be lower
  than the policy's acceptance threshold without an explicit reason.
- If reproducing upstream behavior, retain the best candidate seen since the
  preceding update rather than sampling only the scheduled update frame.

## Acceptance checks

- An approved template extracted from a 0.82 match is stored with score 0.82,
  not 1.0.
- Between two retained templates, 0.90 remains the adaptive template over a
  later 0.72 template.
- The initial template remains unchanged.
- Rejected, weak, clipped (when policy disallows them), recovery, and lost
  frames never enter the template bank.
- Add a focused regression test in `tests/test_mixformerv2_matcher.py` and an
  integration test for the ScoreGated-to-template-update score handoff.

## Priority

High for the small-object / hard-background failure investigation.
