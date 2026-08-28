# Architecture

## Purpose and Design Boundary

BigTracker is a composable single-object tracking library. It separates three concerns:

- A `Predictor` estimates motion before visual inference.
- A `Matcher` searches image content and returns candidate boxes plus scores.
- A `BigTrack` policy decides what evidence to accept, controls lifecycle status, corrects the predictor, owns public output, and authorizes template learning.

This ownership boundary is the most important architectural rule. A matcher may know how to create or activate model-specific templates, but it must not decide whether a frame is safe to learn from. A predictor may express uncertainty, but it must not decide whether a visual result means occlusion or loss.

## Coordinate and Frame Contracts

All public geometry is frame-coordinate floating-point data:

```text
Box   = (x, y, width, height)
Point = (center_x, center_y)
Size  = (width, height)
```

`FrameLike` is a structural protocol. Callers do not need to inherit a library class; an object is accepted if it has:

```python
image: object
idx: int
timestamp: float
```

`image` is normally a BGR NumPy array for built-in matchers. `idx` identifies ordering and is also a fallback time delta for predictors. `timestamp` is preferred for predictor `dt` when it strictly increases.

## Public Type Model

The frozen dataclasses under `BigTracker/types` are request, response, and state boundaries.

### Predictor types

`TrackerPredictionState` holds `target_pos`, `target_velocity`, scalar `uncertainty`, and open-ended `metadata`. Predictor-specific covariance, acceleration, history, timestamps, reject counts, and the last accepted score are stored in metadata so the common state remains small.

`PredictorInitializeInput` supplies a complete state, which supports fresh initialization and restoration. `PredictorPredictInput` carries the current frame. `PredictorUpdateInput` tells the predictor whether the policy accepted visual evidence and supplies the corrected-or-predicted state. Every operation returns its matching output dataclass.

### Matcher types

`MatcherState` contains:

- `init_template`: the first trusted identity template. Shared helpers never replace it.
- `best_templates`: a bounded sequence of `TemplateState(template, template_score)` records.
- `adaptive_template`: the best-scoring record in the current bounded window, with newest winning a tie; it falls back to `init_template` when the bank is empty.
- `metadata`: matcher-level extensibility data.

`MatcherInitializeInput` contains the first frame and box and may carry an existing `matcher_state` for restoration. `MatcherMatchInput` contains one frame and a list of search centers. `MatcherMatchOutput` contains parallel `bboxes` and `scores` lists plus diagnostics in metadata. Template extraction and commitment are deliberately separate requests so policy approval occurs between them.

### BigTrack types

`BigTrackState` composes the latest predictor state, matcher state, internal `TrackerMode`, last public output, last visually accepted frame, and policy metadata.

Internal modes are `INIT`, `TRACKING`, `UNCERTAIN`, `OCCLUDED`, `RECOVERY`, and `LOST`. Public `OutputStatus` intentionally exposes only `ACTIVE`, `UNCERTAIN`, `OCCLUDED`, and `LOST`; recovery is currently surfaced as `UNCERTAIN`.

`BigTrackUpdateOutput` contains `ok`, optional box, frame identity, public status, normalized-ish confidence, and metadata. Confidence values are treated as scores in `[0, 1]` by score-gated helpers, although the base type does not enforce the range.

## Initialization Flow

`BaseBigTrack.initialize()` performs the shared work:

1. Converts the initial box to its center.
2. Builds a zero-velocity `TrackerPredictionState` unless the caller supplied a predictor request.
3. Calls `predictor.initialize()`.
4. Builds a `MatcherInitializeInput` unless the caller supplied one, then calls `matcher.initialize_template()`.
5. Reads matcher state from the supplied restore state or the matcher's `_state` attribute. A matcher that does not expose initialized state causes a runtime error.
6. Creates an `ACTIVE` output for the initial box.
7. Stores `age=1` and `target_size` in BigTrack metadata and enters `TRACKING`.

`initialize_from_state()` requires both explicit predictor and matcher requests and requires `matcher.matcher_state`. It initializes both child objects from those states, but creates a fresh active output and resets top-level metadata to the metadata supplied in the request plus `age=1` and the request box size. Callers that need policy counters restored must pass them through `BigTrackInitializeInput.metadata`.

## Per-Frame Update Flow

`BaseBigTrack.update()` is a template method:

1. Require initialization.
2. Time and call `predictor.predict()`.
3. Ask the concrete policy to `make_candidates()`.
4. Pass all candidate centers to one `matcher.match()` call and time it.
5. Pass previous state, prediction, candidates, boxes, and scores to `decide()`.
6. Ask the policy to `apply_decision()`. That method calls `_update_predictor()` with accepted or rejected feedback.
7. Replace the provisional predictor state with the predictor's actual post-update state.
8. If `allow_template_update` is true, extract a template from the accepted box, commit it to the matcher, and refresh the composed matcher state.
9. Save state/output and create `BigTrackDebugSnapshot` with pre/post predictor state, candidates, matcher results, decision, mode, and stage timings.

Matcher metadata is not passed to `decide()` in the current signature. Policy decisions can therefore use candidate information, boxes, and scores, but not matcher diagnostics such as ambiguity, scale score, or clipping without an API change.

## Tracker Policies

### BaseBigTrack

`big_trackers/base.py` implements orchestration, state restoration, reset/close propagation, output access, debug snapshots, and predictor update timing. It intentionally leaves `make_candidates`, `decide`, and `apply_decision` unimplemented.

`reset()` clears runtime state and resets both children but keeps constructed objects/configuration reusable. `close()` calls reset and then closes both children.

### SimpleBigTrack

`SimpleBigTrack` is a minimal integration policy:

- Creates one candidate at `prediction.target_pos`.
- Requires at least one box and score.
- Accepts the first matcher result without validating its score or geometry.
- Corrects the predictor to the matched center.
- Always emits `ACTIVE` and stays in `TRACKING`.
- Updates age and target size.
- Never authorizes template updates.

Use it for adapter smoke tests and controlled experiments, not robust production lifecycle handling.

### ScoreGatedBigTrack

The score-gated policy also creates one predicted-center candidate. Candidate confidence combines the prior `last_score` with motion uncertainty using:

```text
confidence * 1 / (1 + uncertainty / predictor_uncertainty_scale)
```

It selects the highest normalized matcher score; ties go to the later index. Scores are classified as:

- good: `score >= th_good`
- weak: `th_bad <= score < th_good`
- bad: `score < th_bad`

A good match is accepted while the policy is not already in `RECOVERY` or `LOST`. A weak match is accepted only when center distance and symmetric log-size change agree with the predicted box. A weak accepted match remains `TRACKING` only when the previous mode was `TRACKING`; otherwise it remains `UNCERTAIN`. Bad, missing, or geometrically distant weak evidence uses the predicted box instead of a visual correction.

Rejected output has no box only in `LOST`. `OCCLUDED` maps to public `OCCLUDED`; `UNCERTAIN` and `RECOVERY` map to public `UNCERTAIN`.

Consecutive rejection behavior is controlled by `recovery_after` and `lost_after`. Accepted tracking resets counters. Rejections increment mode-specific counters. Once in recovery, all visual results are currently rejected, so recovery ends only in `LOST`; this is a known extension point rather than a complete recovery search policy.

Template updates require a good accepted match and the configured interval. The current method does not inspect clipping, despite the `template_allow_clipped` config field. The template extractor's returned score—not automatically the accepted match score—is stored in the template bank.

## Decision Helpers

`big_trackers/_decision.py` defines policy-local records and pure helpers:

- `SearchCandidate` records candidate identity, center, confidence, uncertainty, reason, and metadata.
- `BigTrackDecision` records acceptance geometry, output/mode, confidence, template permission, reason, and metadata.
- `BigTrackCounters` records age and uncertainty/loss/recovery counts.
- `clamp01` safely handles bad types, infinities, and NaN.
- `score_band` validates thresholds and returns `bad`, `weak`, or `good`.
- `normalize_predictor_score` reduces confidence as uncertainty rises.
- `box_center_distance_ratio` normalizes center error by predicted-box diagonal.
- `box_size_change_ratio` measures symmetric width/height disagreement in log space.
- `boxes_agree` applies both geometry thresholds.

These helpers do not import neural backends and are the preferred location for reusable policy math.

## Module Map

### Package root

- `BigTracker/__init__.py` exports abstract APIs, predictor implementations/configs, and shared types. It does not export concrete matchers or BigTrack policies.
- `big_track.py`, `predictor.py`, and `matcher.py` define abstract contracts only.

### Maintained subpackages

- `types/` contains common aliases/enums and all immutable boundary dataclasses.
- `big_trackers/` contains orchestration and lifecycle policies.
- `predictor_models/` contains dependency-free motion models and shared motion math.
- `matcher_models/` contains FFT and neural visual adapters plus crop, box, template-bank, and Torch helpers.
- `thirdparty/` contains namespaced runtime portions of four upstream trackers. These are implementation dependencies of neural adapters, not public policy APIs.

See the focused documents for file-by-file details.

## Runtime State and Immutability

Request/state dataclasses are frozen, but contained objects such as tensors, arrays, dictionaries, and backend models may remain mutable. Code uses `dataclasses.replace()` to produce new top-level states. Neural template builders snapshot tensors/features where necessary so subsequent backend activation is less likely to mutate stored templates.

`BigTrackState.matcher_state` is refreshed after approved template updates. Match calls themselves must not mutate the template queue. Tests explicitly protect this rule for MixFormerV2 and the shared bank.

## Extension-Sensitive Invariants

- Construct/load expensive matcher backends in matcher `__init__`, not per target or per initialization.
- Preserve one box and score for every requested target position, in the same order.
- Keep `init_template` stable.
- Change template banks only through approved `update_templates()` calls.
- Keep accept/reject/recovery/lost policy out of matchers and predictors.
- Return the predictor's actual post-update state; do not assume the request state is what the predictor retained.
- Preserve reset, close, restoration, public exports, tools, tests, and docs when adding a component.
