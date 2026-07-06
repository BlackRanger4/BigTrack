# Big Tracker API Explanation

This file explains the Python API skeleton created from `BIG_TRACKER_STRUCTURE.md`.

The code is intentionally API-only. It defines names, data shapes, method signatures, and module boundaries. It does not implement prediction, matching, ranking, template updates, or lifecycle logic.

## How the API was written

The source structure separates the tracker into three stages:

1. Pre-matcher prediction creates candidate search states.
2. Matcher modules return visual evidence.
3. Post-matcher decision modules accept, reject, recover, update state, or declare lost.

The Python files follow that same separation. Data contracts live in `track_state.py`. Pluggable behavior is written as abstract base classes. Abstract methods use `...` bodies because they are API contracts, not algorithms.

## Root package

### `BigTracker/__init__.py`

Exports the main public API from the package. It makes common types available from `BigTracker` directly, such as `TrackState`, `MatchResult`, `TrackerMode`, and `BigTracker`.

It does not run tracking logic. It only defines the package-facing names.

### `BigTracker/big_tracker.py`

Defines the top-level `BigTracker` abstract API.

It describes how an application will use the tracker:

- `initialize(frame, box, track_id)` starts a track from the first known box.
- `update(frame, frame_index)` processes the next frame and returns public output.
- `get_state()` exposes the current internal state.
- `get_history()` exposes previous public outputs.
- `reset()` clears the tracker.

This file owns orchestration at the API level only. The actual implementation will later call pre-matcher, matcher, and post-matcher components.

### `BigTracker/track_state.py`

Defines shared data contracts used by all other modules.

Important API types:

- `Box`: `(x, y, width, height)`.
- `Point`: `(x, y)`.
- `Size`: `(width, height)`.
- `Frame`: generic frame object placeholder.
- `Template`: generic visual template placeholder.

Important enums:

- `TrackerMode`: `INIT`, `TRACKING`, `UNCERTAIN`, `OCCLUDED`, `RECOVERY`, `LOST`.
- `MatcherMode`: `normal`, `uncertain`, `recovery`.
- `OutputStatus`: `ACTIVE`, `UNCERTAIN`, `OCCLUDED`, `LOST`.

Important data classes:

- `KinematicState` stores position, size, velocity, size velocity, and uncertainty.
- `LastResult` stores the latest accepted, predicted, and matched boxes with scores.
- `CandidateState` is the pre-matcher output sent into the matcher.
- `MatchResult` is visual evidence returned by the matcher.
- `TrackerDecision` is the post-matcher lifecycle decision.
- `TrackingOutput` is the public result returned to users.
- `TrackState` is the full internal state of one tracked object.

This file is central because every stage shares these contracts.

## Pre-matcher predictor package

### `BigTracker/pre_matcher_predictor/__init__.py`

Exports all pre-matcher API classes:

- `StatePredictor`
- `SearchRegionBuilder`
- `CandidateGenerator`
- `PreMatcherPredictor`

### `BigTracker/pre_matcher_predictor/state_predictor.py`

Defines the `StatePredictor` API.

It owns motion and size prediction before visual matching:

- `predict_state(...)` returns the next `KinematicState`.
- `predict_box(...)` converts a kinematic state into a box.
- `predict_uncertainty(...)` returns position and size uncertainty.

The actual predictor can later be a Kalman filter, optical-flow model, learned predictor, or custom estimator.

### `BigTracker/pre_matcher_predictor/search_region_builder.py`

Defines the `SearchRegionBuilder` API.

It turns predicted state into matcher search regions:

- `build_search_region(...)` creates the crop/search area around the predicted box.
- `build_expected_scale_range(...)` describes plausible scale changes.
- `clip_to_frame(...)` constrains a search region to image bounds.

This is separate from visual matching because search policy belongs before the matcher.

### `BigTracker/pre_matcher_predictor/candidate_generator.py`

Defines the `CandidateGenerator` API.

It creates one or more `CandidateState` objects:

- `generate_candidates(...)` handles normal, uncertain, and detector-assisted candidates.
- `generate_recovery_candidates(...)` handles wide recovery search after misses.

This matches the structure rule that uncertain or recovery mode should not depend on only one search region.

### `BigTracker/pre_matcher_predictor/pre_matcher_predictor.py`

Defines the high-level `PreMatcherPredictor` API.

It combines state prediction, search-region building, and candidate generation:

- `predict(...)` returns candidate states for the matcher.
- `choose_matcher_mode(...)` converts tracker mode into matcher mode.

It is the public entry point for the pre-matcher stage.

## Matcher manager package

### `BigTracker/matcher_manager/__init__.py`

Exports matcher APIs:

- `FastMatcher`
- `MatcherAdapter`
- `MatcherSearchInput`
- `MatcherTemplateBundle`
- `MultiTemplateMatcher`
- `RawMatcherOutput`
- `RecoveryMatcher`
- `MatcherManager`

### `BigTracker/matcher_manager/matcher_adapter.py`

Defines the adapter boundary between this tracker and a real visual matcher.

This is the missing integration layer. The tracker owns `Frame`, `CandidateState`, and `VisualMemory`, but a matcher usually wants native inputs such as tensors, image crops, template features, response maps, and model-specific scores. The adapter converts between those worlds.

Data contracts:

- `MatcherTemplateBundle` is the prepared identity template, short-term template, template-bank items, and cached features.
- `MatcherSearchInput` is the prepared search input for one candidate region.
- `RawMatcherOutput` is the matcher-native result before it is decoded into tracker evidence.

Adapter methods:

- `prepare_templates(...)` converts `VisualMemory` into matcher-ready template inputs.
- `prepare_search_input(...)` converts a frame and candidate into matcher-ready search input.
- `run_native_matcher(...)` calls the real matcher backend.
- `decode_result(...)` converts raw matcher output into `MatchResult`.
- `refresh_cache(...)` rebuilds cached matcher features after memory changes.

The matcher stage uses this adapter before and after the real matcher. The post-matcher stage does not call the native matcher; it consumes the decoded `MatchResult`.

### `BigTracker/matcher_manager/fast_matcher.py`

Defines the `FastMatcher` API.

It is intended for confident normal tracking:

- `match(...)` matches one candidate quickly.
- `warm_cache(...)` prepares cached template features.
- `get_adapter()` exposes the adapter used to prepare matcher-native inputs and decode outputs.

It returns `MatchResult`, not lifecycle decisions.

### `BigTracker/matcher_manager/multi_template_matcher.py`

Defines the `MultiTemplateMatcher` API.

It compares candidates against identity anchor, short-term template, and template bank:

- `match_many(...)` returns evidence for many candidates.
- `score_template_agreement(...)` measures agreement between a result and visual memory.
- `get_adapter()` exposes the adapter used for matcher-native conversion.

This supports scale, pose, lighting, and appearance variation without allowing template drift decisions inside the matcher.

### `BigTracker/matcher_manager/recovery_matcher.py`

Defines the `RecoveryMatcher` API.

It is intended for occlusion, disappearance, and reappearance:

- `recover(...)` searches recovery candidates.
- `verify_identity_first(...)` checks whether a result strongly agrees with the identity anchor.
- `get_adapter()` exposes the adapter used for recovery matcher conversion.

Recovery is stricter because the tracker must avoid similar distractors.

### `BigTracker/matcher_manager/matcher_manager.py`

Defines the `MatcherManager` API.

It selects and runs matcher behavior by mode:

- `run(...)` returns a list of `MatchResult` objects.
- `supports_mode(...)` reports whether a matcher mode is available.
- `get_adapter(...)` returns the adapter used for the selected matcher mode.

This keeps matcher selection outside the main tracker loop.

## Post-matcher decision package

### `BigTracker/post_matcher_decision/__init__.py`

Exports post-matcher decision APIs:

- `MatchRanker`
- `RankedMatch`
- `StateUpdatePolicy`
- `TemplateUpdateAdapter`
- `TemplateUpdatePolicy`
- `ModeTransitionPolicy`
- `LostTrackPolicy`
- `PostMatcherDecision`

### `BigTracker/post_matcher_decision/match_ranker.py`

Defines match ranking contracts.

- `RankedMatch` binds one `CandidateState`, one `MatchResult`, a final score, and a reason.
- `MatchRanker.rank(...)` orders matches using visual evidence and motion consistency.
- `MatchRanker.select_best(...)` chooses the best ranked result.

This follows the rule that identity and ambiguity should protect the track more than raw motion.

### `BigTracker/post_matcher_decision/state_update_policy.py`

Defines how accepted or rejected evidence updates the internal `TrackState`.

- `update_on_strong_accept(...)` updates state from a strong visual match.
- `update_on_weak_accept(...)` updates state cautiously.
- `update_on_reject(...)` keeps prediction alive without trusting rejected visual evidence.

This file exists so velocity and size updates stay separate from visual matching.

### `BigTracker/post_matcher_decision/template_update_policy.py`

Defines template-memory update rules.

- `can_update(...)` decides whether a result is clean enough for memory update.
- `collect_candidate(...)` extracts the proposed template candidate.
- `get_adapter()` exposes the adapter that builds memory objects from an approved candidate.
- `apply_update(...)` returns updated `VisualMemory`.
- `extract_and_apply_update(...)` combines approved extraction and memory update.

This protects the core rule that templates are not updated during uncertainty, occlusion, recovery, or weak matches.

### `BigTracker/post_matcher_decision/template_update_adapter.py`

Defines the adapter boundary between an approved `MatchResult` and concrete visual-memory objects.

This adapter is separate from `TemplateUpdatePolicy` because policy decides whether an update is allowed, while the adapter knows how to crop, normalize, encode, or package the template for the matcher backend.

Adapter methods:

- `extract_template_candidate(...)` extracts the candidate template from the frame and accepted match.
- `build_short_term_template(...)` converts that candidate into `ShortTermTemplate`.
- `build_template_bank_entry(...)` converts that candidate into one `TemplateBankEntry`.
- `build_variation_state(...)` records how the candidate differs from the identity anchor.
- `refresh_cached_features(...)` rebuilds matcher cache after memory changes.

The post-matcher stage uses this after a strong accepted decision. It should not update memory directly from raw matcher output without passing through the policy gate.

## Matcher Adapter Flow

The intended matcher flow is:

```text
CandidateState + VisualMemory
  -> MatcherAdapter.prepare_templates(...)
  -> MatcherAdapter.prepare_search_input(...)
  -> MatcherAdapter.run_native_matcher(...)
  -> MatcherAdapter.decode_result(...)
  -> MatchResult
```

The `MatchResult` then goes to post-matcher ranking and decision. It is evidence, not a final lifecycle decision.

## Template Update Adapter Flow

The intended template update flow is:

```text
TrackerDecision + MatchResult + Frame
  -> TemplateUpdatePolicy.can_update(...)
  -> TemplateUpdateAdapter.extract_template_candidate(...)
  -> TemplateUpdateAdapter.build_short_term_template(...)
  -> TemplateUpdateAdapter.build_template_bank_entry(...)
  -> TemplateUpdateAdapter.build_variation_state(...)
  -> TemplateUpdateAdapter.refresh_cached_features(...)
  -> VisualMemory
```

This keeps matching and learning separated. The matcher can suggest a candidate, but post-matcher policy decides if it is safe to learn from it.

### `BigTracker/post_matcher_decision/mode_transition_policy.py`

Defines tracker mode transitions.

- `next_mode(...)` chooses the next `TrackerMode`.
- `update_counters(...)` updates lifecycle counters such as lost, uncertain, and recovery counts.

This keeps mode switching counter-based instead of one-frame threshold based.

### `BigTracker/post_matcher_decision/lost_track_policy.py`

Defines lost-track behavior.

- `should_declare_lost(...)` decides whether recovery has failed.
- `build_lost_output(...)` returns the public lost output.

This makes `LOST` a clear state instead of an exception or crash.

### `BigTracker/post_matcher_decision/post_matcher_decision.py`

Defines the high-level post-matcher API.

- `decide(...)` converts candidates and match results into one `TrackerDecision`.
- `build_output(...)` converts internal state and decision into public `TrackingOutput`.

This file is the API boundary where visual evidence becomes lifecycle state.

## Visual memory package

### `BigTracker/visual_memory/__init__.py`

Exports visual memory APIs:

- `IdentityAnchor`
- `ShortTermTemplate`
- `TemplateBank`
- `TemplateBankEntry`
- `VariationState`
- `VisualMemory`

### `BigTracker/visual_memory/identity_anchor.py`

Defines `IdentityAnchor`.

It stores the first clean template for a track:

- `track_id`
- `template`
- `created_frame`
- `metadata`

This object should never be overwritten by future template updates.

### `BigTracker/visual_memory/short_term_template.py`

Defines `ShortTermTemplate`.

It stores the current clean appearance:

- `template`
- `source_frame`
- `quality_score`
- `metadata`

It is allowed to change only after a strong, identity-consistent accepted match.

### `BigTracker/visual_memory/template_bank.py`

Defines template-bank APIs.

- `TemplateBankEntry` is one stored appearance with quality and diversity scores.
- `TemplateBank.entries()` returns stored appearances.
- `TemplateBank.add(...)` returns a bank with an inserted entry.
- `TemplateBank.remove(...)` returns a bank with an entry removed.
- `TemplateBank.clear()` returns an empty bank.

The bank is abstract because different implementations can use circular replacement, ranked replacement, or diversity-aware replacement.

### `BigTracker/visual_memory/variation_state.py`

Defines `VariationState`.

It stores how the current accepted appearance differs from the identity anchor:

- `anchor_difference`
- `source_frame`
- `confidence`
- `metadata`

This supports future appearance variation tracking without changing the identity anchor.

### `BigTracker/visual_memory/visual_memory.py`

Defines `VisualMemory`, the full memory object used by matchers and update policies.

It contains:

- `identity_anchor`
- `short_term_template`
- `template_bank`
- `variation_state`
- `cached_features`

Matchers read this object. Template update policy decides when to replace parts of it.

## What is intentionally not implemented

The API does not yet contain:

- Kalman filtering or any other predictor math.
- Search-region sizing formulas.
- Neural matcher execution.
- Score fusion formulas.
- Template update selection.
- Counter update logic.
- Lost-track thresholds.

Those belong in concrete subclasses later. This first version only fixes the architecture and call contracts.
