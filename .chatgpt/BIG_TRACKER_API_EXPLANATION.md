# Big Tracker API Explanation

This file explains the current Python API skeleton for the tracker architecture.

The code is still API-first. It defines data shapes, boundaries, and ownership rules. It does not implement prediction math, neural matching, score fusion, or lifecycle thresholds.

## Main Rule

The matcher returns visual evidence only.

It does not decide:

- whether a match is accepted,
- whether the track is lost,
- whether recovery succeeded,
- whether templates are updated,
- whether identity memory changes.

Those decisions belong to the post-matcher decision stage.

## Package Shape

```text
BigTracker/
  common_types.py
  track_state.py
  big_tracker.py
  pre_matcher_predictor/
  matcher_manager/
  post_matcher_decision/
  visual_memory/
```

## Common Types

`BigTracker/common_types.py` owns primitive aliases:

- `Box`: `(x, y, width, height)`.
- `Point`: `(x, y)`.
- `Size`: `(width, height)`.
- `Frame`: backend-specific frame object.
- `Template`: backend-specific template object.
- `FeatureMap`: backend-specific cached feature mapping.

These primitives are outside `track_state.py` so visual-memory code does not need to import tracker state contracts.

## Track Contracts

`BigTracker/track_state.py` owns tracker-level domain contracts.

Important enums:

- `TrackerMode`: `INIT`, `TRACKING`, `UNCERTAIN`, `OCCLUDED`, `RECOVERY`, `LOST`.
- `MatcherMode`: `normal`, `uncertain`, `recovery`.
- `OutputStatus`: public status returned to users.
- `AcceptanceLevel`: `STRONG`, `WEAK`, `REJECTED`.
- `OutputBoxSource`: whether output came from a matched, predicted, last accepted, or no box.
- `MemoryUpdateAction`: `FREEZE`, `COLLECT`, `APPLY`.

Important data structures:

- `KinematicState`: position, size, velocity, size velocity, and uncertainty.
- `CandidateState`: pre-matcher hypothesis sent into the matcher, including prediction confidence and uncertainty.
- `MatchScores`: normalized visual scores returned by the matcher.
- `ScaleEvidence`, `AmbiguityEvidence`, `OcclusionEvidence`: structured evidence used by post-matcher policy.
- `MatchEvidence`: visual evidence from the matcher. It has no template update payload and no lifecycle decision.
- `LifecycleTransition`: next mode, public status, counters, and last-seen frame.
- `MemoryUpdatePlan`: whether memory stays frozen, collects a candidate, or applies an approved update.
- `TrackerDecision`: one post-matcher decision containing selected candidate, selected evidence, lifecycle transition, public output intent, and memory update intent.
- `TrackingOutput`: public result returned by the top-level tracker.
- `TrackState`: full internal state for one track.

`TrackState.visual_memory` is explicitly typed as `VisualMemory`, but only through a type-checking import. This keeps runtime imports acyclic.

## Top-Level Tracker

`BigTracker/big_tracker.py` defines the public tracker API:

- `initialize(frame, box, track_id)` creates one track.
- `update(frame, frame_index, external_detections=None)` processes a frame.
- `get_state()` returns internal state.
- `get_history()` returns public outputs.
- `reset()` clears the tracker.

External detections are accepted at the public API because the pre-matcher can use them during uncertainty or recovery.

## Pre-Matcher Predictor

The pre-matcher predicts motion and creates candidate regions before visual matching.

Main APIs:

- `StatePredictor.predict_state(...)`
- `StatePredictor.predict_box(...)`
- `StatePredictor.predict_uncertainty(...)`
- `SearchRegionBuilder.build_search_region(...)`
- `SearchRegionBuilder.build_expected_scale_range(...)`
- `CandidateGenerator.generate_candidates(...)`
- `CandidateGenerator.generate_recovery_candidates(...)`
- `PreMatcherPredictor.predict(...)`
- `PreMatcherPredictor.choose_matcher_mode(...)`

`CandidateGenerator.generate_candidates(...)` receives prediction confidence, motion uncertainty, and size uncertainty. This prevents fake constants in `CandidateState`.

## Matcher Manager

The matcher package contains adapter and matcher orchestration contracts.

Important data structures:

- `MatcherTemplateBundle`: identity, short-term, bank templates, and cached features prepared for a backend.
- `CoordinateTransform`: crop geometry, model input size, and padding used to map native matcher output back to frame coordinates.
- `MatcherSearchInput`: frame/search-region/model-input package for one candidate.
- `RawMatcherOutput`: backend-native result, scores, response map, second-best score, and debug maps.

Important APIs:

- `MatcherAdapter.prepare_templates(...)`
- `MatcherAdapter.prepare_search_input(...)`
- `MatcherAdapter.run_native_matcher(...)`
- `MatcherAdapter.decode_result(...) -> MatchEvidence`
- `MatcherAdapter.refresh_cache(...)`
- `MatcherManager.run(...) -> Sequence[MatchEvidence]`

`RawMatcherOutput` intentionally does not contain a template update candidate. The matcher can expose scores, maps, boxes, and ambiguity evidence. Template extraction happens only after post-matcher approval.

`RecoveryMatcher` returns `MatchEvidence` only. It does not expose a `verify_identity_first()` boolean because recovery acceptance is a lifecycle decision.

## Post-Matcher Decision

The post-matcher stage owns acceptance, lifecycle, output intent, and template-update intent.

Important APIs:

- `MatchRanker.rank(...)`
- `MatchRanker.select_best(...)`
- `PostMatcherDecision.decide(...) -> TrackerDecision`
- `PostMatcherDecision.build_output(...) -> TrackingOutput`
- `StateUpdatePolicy.apply_decision(...) -> TrackState`

There are no separate `ModeTransitionPolicy` or `LostTrackPolicy` exports. Those concerns are part of the single `TrackerDecision` / `LifecycleTransition` result so counters, mode, lost state, and public status cannot drift apart.

## Template Update Flow

Template update is split into policy and backend conversion:

- `TemplateUpdatePolicy.can_update(track_state, decision)`
- `TemplateUpdatePolicy.apply_approved_update(frame, visual_memory, decision, frame_index)`
- `TemplateUpdateAdapter.extract_template_candidate(frame, decision)`
- `TemplateUpdateAdapter.build_short_term_template(...)`
- `TemplateUpdateAdapter.build_template_bank_entry(...)`
- `TemplateUpdateAdapter.build_variation_state(...)`
- `TemplateUpdateAdapter.refresh_cached_features(...)`

The adapter extracts a template from the frame and approved decision. It does not consume a matcher-owned template payload.

## Visual Memory

Visual memory contains:

- `IdentityAnchor`: first clean identity template.
- `ShortTermTemplate`: current clean appearance.
- `TemplateBank`: small immutable concrete template store.
- `VariationState`: difference between accepted appearance and the anchor.
- `cached_features`: backend-specific feature cache.

`VisualMemory.with_template_update(...)` returns a new memory object while preserving the identity anchor. This makes the intended identity-anchor rule explicit in the API.

## Implementation Order

Version 1 should implement:

- one predictor,
- one matcher adapter,
- one matcher manager path,
- one post-matcher decision policy,
- one state updater,
- template freeze by default.

Do not implement template-bank replacement, variation tracking, or complex recovery until V1 can correctly accept, reject, output uncertain/lost states, and avoid updating memory from weak evidence.
