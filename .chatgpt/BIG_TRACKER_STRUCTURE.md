# Big Tracker Structure

This document defines the intended object-tracker architecture.

The design should stay simple:

```text
BigTrack
  owns one Predictor
  owns one Matcher
  owns one BigTrackState
  decides lifecycle, output status, and when templates may update

Predictor
  owns motion prediction
  reads tracker state
  predicts where the target center and size should be
  creates search candidates

Matcher
  owns model-specific template extraction
  owns model-specific search crop rules
  owns visual matching
  returns visual evidence
```

The matcher may create or update model-specific template objects, but only when `BigTrack` asks it to. The matcher must not decide track lifecycle, lost state, recovery success, or whether learning is safe.

---

## Core Classes

```text
BigTrack
Predictor
PredictorModel
Matcher
MatcherModel
BigTrackState
TrackerPredictionState
MatcherState
TrackingOutput
FrameLike
```

Keep these as the main architecture. Avoid many small policy classes until the core tracker works.

---

## Frame Protocol

The frame object is not owned by the tracker. The tracker only needs a protocol describing the required fields.

```text
FrameLike:
  image
    np.ndarray or tensor
    shape can be HxWxC or HxW

  idx
    integer frame index

  timestamp
    capture time or stream timestamp
```

Do not force every project to use the same frame class. Accept any object that exposes these fields, or define an adapter if needed.

---

## Tracker Prediction State

This is the motion-side state. It should stay independent from matcher templates.

```text
TrackerPredictionState:
  target_pos
    center point: cx, cy

  target_size
    width, height

  target_velocity
    vx, vy

  target_size_velocity
    vw, vh

  last_score
    last accepted tracker confidence

  uncertainty
    scalar or structured uncertainty used by Predictor
```

This state is enough for a Kalman filter, alpha-beta filter, constant-velocity model, or simple custom predictor.

---

## Matcher State

This is the visual-side state. It belongs to the matcher domain.

```text
MatcherState:
  init_template
    first trusted identity template
    never overwritten

  best_templates
    bounded queue of approved clean templates
    max size is configurable

  adaptive_template
    current online template
    updated only when BigTrack approves

  cached_features
    optional model-specific cache
```

The matcher defines what a template is. For MixFormer, OSTrack, Siamese trackers, or a classical matcher, the internal template object may be different.

The important rule:

```text
Matcher can build template objects.
BigTrack decides when template building/updating is allowed.
```

---

## Output Data

This is what client code receives. It should not expose internal tracker state.

```text
TrackingOutput:
  box
    optional x, y, width, height

  frame_idx
    source frame index

  timestamp
    source frame timestamp

  status
    ACTIVE / UNCERTAIN / OCCLUDED / LOST

  confidence
    public confidence score
```

`get_output()` should return this object or the latest one.

`get_state()` should return internal state for debugging, checkpointing, or advanced users.

---

## Big Track State

This is the full internal state for one object track.

```text
BigTrackState:
  prediction
    TrackerPredictionState

  matcher
    MatcherState

  output
    last TrackingOutput

  mode
    INIT / TRACKING / UNCERTAIN / OCCLUDED / RECOVERY / LOST

  counters
    age
    lost_count
    uncertain_count
    recovery_count

  last_seen_frame
    last frame index with accepted visual evidence
```

Do not put matcher internals into prediction state. Do not put motion prediction internals into matcher state.

---

## Predictor

The predictor reads `BigTrackState` and creates search candidates. It does not crop images and does not know matcher-specific template/search formulas.

```text
Predictor:
  model: PredictorModel

  predict(state: BigTrackState, frame: FrameLike) -> TrackerPredictionState

  make_candidates(
    state: BigTrackState,
    prediction: TrackerPredictionState,
    frame: FrameLike
  ) -> list[SearchCandidate]
```

### PredictorModel

```text
PredictorModel:
  predict_position(...)
  predict_size(...)
  predict_uncertainty(...)
  update_from_accept(...)
  update_from_reject(...)
```

The model can be:

- Kalman filter
- alpha-beta filter
- constant velocity model
- custom learned motion model

### SearchCandidate

The candidate should describe where the matcher should search, but not how large the final matcher crop must be.

```text
SearchCandidate:
  candidate_id
  search_center
  predicted_target_size
  prediction_confidence
  motion_uncertainty
  reason
```

The matcher decides the actual search crop because each matcher has its own formula. For example, one matcher may use `2 * sqrt(w * h)` for template size and `2.5x` for search size.

---

## Matcher

The matcher owns visual model behavior. It receives frame, templates, and predicted search information. It internally crops, normalizes, runs the model, and decodes boxes.

```text
Matcher:
  model: MatcherModel

  initialize_template(
    frame: FrameLike,
    target_pos,
    target_size
  ) -> MatcherState

  extract_template(
    frame: FrameLike,
    target_pos,
    target_size,
    previous_state: MatcherState
  ) -> TemplateCandidate

  update_templates(
    state: MatcherState,
    template: TemplateCandidate
  ) -> MatcherState

  match(
    frame: FrameLike,
    matcher_state: MatcherState,
    candidate: SearchCandidate,
    mode: TrackerMode
  ) -> MatchEvidence
```

### MatcherModel

```text
MatcherModel:
  build_template_crop(...)
  build_search_crop(...)
  encode_template(...)
  run_match(...)
  decode_box(...)
  score_identity(...)
  score_ambiguity(...)
```

This is where MixFormer, OSTrack, Siamese trackers, or any other visual tracker plugs in.

### TemplateCandidate

```text
TemplateCandidate:
  template
  source_frame_idx
  source_box
  quality_score
  identity_score
  metadata
```

`TemplateCandidate` is not automatically inserted into `MatcherState`. `BigTrack` must approve the update first.

### MatchEvidence

```text
MatchEvidence:
  candidate_id
  box
  match_score
  identity_score
  appearance_score
  localization_score
  ambiguity_score
  scale_score
  occlusion_score
  is_clipped
  metadata
```

This is evidence only. It is not an accept/reject decision.

---

## BigTrack

`BigTrack` is the orchestrator. It owns the predictor, matcher, internal state, and post-process decision logic.

```text
BigTrack:
  predictor: Predictor
  matcher: Matcher
  state: BigTrackState

  initialize(frame: FrameLike, box) -> BigTrackState

  update(frame: FrameLike) -> TrackingOutput

  reset() -> None

  get_state() -> BigTrackState

  get_output() -> TrackingOutput

  decide(
    state: BigTrackState,
    prediction: TrackerPredictionState,
    candidates: list[SearchCandidate],
    matches: list[MatchEvidence]
  ) -> BigTrackDecision
```

### Initialize Flow

```text
initialize(frame, box):
  target_pos, target_size = box_to_center_size(box)

  prediction_state = TrackerPredictionState(
    target_pos,
    target_size,
    velocity = 0,
    last_score = 1
  )

  matcher_state = matcher.initialize_template(
    frame,
    target_pos,
    target_size
  )

  state = BigTrackState(
    prediction = prediction_state,
    matcher = matcher_state,
    mode = TRACKING,
    output = active output
  )
```

### Update Flow

```text
update(frame):
  prediction = predictor.predict(state, frame)

  candidates = predictor.make_candidates(
    state,
    prediction,
    frame
  )

  matches = []
  for candidate in candidates:
    matches.append(
      matcher.match(
        frame,
        state.matcher,
        candidate,
        state.mode
      )
    )

  decision = decide(
    state,
    prediction,
    candidates,
    matches
  )

  state = apply_decision(state, decision)

  if decision.allow_template_update:
    candidate_template = matcher.extract_template(
      frame,
      decision.accepted_target_pos,
      decision.accepted_target_size,
      state.matcher
    )

    state.matcher = matcher.update_templates(
      state.matcher,
      candidate_template
    )

  return state.output
```

---

## BigTrack Decision

The decision object is internal. It is created by `BigTrack.decide(...)`.

```text
BigTrackDecision:
  accepted
    bool

  accepted_box
    optional box

  accepted_target_pos
    optional center point

  accepted_target_size
    optional size

  output_status
    ACTIVE / UNCERTAIN / OCCLUDED / LOST

  next_mode
    next TrackerMode

  confidence
    output confidence

  allow_template_update
    bool

  reason
    short debug string
```

`BigTrack.decide(...)` is where thresholds, counters, recovery rules, lost rules, and template-freeze rules live.

Do not put these rules into `Matcher`.

---

## Ownership Rules

1. `Predictor` owns motion prediction.
2. `Predictor` creates `SearchCandidate`.
3. `Matcher` owns template extraction and visual matching.
4. `Matcher` chooses its own template/search crop formulas.
5. `Matcher` returns `MatchEvidence`, not lifecycle decisions.
6. `BigTrack` owns accept/reject/lost/recovery decisions.
7. `BigTrack` decides when template update is allowed.
8. `MatcherState.init_template` is never overwritten.
9. `TrackingOutput` is small and client-facing.
10. `BigTrackState` is internal and may contain debugging/state details.

---

## What Should Not Be Split Yet

Do not create separate tiny classes for:

- mode transition policy,
- lost policy,
- output policy,
- state update policy,
- template update policy.

Keep those decisions inside `BigTrack.decide(...)` and `apply_decision(...)` until the core tracker is working. Split later only if the code becomes large for a real reason.
