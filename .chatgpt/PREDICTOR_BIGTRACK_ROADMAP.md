# Predictor And BigTrack Roadmap

This roadmap starts after the matcher phase. The four matcher wrappers are now model adapters; they load model/config/checkpoint once, return `MatchEvidence`, and leave lifecycle decisions to `BigTrack`.

The next work should make `BigTrack` smarter:

```text
Predictor
  predicts motion, size, uncertainty, and candidate priors

BigTrack
  creates candidates
  compares matcher evidence
  accepts/rejects/recover/lost decisions
  decides when templates may update
  updates predictor and public output
```

Do not move these rules into NanoTrack, OSTrack, LiteTrack, or MixFormerV2. Matchers should stay visual-evidence providers.

## Current State

Implemented core:

- `BigTracker/big_trackers/base.py`
  - owns the initialize/update flow
  - calls `predictor.predict(...)`
  - calls `make_candidates(...)`
  - calls matcher once per candidate
  - calls `decide(...)`
  - calls `apply_decision(...)`
  - runs template update only when `decision.allow_template_update` is true
- `BigTracker/big_trackers/simple.py`
  - creates one predicted candidate
  - accepts the first match blindly
  - never enters uncertain/recovery/lost
  - never updates templates
- `BigTracker/big_trackers/score_gated.py`
  - creates one predicted candidate
  - accepts strong matcher evidence immediately
  - accepts weak matcher evidence only when it agrees with predictor geometry
  - emits predictor boxes for uncertain/occluded/recovery states
  - enters recovery after repeated rejected visual decisions
  - emits lost after repeated failed recovery cycles
  - updates templates only on good matcher scores at a configured interval
- `BigTracker/predictor_models/kalman.py`
  - constant-velocity Kalman predictor for center and size
  - covariance-backed uncertainty
  - basic reject uncertainty growth
- `BigTracker/predictor_models/adaptive_kalman.py`
  - upgraded constant-velocity Kalman predictor
  - score-adaptive measurement noise
  - uncertainty clamp/decay controls
  - reject velocity damping and covariance growth
  - optional velocity and frame-boundary clamps
- `BigTracker/state.py`
  - already has `TrackerMode`, `BigTrackCounters`, `BigTrackDecision`, `SearchCandidate`, and `MatchEvidence`

Main gap after Phase 2:

- The first real policy exists, but candidate generation is still one-candidate only.
- Recovery mode currently keeps the same search candidate; wider/multiple recovery search is a later phase.
- The Kalman predictor is still basic and can be improved with adaptive uncertainty/velocity behavior.

## Design Goals

1. Keep responsibilities clean.
   - Predictor estimates where and how uncertain.
   - Matcher scores visual evidence.
   - BigTrack decides lifecycle and template updates.

2. Make policy explicit and testable.
   - Thresholds live in config dataclasses.
   - Decisions include reason metadata.
   - Tests cover each mode transition.

3. Prefer one robust policy before adding many predictor models.
   - A stronger BigTrack policy will improve every matcher immediately.

4. Do not overfit to one matcher.
   - Use the common evidence fields:
     - `match_score`
     - `identity_score`
     - `appearance_score`
     - `localization_score`
     - `ambiguity_score`
     - `scale_score`
     - `occlusion_score`
     - `is_clipped`

## Choices We Have

### BigTrack Policy Choices

#### Option A: Score-Gated Policy

Create:

```text
BigTracker/big_trackers/score_gated.py
```

Behavior:

- Accept matcher result when `match_score >= th_good`.
- For `th_bad <= match_score < th_good`, accept matcher only when its box agrees with the predictor box.
- For weak-but-far matches, output the predictor box and enter `UNCERTAIN`.
- For `match_score < th_bad`, output the predictor box and enter `OCCLUDED`.
- Enter `RECOVERY` after `recovery_after` consecutive rejected visual decisions.
- Enter `LOST` after `lost_after` failed recovery cycles.
- Allow template update only when `match_score >= th_good`, the update interval has elapsed, and the box is not clipped unless configured.

Why first:

- Smallest useful upgrade from `SimpleBigTrack`.
- Makes template bank logic actually run.
- Works with all four real matchers.

#### Option B: Recovery Search Policy

Extend score-gated behavior with better recovery search.

Behavior:

- After repeated rejected visual decisions, enter `RECOVERY`.
- A weak match near predictor is accepted for continuity and does not count as rejected.
- A weak match far from predictor is rejected and can contribute to recovery.
- Generate wider search candidates.
- Emit `LOST` after configured recovery budget expires.
- Recover to `TRACKING` only on strong evidence.

Why second:

- Uses the existing `TrackerMode` enum fully.
- Needs more tests and tuning than score gating.

#### Option C: Multi-Candidate Policy

Generate and rank multiple candidates per frame.

Candidate types:

- predicted center
- last accepted center
- velocity-damped center
- local grid around prediction
- scale variants
- wide recovery candidates

Why third:

- It is powerful, but costs multiple matcher calls per frame.
- Needs candidate budgets so heavy matchers do not become too slow.

### Predictor Choices

#### Option A: Improve Current Kalman Predictor

Keep `KalmanPredictorModel`, but add config and metadata support:

- adaptive measurement noise based on accepted match score
- uncertainty cap and decay after clean accepts
- velocity damping after rejected frames
- max velocity and max size-velocity clamps
- frame-boundary aware position clamp
- richer metadata for debugging

Why first:

- Low risk.
- No new dependencies.
- Directly supports better candidate sizing and recovery.

#### Option B: Constant-Acceleration Predictor

Add:

```text
BigTracker/predictor_models/kalman_accel.py
```

Behavior:

- Track position, velocity, and acceleration.
- Useful for fast starts/stops.

Risk:

- More state and more tuning.
- Can overshoot badly after occlusion unless damped.

#### Option C: History-Based Predictor

Add:

```text
BigTracker/predictor_models/history.py
```

Behavior:

- Keep last N accepted boxes.
- Estimate velocity from robust median or exponential smoothing.
- Optionally smooth output boxes.

Why useful:

- Easy to debug.
- Good baseline for comparing Kalman behavior.

#### Option D: Optical-Flow Assisted Predictor

Add optional OpenCV-based motion estimation.

Behavior:

- Track points inside the last accepted box.
- Use optical flow displacement as a motion prior.
- Fall back to Kalman when points are poor.

Risk:

- Adds image-dependent predictor state.
- Needs careful reset/reject behavior.
- Should be optional and later.

## Recommended Implementation Order

### Phase 1: Policy Metrics And Decision Helpers

- [x] Add shared decision helper module:

```text
BigTracker/big_trackers/_decision.py
```

- [x] Add score normalization helpers:
  - clamp scores to `[0, 1]`
  - combine evidence into one acceptance score
  - compute reject reason from failed thresholds
- [x] Add candidate/match pairing helpers:
  - select best match
  - preserve candidate metadata in decision metadata
- [x] Add tests for:
  - strong evidence accepted
  - low match score rejected
  - high ambiguity rejected
  - bad scale rejected
  - clipped result penalized

Acceptance:

- Decision helper tests pass without real matcher models.
- Helpers do not import tracker repos.

### Phase 2: ScoreGatedBigTrack

- [x] Create `ScoreGatedBigTrackConfig`.
- [x] Create `ScoreGatedBigTrack`.
- [x] Implement one-candidate tracking mode first.
- [x] Implement accept/uncertain/reject decisions.
- [x] Enable template updates only when:
  - accepted
  - `match_score >= th_good`
  - `template_update_interval` has elapsed
  - not clipped unless `template_allow_clipped=True`
- [x] Update exports from `BigTracker/big_trackers/__init__.py` and `BigTracker/__init__.py`.
- [x] Add fake matcher tests proving template updates happen only after clean accepts.
- [x] Add `tests/fulltest/main.py` selector support:
  - `POLICY_KIND = "simple"`
  - `POLICY_KIND = "score_gated"`

Acceptance:

- `SimpleBigTrack` still passes existing matcher integration tests.
- `ScoreGatedBigTrack` passes mode/decision/template-update tests.
- Fulltest can switch policies without code edits.

Implemented `ScoreGatedBigTrackConfig` fields:

```text
th_good
th_bad
max_center_error
max_size_error
predictor_uncertainty_scale
recovery_after
lost_after
template_update_interval
template_allow_clipped
```

Rejected visual decisions are:

- no matcher result
- weak score far from predictor
- bad score
- non-good score while already in recovery/lost

Weak score near predictor keeps tracking alive but never updates templates.

### Phase 3: Kalman Predictor Upgrade

- [x] Add upgraded predictor in a separate file:

```text
BigTracker/predictor_models/adaptive_kalman.py
```

- [x] Add `AdaptiveKalmanPredictorConfig`:
  - adaptive measurement noise toggle
  - min/max uncertainty
  - reject velocity damping
  - max position velocity
  - max size velocity
  - uncertainty decay on accept
- [x] Use accepted match score to tune measurement noise.
- [x] Clamp unreasonable velocity and size velocity.
- [x] Improve `update_from_reject(...)`:
  - grow uncertainty
  - damp velocity after repeated rejects
  - preserve predicted position instead of freezing old position when appropriate
- [x] Export `AdaptiveKalmanPredictorConfig` and `AdaptiveKalmanPredictorModel`.
- [x] Add `tests/fulltest/main.py` selector support:
  - `PREDICTOR_KIND = "kalman"`
  - `PREDICTOR_KIND = "adaptive_kalman"`
- [x] Add tests for:
  - adaptive measurement noise
  - reject uncertainty growth
  - velocity damping
  - uncertainty decay after clean accept

Acceptance:

- Existing `KalmanPredictorModel` remains untouched.
- New behavior is selected explicitly through `AdaptiveKalmanPredictorModel`.

### Phase 4: Multi-Candidate Search

- [ ] Add `CandidateGenerationConfig`.
- [ ] Add candidate modes:
  - predicted center
  - last output center
  - velocity-damped center
  - local grid
  - scale variants
- [ ] Use mode-specific budgets:
  - `TRACKING`: cheap, 1-3 candidates
  - `UNCERTAIN`: medium, 3-9 candidates
  - `RECOVERY`: wider, budget-limited
- [ ] Rank matches by combined evidence score plus candidate prior.
- [ ] Add tests proving:
  - best evidence wins
  - candidate prior breaks ties
  - candidate budget is respected

Acceptance:

- Heavy matchers are not called more than configured candidate budget.
- Candidate metadata explains why each candidate exists.

### Phase 5: Recovery Search Expansion

- [ ] Extend `ScoreGatedBigTrack` recovery candidate behavior.
- [x] Implement basic mode transitions:
  - `TRACKING -> UNCERTAIN`
  - `TRACKING -> OCCLUDED`
  - `UNCERTAIN/OCCLUDED -> RECOVERY`
  - `RECOVERY -> TRACKING`
  - `RECOVERY -> LOST`
- [ ] Define output behavior per mode:
  - `ACTIVE`: accepted visual box
  - `UNCERTAIN`: accepted but weak box or predicted box with low confidence
  - `OCCLUDED`: no confident visual box; optional predicted box
  - `LOST`: no box or last known box depending config
- [ ] Add recovery candidates with increasing search radius.
- [ ] Add tests for all transitions and counters.

Acceptance:

- Lost/recovery behavior is deterministic and configured.
- Public `TrackingOutput.status` matches internal lifecycle mode.

### Phase 6: Template Update Policy Refinement

- [ ] Add explicit template update config:
  - update interval/cooldown frames
  - optional clipped-box behavior
  - no update in recovery/lost
  - future optional policy hooks only if needed
- [ ] Add optional update quality score passed into `TemplateCandidate.metadata`.
- [ ] Add tests with fake matcher proving:
  - `init_template` never changes
  - `adaptive_template` changes only after approved clean frames
  - `best_templates` remains bounded
  - cooldown is respected

Acceptance:

- Online/adaptive templates become policy-controlled across all four matchers.

Current Phase 2 rule is intentionally simple:

```text
allow_template_update =
    accepted visual match
    and match_score >= th_good
    and template_update_interval elapsed
    and not clipped unless template_allow_clipped=True
```

### Phase 7: Optional Predictor Models

- [ ] Add `HistoryPredictorModel`.
- [ ] Compare it against Kalman in fake trajectory tests.
- [ ] Add `ConstantAccelerationKalmanPredictorModel` only if history/Kalman tests show a real gap.
- [ ] Add optical-flow predictor last, behind optional OpenCV dependency and clear fallback.

Acceptance:

- Predictor models share the same `PredictorModel` contract.
- Fulltest can select predictor by config.

## Test Plan

Add normal unit tests, not `tests/fulltest`, for deterministic behavior:

```text
tests/test_bigtrack_score_gated.py
tests/test_bigtrack_recovery.py
tests/test_predictor_kalman_policy.py
tests/test_candidate_generation.py
tests/test_template_update_policy.py
```

Use fake predictor and fake matcher objects first. Real tracker/model tests should stay optional.

Required cases:

- clean match accepted
- weak-near match accepted without template update
- weak-far match enters `UNCERTAIN`
- repeated reject enters `RECOVERY`
- recovery success returns to `TRACKING`
- recovery timeout enters `LOST`
- template update happens only on good accepted frames after interval
- matcher errors do not corrupt previous stable state
- candidate budgets are respected
- predictor uncertainty changes search behavior

## Fulltest Additions

`tests/fulltest/main.py` can switch policy, predictor, and matcher:

```python
POLICY_KIND = "score_gated"
PREDICTOR_KIND = "adaptive_kalman"  # or "kalman"
MATCHER_KIND = "mixformerv2"
```

Add visible runtime print fields:

- policy kind
- selected candidate id
- decision reason
- internal mode
- lost/uncertain/recovery counters
- template update count
- match score components

## Done Definition

- At least one production policy exists beyond `SimpleBigTrack`.
- Template updates are driven by BigTrack policy, not matchers.
- Predictor uncertainty changes candidate generation.
- `TRACKING`, `UNCERTAIN`, `RECOVERY`, and `LOST` paths are tested.
- Fulltest can switch predictor, matcher, and policy independently.
- The four external matcher wrappers remain lifecycle-free.
