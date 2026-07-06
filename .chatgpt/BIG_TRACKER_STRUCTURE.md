# Big Tracker Structure

This document describes the full tracker around the visual matcher.

The matcher is only one part of the system. It should decide whether an image region matches the target appearance. The bigger tracker should own prediction, state management, template update policy, recovery, and lost-track decisions.

High-level separation:

```text
Pre-matcher:
  predict where the target may be
  predict expected size and uncertainty
  create candidate search regions

Matcher:
  visually compare target memory against candidate regions
  return boxes, scores, identity evidence, and update candidates

Post-process:
  decide whether to accept, reject, hold, recover, update templates, update states, or declare lost
```

---

## Main Goal

The tracker should keep the correct object identity when:

- the object becomes bigger or smaller,
- the object moves fast,
- the object is partly or fully occluded,
- the object disappears and appears again,
- there are similar distractors,
- several predicted states or candidate boxes exist.

The tracker should not blindly trust either motion prediction or visual matching. It should fuse them through explicit confidence and state rules.

---

## Full Pipeline

```text
Frame t
  |
  v
1. Pre-matcher prediction
  - predict target position, size, velocity, uncertainty
  - create candidate states and search regions
  - choose matcher mode: normal / uncertain / recovery
  |
  v
2. Visual matcher
  - match identity anchor, online templates, and state memories
  - predict box, scale, and confidence for each candidate region
  - return visual evidence, not final lifecycle decision
  |
  v
3. Post-matcher decision
  - accept or reject match
  - update physical state or hold previous state
  - update template memory or freeze it
  - switch mode if confidence changes
  - declare occluded, lost, or recovered
  |
  v
Track state for frame t
```

---

## Track State

Each tracked object should have one main state object.

```text
TrackState:
  id
  mode
  age
  last_seen_frame
  lost_count
  uncertain_count
  recovery_count

  kinematic_state:
    position: cx, cy
    size: w, h
    velocity: vx, vy
    size_velocity: vw, vh
    uncertainty_position
    uncertainty_size

  visual_memory:
    identity_anchor
    short_term_template
    template_bank
    variation_state
    cached_features

  last_result:
    accepted_box
    predicted_box
    matched_box
    match_score
    identity_score
    appearance_score
    localization_score
    ambiguity_score
```

The exact predictor can be a Kalman filter, another estimator, or a custom model. The important design rule is that this predictor lives outside the matcher.

---

## Tracker Modes

The tracker should use modes instead of one flat tracking loop.

| Mode | Meaning | Matcher Behavior | Post-process Behavior |
|---|---|---|---|
| `INIT` | First frame or new track | Create identity anchor | Initialize state and memory |
| `TRACKING` | Confident normal tracking | Compact search, normal matcher | Accept good matches, allow safe template update |
| `UNCERTAIN` | Match quality dropped | Wider search or multiple candidates | Freeze templates, require stronger identity evidence |
| `OCCLUDED` | Target likely hidden | Recovery matcher, identity-first | Hold or predict state, do not update template |
| `RECOVERY` | Search for target after miss | Expanded search, compare candidates | Accept only high identity and low ambiguity |
| `LOST` | Track cannot be trusted | Optional global re-detection | Stop reporting active target or mark inactive |

Mode transitions should be driven by scores and counters, not a single bad frame.

---

## Pre-Matcher Stage

The pre-matcher prepares hypotheses before any visual model runs.

### Inputs

```text
frame
previous TrackState
external detections, optional
scene constraints, optional
```

### Outputs

```text
CandidateState list:
  predicted_box
  search_region
  prediction_confidence
  motion_uncertainty
  expected_scale_range
  priority
  reason
```

### Responsibilities

1. Predict target position.

   Estimate where the target center should be in the new frame.

   ```text
   previous position + velocity -> predicted position
   ```

2. Predict target size.

   Estimate expected width and height. This matters because the matcher search crop depends on target scale.

   ```text
   previous size + size_velocity -> predicted size
   ```

3. Estimate uncertainty.

   If motion is stable, uncertainty is small. If the target is occluded, moving fast, or recently mismatched, uncertainty grows.

4. Build candidate states.

   Do not send only one region when uncertainty is high. Create several candidate search areas.

   Examples:

   ```text
   normal:
     one compact search region around predicted box

   uncertain:
     predicted region
     previous accepted region
     slightly larger search region

   recovery:
     large region around predicted position
     region around last seen position
     detector proposals, if available
     global or tiled search, if needed
   ```

5. Choose matcher mode.

   The pre-matcher tells the matcher how hard the situation is.

   ```text
   TRACKING -> compact, fast matcher
   UNCERTAIN -> wider, stricter matcher
   RECOVERY -> identity-first matcher
   ```

### Search Region Policy

```text
base_search_factor:
  used when tracking is confident

uncertain_search_factor:
  used after low score or high uncertainty

recovery_search_factor:
  used after repeated misses or occlusion
```

Search factor should grow with uncertainty, not with raw speed alone.

Recommended behavior:

```text
if mode == TRACKING:
  search_factor = base

if mode == UNCERTAIN:
  search_factor = base + uncertainty_scale

if mode == RECOVERY:
  search_factor = large
  generate multiple regions if needed
```

---

## Matcher Stage

The matcher is a pluggable visual verification module.

It should not decide track lifecycle by itself. It should return evidence.

### Inputs

```text
frame
CandidateState list
VisualMemory:
  identity_anchor
  short_term_template
  template_bank
  variation_state
  cached_features
MatcherMode:
  normal / uncertain / recovery
```

### Outputs

```text
MatchEvidence list:
  candidate_id
  box
  scores:
    match
    identity
    appearance
    localization
    ambiguity
    scale
    occlusion
  scale evidence
  ambiguity evidence
  occlusion evidence
  search_region_id
  debug_maps, optional
```

### Matcher Responsibilities

1. Compare against identity anchor.

   The identity anchor is the first clean template and must never be overwritten. It is the main protection against drift.

2. Compare against online appearance memory.

   The short-term template and template bank help when the object changes scale, pose, illumination, or viewpoint.

3. Decode location and scale.

   The default decoder should be center-based:

   ```text
   score_map + size_map + offset_map -> box
   ```

4. Score ambiguity.

   A high best score is not enough. The matcher should know whether the second-best object is close.

   ```text
   ambiguity_score = second_best_score / best_score
   ```

5. Return template-update evidence only.

   The matcher can expose box, score, response-map, ambiguity, clipping, and visibility evidence. It must not return a ready-to-insert template. The post-process decides whether memory may update, then the template update adapter extracts and builds memory objects from the approved frame region.

### Matcher Selection

The big tracker can choose different matcher behavior by mode.

| Situation | Matcher Behavior |
|---|---|
| Normal tracking | Fast cached matcher, compact search |
| Scale change | Center head with strong size regression, online template evidence |
| Low confidence | Identity anchor gets higher weight |
| Occlusion | No template update, recovery evidence only |
| Reappearance | Expanded search, identity-first selection |
| Similar distractor | Require identity and appearance agreement |
| Many candidates | Use cached template features and rank candidates |

### Matcher Score Formula

A practical final match score can combine several signals:

```text
match_score =
  w_identity     * identity_score +
  w_appearance   * appearance_score +
  w_localization * localization_score +
  w_scale        * scale_score -
  w_ambiguity    * ambiguity_score
```

Weights should change by mode.

Normal mode:

```text
identity: medium
appearance: high
localization: high
ambiguity penalty: medium
```

Recovery mode:

```text
identity: very high
appearance: medium
localization: medium
ambiguity penalty: high
```

---

## Post-Matcher Stage

The post-matcher turns visual evidence into tracker decisions.

### Inputs

```text
TrackState
CandidateState list
MatchEvidence list
frame index
```

### Outputs

```text
Updated TrackState
public tracking output:
  active / uncertain / occluded / lost
  target_box, optional
  confidence
  reason
```

### Main Decisions

1. Accept match.

   Use when match is visually strong and consistent with predicted state.

2. Reject match but keep track alive.

   Use when match is weak but the track was recently confident.

3. Hold or predict state.

   Use during short occlusion. The tracker may output no box, last box, or predicted box depending on application needs.

4. Enter recovery.

   Use after repeated weak matches or suspected object disappearance.

5. Declare lost.

   Use after recovery fails for too long or ambiguity stays high.

6. Update visual memory.

   Only after strong, stable, identity-consistent matches.

7. Update kinematic state.

   Use accepted boxes. Do not update position and velocity from rejected matches.

---

## Acceptance Policy

The post-process should not use only one threshold.

### Strong Accept

```text
if match_score >= high_threshold
and identity_score >= identity_threshold
and localization_score >= localization_threshold
and ambiguity_score <= ambiguity_threshold:
  accept match
  update kinematic state
  maybe update template memory
  mode = TRACKING
```

### Weak Accept

Use when visual score is okay but not perfect.

```text
if match_score >= medium_threshold
and identity_score >= identity_threshold
and prediction_consistency is good:
  accept match
  update kinematic state carefully
  do not update template
  mode = UNCERTAIN or TRACKING depending on counters
```

### Reject but Keep Alive

```text
if match_score < medium_threshold
and lost_count < max_short_miss:
  reject visual box
  keep predicted state
  freeze template
  mode = UNCERTAIN or OCCLUDED
```

### Enter Recovery

```text
if repeated low score
or object likely outside search region
or occlusion_hint is high:
  mode = RECOVERY
  expand search next frame
  freeze template memory
```

### Declare Lost

```text
if recovery_count > max_recovery_frames
or identity_score stays low
or ambiguity stays high:
  mode = LOST
  stop normal updates
```

---

## Template Update Policy

Template update must happen after post-process approval, not directly inside the matcher.

### Memory Types

```text
identity_anchor:
  created at INIT
  never updated

short_term_template:
  current clean appearance
  updated only after strong accept

template_bank:
  small memory of diverse confident appearances
  fixed-size, ranked or circular

variation_state:
  difference between identity anchor and accepted current appearance
  updated only when template update is approved
```

### Update Allowed

```text
if mode == TRACKING
and strong_accept
and identity_score is high
and ambiguity_score is low
and box is not clipped
and scale jump is plausible
and target is not occluded:
  allow template update from the approved frame region
```

### Update Forbidden

```text
if mode in [UNCERTAIN, OCCLUDED, RECOVERY, LOST]
or match was weak
or identity and appearance disagree
or multiple candidates have similar score
or box is clipped by image border
or object is heavily occluded:
  freeze template memory
```

### Best Practice

Do not update immediately from a single frame. Keep the best candidate over a short interval.

```text
candidate_pool.add(approved_frame_region)

every update_interval:
  choose best stable candidate
  verify identity anchor agreement
  insert into short_term_template or template_bank
```

---

## Kinematic State Update

The kinematic state is updated only from accepted visual evidence.

### Strong Match

```text
position <- matched_box center
size <- matched_box size
velocity <- estimate from previous accepted state and current accepted state
size_velocity <- estimate from size change
uncertainty <- decrease
```

### Weak Match

```text
position <- blend(predicted position, matched position)
size <- blend(predicted size, matched size)
velocity <- update cautiously
uncertainty <- keep or slightly increase
```

### Rejected Match

```text
position <- predicted position
size <- predicted size
velocity <- keep or damp
uncertainty <- increase
```

### Lost Track

```text
do not update from matcher
increase uncertainty or stop active prediction
wait for re-detection if the application supports it
```

---

## Multi-Candidate Decision

When several candidate states exist, the tracker should rank them using both motion and visual evidence.

```text
candidate_score =
  a * match_score +
  b * identity_score +
  c * motion_consistency +
  d * scale_consistency -
  e * ambiguity_score
```

Motion consistency should not override identity. If a nearby object has good motion but poor identity, it should not win.

Recommended priority:

```text
1. identity consistency
2. low ambiguity
3. localization quality
4. motion consistency
5. appearance memory consistency
6. scale consistency
```

---

## Occlusion Handling

Occlusion is not the same as lost identity.

### Suspect Occlusion When

```text
match_score drops
identity_score is still partly present
box becomes unstable or partially visible
scale estimate becomes unreliable
score map is weak or spread out
```

### During Occlusion

```text
freeze template updates
increase search uncertainty
keep identity anchor active
do not learn from occluded crop
hold or predict state
```

### After Reappearance

```text
require high identity score
require low ambiguity
accept only stable box
wait before template update
switch from RECOVERY to TRACKING after confirmation
```

---

## Scale Handling

Scale should be handled in both pre-matcher and matcher.

Pre-matcher:

```text
predict expected size
predict size uncertainty
set search crop based on expected size and uncertainty
generate alternate scale candidates if needed
```

Matcher:

```text
regress width and height through size map
return scale_score and scale_change
compare scale against expected range
```

Post-process:

```text
accept plausible scale changes
reject or mark uncertain for extreme one-frame scale jumps
update size velocity only after accepted matches
freeze template if scale is unstable
```

---

## Public Output Policy

Different applications want different behavior when confidence is low. The tracker should expose status, not only a box.

```text
Output:
  status: ACTIVE / UNCERTAIN / OCCLUDED / LOST
  box: optional
  confidence
  identity_score
  reason
```

Recommended:

- `ACTIVE`: output accepted visual box.
- `UNCERTAIN`: output accepted weak box or predicted box with low confidence.
- `OCCLUDED`: optionally output predicted box, but mark it as not visually confirmed.
- `LOST`: output no active target box.

---

## Pseudocode

```text
for each frame:
  candidates = pre_matcher.predict(track_state, frame)

  match_evidence = matcher.run(
    frame=frame,
    candidates=candidates,
    visual_memory=track_state.visual_memory,
    mode=track_state.mode
  )

  decision = post_process.decide(
    track_state=track_state,
    candidates=candidates,
    match_evidence=match_evidence
  )

  track_state = state_updater.apply_decision(track_state, decision)

  if decision.memory_update.action != FREEZE:
    visual_memory = template_update_policy.apply_approved_update(frame, visual_memory, decision)

  apply lifecycle transition from decision

  return public_output(track_state)
```

---

## Suggested Module Layout

```text
BigTracker
  TrackState
  PreMatcherPredictor
    StatePredictor
    SearchRegionBuilder
    CandidateGenerator
  MatcherManager
    FastMatcher
    MultiTemplateMatcher
    RecoveryMatcher
  PostMatcherDecision
    MatchRanker
    StateUpdatePolicy
    TemplateUpdatePolicy
  VisualMemory
    IdentityAnchor
    ShortTermTemplate
    TemplateBank
    VariationState
```

---

## Implementation Order

### Version 1

Build the lifecycle first.

```text
single predicted state
single matcher call
accept/reject thresholds
TRACKING / UNCERTAIN / LOST modes
no template update
```

### Version 2

Add robust memory.

```text
identity anchor
short-term template
template update gate
template freeze on uncertainty
```

### Version 3

Add recovery.

```text
expanded search
multiple candidate regions
RECOVERY mode
identity-first reacceptance
```

### Version 4

Add multi-state matching.

```text
multiple visual states
template bank
variation state
candidate ranking by identity + motion + scale
```

### Version 5

Optimize.

```text
cached template features
fast matcher in normal mode
expensive matcher only in recovery
candidate pruning
```

---

## Core Rules

1. The matcher returns evidence; post-process makes decisions.
2. Never update the identity anchor.
3. Never update templates during uncertainty, occlusion, or recovery.
4. Do not update velocity from rejected visual matches.
5. A high visual score is not enough if ambiguity is high.
6. Motion consistency should help ranking, but identity consistency should protect the track.
7. Recovery should use larger search and stricter identity checks.
8. Lost is a state, not a crash. The tracker should report it clearly.
