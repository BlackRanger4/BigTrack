# Tracker Matcher Integration TODO

This TODO maps the four external trackers in `ignores/Trackers` into the current `BigTracker` architecture from `BIG_TRACKER_STRUCTURE.md`.

The target shape is:

```text
BigTrack
  owns Predictor
  owns MatcherModel
  owns BigTrackState

MatcherModel
  loads model/config/checkpoint once in __init__
  owns template/search crop rules
  owns visual inference
  returns MatchEvidence only
```

Do not move accept/reject, lost/recovery, or template-learning policy into these matchers. Those rules stay in `BigTrack.decide(...)` and `BigTrack.apply_decision(...)`.

## Hard Rules

1. Model loading belongs in `MatcherModel.__init__(...)`.
   - Load config, build network, load checkpoint, move to device, set eval mode, and build static inference helpers once.
   - `initialize_template(...)` must not build or load the model.
   - `initialize_template(...)` only extracts the first trusted visual template for the current object.

2. Template history is a BigTracker matcher contract, not a tracker-specific feature.
   - Every matcher must represent:
     - `init_template`: first trusted identity template, never overwritten.
     - `best_templates`: bounded bank of latest approved good templates.
     - `adaptive_template`: current online template used for adaptation.
   - Some source trackers do not update templates internally, for example NanoTrack. That does not matter. Our matcher wrapper still supports the same template history shape.
   - `BigTrack` decides when a new template is clean enough. The matcher only builds and stores template objects when asked.

## Current Fit

The existing package already has the right contracts:

- `BigTracker/state.py`
  - `MatcherState`
  - `TemplateCandidate`
  - `SearchCandidate`
  - `MatchEvidence`
- `BigTracker/matcher.py`
  - `MatcherModel`
- `BigTracker/matcher_models/fft.py`
  - useful style reference for a concrete matcher
- `BigTracker/big_trackers/simple.py`
  - first policy for integration smoke tests

The tracker work should be implemented as concrete matcher models under:

```text
BigTracker/matcher_models/
  nanotrack.py
  ostrack.py
  litetrack.py
  mixformer_v2.py
```

Shared utilities should be small and matcher-facing only:

```text
BigTracker/matcher_models/_crop.py
BigTracker/matcher_models/_torch.py
BigTracker/matcher_models/_boxes.py
```

Avoid adding lifecycle policy classes at this stage.

## Source Tracker Summary

### NanoTrack

Source:

```text
ignores/Trackers/NanoTrack/nanotrack/tracker/nano_tracker.py
ignores/Trackers/NanoTrack/nanotrack/tracker/base_tracker.py
ignores/Trackers/NanoTrack/nanotrack/models/model_builder.py
ignores/Trackers/NanoTrack/nanotrack/core/config.py
```

Behavior:

- Uses BGR `np.ndarray` images.
- Builds the first template with `model.template(z_crop)`.
- Search crop uses `get_subwindow(...)`.
- Tracker state is `center_pos`, `size`, and `channel_average`.
- Output is `bbox` and `best_score`.
- Applies internal scale/aspect/window penalties before returning the box.

BigTracker mapping:

- `__init__(...)`
  - load NanoTrack config
  - build separate template-backbone and search-backbone model instances, because template/search shapes differ and shape switching is costly on GPU runtimes
  - load the same checkpoint into both model instances
  - move both models to device and set eval mode
  - precompute static point/window tensors
- `initialize_template(...)`
  - crop template from `frame.image`
  - run `model.template(z_crop)`
  - store template feature state, channel average, config values, and initial target geometry in `MatcherState.init_template`
  - initialize `adaptive_template` from the same template
  - initialize `best_templates` as an empty bounded bank
- `match(...)`
  - use `candidate.search_center` and `candidate.predicted_target_size`, not NanoTrack's old `self.center_pos`
  - select the template object to activate before inference: prefer `adaptive_template`, otherwise `init_template`
  - crop search with NanoTrack formula
  - run `model.track(x_crop)`
  - decode `cls` and `loc`
  - return `MatchEvidence.box` from decoded bbox
  - use `best_score` as `match_score`, `identity_score`, and `appearance_score`
- `extract_template(...)`
  - crop a new template from an approved box
  - encode it using the same NanoTrack template path
  - return a `TemplateCandidate`
- `update_templates(...)`
  - keep `init_template` unchanged
  - append approved template to `best_templates`
  - set `adaptive_template` to the newest approved template
  - enforce `max_best_templates`

Implementation risk:

- NanoTrack uses global `cfg`; adapter config must avoid mutating global settings unexpectedly across tests.
- NanoTrack template and search crops use different input shapes. Keep separate template/search backend instances for PyTorch, ONNX, and later TensorRT paths instead of reusing one dynamic-shape runtime object.
- ONNX Runtime backend should use separate template/search backbone sessions. If `onnx_provider="cuda"` is requested and `CUDAExecutionProvider` is not installed, fail fast instead of silently falling back to CPU.
- Original code clips target size to at least 10 pixels. Preserve this in matcher output metadata and let `BigTrack` decide whether that evidence is acceptable.

### OSTrack

Source:

```text
ignores/Trackers/OSTrack/lib/test/tracker/ostrack.py
ignores/Trackers/OSTrack/lib/test/parameter/ostrack.py
ignores/Trackers/OSTrack/lib/train/data/processing_utils.py
ignores/Trackers/OSTrack/lib/models/ostrack/ostrack.py
```

Behavior:

- Uses RGB `np.ndarray` images.
- Template crop: `sample_target(image, init_bbox, template_factor, output_sz=template_size)`.
- Search crop: `sample_target(image, state, search_factor, output_sz=search_size)`.
- Preprocesses image and attention mask with `Preprocessor`.
- Inference calls `network.forward(template=..., search=..., ce_template_mask=...)`.
- Applies Hann window to `score_map`.
- Decodes box with `box_head.cal_bbox(...)`.
- Maps crop coordinates back using previous state center and `resize_factor`.
- Original class returns only `target_bbox`, not a confidence score.

BigTracker mapping:

- `__init__(...)`
  - load yaml config and checkpoint
  - build network
  - move network to device and set eval mode
  - build preprocessor and Hann window once
- `initialize_template(...)`
  - run template crop/preprocess once
  - store processed template tensors, `box_mask_z`, original template crop, config, and model metadata
  - initialize `adaptive_template` from the same template
  - initialize `best_templates` as an empty bounded bank
- `match(...)`
  - convert `candidate.search_center` plus `candidate.predicted_target_size` to `[x, y, w, h]`
  - select visual templates from `MatcherState`: `init_template`, latest approved templates, and/or `adaptive_template`
  - crop around that candidate with OSTrack `search_factor`
  - run network inference
  - decode `score_map`, `size_map`, `offset_map`
  - map predicted `[cx, cy, w, h]` back around the candidate center, not hidden `self.state`
  - return `MatchEvidence`
- Scores:
  - `match_score`: normalized max of Hann-windowed score map
  - `localization_score`: peak sharpness or distance from candidate center
  - `ambiguity_score`: second peak divided by first peak
  - `scale_score`: area ratio against `candidate.predicted_target_size`
- `extract_template(...)`
  - build a new processed template from an approved box
- `update_templates(...)`
  - keep `init_template` fixed
  - add approved templates to `best_templates`
  - set `adaptive_template` to the newest approved template
  - enforce `max_best_templates`

Implementation risk:

- `OSTrack`, `LiteTrack`, and `MixFormerV2` all use a top-level `lib` package. They cannot be safely imported together by simply appending each repo to `sys.path`.

### LiteTrack

Source:

```text
ignores/Trackers/LiteTrack/lib/test/tracker/litetrack.py
ignores/Trackers/LiteTrack/lib/test/parameter/litetrack.py
ignores/Trackers/LiteTrack/lib/models/litetrack/litetrack.py
```

Behavior:

- Built from OSTrack-style code.
- Template crop and search crop follow the same `sample_target` style.
- Template feature is computed once with `network.forward_z(...)`.
- Search inference calls `network(template_feats=z_feat, search=x_dict.tensors)`.
- Current runtime path is `track_center(...)` for `cfg.MODEL.HEAD.TYPE == 'CENTER'`.
- Original class returns only `target_bbox`, not a confidence score.

BigTracker mapping:

- Reuse the OSTrack crop, map-back, clipping, score-map, and score-normalization utilities.
- `__init__(...)`
  - load yaml config and checkpoint
  - build LiteTrack network
  - move network to device and set eval mode
  - build preprocessor and Hann window once
- `initialize_template(...)`
  - build `z_feat` from approved initial template
  - store `z_feat`, template tensor, template bbox in crop coordinates, config, and checkpoint metadata
  - initialize `adaptive_template` from the same template
  - initialize `best_templates` as an empty bounded bank
- `match(...)`
  - select the active template feature from `adaptive_template`, otherwise `init_template`
  - crop around `SearchCandidate`
  - run LiteTrack center-head inference
  - decode with LiteTrack's `box_head.cal_bbox(...)`
  - return `MatchEvidence`
- `extract_template(...)`
  - recompute `z_feat` for an approved template candidate
- `update_templates(...)`
  - keep `init_template` fixed
  - append approved template to `best_templates`
  - set `adaptive_template` only when BigTrack approves
  - enforce `max_best_templates`

Implementation risk:

- Its source imports `build_LiteTrack` from `lib.models`; isolate package loading before attempting to load it next to OSTrack.
- Checkpoint loading uses `strict=False`; document missing/unexpected keys in matcher metadata.

### MixFormerV2

Source:

```text
ignores/Trackers/MixFormerV2/lib/test/tracker/mixformer2_vit.py
ignores/Trackers/MixFormerV2/lib/test/tracker/mixformer2_vit_online.py
ignores/Trackers/MixFormerV2/lib/test/parameter/mixformer2_vit.py
ignores/Trackers/MixFormerV2/lib/test/parameter/mixformer2_vit_online.py
ignores/Trackers/MixFormerV2/lib/models/mixformer2_vit/
```

Behavior:

- Offline tracker stores `template` and `online_template`, then runs `network(template, online_template, search, softmax=True)`.
- Online tracker additionally returns `conf_score`.
- Online tracker updates templates inside `track()` using score threshold, max-score decay, update interval, and online queue size.
- Search box mapping is the same previous-center plus `resize_factor` formula used by OSTrack.

BigTracker mapping:

- Implement one `MixFormerV2MatcherModel` with a config flag:
  - `variant="offline"`
  - `variant="online"`
- `__init__(...)`
  - load yaml config and checkpoint
  - build selected MixFormerV2 network variant
  - move network to device and set eval mode
  - build preprocessor once
- `initialize_template(...)`
  - build fixed initial template
  - initialize `adaptive_template` from the same template
  - initialize `best_templates` as an empty bounded bank
  - for online mode, store queue size/update metadata but do not auto-update inside `match(...)`
- `match(...)`
  - crop around `SearchCandidate`
  - run network with `init_template`, `best_templates`, and current `adaptive_template` according to the selected variant
  - return `MatchEvidence`
  - use online `conf_score` as primary `match_score` when available
  - otherwise derive score from prediction distribution/box confidence metadata
- `extract_template(...)`
  - create an online template candidate from an approved box
  - set `quality_score` from the accepted match score or score passed through metadata
- `update_templates(...)`
  - implement the bounded latest-good template bank here, not in `match(...)`
  - update `adaptive_template` from the newest approved template
  - only run after `BigTrack` sets `allow_template_update=True`

Implementation risk:

- This is the most policy-sensitive integration because original online MixFormer decides when to learn. That must be removed from the matcher hot path.

## Required Foundation Work

1. Add shared box helpers.
   - `center_size_to_box(center, size) -> Box`
   - `box_to_center_size(box) -> (Point, Size)`
   - `clip_box(box, image_shape, margin=0) -> Box`
   - `map_crop_box_back(pred_box_cxcywh, crop_center, search_size, resize_factor) -> Box`

2. Add shared OSTrack-style crop helper.
   - Equivalent to `sample_target(...)`.
   - Return crop image, resize factor, attention mask, original crop box, and `is_clipped`.
   - Keep this under `BigTracker/matcher_models/_crop.py` so wrappers do not import tracker training packages for crop math.

3. Add PyTorch helper.
   - Device selection.
   - Checkpoint loading.
   - `torch.no_grad()` inference wrapper.
   - Clear error if `torch`, `cv2`, or checkpoint files are missing.
   - Use it from matcher `__init__`, never from `initialize_template`.

4. Add package isolation strategy.
   - Do not load all external repos as top-level `lib`.
   - Preferred: copy or vendor only required runtime modules under unique package names, for example `BigTracker/external/ostrack_src`, `BigTracker/external/litetrack_src`, `BigTracker/external/mixformer2_src`, `BigTracker/external/nanotrack_src`.
   - Alternative: create a separate process per matcher backend. Use only if package isolation becomes too expensive.

5. Add matcher configs.
   - Each matcher should have a frozen dataclass config with:
     - source root
     - config yaml
     - checkpoint path
     - device
     - template/search factors
     - template/search sizes
     - score normalization settings
     - max template queue size
     - template selection mode for matching

6. Add tests before model-heavy integration.
   - Box conversion tests.
   - Crop/map-back tests using synthetic images.
   - Fake network tests for each matcher adapter.
   - One `SimpleBigTrack + fake matcher` integration test proving `SearchCandidate` drives matching.

## Implementation Order

### Phase 1: Matcher Infrastructure

- [x] ✅ Create `_boxes.py`, `_crop.py`, and `_torch.py`.
- [x] ✅ Add unit tests for crop and map-back math.
- [x] ✅ Add a tiny fake tensor network test harness so matcher APIs can be tested without real checkpoints.
- [x] ✅ Decide package isolation strategy before importing any external repo code.

Acceptance:

- Crop/map-back tests pass.
- No external tracker repo is imported by tests unless explicitly requested.

### Phase 2: NanoTrack Matcher

- [x] ✅ Create `BigTracker/matcher_models/nanotrack.py`.
- [x] ✅ Add `NanoTrackMatcherConfig`.
- [x] ✅ Load NanoTrack model and checkpoint once in `NanoTrackMatcherModel.__init__`.
- [x] ✅ Convert `initialize_template()` from `NanoTracker.init(...)`.
- [x] ✅ Convert `match()` from `NanoTracker.track(...)`, driven by `SearchCandidate`.
- [x] ✅ Implement `extract_template()` and `update_templates()` for `init_template`, latest-good bank, and `adaptive_template`.
- [x] ✅ Return full `MatchEvidence` with `best_score`, penalty/window metadata, and clipping metadata.
- [x] ✅ Add fake-model tests.
- [x] ✅ Add optional real checkpoint smoke test behind an environment flag.

Why first:

- Smallest runtime loop.
- Has an explicit `best_score`.
- Its source tracker does not own online template policy, which makes it a clean first wrapper for BigTracker-owned template history.

### Phase 3: OSTrack Matcher

- [x] ✅ Create `BigTracker/matcher_models/ostrack.py`.
- [x] ✅ Add `OSTrackMatcherConfig`.
- [x] ✅ Load yaml config and checkpoint once in `OSTrackMatcherModel.__init__`.
- [x] ✅ Build and store initial processed template plus `box_mask_z`.
- [x] ✅ Convert `track()` into `match()` using `SearchCandidate`.
- [x] ✅ Derive confidence from score-map peak statistics.
- [x] ✅ Implement `extract_template()` and `update_templates()` for latest-good and adaptive templates.
- [x] ✅ Add fake-network tests for `score_map -> MatchEvidence`.
- [x] ✅ Add optional real checkpoint smoke test behind an environment flag.

Why second:

- Strong baseline.
- Its flow is the template for LiteTrack.
- It exposes the crop/search/mapping issues clearly.

### Phase 4: LiteTrack Matcher

- [ ] Create `BigTracker/matcher_models/litetrack.py`.
- [ ] Add `LiteTrackMatcherConfig`.
- [ ] Load yaml config and checkpoint once in `LiteTrackMatcherModel.__init__`.
- [ ] Reuse OSTrack-style crop and map-back helpers.
- [ ] Build initial `z_feat` with `forward_z(...)`.
- [ ] Convert `track_center()` into `match()`.
- [ ] Return `MatchEvidence` from center-head output.
- [ ] Implement approved template extraction as recomputing `z_feat`.
- [ ] Implement latest-good bank and adaptive-template selection.
- [ ] Add fake-network tests.
- [ ] Add optional real checkpoint smoke test behind an environment flag.

Why third:

- Very close to OSTrack after foundation exists.
- Main difference is cached template feature rather than raw template tensor inference.

### Phase 5: MixFormerV2 Matcher

- [ ] Create `BigTracker/matcher_models/mixformer_v2.py`.
- [ ] Add `MixFormerV2MatcherConfig` with `variant`.
- [ ] Load yaml config and checkpoint once in `MixFormerV2MatcherModel.__init__`.
- [ ] Implement offline variant first.
- [ ] Implement online variant second, with all template queue updates moved to `update_templates()`.
- [ ] Use `conf_score` as `match_score` for online mode.
- [ ] Define offline score derivation if no score head is available.
- [ ] Add tests proving `match()` never mutates template queues.
- [ ] Add tests proving `update_templates()` changes `adaptive_template` only after an approved `TemplateCandidate`.

Why last:

- Online MixFormer currently mixes matching and template update policy.
- It needs the strongest separation from BigTrack lifecycle rules.

## Integration With BigTrack Policies

Use `SimpleBigTrack` only for first smoke tests. It blindly accepts matcher evidence and never updates templates.

After matchers work, add a real policy tracker, for example:

```text
BigTracker/big_trackers/score_gated.py
```

Responsibilities:

- accept if `match_score`, `identity_score`, `ambiguity_score`, and `scale_score` pass thresholds
- set `UNCERTAIN` on weak but usable evidence
- set `OCCLUDED` or `LOST` after repeated rejects
- set `allow_template_update=True` only when evidence is clean
- create wider recovery candidates when uncertainty grows

Do not put those rules into `NanoTrackMatcherModel`, `OSTrackMatcherModel`, `LiteTrackMatcherModel`, or `MixFormerV2MatcherModel`.

## Done Definition

- All four matcher models implement `MatcherModel`.
- All four matcher models load model/config/checkpoint once in `__init__`, not in `initialize_template`.
- Each matcher can initialize, match one candidate, and return `MatchEvidence`.
- `match()` does not mutate lifecycle state.
- `match()` does not approve template updates.
- `MatcherState.init_template` is never overwritten.
- Every matcher supports `init_template`, `best_templates` as a latest-good bank, and `adaptive_template`.
- Template updates only happen through `extract_template()` and `update_templates()` after `BigTrack` approval.
- Synthetic tests pass without checkpoints.
- Real-model smoke tests are optional and skipped unless checkpoint paths are configured.
