# Matcher Template Lifecycle

This note explains how template extraction and template updates are implemented in:

- `BigTracker/matcher_models/nanotrack.py`
- `BigTracker/matcher_models/ostrack.py`
- `BigTracker/matcher_models/litetrack.py`
- `BigTracker/matcher_models/mixformerv2.py`

The important design rule is: matcher models know how to build and store visual templates, but BigTrack decides when a new template is trusted enough to enter the template history.

## Shared BigTrack Flow

Template lifecycle is driven by `BaseBigTrack.update()`:

1. `predictor.predict(...)` predicts the next target state.
2. `make_candidates(...)` creates one or more `SearchCandidate` objects.
3. `matcher.match(...)` returns `MatchEvidence` for each candidate.
4. `decide(...)` accepts or rejects the visual evidence.
5. `apply_decision(...)` updates prediction/output/lifecycle state.
6. Only if `decision.allow_template_update` is true:
   - `matcher.extract_template(...)` builds a new `TemplateCandidate` from the accepted box.
   - `matcher.update_templates(...)` commits that template into `MatcherState`.

So a matcher never updates its own template history during `match()`. `match()` only reads `MatcherState` and returns evidence.

Current `SimpleBigTrack` always sets `allow_template_update=False`, so it initializes templates but does not refresh them. The future BigTrack policy phase must turn template updates on under controlled score/quality rules.

## Shared MatcherState Shape

Every matcher returns this structure from `initialize_template(...)`:

```python
MatcherState(
    init_template=template,
    best_templates=(),
    adaptive_template=template,
    cached_features={...},
)
```

The meaning is the same for all four trackers:

- `init_template`: fixed identity anchor from the first ROI. It is never replaced by `update_templates()`.
- `best_templates`: bounded bank of latest approved templates. The current wrappers append approved templates and keep only the last `max_best_templates`.
- `adaptive_template`: newest approved online template. At initialization it points to `init_template`; after approved updates it points to the newest template candidate.

## Shared Update Rule

All four wrappers implement the same `update_templates(...)` logic:

1. Append `template.template` to `state.best_templates`.
2. Read `config.max_best_templates`.
3. If `max_best_templates == 0`, clear the bank.
4. If the bank is too long, keep only the newest templates.
5. Return a replaced `MatcherState` with:
   - unchanged `init_template`
   - updated `best_templates`
   - `adaptive_template=template.template`

That gives us the three-template-history structure we wanted:

- initialization template
- latest-good template bank
- active online/adaptive template

## NanoTrack

File: `BigTracker/matcher_models/nanotrack.py`

Template class: `NanoTrackTemplate`

Stored template payload:

- `feature_state`: encoded NanoTrack template feature, usually backend `zf`
- `pad_value`: constant padding value, currently `0`
- `target_size`
- `source_frame_idx`
- `source_box`
- `crop_box`
- `crop_resize_factor`
- `was_clipped`
- metadata with matcher name, exemplar size, and context amount

### Initialization

`initialize_template(...)` calls `_build_template(...)`.

`_build_template(...)`:

1. Converts target center/size to source box.
2. Uses a constant zero padding value.
3. Computes NanoTrack exemplar crop size from target size and `context_amount`.
4. Uses `nanotrack_subwindow(...)` with `exemplar_size`.
5. Calls `_encode_template(...)`.
6. Stores a snapshot of the encoded feature in `NanoTrackTemplate.feature_state`.

The model/backend is already loaded in `__init__`; initialization does not load weights.

### Matching

`match(...)`:

1. Selects `adaptive_template` if available, otherwise `init_template`.
2. Activates the stored feature with `_activate_template(...)`.
3. Builds a search crop around `SearchCandidate.search_center`.
4. Runs backend `track(...)`.
5. Decodes NanoTrack score/location outputs into a frame box.
6. Returns `MatchEvidence`.

NanoTrack is special because the source model keeps an active template feature internally. The wrapper avoids hidden lifecycle ownership by restoring the selected `NanoTrackTemplate.feature_state` before every search.

### Approved Extraction And Update

`extract_template(...)` only runs when BigTrack asks. It builds a new `NanoTrackTemplate` from the accepted target position/size.

`update_templates(...)` preserves `init_template`, appends the approved template to `best_templates`, and makes it the new `adaptive_template`.

## OSTrack

File: `BigTracker/matcher_models/ostrack.py`

Template class: `OSTrackTemplate`

Stored template payload:

- `template_tensor`: preprocessed template crop tensor
- `box_mask_z`: optional candidate-elimination mask for CE-based OSTrack variants
- `target_size`
- `source_frame_idx`
- `source_box`
- `crop_box`
- `crop_resize_factor`
- `was_clipped`
- metadata with matcher name, template factor, and template size

### Initialization

`initialize_template(...)` calls `_build_template(...)`.

`_build_template(...)`:

1. Converts target center/size to source box.
2. Uses `sample_target(...)` with `template_factor` and `template_size`.
3. Backend preprocesses the crop plus attention mask into a tensor/NestedTensor.
4. Builds `box_mask_z` through `backend.build_box_mask(...)`.
5. Stores the tensor and mask in `OSTrackTemplate`.

The OSTrack network/config/checkpoint are loaded once in `__init__`.

### Matching

`match(...)`:

1. Selects `adaptive_template` if available, otherwise `init_template`.
2. Builds a search crop around `SearchCandidate.search_center`.
3. Backend preprocesses the search crop.
4. Calls backend `forward(template.template_tensor, search_tensor, template.box_mask_z)`.
5. Applies the output window to `score_map`.
6. Decodes the center-head outputs.
7. Maps crop-local prediction back into frame coordinates.
8. Returns `MatchEvidence`.

### Approved Extraction And Update

`extract_template(...)` builds a fresh `OSTrackTemplate` for the accepted box. `update_templates(...)` then makes that template the active adaptive template and appends it to the bounded latest-good bank.

## LiteTrack

File: `BigTracker/matcher_models/litetrack.py`

Template class: `LiteTrackTemplate`

Stored template payload:

- `template_features`: encoded LiteTrack template features from `forward_z(...)`
- `template_tensor`: snapshot of the preprocessed template tensor
- `target_size`
- `source_frame_idx`
- `source_box`
- `crop_box`
- `crop_resize_factor`
- `was_clipped`
- metadata with matcher name, template factor, and template size

### Initialization

`initialize_template(...)` calls `_build_template(...)`.

`_build_template(...)`:

1. Converts target center/size to source box.
2. Uses `sample_target(...)` with `template_factor` and `template_size`.
3. Backend preprocesses crop plus attention mask.
4. Calls `backend.encode_template(...)`.
5. Snapshots both encoded features and the template tensor.
6. Stores them in `LiteTrackTemplate`.

LiteTrack differs from OSTrack because the template is encoded once into `template_features`; matching uses those features directly.

### Matching

`match(...)`:

1. Selects `adaptive_template` if available, otherwise `init_template`.
2. Builds a search crop from the BigTrack candidate.
3. Backend preprocesses the search crop.
4. Calls `backend.forward(template.template_features, search_tensor)`.
5. Applies the output window to `score_map`.
6. Decodes center-head maps and maps the box back to frame coordinates.
7. Returns `MatchEvidence`.

### Approved Extraction And Update

Approved extraction builds and encodes a new LiteTrack template immediately. The update step preserves the fixed initial template, appends the approved encoded template to `best_templates`, and makes it the new adaptive template.

## MixFormerV2

File: `BigTracker/matcher_models/mixformerv2.py`

Template class: `MixFormerV2Template`

Stored template payload:

- `template_tensor`: preprocessed template crop tensor
- `target_size`
- `source_frame_idx`
- `source_box`
- `crop_box`
- `crop_resize_factor`
- `was_clipped`
- metadata with matcher name, variant, template factor, and template size

### Initialization

`initialize_template(...)` calls `_build_template(...)`.

`_build_template(...)`:

1. Converts target center/size to source box.
2. Uses `sample_target(...)` with `template_factor` and `template_size`.
3. Backend preprocesses the crop.
4. Snapshots the tensor into `MixFormerV2Template`.

The MixFormerV2 network/config/checkpoint are loaded once in `__init__`.

### Matching

MixFormerV2 uses two template inputs during forward:

- fixed init template
- online/adaptive template

`match(...)`:

1. Selects the fixed template with `_select_init_template(...)`.
2. Selects the online template with `_select_online_template(...)`.
3. Builds a search crop around `SearchCandidate.search_center`.
4. Backend preprocesses the search crop.
5. Calls `backend.forward(init_template.template_tensor, online_template.template_tensor, search_tensor)`.
6. Decodes `pred_boxes` into frame coordinates.
7. Uses `pred_scores` when available, otherwise estimates confidence from box agreement.
8. Returns `MatchEvidence`.

This is the clearest place where our own template structure maps to the original tracker design: `init_template` stays fixed as the identity anchor, while `adaptive_template` acts as the online template.

### Approved Extraction And Update

Approved extraction builds a new `MixFormerV2Template`. The update step does not touch `init_template`; it only updates the latest-good bank and replaces `adaptive_template`.

## Important Differences Between The Four

NanoTrack:

- Stores encoded feature state and activates it before each search.
- Uses constant zero padding in our wrapper.
- Supports torch and ONNX backends.

OSTrack:

- Stores a preprocessed template tensor and optional CE mask.
- Does not separately encode template features before match.

LiteTrack:

- Stores encoded template features from `forward_z(...)`.
- Matching uses `template_features`, not the raw template tensor.

MixFormerV2:

- Always passes both fixed initial template and online adaptive template.
- Our `init_template` and `adaptive_template` map directly to those two model inputs.

## What Future BigTrack Policy Must Add

The wrappers are ready for template updates, but current `SimpleBigTrack` disables them. The next BigTrack policy should decide when `allow_template_update=True`, using signals such as:

- `match_score` above a safe threshold
- low `ambiguity_score`
- strong `localization_score`
- acceptable `scale_score`
- not clipped, or clipped only under controlled conditions
- enough frame gap since last template update
- current tracker mode is stable tracking, not recovery

That policy should be implemented in BigTrack, not inside the matcher wrappers. The matcher wrappers should continue to only build, store, select, and apply model-specific template payloads.

## Original Source Tracker Behavior

This section compares our wrapper behavior against the original tracker source files under `ignores/Trackers`.

The source trackers are written as complete standalone trackers. They usually own:

- model loading
- initial template extraction
- current tracking box/state
- search crop center
- output box smoothing/clipping
- sometimes online template update policy

Our wrappers deliberately split that apart:

- model loading stays in matcher `__init__`
- template building stays in matcher methods
- current box, prediction, acceptance, and template update permission move to BigTrack/Predictor
- `match()` does not mutate tracker lifecycle state

### Original NanoTrack

Source file:

- `ignores/Trackers/NanoTrack/nanotrack/tracker/nano_tracker.py`

Original `NanoTracker.init(...)`:

1. Converts the initial box to `self.center_pos` and `self.size`.
2. Computes exemplar crop size using `CONTEXT_AMOUNT`.
3. Computes `self.channel_average` from the full image.
4. Crops template with `get_subwindow(...)`.
5. Calls `self.model.template(z_crop)`.

Original `NanoTracker.track(...)`:

1. Uses `self.center_pos` and `self.size` as the search center/size source.
2. Computes search crop size using the current target size and `INSTANCE_SIZE / EXEMPLAR_SIZE`.
3. Crops search image with the saved `self.channel_average`.
4. Calls `self.model.track(x_crop)`.
5. Converts class/location outputs into scores and boxes.
6. Applies scale/aspect penalty, Hann window penalty, and learning-rate smoothing.
7. Clips the center/size to the image boundary.
8. Mutates `self.center_pos` and `self.size`.
9. Returns the new box and best score.

What our NanoTrack wrapper does differently:

- It stores encoded template state in `NanoTrackTemplate.feature_state`.
- It restores that feature state before each search instead of letting the source tracker own one hidden active template forever.
- It uses `SearchCandidate.search_center` and `SearchCandidate.predicted_target_size` instead of `self.center_pos` and `self.size`.
- It uses constant zero padding instead of the source tracker's full-frame `channel_average`.
- It preserves NanoTrack decode behavior: point grid, score conversion, scale/aspect penalty, Hann window, smoothing, and min-size clipping.
- It does not mutate the current object position inside `match()`.
- A new NanoTrack template is only built when BigTrack calls `extract_template(...)`.

### Original OSTrack

Source file:

- `ignores/Trackers/OSTrack/lib/test/tracker/ostrack.py`

Original `OSTrack.initialize(...)`:

1. Crops the template from `info["init_bbox"]` with `sample_target(...)`.
2. Preprocesses the crop and attention mask.
3. Stores the processed template as `self.z_dict1`.
4. Builds `self.box_mask_z` for candidate elimination variants when `CE_LOC` is enabled.
5. Stores the current box in `self.state`.

Original `OSTrack.track(...)`:

1. Crops the search region from `self.state`.
2. Preprocesses the search crop.
3. Calls `network.forward(template=self.z_dict1.tensors, search=x_dict.tensors, ce_template_mask=self.box_mask_z)`.
4. Multiplies `score_map` by `output_window`.
5. Decodes center-head maps with `box_head.cal_bbox(...)`.
6. Averages predicted boxes.
7. Maps the crop-local box back around the previous `self.state` center.
8. Clips the box and overwrites `self.state`.

What our OSTrack wrapper does differently:

- It stores `template_tensor` and `box_mask_z` in `OSTrackTemplate`.
- It selects `adaptive_template` or `init_template` from `MatcherState`, not from `self.z_dict1`.
- It crops search from the BigTrack candidate, not from a private `self.state`.
- It preserves the source search/decode math: `sample_target`, output window, `cal_bbox`, mean box, map-back, and margin clipping.
- It returns `MatchEvidence`; BigTrack decides whether that box becomes state.
- OSTrack source has no online template update in this tracker class; our wrapper adds the capability through BigTrack-approved `update_templates(...)`.

### Original LiteTrack

Source file:

- `ignores/Trackers/LiteTrack/lib/test/tracker/litetrack.py`

Original `LiteTrack.initialize(...)`:

1. Crops the template from `info["init_bbox"]` with `sample_target(...)`.
2. Preprocesses the crop and attention mask.
3. Builds a template bounding box in template-crop coordinates.
4. Converts that box to `xyxy`.
5. Calls `network.forward_z(template.tensors, template_bb=template_bbox)`.
6. Stores the encoded feature as `self.z_feat`.
7. Stores the current box in `self.state`.

Original `LiteTrack.track_center(...)`:

1. Crops search from `self.state`.
2. Preprocesses the search crop.
3. Calls `network(template_feats=self.z_feat, search=x_dict.tensors)`.
4. Multiplies score map by `output_window`.
5. Decodes center-head maps with `box_head.cal_bbox(...)`.
6. Averages predicted boxes.
7. Maps the crop-local box back around the previous `self.state` center.
8. Clips the box and overwrites `self.state`.

What our LiteTrack wrapper does differently:

- It stores encoded features in `LiteTrackTemplate.template_features`.
- It keeps the preprocessed template tensor as debug/state payload, but matching uses `template_features`, like the source tracker.
- It uses BigTrack candidates for search center/size instead of hidden `self.state`.
- It keeps the source model behavior: template `forward_z`, search forward with encoded features, Hann response, center-head decode, mean box, map-back, and clipping.
- It can refresh `template_features` later through BigTrack-approved extraction, while the source class keeps only one `self.z_feat` from initialization.

### Original MixFormerV2

Source files:

- `ignores/Trackers/MixFormerV2/lib/test/tracker/mixformer2_vit.py`
- `ignores/Trackers/MixFormerV2/lib/test/tracker/mixformer2_vit_online.py`

Original offline `MixFormer.initialize(...)`:

1. Crops template from `info["init_bbox"]`.
2. Preprocesses the template crop.
3. Stores `self.template`.
4. Sets `self.online_template = template`.
5. Stores current box in `self.state`.

Original offline `MixFormer.track(...)`:

1. Crops search from `self.state`.
2. Preprocesses search crop.
3. Calls `network(self.template, self.online_template, search, softmax=True)`.
4. Averages predicted boxes.
5. Maps the crop-local result around previous `self.state`.
6. Clips and overwrites `self.state`.

Original online `MixFormerOnline.initialize(...)`:

1. Crops and preprocesses the initial template.
2. Stores `self.template` as the fixed template.
3. Stores `self.online_template` as the online template.
4. For `online_size > 1`, calls `network.set_online(self.template, self.online_template)`.
5. Initializes `self.online_max_template`, score bookkeeping, update interval, and current `self.state`.

Original online `MixFormerOnline.track(...)`:

1. Crops search from `self.state`.
2. Runs `network(self.template, self.online_template, search, softmax=True, run_score_head=True)`.
3. Uses `pred_scores.sigmoid()` as confidence.
4. Decodes and clips the output box, then overwrites `self.state`.
5. If score is strong enough, crops a candidate online template from the current prediction.
6. On update interval, replaces or rotates `self.online_template`.
7. Calls `network.set_online(...)` when the online bank has more than one template.

What our MixFormerV2 wrapper does differently:

- It maps source `self.template` to `MatcherState.init_template`.
- It maps source `self.online_template` to `MatcherState.adaptive_template`.
- It keeps both template inputs in every forward call.
- It crops search from BigTrack candidate state, not from private `self.state`.
- It does not run the source score-threshold/update-interval online policy inside `match()`.
- Instead, BigTrack will decide when a prediction is approved, then call `extract_template(...)` and `update_templates(...)`.
- Current wrapper uses one active `adaptive_template`; the `best_templates` bank is available for future BigTrack policy if we want to emulate MixFormerV2's multi-template online bank more closely.

## Why The Wrapper Design Is Different

The original trackers are single-object tracker loops. They combine motion state, visual state, confidence policy, and template update policy in one class.

BigTracker separates those responsibilities:

- Predictor owns motion prediction.
- Matcher owns model-specific visual evidence and template payloads.
- BigTrack owns accept/reject policy, lifecycle mode, and whether a new template is trusted.

That is why our matchers look slightly more indirect than the source code. The original source uses `self.state` as both the search center and the accepted output state. Our code uses `SearchCandidate` for search and waits for `BigTrackDecision` before committing anything.

## Clean Template Creation And Update Summary

This is the practical template rule for the four current matchers:

```text
NanoTrack + LiteTrack:
    store template features
    feature-space template blending is possible later

OSTrack + MixFormerV2:
    store normalized image-crop tensors
    feature-space template blending is not available unless we expose/cache model internals
```

Mathematically, an image tensor can be averaged, but that is not the same as averaging learned template features. Averaging image crops usually creates a blurred/ghost template and is not what these tracker source implementations do. For OSTrack and MixFormerV2, the correct update primitive is to crop a new approved image patch and replace/add it as a template tensor, not to average feature tensors.

### NanoTrack Template

Original source:

- `ignores/Trackers/NanoTrack/nanotrack/tracker/nano_tracker.py`

Original template creation:

1. Initial box becomes `center_pos` and `size`.
2. Source computes context crop size from `CONTEXT_AMOUNT`.
3. Source computes `channel_average` from the full image.
4. Source crops an exemplar image with `get_subwindow(...)`.
5. Source calls `model.template(z_crop)`.
6. The model stores encoded template feature internally as `zf`.

Original template update behavior:

- The original NanoTrack tracker does not update the template after initialization.
- During `track(...)`, it updates only `center_pos` and `size`.
- The same encoded template feature is used for all future searches.

Our wrapper:

- Stores the encoded feature in `NanoTrackTemplate.feature_state`.
- Restores that feature before every match.
- Stores `pad_value=0` and uses it for template/search crop padding.
- Can create a new encoded feature only when BigTrack calls `extract_template(...)`.
- Because this is feature state, future BigTrack policy could implement weighted feature averaging if the backend feature shapes are compatible.

### LiteTrack Template

Original source:

- `ignores/Trackers/LiteTrack/lib/test/tracker/litetrack.py`

Original template creation:

1. Initial box is cropped with `sample_target(...)`.
2. Crop is resized to template size and padded if needed.
3. Preprocessor converts it into a normalized tensor.
4. Source builds a template target box in crop coordinates.
5. Source calls `network.forward_z(template.tensors, template_bb=template_bbox)`.
6. Encoded template feature is stored as `self.z_feat`.

Original template update behavior:

- The original LiteTrack test tracker does not refresh `self.z_feat`.
- It uses the initialization template feature for all future searches.
- It updates only `self.state` after each accepted prediction inside `track_center(...)`.

Our wrapper:

- Stores encoded features in `LiteTrackTemplate.template_features`.
- Uses those features directly in search forward.
- Also stores `template_tensor`, but matching is driven by `template_features`.
- Because the saved active template is a feature tensor, future BigTrack policy could do weighted feature averaging or feature-bank selection.

### OSTrack Template

Original source:

- `ignores/Trackers/OSTrack/lib/test/tracker/ostrack.py`

Original template creation:

1. Initial box is cropped with `sample_target(...)`.
2. Crop is resized to template size and padded if needed.
3. Preprocessor converts image crop plus mask into a normalized `NestedTensor`.
4. Source stores the processed template as `self.z_dict1`.
5. If candidate elimination is enabled, source builds `self.box_mask_z`.

Original template update behavior:

- The original OSTrack test tracker does not update `self.z_dict1`.
- It uses the initial normalized template image tensor for all future searches.
- It updates only `self.state` after decoding each search result.

Our wrapper:

- Stores `OSTrackTemplate.template_tensor`, which is still a normalized 3-channel image crop tensor.
- Stores `OSTrackTemplate.box_mask_z`.
- Feature extraction happens inside `network.forward(template=..., search=...)`.
- Since we do not store exposed backbone features, weighted feature averaging is not available here.
- Approved template updates should crop a new image tensor and replace/adapt `adaptive_template`.

### MixFormerV2 Template

Original sources:

- `ignores/Trackers/MixFormerV2/lib/test/tracker/mixformer2_vit.py`
- `ignores/Trackers/MixFormerV2/lib/test/tracker/mixformer2_vit_online.py`

Original offline template creation:

1. Initial box is cropped with `sample_target(...)`.
2. Crop is resized to template size and padded if needed.
3. Preprocessor converts it into a normalized image tensor.
4. Source stores it as `self.template`.
5. Source sets `self.online_template = template`.

Original offline template update behavior:

- The offline source tracker does not update templates.
- `self.template` and `self.online_template` stay equal to the initialization template.
- It updates only `self.state`.

Original online template creation:

1. Initial crop is preprocessed into `self.template`.
2. `self.online_template` starts as the same initial tensor.
3. Source initializes `self.online_max_template`, score tracking, `online_size`, and `update_interval`.
4. For multi-template online mode, source can call `network.set_online(self.template, self.online_template)`.

Original online template update behavior:

1. Each frame, source scores the prediction with `pred_scores`.
2. If score is better than threshold and current best, it crops a new image patch from current `self.state`.
3. That crop becomes `self.online_max_template`.
4. On update interval, source replaces or rotates entries in `self.online_template`.
5. It does not average features; it replaces/adds normalized image template tensors.

Our wrapper:

- Stores `MixFormerV2Template.template_tensor`, a normalized 3-channel image crop tensor.
- Maps fixed source `self.template` to `MatcherState.init_template`.
- Maps source `self.online_template` to `MatcherState.adaptive_template`.
- Feature extraction and fusion happen inside the model forward call.
- Weighted feature averaging is not available without changing the backend to expose/cache internal features.
- The right BigTrack update operation is to approve a box, crop a new template image tensor, then replace `adaptive_template` or manage `best_templates`.

## Visual Data-Flow Shapes

These diagrams show what is stored as the template and what is recomputed during search.

### BigTracker Wrapper Flow

```text
initialize_template(frame, ROI)
    |
    |--> crop template patch
    |--> tracker-specific preprocessing / encoding
    |
    |--> MatcherState
         |
         |--> init_template      fixed identity anchor
         |--> best_templates     latest approved template bank
         |--> adaptive_template  active online template


update(frame)
    |
    |--> Predictor
    |       |
    |       |--> SearchCandidate(search_center, predicted_target_size)
    |
    |--> Matcher.match(candidate, MatcherState)
    |       |
    |       |--> crop search patch from candidate
    |       |--> compare search against selected template
    |       |--> MatchEvidence(box, score, quality signals)
    |
    |--> BigTrack.decide(...)
            |
            |--> reject:
            |       |--> do not update template
            |
            |--> accept, allow_template_update=False:
            |       |--> update motion/output state only
            |
            |--> accept, allow_template_update=True:
                    |
                    |--> extract_template(approved box)
                    |--> update_templates(...)
```

### NanoTrack

```text
Template path
-------------
frame + ROI
    |
    |--> exemplar crop image [3, 127, 127]
    |       |
    |       |--> if crop crosses image boundary: pad with 0
    |--> NanoTrack model.template(...)
    |--> encoded feature zf
    |
    |--> NanoTrackTemplate.feature_state

Search path
-----------
frame + SearchCandidate
    |
    |--> search crop image [3, 255, 255]
    |       |
    |       |--> if crop crosses image boundary: pad with 0
    |--> restore selected feature_state as backend zf
    |--> NanoTrack model.track(search crop)
    |--> cls + loc
    |--> score/penalty/window decode
    |
    |--> MatchEvidence

Template update style
---------------------
approved box
    |
    |--> crop new exemplar
    |--> encode new zf
    |--> replace adaptive_template
    |--> append to best_templates

Storage type: feature tensor/state
Feature averaging later: possible if shapes/backend semantics match
Original git behavior: init zf once, no template update
```

### LiteTrack

```text
Template path
-------------
frame + ROI
    |
    |--> template crop image [3, template_size, template_size]
    |--> normalize image tensor
    |--> build template bbox in crop coordinates
    |--> network.forward_z(template_tensor, template_bbox)
    |--> encoded template_features
    |
    |--> LiteTrackTemplate.template_features

Search path
-----------
frame + SearchCandidate
    |
    |--> search crop image [3, search_size, search_size]
    |--> normalize image tensor
    |--> network(template_feats, search_tensor)
    |--> score_map + size_map + offset_map
    |--> Hann window + center-head decode
    |
    |--> MatchEvidence

Template update style
---------------------
approved box
    |
    |--> crop new template image
    |--> normalize
    |--> forward_z(...)
    |--> replace adaptive_template
    |--> append to best_templates

Storage type: feature tensor
Feature averaging later: possible
Original git behavior: init z_feat once, no template update
```

### OSTrack

```text
Template path
-------------
frame + ROI
    |
    |--> template crop image [3, template_size, template_size]
    |--> normalize image tensor + attention mask
    |--> optional CE box mask
    |
    |--> OSTrackTemplate.template_tensor
    |--> OSTrackTemplate.box_mask_z

Search path
-----------
frame + SearchCandidate
    |
    |--> search crop image [3, search_size, search_size]
    |--> normalize image tensor + attention mask
    |--> network.forward(template_tensor, search_tensor, box_mask_z)
    |       |
    |       |--> internal feature extraction/fusion
    |       |--> score_map + size_map + offset_map
    |
    |--> Hann window + center-head decode
    |--> MatchEvidence

Template update style
---------------------
approved box
    |
    |--> crop new template image
    |--> normalize image tensor
    |--> build new box_mask_z
    |--> replace adaptive_template
    |--> append to best_templates

Storage type: normalized image-crop tensor
Feature averaging later: not available unless backend exposes cached features
Original git behavior: init z_dict1 once, no template update
```

### MixFormerV2

```text
Template path
-------------
frame + ROI
    |
    |--> template crop image [3, template_size, template_size]
    |--> normalize image tensor
    |
    |--> MixFormerV2Template.template_tensor

MatcherState mapping
--------------------
init_template
    |
    |--> fixed source self.template

adaptive_template
    |
    |--> source self.online_template

Search path
-----------
frame + SearchCandidate
    |
    |--> search crop image [3, search_size, search_size]
    |--> normalize image tensor
    |
    |--> network(init_template_tensor,
    |            online_template_tensor,
    |            search_tensor,
    |            ...)
            |
            |--> internal feature extraction/fusion
            |--> pred_boxes + pred_scores
    |
    |--> MatchEvidence

Template update style
---------------------
approved box
    |
    |--> crop new template image
    |--> normalize image tensor
    |--> replace adaptive_template
    |--> append to best_templates

Storage type: normalized image-crop tensor
Feature averaging later: not available unless backend exposes cached features
Original git behavior:
    offline: init template and online_template once, no update
    online: replace/rotate online image-template tensors by score and interval
```
