# Matchers

## Shared Contract and Ownership

A matcher owns visual preprocessing, templates, search crops, backend inference, box decoding, and matcher diagnostics. It returns evidence as parallel boxes and scores. It does not own acceptance, lifecycle, recovery, or permission to learn a template.

The operation sequence is:

1. Matcher construction loads/builds the backend and reusable inference helpers.
2. `initialize_template()` creates or restores object-specific state.
3. `match()` evaluates each requested center without changing the template bank.
4. If BigTrack approves learning, `extract_template()` creates a model-specific object.
5. `update_templates()` commits it through the shared bounded bank.

All built-in implementations store runtime state in `_state`, which `BaseBigTrack` currently reads when composing `BigTrackState`.

## Template Bank

`_templates.update_template_bank()` appends a clamped `TemplateState`, keeps only the newest `max_templates` entries, and chooses the highest score as `adaptive_template`; the newest wins equal-score ties. A zero-sized bank clears history and activates `init_template`.

The bank is “bounded recent approved templates,” not an all-time top-k collection. An older high score eventually expires. `init_template` never enters this eviction window and remains the identity fallback.

Built-in matcher `extract_template()` methods currently return score `1.0`. A caller or alternative matcher can return another score; `BaseBigTrack` forwards the extractor's score to the bank.

## Shared Geometry and Runtime Helpers

- `_boxes.py` converts between xywh and center/size, clips boxes to image bounds/margins, and maps crop-local center boxes back to frame coordinates.
- `_crop.py` creates constant-padded crops and attention masks. `sample_target()` uses a square side of `ceil(sqrt(width*height) * factor)` and optionally resizes through OpenCV.
- `_torch.py` lazily imports Torch, resolves explicit or CUDA/CPU device choice, loads checkpoints, and provides an inference-mode context.

## FFT Matcher

`FftMatcherModel` is the dependency-light baseline. It converts images to normalized grayscale, crops a square template, stores its FFT spectrum, and cross-correlates a centered template canvas against each search crop.

The response peak supplies translation. Peak z-score becomes `match_score`; the ratio of the second peak to the main peak becomes ambiguity metadata; distance from the search center becomes localization metadata. The matched box keeps the template's target size.

The active template is `adaptive_template` or `init_template`. `template_area_factor`, `search_area_factor`, `min_crop_size`, bank size, and peak exclusion radius are active. `uncertain_search_area_factor` and `recovery_search_area_factor` are presently unused because matcher input does not include lifecycle mode.

## NanoTrack Matcher

`NanoTrackMatcherModel` supports a Torch backend and split ONNX runtime. Its constructor resolves configuration, loads the backend, switches to evaluation mode, and precomputes the anchor-free point grid and Hann window.

Template extraction computes NanoTrack's context crop, padding value, resized exemplar, and encoded feature state. Before matching, the selected template feature is activated in the backend. For each candidate, the adapter builds an instance crop, obtains classification and location outputs, converts them into point-relative boxes, applies scale/aspect penalty and window influence, selects the best index, smooths size using `lr`, clips to minimum/image size, and returns the decoded box and classification score.

Diagnostics include best index, penalty, penalized score, resize scale, localization, ambiguity, scale score, crop box, and clipping. The ONNX path can use distinct template/search backbone sessions plus a head session; provider selection supports CPU or available CUDA providers.

## OSTrack Matcher

`OSTrackMatcherModel` loads a YAML config, vendored network, checkpoint, device, preprocessing, output Hann window, and optional candidate-elimination template mask. Templates store the processed template tensor and its box mask.

Each search candidate uses an OSTrack-style crop and attention mask. The backend returns score/size/offset maps; the response is multiplied by the output window, the box head decodes predictions, predictions are averaged, mapped to frame coordinates, and clipped with `clip_margin`.

The peak response supplies confidence and response statistics supply localization/ambiguity diagnostics. This wrapper uses one active adaptive template.

## LiteTrack Matcher

`LiteTrackMatcherModel` shares OSTrack crop/mapping structure but caches encoded template features through the LiteTrack template path. Match inference combines cached features with each processed search crop, uses the center head to decode a box, maps/clips it, and reports peak-based score diagnostics.

Template updates recompute and store template features, so they can be materially more expensive than merely retaining a crop. The active adaptive template's encoded features drive inference.

## MixFormerV2 Matcher

`MixFormerV2MatcherModel` supports `online` and `offline` variants. It preserves the fixed `init_template` separately from the active adaptive template and passes both to backend inference. Search predictions are averaged, mapped, and clipped.

When `pred_scores` exist, they provide match confidence and may be converted from logits. Without a score head, the wrapper uses configured fallback confidence and prediction agreement statistics. Diagnostics record score source, localization/ambiguity, scale agreement, both template source frames, crop geometry, and clipping.

Unlike the upstream online tracker, matching does not autonomously update templates. Only the shared approved update path changes the active template.

## Runtime Dependencies and Assets

The base project dependencies are NumPy and OpenCV. Actual neural backends may additionally require:

- Torch and torchvision.
- `timm` for OSTrack, LiteTrack, and MixFormerV2 model layers.
- PyYAML and `easydict` for YAML configuration trees.
- `onnxruntime` for NanoTrack ONNX mode.
- Compatible model YAML files and checkpoints.

The current `pyproject.toml` `torch` extra lists only `torch`; it is not a complete neural-backend environment. Checkpoint/model assets live outside the package, commonly under ignored `ignores/Models/...` paths in a developer checkout. Do not assume those paths exist in an installed wheel.

Real-model smoke tests are opt-in and document their expected local assets in [Testing and Tools](testing-and-tools.md#optional-real-checkpoint-tests).

## Vendored Runtime Code

`BigTracker/thirdparty` contains uniquely namespaced runtime portions of NanoTrack, OSTrack, LiteTrack, and MixFormerV2. Wrapper imports go through each vendor's top-level facade:

- NanoTrack facade loads its global YAML config and builds `ModelBuilder`; its runtime tree contains config, cross-correlation, backbones, necks, BAN heads, losses, and model assembly.
- OSTrack facade deep-copies default config and builds the network; its runtime tree contains config, ViT/CE backbone, heads, attention/layer utilities, token helpers, and general tensor/box utilities.
- LiteTrack facade deep-copies config and builds LiteTrack; its runtime tree contains CAE asynchronous ViT, cached-template model, center head/layers, configuration, box utilities, and Hann utilities.
- MixFormerV2 facade selects online/offline config and builder; its runtime tree contains transformer variants, score/box heads, positional encoding, config, and supporting utilities.

Treat these trees as vendored implementation code. Prefer changing the wrapper or facade. If an upstream runtime change is unavoidable, add focused tests, record provenance and compatibility in the vendor README, and verify every wrapper using the shared module.

## Adding a Matcher

Follow the exact checklist in [Adding a Matcher](development.md#adding-a-matcher). At minimum, implement every abstract operation, provide fake-backend tests without weights, preserve template-bank invariants, add optional real smoke coverage when appropriate, export the component, register it in the full-test UI, and update this document plus the matcher package README.
