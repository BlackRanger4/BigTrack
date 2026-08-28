# Development Guide

## Definition of Done for Any Component

A change is complete only when code, tests, tool integration, exports, and documentation agree. Before implementation, identify which layer owns the new behavior. Lifecycle decisions belong in a BigTrack policy, motion estimation belongs in a predictor, and visual inference/template representation belongs in a matcher.

For every public component:

1. Implement the correct abstract contract.
2. Add deterministic unit tests, including reset and failure/edge behavior.
3. Export it from its subpackage; export it from root only if root-level API policy calls for that.
4. Register it in relevant interactive tools.
5. Update package-local and central docs.
6. Run the full default suite and any relevant optional smoke test.

## Adding a BigTrack Policy

Create `BigTracker/big_trackers/<name>.py`. Usually subclass `BaseBigTrack` so initialization, restoration, matcher invocation, template update execution, timing, and debug snapshots remain consistent.

Implement:

- `make_candidates(state, prediction, frame)`: return ordered `SearchCandidate` objects with stable IDs, priors/reasons, and enough metadata to explain candidate generation.
- `decide(state, prediction, candidates, bboxes, scores)`: validate parallel results, select evidence, produce a complete `BigTrackDecision`, and avoid mutating child state.
- `apply_decision(state, prediction, decision, frame)`: call `_update_predictor()` exactly once, build public output, update counters/metadata/target size, and return a replaced state.

Design questions that must be answered in code and docs:

- Which score/geometry conditions accept a visual box?
- What does each internal mode output publicly?
- Can `RECOVERY` accept a match, and how strong must it be?
- What resets or increments each counter?
- When is the output box omitted?
- When can templates update, and which matcher diagnostics are required?
- How many candidates are allowed per mode and how does uncertainty affect them?

Tests should use fake predictor and matcher implementations. Cover initialization, each decision band, every mode transition in and out, no-result/mismatched-result behavior, template allow/deny, predictor accepted/rejected feedback, counter resets, `last_seen_frame`, state restoration, reset/close, and debug snapshot output.

Then:

- export the policy/config in `BigTracker/big_trackers/__init__.py`;
- decide explicitly whether to export at root `BigTracker/__init__.py` (current policies are not root exports);
- add a `ComponentSpec` to `POLICIES` in `tools/bigtrack_fulltest/app.py`;
- update `BigTracker/big_trackers/README.md`, `docs/architecture.md`, this guide if the process changed, and root README's component list.

If the policy needs ambiguity, scale, or clipping, first extend the base decision boundary to receive `MatcherMatchOutput` or its metadata, then update every policy and test. Do not reach into a concrete matcher from policy code.

## Adding a Matcher

Create `BigTracker/matcher_models/<name>.py` with a frozen config dataclass, a model-specific immutable template dataclass, an optional backend `Protocol`, and a `MatcherModel` implementation.

Constructor responsibilities:

- validate/resolve config;
- import optional frameworks lazily when practical;
- build/load the backend and checkpoint once;
- select device and evaluation mode;
- precompute static windows/grids/helpers;
- accept injected fake backend or factory if the model is heavy.

Method responsibilities:

- `initialize_template`: restore supplied `MatcherState` without re-extracting, or create the protected initial template and set it active.
- `extract_template`: build a new template only from the policy-approved frame/box and return meaningful metadata and a quality score.
- `update_templates`: call `update_template_bank()` unless the model has a documented compatible alternative; never overwrite `init_template`.
- `match`: require state, select active templates, return exactly one box and score per input center in order, put rich diagnostics in metadata, and never mutate the template bank.
- `reset`: clear object-specific state without unnecessarily rebuilding/unloading reusable model resources.
- `close`: release resources and leave the object unusable or reset as documented.

Geometry must use frame xywh at public boundaries. Record crop box, resize factor, source box/frame, active template source, and clipping where applicable. Normalize the primary score so policies can compare it consistently; document score derivation and limitations.

Testing layers:

1. Pure crop/box/decoder math with small synthetic arrays.
2. Fake backend contract tests: constructor loading once, init/restore, multiple centers, output order, metadata, template immutability, bank updates, reset/close, invalid output.
3. BigTrack integration with a fake/light predictor.
4. Optional real config/checkpoint smoke behind a unique environment flag.

Integration steps:

- export config/model/template from `BigTracker/matcher_models/__init__.py`;
- root-export only after an explicit API decision;
- register import, config, and factory in `MATCHERS` in `tools/bigtrack_fulltest/app.py`;
- add sensible local defaults only to `DEFAULT_OVERRIDES`, never hard-code them into library defaults;
- if vendoring runtime code, namespace it under `BigTracker/thirdparty/<name>`, add a narrow facade, document upstream provenance/dependencies, and avoid top-level module collisions;
- update `BigTracker/matcher_models/README.md`, `BigTracker/thirdparty/README.md` when applicable, `docs/matchers.md`, `docs/testing-and-tools.md`, and root README.

If new dependencies are required, add correct dependency/extra metadata to `pyproject.toml` and installation documentation. Never make default tests depend on downloading weights.

## Adding a Predictor

Create `BigTracker/predictor_models/<name>.py` with a frozen config dataclass and `PredictorModel` implementation.

Required behavior:

- `initialize` must accept a full `TrackerPredictionState`, including restored metadata.
- `predict` must use current frame time through `frame_dt()` and store updated timing metadata under a unique prefix.
- `update` must define both accepted measurement correction and rejected prediction behavior.
- return the actual retained state in `PredictorUpdateOutput.predictor_state`.
- clamp uncertainty and any configured velocity/acceleration through shared helpers.
- `reset`/`close` must be safe and tested.

Document the state model, equations/assumptions, metadata keys, score use, uncertainty meaning, and rejection behavior. Avoid claiming uncertainty is calibrated across model families.

Tests must cover time delta, stationary and moving prediction, high/low score acceptance when relevant, rejection, bounds, metadata restoration, reset, and the shared predictor contract. Add the model to the deterministic trajectory evaluation if its output is comparable.

Then update:

- `BigTracker/predictor_models/__init__.py`;
- `BigTracker/__init__.py` if following the current predictor export policy;
- `PREDICTORS` in `tools/bigtrack_fulltest/app.py`;
- `PREDICTORS` in `tools/predictor_trajectory_ui.py`;
- predictor construction/specs in `tests/test_predictor_trajectory_evaluation.py`;
- `BigTracker/predictor_models/README.md`, `docs/predictors.md`, `docs/testing-and-tools.md`, and root README.

Regenerate the trajectory report after behavior changes. Review metric differences rather than blindly accepting new numbers.

## Adding Shared Types or Changing Contracts

Types belong in the appropriate file under `BigTracker/types`, then must be re-exported from `BigTracker/types/__init__.py`. Public types also need consideration in `BigTracker/__init__.py`.

Contract changes have broad impact. Search all constructors and field reads with `rg`, update fake objects and tools, add migration notes, test state restoration, and update architecture diagrams. Adding required dataclass fields without defaults is a breaking change. Open-ended metadata is suitable for experimental/model-specific values; promote a value to a field only when multiple components require a stable typed contract.

## Adding or Changing Tools

Keep calculation/source/runner logic separate from GUI widgets so it remains testable. A new selectable component needs a registry entry and a config dataclass the UI can inspect. A new log field must pass through `_json_value()` safely and should not serialize full tensors or image arrays.

Update `docs/testing-and-tools.md` with launch command, inputs, controls, outputs, state mutations, config defaults, and any new environment variables. Add a focused test for non-visual logic. GUI smoke execution should not replace deterministic unit tests.

## Keeping Documentation Current

Use this checklist in every feature PR:

- [ ] Code docstrings describe local behavior without promising unimplemented features.
- [ ] Package-local README lists the new/changed file and responsibility.
- [ ] Central architecture/matcher/predictor document matches actual control flow and configuration.
- [ ] Development steps list every new export, registry, test, asset, and dependency location.
- [ ] Testing/tools document lists new tests, commands, flags, and UI registry changes.
- [ ] Root README still has correct imports and a working minimal example.
- [ ] Known limitations are removed only after tests prove the feature now exists.
- [ ] Generated benchmark tables are regenerated when underlying model behavior/config/scenarios change.
- [ ] Links resolve and renamed/deleted files have no remaining references (`rg` is useful here).
- [ ] Version and installation examples change together when publishing a release.

Do not create temporary roadmap documents that duplicate current docs. Put unfinished work under a clearly marked “Known boundaries” or “Future work” section in the most relevant maintained file, then update that section in the same change that implements it.

## Recommended Verification Sequence

```powershell
python -m compileall BigTracker tests tools
python -m unittest discover -s tests -v
python -m pytest
python tests\test_predictor_trajectory_evaluation.py
```

Run only relevant real-checkpoint smoke tests after unit tests pass. Finally launch affected interactive tools against a small local source and confirm construction, selection, display, reset, and close behavior.
