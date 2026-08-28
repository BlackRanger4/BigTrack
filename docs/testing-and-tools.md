# Testing and Tools

## Test Commands

Preferred development command after installing the dev extra:

```powershell
python -m pip install -e ".[dev]"
python -m pytest
```

The suite is also written with `unittest`, so it can run without pytest:

```powershell
python -m unittest discover -s tests -v
```

On the 2026-08-28 documentation audit, the active interpreter did not have pytest, NumPy, or OpenCV. Unittest discovery ran 62 tests successfully with 26 dependency-based skips. This is an environment snapshot, not a permanent expected count.

## Test Module Map

- `test_bigtrack_decision_helpers.py`: score normalization/bands, invalid scores, predictor uncertainty, geometry agreement, and candidate context.
- `test_bigtrack_score_gated.py`: good/weak/bad decisions, prediction fallback, lifecycle counters, template interval, uncertain recovery, and debug timings/post-update state.
- `test_bigtrack_state_restore.py`: simple and score-gated restoration plus required restore inputs.
- `test_bigtrack_template_score.py`: verifies that the extractor-provided template score reaches `update_templates()`.
- `test_template_bank.py`: bounded-window selection, newest tie-breaking, and zero-bank fallback.
- `test_matcher_utils.py`: box conversion/clipping/map-back and padded/resized crops.
- `test_nanotrack_matcher.py`: fake backend construction timing, template shape/padding, multi-center output ordering, approved updates, split Torch/ONNX behavior, and optional real smoke.
- `test_ostrack_matcher.py`: fake backend lifecycle, templates, multi-center output, updates, and optional real smoke.
- `test_litetrack_matcher.py`: fake backend lifecycle, cached features, matching, updates, and optional real smoke.
- `test_mixformerv2_matcher.py`: fake backend lifecycle, fixed/adaptive templates, match immutability, approved updates, and optional real smoke.
- `test_predictor_kalman_clamp.py`: time delta, constant-velocity prediction, measurement correction, and rejection uncertainty.
- `test_predictor_adaptive_kalman.py`: score-aware noise, velocity clamping, reject damping/growth, and clean-accept decay.
- `test_optional_predictors.py`: alpha-beta limits, history bounds, acceleration Kalman behavior, and common predictor contract.
- `test_predictor_trajectory_evaluation.py`: deterministic multi-scenario predictor regression and report generation.
- `test_predictor_trajectory_ui.py`: correct future-horizon cache semantics for the UI.

## What New Tests Belong Where

- Pure policy math belongs in a decision-helper test.
- Policy transitions use fake predictor/matcher objects and do not require images or checkpoints.
- Predictor behavior gets a focused model test and should also be added to the shared-contract and trajectory evaluation when comparable.
- Matcher geometry belongs in utility tests.
- A matcher adapter must use a fake backend for deterministic, fast contract tests.
- Real checkpoint loading is a separate opt-in smoke test; it must not make the default suite download or require weights.
- Tool calculation logic should be separated from GUI creation enough to test without opening a window.

## Optional Real-Checkpoint Tests

Set only the matcher you want to test. Each test also skips if its expected assets are absent.

```powershell
$env:BIGTRACK_REAL_DEVICE = "cuda"  # optional; omit for auto/CPU behavior
$env:BIGTRACK_RUN_NANOTRACK_REAL_SMOKE = "1"
python -m unittest tests.test_nanotrack_matcher
```

Equivalent enable flags are:

- `BIGTRACK_RUN_NANOTRACK_REAL_SMOKE`
- `BIGTRACK_RUN_OSTRACK_REAL_SMOKE`
- `BIGTRACK_RUN_LITETRACK_REAL_SMOKE`
- `BIGTRACK_RUN_MIXFORMERV2_REAL_SMOKE`

Read the corresponding test before running it because asset filenames are explicit and may differ from tool defaults.

## Full-Test UI

Launch:

```powershell
python tools\run_bigtrack_fulltest.py
```

`tools/bigtrack_fulltest/app.py` is a Tkinter setup application. It owns three component registries:

- `PREDICTORS`: all five predictors and their config dataclasses.
- `MATCHERS`: FFT plus four neural wrappers and configs.
- `POLICIES`: score-gated and simple policies.

The config tab is generated from dataclass fields. `DEFAULT_OVERRIDES` supplies checkout-specific model paths and NanoTrack ONNX defaults. The source tab accepts a video or an image folder, folder FPS, and optional JSONL logging.

`frame_source.py` defines the concrete `Frame`, video capture, sorted image-folder input, reset/close, and source factory.

`runner.py` provides:

- interactive ROI initialization and saved-state reinitialization;
- pause/continuous/single-frame controls;
- zoom and pan for main/debug views;
- output, candidate, prediction, matcher, history, and timing overlays;
- bounded timing statistics;
- optional JSONL state/event records with tensor/template-safe summaries;
- Windows timer-resolution adjustment for smoother stepping.

When adding a component, update the appropriate registry import/spec. If its configuration cannot be represented by simple dataclass fields parsed through `ast.literal_eval`, extend the UI builder and test that behavior.

## Predictor Trajectory UI

Launch:

```powershell
python tools\predictor_trajectory_ui.py
```

This Tkinter tool generates a seeded bounded trajectory with spring, swirl, damping, acceleration noise, observation noise, and bounce behavior. It precomputes each registered predictor at selected future horizons `m`, renders truth/observations/predictions, and reports per-point errors.

Its `PREDICTORS` tuple is separate from the full-test registry. A new comparable predictor must be added here and to `tests/test_predictor_trajectory_evaluation.py`; otherwise it will be available in tracking but absent from predictor comparison.

## Packaging Checks

The project uses setuptools package discovery for `BigTracker*` and excludes tests/ignored assets. Useful checks are:

```powershell
python -m compileall BigTracker tests tools
python -m pip wheel . --no-deps --no-build-isolation -w .pkg-check
```

Do not commit `.pkg-check`, build artifacts, checkpoints, logs, caches, or local videos. If a new runtime dependency is required for normal installed use, update `pyproject.toml` and installation docs rather than relying on a developer environment.
