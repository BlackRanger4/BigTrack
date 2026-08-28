# BigTracker Documentation

This directory is the maintained documentation hub for the repository. It describes the code that exists now, the contracts that extensions must follow, how to test changes, how to register them in the interactive tools, and which documentation must change with the code.

## Start Here

- [Architecture](architecture.md) explains the end-to-end frame flow, public contracts, state ownership, tracker policies, and every maintained package module.
- [Matchers](matchers.md) explains the FFT and neural matcher adapters, template lifecycle, backend loading, crop/box utilities, and vendored runtime code.
- [Predictors](predictors.md) explains the five motion models, their configuration, metadata, and accept/reject behavior.
- [Development Guide](development.md) gives step-by-step checklists for adding a BigTrack policy, matcher, predictor, test coverage, tool registration, exports, and documentation.
- [Testing and Tools](testing-and-tools.md) maps every test module and both interactive tools, including optional real-checkpoint smoke tests.
- [Predictor Trajectory Report](predictor-trajectory-report.md) preserves the reproducible synthetic comparison that was previously stored as an internal note.

Package-local guides are also available beside the code:

- [`BigTracker/README.md`](../BigTracker/README.md)
- [`BigTracker/types/README.md`](../BigTracker/types/README.md)
- [`BigTracker/big_trackers/README.md`](../BigTracker/big_trackers/README.md)
- [`BigTracker/predictor_models/README.md`](../BigTracker/predictor_models/README.md)
- [`BigTracker/matcher_models/README.md`](../BigTracker/matcher_models/README.md)
- [`BigTracker/thirdparty/README.md`](../BigTracker/thirdparty/README.md)
- [`tests/README.md`](../tests/README.md)
- [`tools/README.md`](../tools/README.md)

## Repository Layout

- `BigTracker/`: installable library and vendored runtime packages.
- `tests/`: deterministic unit, integration, optional checkpoint, and synthetic evaluation tests.
- `tools/`: full tracking UI/runner and predictor trajectory UI.
- `docs/`: maintained cross-cutting architecture and development documentation.
- `ignores/`: git-ignored local research repositories, checkpoints, configs, videos, and archives. Library code must not require these paths to exist unless the caller explicitly selects a local model asset.
- `logs/`: runtime output from tools; generated logs are not library source.
- `.vscode/`: developer editor settings.
- `.chatgpt/`: the former internal-note location; its five stale/duplicated documents were retired after consolidation.
- `pyproject.toml`: package metadata, base/optional dependencies, setuptools discovery, and pytest test path.
- `.gitignore`: excludes local assets and generated output.
- `README.md`: concise public entry point with a working API example.

README files are placed at maintained component boundaries and at each vendored tracker boundary. They are intentionally not added to `__pycache__` directories or every internal leaf of vendored upstream code; doing that would mix project documentation into implementation snapshots without adding a distinct ownership boundary.

## Current Implementation Snapshot

The central composition is:

```text
FrameLike + previous BigTrackState
        |
        v
Predictor.predict() -> TrackerPredictionState
        |
        v
BigTrack.make_candidates() -> SearchCandidate(s)
        |
        v
Matcher.match() -> parallel bboxes + scores
        |
        v
BigTrack.decide() -> BigTrackDecision
        |
        v
BigTrack.apply_decision() -> output + predictor correction + lifecycle state
        |
        +-- if allowed: Matcher.extract_template() -> update_templates()
```

Implemented policies are `SimpleBigTrack` and `ScoreGatedBigTrack`. Implemented predictors are basic Kalman, adaptive Kalman, alpha-beta, history, and constant-acceleration Kalman. Implemented matchers are FFT, NanoTrack, OSTrack, LiteTrack, and MixFormerV2.

## Known Boundaries

These are current facts, not future intentions:

- Predictors model the target center and velocity. They do not predict width or height; target size is retained by BigTrack/matcher state.
- Both policies currently create exactly one search candidate at the predicted center.
- `ScoreGatedBigTrack` cannot leave `RECOVERY` or `LOST` from matcher evidence; it rejects every match in those modes. A good match can recover from `UNCERTAIN` or `OCCLUDED` before recovery starts.
- `ScoreGatedBigTrackConfig.template_allow_clipped` exists but is not consulted. Matcher clipping information lives in `MatcherMatchOutput.metadata`, while the policy currently receives only boxes and scores.
- The FFT config contains uncertain/recovery search factors, but matching currently uses only `search_area_factor`.
- Concrete matcher and policy classes are exported from their subpackages, not from the root `BigTracker` package.
- Neural matcher dependencies and checkpoint files are not fully represented by the `torch` package extra. See [Matchers](matchers.md#runtime-dependencies-and-assets).

## Documentation Rule

Documentation is part of the definition of done. Any new or changed public component must update:

1. Its package-local `README.md`.
2. The relevant detailed document in this directory.
3. [Development Guide](development.md) if the extension process or registry changes.
4. [Testing and Tools](testing-and-tools.md) when tests, environment flags, tools, or registries change.
5. The root [README](../README.md) when installation, public usage, or the top-level component list changes.

The full checklist is in [Keeping Documentation Current](development.md#keeping-documentation-current).

## Legacy Notes

The five former `.chatgpt/*.md` files were reviewed. Their still-correct principles and benchmark data have been incorporated here, while stale names such as `state.py`, `MatchEvidence`, `TemplateCandidate`, `TrackingOutput`, and `tests/fulltest/main.py` were not carried forward. The original files were retired to leave one maintained source of truth.
