# BigTracker Package

This is the installable Python package. The main abstractions are `BigTrack`, `Predictor`, and `Matcher`; immutable request/state/result contracts live in `types/`; implementations live in `big_trackers/`, `predictor_models/`, and `matcher_models/`.

## Files

- `big_track.py`: abstract orchestration/lifecycle API.
- `predictor.py`: abstract motion API and `PredictorModel` marker base.
- `matcher.py`: abstract visual/template API and `MatcherModel` marker base.
- `__init__.py`: root public exports. It currently exports types and predictors, not concrete policies/matchers.

## Subpackages

- [types](types/README.md): coordinates, modes/statuses, and boundary dataclasses.
- [big_trackers](big_trackers/README.md): shared update flow and policies.
- [predictor_models](predictor_models/README.md): motion estimators.
- [matcher_models](matcher_models/README.md): visual adapters and shared matcher utilities.
- [thirdparty](thirdparty/README.md): vendored neural tracker runtime code.

The maintained documentation begins at [`docs/README.md`](../docs/README.md). Extension checklists are in [`docs/development.md`](../docs/development.md).

## Dependency Direction

```text
types <- abstract APIs <- implementations
  ^                         |
  +-------------------------+

matcher_models -> thirdparty (neural adapters only)
big_trackers -> predictor + matcher contracts
```

Keep `types` independent of concrete models. Keep lifecycle policy out of matchers and predictors.
