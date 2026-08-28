# Types

This package defines immutable boundary data shared by policies, predictors, matchers, tools, and callers.

- `common.py`: `Box`, `Point`, `Size`, `ImageLike`, structural `FrameLike`, internal `TrackerMode`, and public `OutputStatus`.
- `predictor.py`: predictor state plus initialize/predict/update request and output dataclasses.
- `matcher.py`: template/matcher state plus initialize/template/update/match request and output dataclasses.
- `big_track.py`: BigTrack initialization, composed state, and update input/output dataclasses.
- `__init__.py`: the supported import surface for these types.

Coordinates use frame `xywh` externally. Dataclasses are frozen, but objects stored inside `image`, `template`, and `metadata` may still be mutable.

When adding a field, prefer a default for compatibility, update all constructors and restoration tests, export new types here and at root when public, and update [`docs/architecture.md`](../../docs/architecture.md). Model-specific experimental state belongs in metadata until it needs a stable cross-component contract.
