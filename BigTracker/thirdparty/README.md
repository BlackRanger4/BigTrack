# Vendored Tracker Runtimes

This package contains runtime portions of external neural trackers under unique `BigTracker.thirdparty.*` namespaces. It prevents the top-level `lib`/module collisions common when several research repositories are placed on `sys.path` together.

- [NanoTrack](nanotrack/README.md): config, model builder, backbones, BAN heads, neck, cross-correlation, and losses.
- [OSTrack](ostrack/README.md): config, ViT/candidate-elimination model, prediction heads, attention layers, and utilities.
- [LiteTrack](litetrack/README.md): config, asynchronous CAE ViT, cached-template model, center head/layers, and utilities.
- [MixFormerV2](mixformerv2/README.md): online/offline config, transformer models, box/score heads, positional encoding, and utilities.

The top-level `__init__.py` is intentionally empty. Each vendor package exposes a narrow facade consumed by `matcher_models`.

Prefer changes in adapters over changes here. For unavoidable vendored edits, preserve namespace imports, document upstream source/version or snapshot, list compatibility changes in the vendor README, and add a wrapper-level regression test. These trees require optional packages beyond the base install; see [`docs/matchers.md#runtime-dependencies-and-assets`](../../docs/matchers.md#runtime-dependencies-and-assets).
