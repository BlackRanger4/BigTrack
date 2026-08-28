# BigTrack Full-Test Tool

- `app.py`: Tkinter setup UI, component registries/factories, reflected dataclass config fields, local default overrides, source/log selection, and runner construction.
- `frame_source.py`: common frame record and video/image-folder source implementations.
- `runner.py`: ROI initialization, frame stepping, state restore, main/debug rendering, zoom/pan, timing, controls, and safe JSONL state summaries.
- `__init__.py`: package marker.

The tool is a manual integration/visualization environment, not a replacement for unit tests. Default model paths are checkout conveniences under `ignores/`; callers can override every displayed config field. See [`docs/testing-and-tools.md#full-test-ui`](../../docs/testing-and-tools.md#full-test-ui) for behavior and [`docs/development.md#adding-or-changing-tools`](../../docs/development.md#adding-or-changing-tools) for extension rules.
