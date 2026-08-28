# Vendored NanoTrack Runtime

`__init__.py` is the facade: it loads the vendored global config and builds `ModelBuilder`.

The nested `nanotrack/` tree contains `core/config.py` and cross-correlation, model assembly/losses, MobileNetV3/AlexNet backbones, BAN head variants, and adjustment necks. The BigTracker wrapper performs tracker-loop cropping, windowing, penalties, decoding, clipping, and template-bank policy outside this tree.

Torch mode requires Torch plus a YAML config/checkpoint. ONNX mode is implemented by the wrapper and requires `onnxruntime` and split backbone/head assets. See [`matcher_models/nanotrack.py`](../../matcher_models/nanotrack.py) and [`docs/matchers.md`](../../../docs/matchers.md).
