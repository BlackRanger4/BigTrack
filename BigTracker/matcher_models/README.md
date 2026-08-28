# Matcher Models

Matchers turn image content around requested centers into boxes and scores. They own visual templates but never decide lifecycle or whether learning is safe.

- `fft.py`: normalized grayscale FFT cross-correlation baseline.
- `nanotrack.py`: Torch/split-ONNX NanoTrack adapter and decoder.
- `ostrack.py`: OSTrack template/search transformer adapter.
- `litetrack.py`: LiteTrack cached-template-feature adapter.
- `mixformerv2.py`: online/offline MixFormerV2 adapter using fixed and adaptive templates.
- `_boxes.py`: public/crop coordinate conversion, clipping, and map-back.
- `_crop.py`: padded centered and OSTrack-style square crops with masks.
- `_templates.py`: bounded approved template bank and active-template selection.
- `_torch.py`: lazy Torch import, device/checkpoint helpers, and inference context.
- `__init__.py`: concrete config/model/template exports.

Every heavy adapter accepts an injected backend/factory so default unit tests can run without model weights. Construction owns backend loading; initialization owns only per-object templates. `match()` must not mutate template history.

See [`docs/matchers.md`](../../docs/matchers.md) and [`docs/development.md#adding-a-matcher`](../../docs/development.md#adding-a-matcher).
