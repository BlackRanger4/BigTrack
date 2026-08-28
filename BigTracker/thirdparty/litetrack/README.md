# Vendored LiteTrack Runtime

`__init__.py` deep-copies the default config, applies YAML, and exposes the LiteTrack builder.

`lib/models/litetrack` contains the cached-template LiteTrack network and asynchronous CAE ViT backbone. `lib/models/layers` contains attention, positional encoding, patch embedding, normalization, and prediction heads. `lib/config` contains defaults/loading; `lib/test/utils/hann.py` supplies runtime Hann helpers; `lib/utils` contains box and general research utilities.

The BigTracker adapter owns crops, preprocessing, cached template objects, decoding/mapping, scoring, and approved template history. Typical dependencies include Torch, torchvision, timm, PyYAML, and easydict. See [`matcher_models/litetrack.py`](../../matcher_models/litetrack.py).
