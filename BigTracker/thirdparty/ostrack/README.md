# Vendored OSTrack Runtime

`__init__.py` deep-copies the default configuration, applies a YAML file, and exposes the OSTrack network builder.

`lib/config/ostrack` defines the configuration tree. `lib/models/ostrack` contains token handling, base backbone, ViT and candidate-elimination ViT, and network assembly. `lib/models/layers` contains attention, relative position, patch embedding, normalization, and center/corner heads. `lib/utils` contains box, mask, tensor, distributed, and supporting research utilities.

The BigTracker adapter owns preprocessing, crop mapping, template state, checkpoint loading, confidence extraction, and policy-safe updates. Typical dependencies include Torch, torchvision, timm, PyYAML, and easydict. See [`matcher_models/ostrack.py`](../../matcher_models/ostrack.py).
