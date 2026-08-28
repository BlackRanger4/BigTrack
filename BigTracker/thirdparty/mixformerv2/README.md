# Vendored MixFormerV2 Runtime

`__init__.py` selects online/offline configuration and network builders. Offline config loading can fall back to the online config schema for compatible files.

`lib/config` contains both variant schemas. `lib/models/mixformer2_vit` contains offline and online transformer implementations plus box/score heads. `lib/models/mixformer_vit/pos_util.py` contains positional embedding helpers. `lib/utils` and the small `lib/train/admin` subset provide runtime/config support retained from upstream.

The BigTracker wrapper owns template/search crops, fixed versus adaptive templates, backend preprocessing, score interpretation, mapping/clipping, checkpoint compatibility handling, and approved template updates. Typical dependencies include Torch, torchvision, timm, PyYAML, and easydict. See [`matcher_models/mixformerv2.py`](../../matcher_models/mixformerv2.py).
