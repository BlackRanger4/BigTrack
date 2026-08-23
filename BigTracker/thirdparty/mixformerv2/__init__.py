from __future__ import annotations

from pathlib import Path
from typing import Any


def load_mixformerv2_config(config_path: Path, variant: str) -> Any:
    """Load a local MixFormerV2 config object."""

    if str(variant).lower() == "online":
        from BigTracker.thirdparty.mixformerv2.lib.config.mixformer2_vit_online.config import (
            update_new_config_from_file,
        )
    elif str(variant).lower() == "offline":
        from BigTracker.thirdparty.mixformerv2.lib.config.mixformer2_vit.config import (
            update_new_config_from_file,
        )
    else:
        raise ValueError(f"Unknown MixFormerV2 variant: {variant!r}")

    return update_new_config_from_file(str(config_path))


def build_mixformerv2_network(config: Any, variant: str, training: bool = False) -> Any:
    """Build a local MixFormerV2 network from a loaded config."""

    if str(variant).lower() == "online":
        from BigTracker.thirdparty.mixformerv2.lib.models.mixformer2_vit.mixformer2_vit_online import (
            build_mixformer_vit_online,
        )

        return build_mixformer_vit_online(config, train=training)
    if str(variant).lower() == "offline":
        from BigTracker.thirdparty.mixformerv2.lib.models.mixformer2_vit.mixformer2_vit import (
            build_mixformer_vit,
        )

        return build_mixformer_vit(config, train=training)
    raise ValueError(f"Unknown MixFormerV2 variant: {variant!r}")
