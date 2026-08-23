from __future__ import annotations

from pathlib import Path
from typing import Any


def load_mixformerv2_config(config_path: Path, variant: str) -> Any:
    """Load a local MixFormerV2 config object."""

    variant_name = str(variant).lower()
    if variant_name == "online":
        from BigTracker.thirdparty.mixformerv2.lib.config.mixformer2_vit_online.config import (
            update_new_config_from_file,
        )
        return update_new_config_from_file(str(config_path))
    if variant_name == "offline":
        from BigTracker.thirdparty.mixformerv2.lib.config.mixformer2_vit.config import (
            update_new_config_from_file as update_offline_config_from_file,
        )

        try:
            return update_offline_config_from_file(str(config_path))
        except ValueError:
            from BigTracker.thirdparty.mixformerv2.lib.config.mixformer2_vit_online.config import (
                update_new_config_from_file as update_online_config_from_file,
            )

            return update_online_config_from_file(str(config_path))
    raise ValueError(f"Unknown MixFormerV2 variant: {variant!r}")


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
