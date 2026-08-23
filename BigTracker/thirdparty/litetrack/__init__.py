from __future__ import annotations

import copy
from pathlib import Path
from typing import Any


def load_litetrack_config(config_path: Path) -> Any:
    """Load a local LiteTrack config object."""

    from BigTracker.thirdparty.litetrack.lib.config.litetrack.config import cfg as default_cfg
    from BigTracker.thirdparty.litetrack.lib.config.litetrack.config import update_config_from_file

    config = copy.deepcopy(default_cfg)
    update_config_from_file(str(config_path), base_cfg=config)
    return config


def build_litetrack_network(config: Any, training: bool = False) -> Any:
    """Build a local LiteTrack network from a loaded config."""

    from BigTracker.thirdparty.litetrack.lib.models.litetrack.litetrack import build_LiteTrack

    return build_LiteTrack(config, training=training)
