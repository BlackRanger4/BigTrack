from __future__ import annotations

import copy
from pathlib import Path
from typing import Any


def load_ostrack_config(config_path: Path | str) -> Any:
    """Load an OSTrack YAML into a fresh vendored config object."""

    from BigTracker.thirdparty.ostrack.lib.config.ostrack.config import cfg as default_cfg
    from BigTracker.thirdparty.ostrack.lib.config.ostrack.config import update_config_from_file

    loaded_cfg = copy.deepcopy(default_cfg)
    update_config_from_file(str(config_path), base_cfg=loaded_cfg)
    return loaded_cfg


def build_ostrack_network(config: Any, training: bool = False) -> Any:
    """Build the vendored OSTrack network."""

    from BigTracker.thirdparty.ostrack.lib.models.ostrack import build_ostrack

    return build_ostrack(config, training=training)
