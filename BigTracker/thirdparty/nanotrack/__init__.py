from __future__ import annotations

from pathlib import Path
from typing import Any


def load_nanotrack_config(config_path: Path | str) -> Any:
    """Load NanoTrack YAML into the vendored source config object."""

    from BigTracker.thirdparty.nanotrack.nanotrack.core.config import cfg

    cfg.merge_from_file(str(config_path))
    return cfg


def build_nanotrack_model() -> Any:
    """Build the vendored NanoTrack model."""

    from BigTracker.thirdparty.nanotrack.nanotrack.models.model_builder import ModelBuilder

    return ModelBuilder()
