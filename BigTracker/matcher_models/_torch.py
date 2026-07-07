from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional


def require_torch() -> Any:
    """Import torch only when a torch-backed matcher is constructed."""

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("This matcher requires torch") from error
    return torch


def resolve_device(device: Optional[str] = None) -> Any:
    """Return a torch device, preferring CUDA only when requested or available."""

    torch = require_torch()
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(checkpoint_path: str, map_location: Any = "cpu") -> Any:
    """Load a torch checkpoint with a clear missing-file error."""

    torch = require_torch()
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    return torch.load(str(path), map_location=map_location)


def move_to_device(value: Any, device: Any) -> Any:
    """Move a torch-like value to a device when it supports `.to(...)`."""

    if hasattr(value, "to"):
        return value.to(device)
    return value


@contextmanager
def inference_context() -> Iterator[None]:
    """Run torch inference without gradients."""

    torch = require_torch()
    with torch.no_grad():
        yield

