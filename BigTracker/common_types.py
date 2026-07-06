from __future__ import annotations

from typing import Any, Mapping, Tuple


# Bounding box in frame coordinates: x, y, width, height.
Box = Tuple[float, float, float, float]

# 2D point in frame coordinates.
Point = Tuple[float, float]

# Width and height pair.
Size = Tuple[float, float]

# Backend-specific image/frame object, such as a numpy array or tensor.
Frame = Any

# Backend-specific visual template object.
Template = Any

# Backend-specific cached features keyed by stable feature names.
FeatureMap = Mapping[str, Any]
