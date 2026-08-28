# BigTracker

BigTracker is a Python library for composable single-object tracking experiments. It separates motion prediction, visual matching, and lifecycle policy so each part can be implemented, tested, and replaced independently.

The maintained documentation starts at [docs/README.md](docs/README.md). Read [Architecture](docs/architecture.md) for the complete frame flow and [Development Guide](docs/development.md) before adding a tracker policy, matcher, predictor, test, or tool integration.

## Install

For local development:

```powershell
git clone https://github.com/BlackRanger4/BigTrack.git
cd BigTrack
python -m pip install -e ".[dev]"
```

For a normal local install:

```powershell
python -m pip install .
```

The base package requires Python 3.9+, NumPy, and OpenCV. Neural matchers require additional frameworks/configs/checkpoints; the current `torch` extra is not a complete environment for every vendored backend. See [matcher dependencies and assets](docs/matchers.md#runtime-dependencies-and-assets).

## Architecture

```text
BigTrack policy
  owns one Predictor
  owns one Matcher
  owns composed state and public output
  decides accept/reject/mode/template permission

Predictor
  predicts target-center motion and uncertainty
  corrects or rejects its prediction after policy feedback

Matcher
  owns visual crops, inference, decoding, and templates
  returns boxes, scores, and diagnostics
```

The key rule is: matchers may build templates, but BigTrack policy decides when a template update is allowed.

## Components

Policies:

- `SimpleBigTrack`: accepts the first matcher result and never updates templates.
- `ScoreGatedBigTrack`: accepts good scores, conditionally accepts weak geometrically consistent matches, falls back to prediction on rejection, and advances uncertainty/recovery/lost counters.

Predictors:

- `KalmanPredictorModel`
- `AdaptiveKalmanPredictorModel`
- `AlphaBetaPredictorModel`
- `HistoryPredictorModel`
- `ConstantAccelerationKalmanPredictorModel`

Matchers:

- `FftMatcherModel`
- `NanoTrackMatcherModel`
- `OSTrackMatcherModel`
- `LiteTrackMatcherModel`
- `MixFormerV2MatcherModel`

Concrete policies and matchers are imported from their subpackages. Predictors and shared types are also available from the root package.

## Minimal Example

```python
from dataclasses import dataclass

import numpy as np

from BigTracker.big_trackers import ScoreGatedBigTrack
from BigTracker.matcher_models import FftMatcherModel
from BigTracker.predictor_models import AdaptiveKalmanPredictorModel
from BigTracker.types import BigTrackInitializeInput, BigTrackUpdateInput


@dataclass
class Frame:
    image: np.ndarray
    idx: int
    timestamp: float


tracker = ScoreGatedBigTrack(
    predictor=AdaptiveKalmanPredictorModel(),
    matcher=FftMatcherModel(),
)

first = Frame(np.zeros((360, 640, 3), dtype=np.uint8), idx=0, timestamp=0.0)
tracker.initialize(BigTrackInitializeInput(frame=first, box=(100.0, 80.0, 40.0, 60.0)))

second = Frame(np.zeros((360, 640, 3), dtype=np.uint8), idx=1, timestamp=1.0)
output = tracker.update(BigTrackUpdateInput(frame=second))
print(output.box, output.status, output.confidence)
```

The request dataclasses are required; `initialize(frame, box)` and `update(frame)` are not the current API.

## Tests and Tools

```powershell
python -m pytest
# or, without pytest:
python -m unittest discover -s tests -v
```

Interactive tools:

```powershell
python tools\run_bigtrack_fulltest.py
python tools\predictor_trajectory_ui.py
```

See [Testing and Tools](docs/testing-and-tools.md) for test ownership, optional real-checkpoint flags, tool registries, controls, and packaging checks.

## Package Guides

- [BigTracker package](BigTracker/README.md)
- [BigTrack policies](BigTracker/big_trackers/README.md)
- [Predictors](BigTracker/predictor_models/README.md)
- [Matchers](BigTracker/matcher_models/README.md)
- [Shared types](BigTracker/types/README.md)
- [Vendored runtimes](BigTracker/thirdparty/README.md)
