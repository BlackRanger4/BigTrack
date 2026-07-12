# BigTracker

BigTracker is a Python library for single-object tracking experiments. It separates motion prediction, visual matching, and lifecycle policy so matcher models can return evidence while `BigTrack` decides whether to accept, reject, recover, lose, or update templates.

The package is prepared for reuse from other repositories through `pyproject.toml`.

## Install

Install directly from GitHub:

```powershell
python -m pip install "bigtracker @ git+https://github.com/BlackRanger4/BigTrack.git@v0.2.0"
```

Or add it to `requirements.txt`:

```text
bigtracker @ git+https://github.com/BlackRanger4/BigTrack.git@v0.2.0
```

For local development, clone the repository first:

```powershell
git clone https://github.com/BlackRanger4/BigTrack.git
cd BigTrack
```

Then install from the checkout:

```powershell
python -m pip install .
```

For editable local development:

```powershell
python -m pip install -e .
```

The base package depends on:

- `numpy`
- `opencv-python`

Torch-backed matcher wrappers import `torch` lazily. Install the optional extra when you need those backends:

```powershell
python -m pip install "bigtracker[torch] @ git+https://github.com/BlackRanger4/BigTrack.git@v0.2.0"
```

## Architecture

The core design is intentionally split:

```text
BigTrack
  owns one Predictor
  owns one Matcher
  owns one BigTrackState
  decides lifecycle, public output, and template update permission

Predictor
  owns motion prediction
  reads tracker state
  predicts the next target center, size, velocity, and uncertainty

Matcher
  owns model-specific template extraction
  owns model-specific search crop rules
  owns visual matching
  returns MatchEvidence only
```

Important ownership rule:

```text
Matcher can build template objects.
BigTrack decides when template building/updating is allowed.
```

That keeps visual model wrappers free of accept/reject, lost/recovery, and template-learning policy.

## Main Components

Package layout:

```text
BigTracker/
  big_track.py
  big_trackers/
    base.py
    simple.py
    score_gated.py
  matcher.py
  matcher_models/
    fft.py
    nanotrack.py
    ostrack.py
    litetrack.py
    mixformerv2.py
  predictor.py
  predictor_models/
    kalman.py
    adaptive_kalman.py
    alpha_beta.py
    history.py
    kalman_accel.py
  state.py
  types.py
```

Tracker policies:

- `SimpleBigTrack`: basic integration policy. It accepts the first matcher result and does not update templates.
- `ScoreGatedBigTrack`: current production-oriented policy. It accepts strong visual evidence, accepts weak evidence only when it agrees with prediction geometry, emits predictor boxes during uncertain/occluded states, enters recovery/lost by counters, and updates templates only after clean accepted matches.

Predictor models:

- `KalmanPredictorModel`
- `AdaptiveKalmanPredictorModel`
- `AlphaBetaPredictorModel`
- `HistoryPredictorModel`
- `ConstantAccelerationKalmanPredictorModel`

Matcher models:

- `FftMatcherModel`
- `NanoTrackMatcherModel`
- `OSTrackMatcherModel`
- `LiteTrackMatcherModel`
- `MixFormerV2MatcherModel`

The deep matcher wrappers expect external source/config/checkpoint paths through their config objects. Model weights and external tracker repositories are not packaged into this wheel.

## Template Lifecycle

Every matcher uses the same `MatcherState` shape:

```text
init_template
  first trusted identity template, never overwritten

best_templates
  bounded bank of approved clean templates

adaptive_template
  active online template selected from approved templates
```

The shared update flow is:

1. `predictor.predict(...)` predicts the next target state.
2. `make_candidates(...)` creates `SearchCandidate` objects.
3. `matcher.match(...)` returns `MatchEvidence`.
4. `BigTrack.decide(...)` accepts or rejects the evidence.
5. `BigTrack.apply_decision(...)` updates state and output.
6. Only if `decision.allow_template_update` is true, `matcher.extract_template(...)` and `matcher.update_templates(...)` run.

`ScoreGatedBigTrack` allows template updates only when:

```text
accepted visual match
and match_score >= th_good
and template_update_interval elapsed
and not clipped unless template_allow_clipped=True
```

Weak matches can keep tracking alive, but they do not update matcher templates.

## Basic Usage

The frame object can be any object with these fields:

```text
image
idx
timestamp
```

Example with the built-in FFT matcher and adaptive Kalman predictor:

```python
from dataclasses import dataclass

import numpy as np

from BigTracker import (
    AdaptiveKalmanPredictorModel,
    FftMatcherModel,
    ScoreGatedBigTrack,
    TrackingOutput,
)


@dataclass
class Frame:
    image: np.ndarray
    idx: int
    timestamp: float


predictor = AdaptiveKalmanPredictorModel()
matcher = FftMatcherModel()
tracker = ScoreGatedBigTrack(predictor=predictor, matcher=matcher)

first_frame = Frame(image=np.zeros((360, 640, 3), dtype=np.uint8), idx=0, timestamp=0.0)
initial_box = (100.0, 80.0, 40.0, 60.0)  # x, y, width, height

tracker.initialize(first_frame, initial_box)

next_frame = Frame(image=np.zeros((360, 640, 3), dtype=np.uint8), idx=1, timestamp=1.0)
output: TrackingOutput = tracker.update(next_frame)

print(output.box, output.status, output.confidence)
```

## Tests

Run the test suite:

```powershell
python -m pytest
```

Current validation after packaging:

```text
60 passed, 4 skipped
```

Build a wheel for a packaging sanity check:

```powershell
python -m pip wheel . --no-deps --no-build-isolation -w .pkg-check
```

## Predictor Evaluation

The synthetic predictor trajectory evaluation lives in:

```text
tests/test_predictor_trajectory_evaluation.py
```

It compares every predictor across deterministic trajectory families with noise, outliers, occlusion windows, frame-boundary bounces, and size changes. The current report in `.chatgpt/PREDICTOR_TRAJECTORY_REPORT.md` found `adaptive_kalman` best overall on the synthetic aggregate, but real video should drive final predictor choice and tuning.

Regenerate the report output:

```powershell
python tests\test_predictor_trajectory_evaluation.py
```

## Development Notes

Internal design notes are kept under `.chatgpt/`:

- `.chatgpt/BIG_TRACKER_STRUCTURE.md`
- `.chatgpt/MATCHER_TEMPLATE_LIFECYCLE.md`
- `.chatgpt/PREDICTOR_BIGTRACK_ROADMAP.md`
- `.chatgpt/PREDICTOR_TRAJECTORY_REPORT.md`
- `.chatgpt/TRACKER_MATCHER_INTEGRATION_DONE.md`

Future work from the roadmap:

- multi-candidate search
- richer recovery policy
- better public diagnostics
- optional C++/pybind11 packaging later if native code is added
