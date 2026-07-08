# Predictor Trajectory Evaluation Report

This report evaluates every BigTracker predictor with the deterministic synthetic
trajectory implemented in:

```text
tests/test_predictor_trajectory_evaluation.py
```

## Scenario

- Seed: `1337`
- Frames: `180`
- Frame size: `640x360`
- Motion includes:
  - smooth sinusoidal acceleration
  - sharp direction changes
  - random acceleration disturbances
  - frame-boundary bounces
  - target width and height changes
- Measurement input includes:
  - Gaussian center noise
  - Gaussian size noise
  - four deterministic outlier measurements
  - two structured occlusion windows
- Accepted measurements: `155`
- Rejected/occluded measurements: `24`

Every predictor receives the same truth trajectory, measurement noise, scores,
outliers, and rejected frames.

## Results

| Predictor | Position RMSE | Size RMSE | Box RMSE | Accept Pos RMSE | Reject Pos RMSE | Max Pos Err | Final Pos Err | Avg Uncertainty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `constant_accel_kalman` | 19.586 | 5.322 | 10.659 | 13.160 | 41.744 | 98.042 | 3.627 | 9.447 |
| `history` | 26.970 | 5.256 | 13.758 | 17.082 | 59.503 | 148.671 | 3.970 | 18.728 |
| `adaptive_kalman` | 32.376 | 3.553 | 16.468 | 17.193 | 76.868 | 182.257 | 3.517 | 6.659 |
| `alpha_beta` | 32.706 | 2.595 | 16.462 | 18.178 | 76.446 | 180.253 | 3.473 | 14.397 |
| `kalman` | 33.021 | 4.617 | 16.754 | 18.312 | 77.246 | 182.026 | 3.553 | 3.621 |

All errors are measured in pixels.

## Metric Meaning

- `Position RMSE`: center prediction error over every evaluated frame.
- `Size RMSE`: width/height prediction error over every evaluated frame.
- `Box RMSE`: combined `x, y, width, height` prediction error.
- `Accept Pos RMSE`: center prediction error when a measurement was available.
- `Reject Pos RMSE`: center prediction error during occlusion/rejected frames.
- `Max Pos Err`: worst center prediction error in the run.
- `Final Pos Err`: corrected state error on the final frame.
- `Avg Uncertainty`: average predictor uncertainty before accept/reject update.

## Interpretation

For this trajectory and these configs:

1. `constant_accel_kalman` has the best overall position and occlusion prediction.
2. `history` is second for position prediction but has higher uncertainty and larger worst-case drift.
3. `alpha_beta` has the best target-size prediction.
4. All predictors recover to a similar low final position error after measurements return.
5. The uncertainty values are not directly calibrated across model types, so they
   should not be compared as if they use the same probability scale.

This is a synthetic regression and stress test, not a final production ranking.
Real video evaluation should determine the selected predictor and tuned config.

## Run

Run as a unittest:

```powershell
python -m unittest tests.test_predictor_trajectory_evaluation
```

Print a regenerated markdown report:

```powershell
python tests\test_predictor_trajectory_evaluation.py
```

The fixed seed makes the report repeatable. Predictor behavior changes should be
visible as metric changes.
