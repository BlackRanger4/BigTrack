# Predictor Trajectory Evaluation Report

This report evaluates every BigTracker predictor across deterministic synthetic
trajectory families implemented in:

```text
tests/test_predictor_trajectory_evaluation.py
```

## Scenario

- Runs: `80`
- Seeds: `1337-1346`
- Motion scenarios: `8`
- Frames: `180`
- Frame size: `640x360`
- Motion includes:
  - mixed sinusoidal acceleration with sharp direction changes
  - constant velocity
  - constant acceleration
  - zig-zag target velocity changes
  - sine-wave motion
  - exponential direct return toward frame center
  - exponential circular/orbit return toward frame center
  - random maneuver changes
  - frame-boundary bounces and target width/height changes
- Measurement input includes:
  - Gaussian center noise
  - Gaussian size noise
  - four deterministic outlier measurements
  - two structured occlusion windows
- Accepted measurements per run: `155`
- Rejected/occluded measurements per run: `24`

Within each run, every predictor receives the same truth trajectory, measurement
noise, scores, outliers, and rejected frames. Each run creates fresh predictor
instances.

## Results

Each cell is `average (minimum-maximum)` across all scenario and seed runs.

| Predictor | Position RMSE | Size RMSE | Box RMSE | Accept Pos RMSE | Reject Pos RMSE | Max Pos Err | Final Pos Err | Avg Uncertainty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `adaptive_kalman` | 16.145 (6.576-32.200) | 3.848 (3.369-4.356) | 8.303 (3.576-16.390) | 10.739 (6.581-17.239) | 34.106 (5.299-76.246) | 80.887 (25.036-180.876) | 3.422 (0.528-10.219) | 6.654 (6.598-6.683) |
| `kalman` | 17.694 (8.561-35.019) | 5.071 (4.619-5.867) | 9.040 (4.587-17.722) | 12.666 (7.825-20.295) | 34.903 (5.288-80.806) | 91.111 (35.401-211.764) | 3.549 (0.441-10.440) | 3.621 (3.621-3.621) |
| `constant_accel_kalman` | 17.697 (9.342-32.281) | 6.070 (4.380-7.357) | 9.312 (5.249-17.056) | 12.108 (8.500-18.395) | 36.798 (13.573-75.692) | 80.474 (34.796-179.624) | 3.581 (0.501-8.658) | 9.440 (9.401-9.459) |
| `alpha_beta` | 17.945 (5.642-33.307) | 2.816 (2.608-3.378) | 9.073 (3.051-16.562) | 11.778 (5.643-20.325) | 38.328 (3.904-77.146) | 86.763 (20.279-185.106) | 3.546 (0.615-12.381) | 14.437 (14.303-14.535) |
| `history` | 18.253 (10.621-31.558) | 5.654 (5.046-6.263) | 9.259 (5.662-15.939) | 13.871 (10.064-19.257) | 34.086 (7.344-72.387) | 91.195 (52.806-172.915) | 4.504 (1.592-9.869) | 18.796 (18.604-18.930) |

All errors are measured in pixels.

## Position RMSE By Scenario

Each cell is average position RMSE across the ten seeds for that motion scenario.

| Scenario | adaptive_kalman | kalman | constant_accel_kalman | alpha_beta | history |
|---|---:|---:|---:|---:|---:|
| mixed_accel | 24.728 | 23.241 | 25.347 | 29.597 | 24.291 |
| constant_velocity | 9.512 | 10.145 | 14.524 | 8.486 | 13.054 |
| constant_acceleration | 13.255 | 14.609 | 16.529 | 12.898 | 16.273 |
| zig_zag | 27.171 | 32.443 | 26.372 | 30.236 | 27.755 |
| sine_wave | 11.950 | 11.016 | 13.515 | 12.672 | 13.761 |
| center_direct | 7.591 | 10.129 | 13.048 | 6.303 | 12.050 |
| center_orbit | 23.539 | 28.018 | 18.463 | 29.520 | 24.756 |
| random_maneuver | 11.413 | 11.948 | 13.781 | 13.851 | 14.083 |

## Metric Meaning

- `Position RMSE`: center prediction error over every evaluated frame.
- `Size RMSE`: width/height prediction error over every evaluated frame.
- `Box RMSE`: combined `x, y, width, height` prediction error.
- `Accept Pos RMSE`: center prediction error when a measurement was available.
- `Reject Pos RMSE`: center prediction error during occlusion/rejected frames.
- `Max Pos Err`: worst center prediction error within a run; its table entry
  aggregates those eighty scenario-seed worst errors.
- `Final Pos Err`: corrected state error on the final frame.
- `Avg Uncertainty`: average predictor uncertainty before accept/reject update.

## Interpretation

Across these eight trajectory families, ten seeds per family, and these configs:

1. `adaptive_kalman` has the best overall average position RMSE at `16.145` pixels.
2. `kalman`, `constant_accel_kalman`, `alpha_beta`, and `history` are close in
   the overall aggregate, all between `17.694` and `18.253` pixels.
3. `constant_accel_kalman` is strongest on the circular center-return scenario
   and remains good on zig-zag motion, but it no longer wins overall once
   constant-speed, sine, and center-direct paths are included.
4. `alpha_beta` has the best average target-size prediction and is strongest on
   constant-velocity and direct center-return motion.
5. All predictors recover to a low average final position error after measurements return.
6. The uncertainty values are not directly calibrated across model types, so they
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

The fixed seed set makes the report repeatable. Predictor behavior changes should
be visible as aggregate metric changes.
