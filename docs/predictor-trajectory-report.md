# Predictor Trajectory Evaluation Report

Deterministic synthetic evaluation for the BigTracker predictor models.

## Scenario

- Runs: `80`
- Seeds: `1337-1346`
- Motion scenarios: `8`
- Frames: `180`
- Frame size: `640x360`
- Measurements: Gaussian noise, structured occlusion windows, and deterministic outliers.
- Accepted measurements per run: `155`
- Rejected measurements per run: `24`

## Metrics

Each cell is `average (minimum-maximum)` across all scenario and seed runs.

| Predictor | Position RMSE | Accept Pos RMSE | Reject Pos RMSE | Max Pos Err | Final Pos Err | Avg Uncertainty |
|---|---:|---:|---:|---:|---:|---:|
| adaptive_kalman | 18.367 (6.097-42.294) | 11.476 (6.233-21.823) | 40.087 (2.524-101.318) | 90.806 (23.045-240.786) | 4.251 (0.719-10.329) | 4.088 (4.065-4.132) |
| constant_accel_kalman | 18.469 (8.905-44.681) | 12.432 (8.212-23.525) | 38.462 (7.225-106.373) | 88.752 (31.228-244.393) | 4.787 (0.618-10.679) | 8.044 (8.014-8.105) |
| alpha_beta | 19.994 (5.289-47.569) | 12.542 (5.368-25.879) | 43.773 (2.596-112.032) | 96.220 (17.853-243.490) | 4.155 (0.840-10.710) | 1.796 (1.784-1.815) |
| kalman | 20.332 (8.069-51.009) | 13.676 (7.819-27.338) | 41.710 (2.422-120.743) | 107.759 (33.428-314.761) | 4.387 (0.861-10.778) | 1.623 (1.623-1.623) |
| history | 20.647 (10.675-44.212) | 14.637 (10.204-24.221) | 40.711 (6.458-103.875) | 101.238 (45.920-245.778) | 5.039 (0.358-12.265) | 2.307 (2.292-2.326) |
| matcher_target | 25.818 (8.229-47.982) | 16.382 (8.552-27.152) | 55.738 (3.280-111.736) | 109.353 (36.210-194.663) | 5.039 (0.358-12.265) | 0.000 (0.000-0.000) |

## Position RMSE by Scenario

Each cell is average position RMSE across the ten seeds for that motion scenario.

| Scenario | adaptive_kalman | constant_accel_kalman | alpha_beta | kalman | history | matcher_target |
|---|---:|---:|---:|---:|---:|---:|
| mixed_accel | 34.888 | 31.874 | 38.547 | 41.203 | 35.639 | 38.035 |
| constant_velocity | 8.636 | 12.480 | 8.440 | 9.427 | 12.145 | 16.375 |
| constant_acceleration | 13.083 | 15.189 | 13.366 | 10.844 | 15.767 | 31.520 |
| zig_zag | 28.519 | 29.181 | 31.111 | 34.110 | 29.966 | 27.685 |
| sine_wave | 11.708 | 11.131 | 13.082 | 11.783 | 13.440 | 18.457 |
| center_direct | 7.010 | 11.686 | 5.851 | 9.405 | 11.284 | 8.893 |
| center_orbit | 31.145 | 24.303 | 35.234 | 32.783 | 32.300 | 46.494 |
| random_maneuver | 11.946 | 11.909 | 14.319 | 13.106 | 14.632 | 19.086 |

All errors are pixels. `Reject Pos RMSE` measures prediction during occlusion/rejected measurement frames. Uncertainty values are model-relative, not calibrated probabilities, so they should not be compared directly across families.

Adaptive Kalman has the best aggregate position RMSE for the checked configuration. Individual scenarios favor different models. This is a synthetic stress test, not a final predictor ranking; representative video must drive production choice and tuning.

## Reproduce

```powershell
python -m unittest tests.test_predictor_trajectory_evaluation
python tests\test_predictor_trajectory_evaluation.py
```

The second command prints the generated report. When predictor behavior, evaluation configuration, seeds, or scenarios change, replace the tables here with that output and review the differences.
