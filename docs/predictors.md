# Predictors

## Shared Contract

Every predictor implements `initialize`, `predict`, `update`, `reset`, and `close` from `Predictor`. Predictors operate on target center position and velocity. They do not currently model target width/height.

`predict()` advances the internally retained state to the current frame. `update()` then receives policy feedback:

- On accept, `predictor_state.target_pos` is the visual measurement selected by BigTrack and metadata contains the match `score`.
- On reject, the state remains at the predicted position; models generally increase uncertainty and may damp velocity/acceleration.

All models use `frame_dt()` from `_motion.py`. A positive timestamp difference wins; otherwise a positive frame-index difference is used; otherwise `dt=1.0`.

## Shared Motion Utilities

`predictor_models/_motion.py` contains scalar/pair clamps, uncertainty bounds, velocity and acceleration limiting, damping, and time-delta calculation. Reuse these helpers so model behavior remains consistent around invalid time steps and configured limits.

## KalmanPredictorModel

This is a lightweight constant-velocity Kalman filter with independent 2x2 covariance matrices for x and y. State per axis is position and velocity.

Prediction applies the constant-velocity transition and adds `process_noise_position` to both covariance diagonals. Accepted updates use position-only measurement correction with `measurement_noise_position`. Rejection preserves predicted motion and adds `reject_uncertainty_growth`.

Uncertainty is derived from the summed covariance diagonals divided by 100 and then clamped. Metadata keys use the `kalman_` prefix and store covariance, timing, last stage, and reject count.

This model does not adapt measurement trust to visual score and does not damp rejected velocity.

## AdaptiveKalmanPredictorModel

This model extends the constant-velocity idea with score-aware and reject-aware behavior:

- High accepted scores reduce measurement noise, so the filter follows the visual measurement more strongly.
- Accepted covariance is decayed according to score.
- Rejection scales covariance by `reject_covariance_growth`, adds uncertainty growth, and damps velocity.
- Optional `max_position_velocity` bounds both prediction and retained velocity.
- Uncertainty is bounded by configured min/max values.

Metadata uses the `adaptive_kalman_` prefix and records covariance, last score, measurement-noise scale, accept decay, timing, stage, and reject count.

The synthetic regression currently ranks this model best overall for the checked configurations, but the report is not a production guarantee.

## AlphaBetaPredictorModel

The alpha-beta model is the smallest adaptive predictor. Prediction is constant velocity. On accept, the residual between predicted and measured position is corrected by score-weighted alpha, while velocity is corrected by score-weighted beta divided by `dt`.

Optional velocity and acceleration clamps prevent abrupt changes. Rejection damps velocity and increases uncertainty. Accepted uncertainty decays more when confidence is high. Metadata uses the `alpha_beta_` prefix.

This is useful when low overhead and directly understandable tuning are more valuable than covariance modeling.

## HistoryPredictorModel

The history model maintains a bounded tuple of accepted observations in metadata. It estimates desired velocity from the oldest and newest records within `velocity_window`, smooths that estimate against current velocity, and optionally limits acceleration and velocity.

Rejected frames are never appended to history. They damp velocity and grow uncertainty. `history_length` must effectively support at least two records; the implementation enforces a minimum retained length of two. Metadata uses the `history_` prefix.

This model is easy to inspect and robust to isolated prediction calls, but it depends on representative accepted measurements in its recent window.

## ConstantAccelerationKalmanPredictorModel

This filter uses a three-value state per axis: position, velocity, and acceleration, with independent 3x3 covariances. Acceleration is stored in metadata because the common public state exposes only position and velocity.

Prediction applies:

```text
position' = position + velocity*dt + 0.5*acceleration*dt^2
velocity' = velocity + acceleration*dt
```

Accepted updates correct all three values from a position measurement through Kalman gains. Measurement noise is score-adaptive. Rejection grows covariance, damps velocity and acceleration independently, and increases uncertainty. Optional velocity and acceleration limits apply. Metadata uses the `constant_accel_kalman_` prefix.

This model can represent sustained acceleration but has more state and tuning sensitivity than constant-velocity alternatives.

## MatcherTargetPredictorModel

This is the intentional no-motion baseline. `predict()` returns its retained state unchanged: it does not advance time, extrapolate velocity, filter a measurement, adjust uncertainty, or use score. On an accepted policy decision, BigTrack has already replaced `predictor_state.target_pos` with the selected matcher-box center, and `update()` retains that state. On a rejected decision, the policy supplies the unchanged prediction, so the retained position remains the latest accepted matcher center.

The predictor has an empty frozen `MatcherTargetPredictorConfig` solely so it can participate in the standard component registries. It has no model-specific metadata or calibrated uncertainty. It is useful as a simple matcher-only baseline, not for recovering through missed or rejected matches.

## Choosing a Predictor

- Start with `AdaptiveKalmanPredictorModel` for the most complete current accept/reject behavior.
- Use `KalmanPredictorModel` as a stable, simple covariance baseline.
- Use `AlphaBetaPredictorModel` for minimal computation and transparent gains.
- Use `HistoryPredictorModel` when recent accepted motion is the desired estimator.
- Use `ConstantAccelerationKalmanPredictorModel` for paths with sustained acceleration and sufficient tuning data.
- Use `MatcherTargetPredictorModel` to search at the latest accepted matcher center without motion prediction.

Always tune against representative videos. See [Predictor Trajectory Report](predictor-trajectory-report.md) for the deterministic synthetic baseline and [Development Guide](development.md#adding-a-predictor) before adding a model.

## Files and Exports

- `_motion.py`: shared math and time handling.
- `kalman.py`: basic constant-velocity Kalman.
- `adaptive_kalman.py`: score-aware constant-velocity Kalman.
- `alpha_beta.py`: alpha-beta residual correction.
- `history.py`: bounded accepted-history velocity estimation.
- `kalman_accel.py`: constant-acceleration Kalman.
- `matcher_target.py`: retains the latest accepted matcher center without motion prediction.
- `__init__.py`: predictor subpackage exports.

All six configs and models are also exported by `BigTracker/__init__.py`. New predictors must update both export layers and the registries described in [Testing and Tools](testing-and-tools.md).
