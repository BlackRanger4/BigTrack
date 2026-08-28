# Predictor Models

Predictors estimate target-center motion and uncertainty. They do not currently estimate target size.

- `_motion.py`: shared time delta, clamps, damping, acceleration limits, and uncertainty bounds.
- `kalman.py`: basic constant-velocity 2x2-per-axis Kalman filter.
- `adaptive_kalman.py`: score-aware Kalman correction plus reject covariance growth/velocity damping.
- `alpha_beta.py`: lightweight score-weighted residual correction.
- `history.py`: bounded accepted-observation history and smoothed velocity.
- `kalman_accel.py`: 3x3-per-axis position/velocity/acceleration Kalman filter.
- `matcher_target.py`: no-motion baseline that retains the latest matcher center accepted by the policy.
- `__init__.py`: config/model exports.

Models with persistent model-specific data store it under a unique metadata prefix. Every predictor must define accepted and rejected update behavior and return its retained post-update state.

See [`docs/predictors.md`](../../docs/predictors.md) for detailed behavior and [`docs/development.md#adding-a-predictor`](../../docs/development.md#adding-a-predictor) for the implementation, testing, export, tool, evaluation, and documentation checklist.
