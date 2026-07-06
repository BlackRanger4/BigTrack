from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from BigTracker.predictor import PredictorModel
from BigTracker.state import BigTrackState, TrackerPredictionState
from BigTracker.types import FrameLike, Point, Size, TrackerMode


Covariance2x2 = Tuple[float, float, float, float]


@dataclass(frozen=True)
class KalmanPredictorConfig:
    """Configuration for the lightweight constant-velocity Kalman model."""

    process_noise_position: float = 1.0
    process_noise_size: float = 0.5
    measurement_noise_position: float = 4.0
    measurement_noise_size: float = 2.0
    default_covariance: Covariance2x2 = (10.0, 0.0, 0.0, 10.0)
    min_size: Size = (1.0, 1.0)
    reject_uncertainty_growth: float = 1.5


class KalmanPredictorModel(PredictorModel):
    """Small dependency-free Kalman model for position and target size."""

    def __init__(self, config: Optional[KalmanPredictorConfig] = None) -> None:
        """Store Kalman parameters used by predict and update operations."""
        self.config = config or KalmanPredictorConfig()

    def predict(self, state: BigTrackState, frame: FrameLike) -> TrackerPredictionState:
        """Predict position, size, velocity, covariance, and uncertainty."""
        dt = self._frame_dt(state, frame)
        prediction = state.prediction
        covariances = self._covariances(prediction.metadata)

        x, vx, cov_x = self._predict_1d(
            prediction.target_pos[0],
            prediction.target_velocity[0],
            covariances["x"],
            dt,
            self.config.process_noise_position,
        )
        y, vy, cov_y = self._predict_1d(
            prediction.target_pos[1],
            prediction.target_velocity[1],
            covariances["y"],
            dt,
            self.config.process_noise_position,
        )
        w, vw, cov_w = self._predict_1d(
            prediction.target_size[0],
            prediction.target_size_velocity[0],
            covariances["w"],
            dt,
            self.config.process_noise_size,
        )
        h, vh, cov_h = self._predict_1d(
            prediction.target_size[1],
            prediction.target_size_velocity[1],
            covariances["h"],
            dt,
            self.config.process_noise_size,
        )

        size = (
            max(self.config.min_size[0], w),
            max(self.config.min_size[1], h),
        )
        metadata = self._metadata(prediction.metadata, cov_x, cov_y, cov_w, cov_h, frame)

        return TrackerPredictionState(
            target_pos=(x, y),
            target_size=size,
            target_velocity=(vx, vy),
            target_size_velocity=(vw, vh),
            last_score=prediction.last_score,
            uncertainty=self._uncertainty_from_covariances(metadata, state.mode),
            metadata=metadata,
        )

    def predict_position(self, state: BigTrackState, frame: FrameLike) -> Point:
        """Predict the target center using constant velocity."""
        dt = self._frame_dt(state, frame)
        prediction = state.prediction
        return (
            prediction.target_pos[0] + prediction.target_velocity[0] * dt,
            prediction.target_pos[1] + prediction.target_velocity[1] * dt,
        )

    def predict_size(self, state: BigTrackState, frame: FrameLike) -> Size:
        """Predict target size using constant size velocity."""
        dt = self._frame_dt(state, frame)
        prediction = state.prediction
        return (
            max(
                self.config.min_size[0],
                prediction.target_size[0] + prediction.target_size_velocity[0] * dt,
            ),
            max(
                self.config.min_size[1],
                prediction.target_size[1] + prediction.target_size_velocity[1] * dt,
            ),
        )

    def predict_uncertainty(self, state: BigTrackState, frame: FrameLike) -> float:
        """Estimate uncertainty from stored covariance and tracker mode."""
        return self._uncertainty_from_covariances(state.prediction.metadata, state.mode)

    def update_from_accept(
        self,
        state: BigTrackState,
        accepted_pos: Point,
        accepted_size: Size,
        score: float,
    ) -> TrackerPredictionState:
        """Run a measurement update after BigTrack accepts a visual match."""
        prediction = state.prediction
        covariances = self._covariances(prediction.metadata)

        x, vx, cov_x = self._update_1d(
            prediction.target_pos[0],
            prediction.target_velocity[0],
            covariances["x"],
            accepted_pos[0],
            self.config.measurement_noise_position,
        )
        y, vy, cov_y = self._update_1d(
            prediction.target_pos[1],
            prediction.target_velocity[1],
            covariances["y"],
            accepted_pos[1],
            self.config.measurement_noise_position,
        )
        w, vw, cov_w = self._update_1d(
            prediction.target_size[0],
            prediction.target_size_velocity[0],
            covariances["w"],
            accepted_size[0],
            self.config.measurement_noise_size,
        )
        h, vh, cov_h = self._update_1d(
            prediction.target_size[1],
            prediction.target_size_velocity[1],
            covariances["h"],
            accepted_size[1],
            self.config.measurement_noise_size,
        )

        metadata = self._metadata(prediction.metadata, cov_x, cov_y, cov_w, cov_h)
        return TrackerPredictionState(
            target_pos=(x, y),
            target_size=(
                max(self.config.min_size[0], w),
                max(self.config.min_size[1], h),
            ),
            target_velocity=(vx, vy),
            target_size_velocity=(vw, vh),
            last_score=max(0.0, min(1.0, score)),
            uncertainty=self._uncertainty_from_covariances(metadata, TrackerMode.TRACKING),
            metadata=metadata,
        )

    def update_from_reject(self, state: BigTrackState) -> TrackerPredictionState:
        """Increase uncertainty when BigTrack rejects visual evidence."""
        prediction = state.prediction
        metadata = dict(prediction.metadata)
        metadata["kalman_reject_count"] = int(metadata.get("kalman_reject_count", 0)) + 1
        uncertainty = prediction.uncertainty + self.config.reject_uncertainty_growth

        return TrackerPredictionState(
            target_pos=prediction.target_pos,
            target_size=prediction.target_size,
            target_velocity=prediction.target_velocity,
            target_size_velocity=prediction.target_size_velocity,
            last_score=prediction.last_score,
            uncertainty=uncertainty,
            metadata=metadata,
        )

    def _predict_1d(
        self,
        value: float,
        velocity: float,
        covariance: Covariance2x2,
        dt: float,
        process_noise: float,
    ) -> Tuple[float, float, Covariance2x2]:
        """Predict one constant-velocity scalar state."""
        p00, p01, p10, p11 = covariance
        next_value = value + velocity * dt
        next_velocity = velocity
        next_covariance = (
            p00 + dt * (p10 + p01) + dt * dt * p11 + process_noise,
            p01 + dt * p11,
            p10 + dt * p11,
            p11 + process_noise,
        )
        return next_value, next_velocity, next_covariance

    def _update_1d(
        self,
        value: float,
        velocity: float,
        covariance: Covariance2x2,
        measurement: float,
        measurement_noise: float,
    ) -> Tuple[float, float, Covariance2x2]:
        """Apply one scalar Kalman measurement update."""
        p00, p01, p10, p11 = covariance
        innovation_covariance = p00 + measurement_noise
        if innovation_covariance <= 0.0:
            return measurement, velocity, covariance

        gain_value = p00 / innovation_covariance
        gain_velocity = p10 / innovation_covariance
        residual = measurement - value
        next_value = value + gain_value * residual
        next_velocity = velocity + gain_velocity * residual
        next_covariance = (
            (1.0 - gain_value) * p00,
            (1.0 - gain_value) * p01,
            p10 - gain_velocity * p00,
            p11 - gain_velocity * p01,
        )
        return next_value, next_velocity, next_covariance

    def _frame_dt(self, state: BigTrackState, frame: FrameLike) -> float:
        """Estimate frame delta from timestamp, then frame index, then fallback to one."""
        previous_timestamp = state.output.timestamp
        if frame.timestamp > previous_timestamp:
            return max(1e-6, frame.timestamp - previous_timestamp)

        previous_idx = state.output.frame_idx
        if frame.idx > previous_idx:
            return float(frame.idx - previous_idx)

        return 1.0

    def _covariances(self, metadata: Mapping[str, Any]) -> Dict[str, Covariance2x2]:
        """Read per-dimension covariance from prediction metadata."""
        stored = metadata.get("kalman_covariance", {})
        return {
            "x": tuple(stored.get("x", self.config.default_covariance)),
            "y": tuple(stored.get("y", self.config.default_covariance)),
            "w": tuple(stored.get("w", self.config.default_covariance)),
            "h": tuple(stored.get("h", self.config.default_covariance)),
        }

    def _metadata(
        self,
        metadata: Mapping[str, Any],
        cov_x: Covariance2x2,
        cov_y: Covariance2x2,
        cov_w: Covariance2x2,
        cov_h: Covariance2x2,
        frame: Optional[FrameLike] = None,
    ) -> Mapping[str, Any]:
        """Return metadata with updated Kalman covariance."""
        updated = dict(metadata)
        updated["kalman_covariance"] = {
            "x": cov_x,
            "y": cov_y,
            "w": cov_w,
            "h": cov_h,
        }
        updated["kalman_reject_count"] = 0
        if frame is not None:
            updated["kalman_last_frame_idx"] = frame.idx
            updated["kalman_last_timestamp"] = frame.timestamp
        return updated

    def _uncertainty_from_covariances(
        self,
        metadata: Mapping[str, Any],
        mode: TrackerMode,
    ) -> float:
        """Convert covariance and mode into a scalar uncertainty."""
        covariances = self._covariances(metadata)
        diagonal_sum = sum(covariance[0] + covariance[3] for covariance in covariances.values())
        base_uncertainty = max(0.0, diagonal_sum / 100.0)

        if mode in (TrackerMode.RECOVERY, TrackerMode.LOST):
            return base_uncertainty * 2.0
        if mode in (TrackerMode.UNCERTAIN, TrackerMode.OCCLUDED):
            return base_uncertainty * 1.5
        return base_uncertainty
