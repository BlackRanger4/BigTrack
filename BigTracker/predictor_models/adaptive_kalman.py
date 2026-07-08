from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from BigTracker.predictor import PredictorModel
from BigTracker.state import BigTrackState, TrackerPredictionState
from BigTracker.types import FrameLike, Point, Size, TrackerMode


Covariance2x2 = Tuple[float, float, float, float]


@dataclass(frozen=True)
class AdaptiveKalmanPredictorConfig:
    """Config for a constant-velocity Kalman predictor with policy feedback."""

    process_noise_position: float = 1.0
    process_noise_size: float = 0.5
    measurement_noise_position: float = 4.0
    measurement_noise_size: float = 2.0
    default_covariance: Covariance2x2 = (10.0, 0.0, 0.0, 10.0)
    min_size: Size = (1.0, 1.0)
    adaptive_measurement_noise: bool = True
    min_measurement_noise_scale: float = 0.25
    max_measurement_noise_scale: float = 3.0
    min_uncertainty: float = 0.0
    max_uncertainty: Optional[float] = 100.0
    reject_uncertainty_growth: float = 1.5
    reject_covariance_growth: float = 1.15
    reject_velocity_damping: float = 0.85
    max_position_velocity: Optional[float] = None
    max_size_velocity: Optional[float] = None
    uncertainty_accept_decay: float = 0.90
    clamp_to_frame: bool = True


class AdaptiveKalmanPredictorModel(PredictorModel):
    """Constant-velocity Kalman predictor with score-aware accept/reject behavior."""

    def __init__(self, config: Optional[AdaptiveKalmanPredictorConfig] = None) -> None:
        self.config = config or AdaptiveKalmanPredictorConfig()

    def predict(self, state: BigTrackState, frame: FrameLike) -> TrackerPredictionState:
        """Predict center, size, velocity, covariance, and scalar uncertainty."""

        dt = self._frame_dt(state, frame)
        prediction = state.prediction
        covariances = self._covariances(prediction.metadata)

        vx0 = self._clamp_scalar_velocity(prediction.target_velocity[0], self.config.max_position_velocity)
        vy0 = self._clamp_scalar_velocity(prediction.target_velocity[1], self.config.max_position_velocity)
        vw0 = self._clamp_scalar_velocity(
            prediction.target_size_velocity[0],
            self.config.max_size_velocity,
        )
        vh0 = self._clamp_scalar_velocity(
            prediction.target_size_velocity[1],
            self.config.max_size_velocity,
        )

        x, vx, cov_x = self._predict_1d(
            prediction.target_pos[0],
            vx0,
            covariances["x"],
            dt,
            self.config.process_noise_position,
        )
        y, vy, cov_y = self._predict_1d(
            prediction.target_pos[1],
            vy0,
            covariances["y"],
            dt,
            self.config.process_noise_position,
        )
        w, vw, cov_w = self._predict_1d(
            prediction.target_size[0],
            vw0,
            covariances["w"],
            dt,
            self.config.process_noise_size,
        )
        h, vh, cov_h = self._predict_1d(
            prediction.target_size[1],
            vh0,
            covariances["h"],
            dt,
            self.config.process_noise_size,
        )

        target_size = (
            max(float(self.config.min_size[0]), w),
            max(float(self.config.min_size[1]), h),
        )
        target_pos = self._clamp_position_to_frame((x, y), target_size, frame)
        metadata = self._metadata(
            prediction.metadata,
            cov_x,
            cov_y,
            cov_w,
            cov_h,
            frame=frame,
            stage="predict",
        )

        return TrackerPredictionState(
            target_pos=target_pos,
            target_size=target_size,
            target_velocity=(
                self._clamp_scalar_velocity(vx, self.config.max_position_velocity),
                self._clamp_scalar_velocity(vy, self.config.max_position_velocity),
            ),
            target_size_velocity=(
                self._clamp_scalar_velocity(vw, self.config.max_size_velocity),
                self._clamp_scalar_velocity(vh, self.config.max_size_velocity),
            ),
            last_score=prediction.last_score,
            uncertainty=self._uncertainty_from_covariances(metadata, state.mode),
            metadata=metadata,
        )

    def predict_position(self, state: BigTrackState, frame: FrameLike) -> Point:
        """Predict target center using clamped constant velocity."""

        dt = self._frame_dt(state, frame)
        prediction = state.prediction
        vx = self._clamp_scalar_velocity(prediction.target_velocity[0], self.config.max_position_velocity)
        vy = self._clamp_scalar_velocity(prediction.target_velocity[1], self.config.max_position_velocity)
        return (
            prediction.target_pos[0] + vx * dt,
            prediction.target_pos[1] + vy * dt,
        )

    def predict_size(self, state: BigTrackState, frame: FrameLike) -> Size:
        """Predict target size using clamped constant size velocity."""

        dt = self._frame_dt(state, frame)
        prediction = state.prediction
        vw = self._clamp_scalar_velocity(
            prediction.target_size_velocity[0],
            self.config.max_size_velocity,
        )
        vh = self._clamp_scalar_velocity(
            prediction.target_size_velocity[1],
            self.config.max_size_velocity,
        )
        return (
            max(float(self.config.min_size[0]), prediction.target_size[0] + vw * dt),
            max(float(self.config.min_size[1]), prediction.target_size[1] + vh * dt),
        )

    def predict_uncertainty(self, state: BigTrackState, frame: FrameLike) -> float:
        """Estimate uncertainty from stored covariance and current mode."""

        return self._uncertainty_from_covariances(state.prediction.metadata, state.mode)

    def update_from_accept(
        self,
        state: BigTrackState,
        accepted_pos: Point,
        accepted_size: Size,
        score: float,
    ) -> TrackerPredictionState:
        """Update from accepted visual evidence with score-dependent noise."""

        prediction = state.prediction
        covariances = self._covariances(prediction.metadata)
        score = _clamp01(score)
        noise_scale = self._measurement_noise_scale(score)
        position_noise = self.config.measurement_noise_position * noise_scale
        size_noise = self.config.measurement_noise_size * noise_scale

        x, vx, cov_x = self._update_1d(
            prediction.target_pos[0],
            prediction.target_velocity[0],
            covariances["x"],
            accepted_pos[0],
            position_noise,
        )
        y, vy, cov_y = self._update_1d(
            prediction.target_pos[1],
            prediction.target_velocity[1],
            covariances["y"],
            accepted_pos[1],
            position_noise,
        )
        w, vw, cov_w = self._update_1d(
            prediction.target_size[0],
            prediction.target_size_velocity[0],
            covariances["w"],
            accepted_size[0],
            size_noise,
        )
        h, vh, cov_h = self._update_1d(
            prediction.target_size[1],
            prediction.target_size_velocity[1],
            covariances["h"],
            accepted_size[1],
            size_noise,
        )

        decay = self._accept_decay(score)
        cov_x = self._scale_covariance(cov_x, decay)
        cov_y = self._scale_covariance(cov_y, decay)
        cov_w = self._scale_covariance(cov_w, decay)
        cov_h = self._scale_covariance(cov_h, decay)
        metadata = self._metadata(
            prediction.metadata,
            cov_x,
            cov_y,
            cov_w,
            cov_h,
            stage="accept",
            reject_count=0,
        )
        metadata = dict(metadata)
        metadata["adaptive_kalman_last_score"] = score
        metadata["adaptive_kalman_measurement_noise_scale"] = noise_scale
        metadata["adaptive_kalman_measurement_noise_position"] = position_noise
        metadata["adaptive_kalman_measurement_noise_size"] = size_noise
        metadata["adaptive_kalman_accept_decay"] = decay

        return TrackerPredictionState(
            target_pos=(x, y),
            target_size=(
                max(float(self.config.min_size[0]), w),
                max(float(self.config.min_size[1]), h),
            ),
            target_velocity=(
                self._clamp_scalar_velocity(vx, self.config.max_position_velocity),
                self._clamp_scalar_velocity(vy, self.config.max_position_velocity),
            ),
            target_size_velocity=(
                self._clamp_scalar_velocity(vw, self.config.max_size_velocity),
                self._clamp_scalar_velocity(vh, self.config.max_size_velocity),
            ),
            last_score=score,
            uncertainty=self._uncertainty_from_covariances(metadata, TrackerMode.TRACKING),
            metadata=metadata,
        )

    def update_from_reject(self, state: BigTrackState) -> TrackerPredictionState:
        """Keep the predicted state, grow uncertainty, and damp velocity."""

        prediction = state.prediction
        covariances = self._covariances(prediction.metadata)
        reject_count = int(prediction.metadata.get("adaptive_kalman_reject_count", 0)) + 1
        covariance_growth = max(1.0, float(self.config.reject_covariance_growth))
        cov_x = self._scale_covariance(covariances["x"], covariance_growth)
        cov_y = self._scale_covariance(covariances["y"], covariance_growth)
        cov_w = self._scale_covariance(covariances["w"], covariance_growth)
        cov_h = self._scale_covariance(covariances["h"], covariance_growth)
        metadata = self._metadata(
            prediction.metadata,
            cov_x,
            cov_y,
            cov_w,
            cov_h,
            stage="reject",
            reject_count=reject_count,
        )

        damping = _clamp(self.config.reject_velocity_damping, 0.0, 1.0)
        target_velocity = (
            self._clamp_scalar_velocity(
                prediction.target_velocity[0] * damping,
                self.config.max_position_velocity,
            ),
            self._clamp_scalar_velocity(
                prediction.target_velocity[1] * damping,
                self.config.max_position_velocity,
            ),
        )
        target_size_velocity = (
            self._clamp_scalar_velocity(
                prediction.target_size_velocity[0] * damping,
                self.config.max_size_velocity,
            ),
            self._clamp_scalar_velocity(
                prediction.target_size_velocity[1] * damping,
                self.config.max_size_velocity,
            ),
        )
        uncertainty = self._clamp_uncertainty(
            self._uncertainty_from_covariances(metadata, state.mode)
            + max(0.0, float(self.config.reject_uncertainty_growth))
        )

        return TrackerPredictionState(
            target_pos=prediction.target_pos,
            target_size=prediction.target_size,
            target_velocity=target_velocity,
            target_size_velocity=target_size_velocity,
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
        p00, p01, p10, p11 = covariance
        innovation_covariance = p00 + max(1e-9, measurement_noise)
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
        previous_timestamp = state.output.timestamp
        if frame.timestamp > previous_timestamp:
            return max(1e-6, frame.timestamp - previous_timestamp)

        previous_idx = state.output.frame_idx
        if frame.idx > previous_idx:
            return float(frame.idx - previous_idx)

        return 1.0

    def _covariances(self, metadata: Mapping[str, Any]) -> Dict[str, Covariance2x2]:
        stored = metadata.get("adaptive_kalman_covariance")
        if stored is None:
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
        *,
        frame: Optional[FrameLike] = None,
        stage: str,
        reject_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        updated = dict(metadata)
        updated["adaptive_kalman_covariance"] = {
            "x": cov_x,
            "y": cov_y,
            "w": cov_w,
            "h": cov_h,
        }
        if reject_count is not None:
            updated["adaptive_kalman_reject_count"] = int(reject_count)
        else:
            updated["adaptive_kalman_reject_count"] = int(
                metadata.get("adaptive_kalman_reject_count", 0)
            )
        updated["adaptive_kalman_last_stage"] = stage
        if frame is not None:
            updated["adaptive_kalman_last_frame_idx"] = frame.idx
            updated["adaptive_kalman_last_timestamp"] = frame.timestamp
        return updated

    def _uncertainty_from_covariances(
        self,
        metadata: Mapping[str, Any],
        mode: TrackerMode,
    ) -> float:
        covariances = self._covariances(metadata)
        diagonal_sum = sum(covariance[0] + covariance[3] for covariance in covariances.values())
        base_uncertainty = max(0.0, diagonal_sum / 100.0)

        if mode in (TrackerMode.RECOVERY, TrackerMode.LOST):
            base_uncertainty *= 2.0
        elif mode in (TrackerMode.UNCERTAIN, TrackerMode.OCCLUDED):
            base_uncertainty *= 1.5

        return self._clamp_uncertainty(base_uncertainty)

    def _measurement_noise_scale(self, score: float) -> float:
        if not self.config.adaptive_measurement_noise:
            return 1.0
        min_scale = max(1e-9, float(self.config.min_measurement_noise_scale))
        max_scale = max(min_scale, float(self.config.max_measurement_noise_scale))
        return max_scale - (max_scale - min_scale) * _clamp01(score)

    def _accept_decay(self, score: float) -> float:
        base_decay = _clamp(float(self.config.uncertainty_accept_decay), 0.0, 1.0)
        return 1.0 - (1.0 - base_decay) * _clamp01(score)

    def _scale_covariance(
        self,
        covariance: Covariance2x2,
        scale: float,
    ) -> Covariance2x2:
        scale = max(0.0, float(scale))
        return tuple(float(value) * scale for value in covariance)  # type: ignore[return-value]

    def _clamp_scalar_velocity(self, velocity: float, max_velocity: Optional[float]) -> float:
        if max_velocity is None:
            return float(velocity)
        limit = abs(float(max_velocity))
        return _clamp(float(velocity), -limit, limit)

    def _clamp_position_to_frame(
        self,
        target_pos: Point,
        target_size: Size,
        frame: FrameLike,
    ) -> Point:
        if not self.config.clamp_to_frame:
            return target_pos

        image = getattr(frame, "image", None)
        shape = getattr(image, "shape", None)
        if shape is None or len(shape) < 2:
            return target_pos

        frame_height = float(shape[0])
        frame_width = float(shape[1])
        if frame_width <= 0.0 or frame_height <= 0.0:
            return target_pos

        half_w = max(0.0, float(target_size[0]) / 2.0)
        half_h = max(0.0, float(target_size[1]) / 2.0)

        if half_w * 2.0 >= frame_width:
            min_x = max_x = frame_width / 2.0
        else:
            min_x = half_w
            max_x = frame_width - half_w

        if half_h * 2.0 >= frame_height:
            min_y = max_y = frame_height / 2.0
        else:
            min_y = half_h
            max_y = frame_height - half_h

        return (
            _clamp(float(target_pos[0]), min_x, max_x),
            _clamp(float(target_pos[1]), min_y, max_y),
        )

    def _clamp_uncertainty(self, uncertainty: float) -> float:
        lower = max(0.0, float(self.config.min_uncertainty))
        upper = self.config.max_uncertainty
        if upper is None:
            return max(lower, float(uncertainty))
        return _clamp(float(uncertainty), lower, max(lower, float(upper)))


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(float(minimum), min(float(maximum), float(value)))


def _clamp01(value: float) -> float:
    return _clamp(value, 0.0, 1.0)
