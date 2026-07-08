from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from BigTracker.predictor import PredictorModel
from BigTracker.predictor_models._motion import (
    clamp,
    clamp01,
    clamp_center_size,
    clamp_pair_abs,
    clamp_size_pair_abs,
    clamp_uncertainty,
    damp_pair,
    damp_size_pair,
    frame_dt,
    frame_shape_from_frame,
    frame_shape_from_metadata,
    put_frame_shape,
)
from BigTracker.state import BigTrackState, TrackerPredictionState
from BigTracker.types import FrameLike, Point, Size, TrackerMode


Covariance3x3 = Tuple[float, float, float, float, float, float, float, float, float]


@dataclass(frozen=True)
class ConstantAccelerationKalmanPredictorConfig:
    """Config for a scalar Kalman filter tracking value, velocity, acceleration."""

    process_noise_position: float = 1.0
    process_noise_size: float = 0.5
    process_noise_velocity: float = 0.5
    process_noise_size_velocity: float = 0.25
    process_noise_acceleration: float = 0.25
    process_noise_size_acceleration: float = 0.10
    measurement_noise_position: float = 4.0
    measurement_noise_size: float = 2.0
    default_covariance: Covariance3x3 = (10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0)
    min_size: Size = (1.0, 1.0)
    adaptive_measurement_noise: bool = True
    min_measurement_noise_scale: float = 0.25
    max_measurement_noise_scale: float = 3.0
    max_position_velocity: Optional[float] = None
    max_size_velocity: Optional[float] = None
    max_position_acceleration: Optional[float] = None
    max_size_acceleration: Optional[float] = None
    reject_velocity_damping: float = 0.85
    reject_acceleration_damping: float = 0.65
    reject_uncertainty_growth: float = 1.5
    reject_covariance_growth: float = 1.15
    uncertainty_accept_decay: float = 0.90
    min_uncertainty: float = 0.0
    max_uncertainty: Optional[float] = 100.0
    clamp_to_frame: bool = True


class ConstantAccelerationKalmanPredictorModel(PredictorModel):
    """Kalman predictor with value, velocity, and acceleration per dimension."""

    _META_PREFIX = "constant_accel_kalman"

    def __init__(self, config: Optional[ConstantAccelerationKalmanPredictorConfig] = None) -> None:
        self.config = config or ConstantAccelerationKalmanPredictorConfig()

    def predict(self, state: BigTrackState, frame: FrameLike) -> TrackerPredictionState:
        prediction = state.prediction
        dt = frame_dt(state, frame)
        accelerations = self._accelerations(prediction.metadata)
        covariances = self._covariances(prediction.metadata)

        ax, ay = self._clamp_acceleration_pair(
            accelerations["pos"],
            self.config.max_position_acceleration,
        )
        aw, ah = self._clamp_acceleration_pair(
            accelerations["size"],
            self.config.max_size_acceleration,
        )

        x, vx, ax, cov_x = self._predict_1d(
            prediction.target_pos[0],
            prediction.target_velocity[0],
            ax,
            covariances["x"],
            dt,
            (
                self.config.process_noise_position,
                self.config.process_noise_velocity,
                self.config.process_noise_acceleration,
            ),
        )
        y, vy, ay, cov_y = self._predict_1d(
            prediction.target_pos[1],
            prediction.target_velocity[1],
            ay,
            covariances["y"],
            dt,
            (
                self.config.process_noise_position,
                self.config.process_noise_velocity,
                self.config.process_noise_acceleration,
            ),
        )
        w, vw, aw, cov_w = self._predict_1d(
            prediction.target_size[0],
            prediction.target_size_velocity[0],
            aw,
            covariances["w"],
            dt,
            (
                self.config.process_noise_size,
                self.config.process_noise_size_velocity,
                self.config.process_noise_size_acceleration,
            ),
        )
        h, vh, ah, cov_h = self._predict_1d(
            prediction.target_size[1],
            prediction.target_size_velocity[1],
            ah,
            covariances["h"],
            dt,
            (
                self.config.process_noise_size,
                self.config.process_noise_size_velocity,
                self.config.process_noise_size_acceleration,
            ),
        )

        velocity = clamp_pair_abs((vx, vy), self.config.max_position_velocity)
        size_velocity = clamp_size_pair_abs((vw, vh), self.config.max_size_velocity)
        frame_shape = frame_shape_from_frame(frame)
        target_pos, target_size = clamp_center_size(
            (x, y),
            (w, h),
            frame_shape=frame_shape,
            min_size=self.config.min_size,
            clamp_to_frame=self.config.clamp_to_frame,
        )
        metadata = self._metadata(
            prediction.metadata,
            cov_x,
            cov_y,
            cov_w,
            cov_h,
            acceleration=(ax, ay),
            size_acceleration=(aw, ah),
            stage="predict",
            frame=frame,
        )
        metadata[f"{self._META_PREFIX}_last_dt"] = dt

        return TrackerPredictionState(
            target_pos=target_pos,
            target_size=target_size,
            target_velocity=velocity,
            target_size_velocity=size_velocity,
            last_score=prediction.last_score,
            uncertainty=self._uncertainty_from_covariances(metadata, state.mode),
            metadata=metadata,
        )

    def update_from_accept(
        self,
        state: BigTrackState,
        accepted_pos: Point,
        accepted_size: Size,
        score: float,
    ) -> TrackerPredictionState:
        prediction = state.prediction
        metadata = dict(prediction.metadata)
        covariances = self._covariances(metadata)
        accelerations = self._accelerations(metadata)
        score = clamp01(score)
        noise_scale = self._measurement_noise_scale(score)
        position_noise = self.config.measurement_noise_position * noise_scale
        size_noise = self.config.measurement_noise_size * noise_scale

        x, vx, ax, cov_x = self._update_1d(
            prediction.target_pos[0],
            prediction.target_velocity[0],
            accelerations["pos"][0],
            covariances["x"],
            accepted_pos[0],
            position_noise,
        )
        y, vy, ay, cov_y = self._update_1d(
            prediction.target_pos[1],
            prediction.target_velocity[1],
            accelerations["pos"][1],
            covariances["y"],
            accepted_pos[1],
            position_noise,
        )
        w, vw, aw, cov_w = self._update_1d(
            prediction.target_size[0],
            prediction.target_size_velocity[0],
            accelerations["size"][0],
            covariances["w"],
            accepted_size[0],
            size_noise,
        )
        h, vh, ah, cov_h = self._update_1d(
            prediction.target_size[1],
            prediction.target_size_velocity[1],
            accelerations["size"][1],
            covariances["h"],
            accepted_size[1],
            size_noise,
        )

        ax, ay = self._clamp_acceleration_pair((ax, ay), self.config.max_position_acceleration)
        aw, ah = self._clamp_acceleration_pair((aw, ah), self.config.max_size_acceleration)
        velocity = clamp_pair_abs((vx, vy), self.config.max_position_velocity)
        size_velocity = clamp_size_pair_abs((vw, vh), self.config.max_size_velocity)
        target_pos, target_size = self._clamp_from_metadata(metadata, (x, y), (w, h))
        decay = self._accept_decay(score)
        cov_x = self._scale_covariance(cov_x, decay)
        cov_y = self._scale_covariance(cov_y, decay)
        cov_w = self._scale_covariance(cov_w, decay)
        cov_h = self._scale_covariance(cov_h, decay)
        metadata = self._metadata(
            metadata,
            cov_x,
            cov_y,
            cov_w,
            cov_h,
            acceleration=(ax, ay),
            size_acceleration=(aw, ah),
            stage="accept",
            reject_count=0,
        )
        metadata[f"{self._META_PREFIX}_last_score"] = score
        metadata[f"{self._META_PREFIX}_measurement_noise_scale"] = noise_scale

        return TrackerPredictionState(
            target_pos=target_pos,
            target_size=target_size,
            target_velocity=velocity,
            target_size_velocity=size_velocity,
            last_score=score,
            uncertainty=self._uncertainty_from_covariances(metadata, TrackerMode.TRACKING),
            metadata=metadata,
        )

    def update_from_reject(self, state: BigTrackState) -> TrackerPredictionState:
        prediction = state.prediction
        metadata = dict(prediction.metadata)
        covariances = self._covariances(metadata)
        accelerations = self._accelerations(metadata)
        covariance_growth = max(1.0, float(self.config.reject_covariance_growth))
        acceleration_damping = clamp(self.config.reject_acceleration_damping, 0.0, 1.0)
        reject_count = int(metadata.get(f"{self._META_PREFIX}_reject_count", 0)) + 1
        metadata = self._metadata(
            metadata,
            self._scale_covariance(covariances["x"], covariance_growth),
            self._scale_covariance(covariances["y"], covariance_growth),
            self._scale_covariance(covariances["w"], covariance_growth),
            self._scale_covariance(covariances["h"], covariance_growth),
            acceleration=(
                accelerations["pos"][0] * acceleration_damping,
                accelerations["pos"][1] * acceleration_damping,
            ),
            size_acceleration=(
                accelerations["size"][0] * acceleration_damping,
                accelerations["size"][1] * acceleration_damping,
            ),
            stage="reject",
            reject_count=reject_count,
        )

        return TrackerPredictionState(
            target_pos=prediction.target_pos,
            target_size=prediction.target_size,
            target_velocity=clamp_pair_abs(
                damp_pair(prediction.target_velocity, self.config.reject_velocity_damping),
                self.config.max_position_velocity,
            ),
            target_size_velocity=clamp_size_pair_abs(
                damp_size_pair(prediction.target_size_velocity, self.config.reject_velocity_damping),
                self.config.max_size_velocity,
            ),
            last_score=prediction.last_score,
            uncertainty=clamp_uncertainty(
                self._uncertainty_from_covariances(metadata, state.mode)
                + max(0.0, float(self.config.reject_uncertainty_growth)),
                min_uncertainty=self.config.min_uncertainty,
                max_uncertainty=self.config.max_uncertainty,
            ),
            metadata=metadata,
        )

    def _predict_1d(
        self,
        value: float,
        velocity: float,
        acceleration: float,
        covariance: Covariance3x3,
        dt: float,
        process_noise: tuple[float, float, float],
    ) -> tuple[float, float, float, Covariance3x3]:
        next_value = value + velocity * dt + 0.5 * acceleration * dt * dt
        next_velocity = velocity + acceleration * dt
        next_acceleration = acceleration
        return (
            next_value,
            next_velocity,
            next_acceleration,
            self._predict_covariance(covariance, dt, process_noise),
        )

    def _update_1d(
        self,
        value: float,
        velocity: float,
        acceleration: float,
        covariance: Covariance3x3,
        measurement: float,
        measurement_noise: float,
    ) -> tuple[float, float, float, Covariance3x3]:
        p00, p01, p02, p10, p11, p12, p20, p21, p22 = covariance
        innovation_covariance = p00 + max(1e-9, measurement_noise)
        gain_value = p00 / innovation_covariance
        gain_velocity = p10 / innovation_covariance
        gain_acceleration = p20 / innovation_covariance
        residual = measurement - value
        next_value = value + gain_value * residual
        next_velocity = velocity + gain_velocity * residual
        next_acceleration = acceleration + gain_acceleration * residual
        next_covariance = (
            (1.0 - gain_value) * p00,
            (1.0 - gain_value) * p01,
            (1.0 - gain_value) * p02,
            p10 - gain_velocity * p00,
            p11 - gain_velocity * p01,
            p12 - gain_velocity * p02,
            p20 - gain_acceleration * p00,
            p21 - gain_acceleration * p01,
            p22 - gain_acceleration * p02,
        )
        return next_value, next_velocity, next_acceleration, next_covariance

    def _predict_covariance(
        self,
        covariance: Covariance3x3,
        dt: float,
        process_noise: tuple[float, float, float],
    ) -> Covariance3x3:
        p = (
            (covariance[0], covariance[1], covariance[2]),
            (covariance[3], covariance[4], covariance[5]),
            (covariance[6], covariance[7], covariance[8]),
        )
        f = ((1.0, dt, 0.5 * dt * dt), (0.0, 1.0, dt), (0.0, 0.0, 1.0))
        fp = tuple(
            tuple(sum(f[row][k] * p[k][col] for k in range(3)) for col in range(3))
            for row in range(3)
        )
        fpf_t = tuple(
            tuple(sum(fp[row][k] * f[col][k] for k in range(3)) for col in range(3))
            for row in range(3)
        )
        return (
            fpf_t[0][0] + process_noise[0],
            fpf_t[0][1],
            fpf_t[0][2],
            fpf_t[1][0],
            fpf_t[1][1] + process_noise[1],
            fpf_t[1][2],
            fpf_t[2][0],
            fpf_t[2][1],
            fpf_t[2][2] + process_noise[2],
        )

    def _covariances(self, metadata: Mapping[str, Any]) -> Dict[str, Covariance3x3]:
        stored = metadata.get(f"{self._META_PREFIX}_covariance", {})
        return {
            "x": tuple(stored.get("x", self.config.default_covariance)),
            "y": tuple(stored.get("y", self.config.default_covariance)),
            "w": tuple(stored.get("w", self.config.default_covariance)),
            "h": tuple(stored.get("h", self.config.default_covariance)),
        }

    def _accelerations(self, metadata: Mapping[str, Any]) -> dict[str, Point]:
        acceleration = metadata.get(f"{self._META_PREFIX}_acceleration", (0.0, 0.0))
        size_acceleration = metadata.get(f"{self._META_PREFIX}_size_acceleration", (0.0, 0.0))
        return {
            "pos": (float(acceleration[0]), float(acceleration[1])),
            "size": (float(size_acceleration[0]), float(size_acceleration[1])),
        }

    def _metadata(
        self,
        metadata: Mapping[str, Any],
        cov_x: Covariance3x3,
        cov_y: Covariance3x3,
        cov_w: Covariance3x3,
        cov_h: Covariance3x3,
        *,
        acceleration: Point,
        size_acceleration: Size,
        stage: str,
        frame: Optional[FrameLike] = None,
        reject_count: Optional[int] = None,
    ) -> dict[str, Any]:
        updated = dict(metadata)
        updated[f"{self._META_PREFIX}_covariance"] = {
            "x": cov_x,
            "y": cov_y,
            "w": cov_w,
            "h": cov_h,
        }
        updated[f"{self._META_PREFIX}_acceleration"] = acceleration
        updated[f"{self._META_PREFIX}_size_acceleration"] = size_acceleration
        updated[f"{self._META_PREFIX}_last_stage"] = stage
        if reject_count is not None:
            updated[f"{self._META_PREFIX}_reject_count"] = int(reject_count)
        if frame is not None:
            updated[f"{self._META_PREFIX}_last_frame_idx"] = frame.idx
            updated[f"{self._META_PREFIX}_last_timestamp"] = frame.timestamp
            put_frame_shape(updated, f"{self._META_PREFIX}_frame_shape", frame)
        return updated

    def _clamp_from_metadata(
        self,
        metadata: Mapping[str, Any],
        target_pos: Point,
        target_size: Size,
    ) -> tuple[Point, Size]:
        return clamp_center_size(
            target_pos,
            target_size,
            frame_shape=frame_shape_from_metadata(metadata, f"{self._META_PREFIX}_frame_shape"),
            min_size=self.config.min_size,
            clamp_to_frame=self.config.clamp_to_frame,
        )

    def _uncertainty_from_covariances(
        self,
        metadata: Mapping[str, Any],
        mode: TrackerMode,
    ) -> float:
        covariances = self._covariances(metadata)
        diagonal_sum = sum(
            covariance[0] + covariance[4] + covariance[8] for covariance in covariances.values()
        )
        uncertainty = max(0.0, diagonal_sum / 150.0)
        if mode in (TrackerMode.RECOVERY, TrackerMode.LOST):
            uncertainty *= 2.0
        elif mode in (TrackerMode.UNCERTAIN, TrackerMode.OCCLUDED):
            uncertainty *= 1.5
        return clamp_uncertainty(
            uncertainty,
            min_uncertainty=self.config.min_uncertainty,
            max_uncertainty=self.config.max_uncertainty,
        )

    def _measurement_noise_scale(self, score: float) -> float:
        if not self.config.adaptive_measurement_noise:
            return 1.0
        min_scale = max(1e-9, float(self.config.min_measurement_noise_scale))
        max_scale = max(min_scale, float(self.config.max_measurement_noise_scale))
        return max_scale - (max_scale - min_scale) * clamp01(score)

    def _accept_decay(self, score: float) -> float:
        base_decay = clamp01(self.config.uncertainty_accept_decay)
        return 1.0 - (1.0 - base_decay) * clamp01(score)

    def _scale_covariance(self, covariance: Covariance3x3, scale: float) -> Covariance3x3:
        scale = max(0.0, float(scale))
        return tuple(float(value) * scale for value in covariance)  # type: ignore[return-value]

    def _clamp_acceleration_pair(
        self,
        values: Point,
        max_acceleration: Optional[float],
    ) -> Point:
        if max_acceleration is None:
            return float(values[0]), float(values[1])
        limit = abs(float(max_acceleration))
        return clamp(float(values[0]), -limit, limit), clamp(float(values[1]), -limit, limit)
