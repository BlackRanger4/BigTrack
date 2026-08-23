from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from BigTracker.predictor import PredictorModel
from BigTracker.predictor_models._motion import (
    clamp,
    clamp01,
    clamp_pair_abs,
    clamp_uncertainty,
    damp_pair,
    frame_dt,
)
from BigTracker.types import (
    Point,
    PredictorInitializeInput,
    PredictorInitializeOutput,
    PredictorPredictInput,
    PredictorPredictOutput,
    PredictorUpdateInput,
    PredictorUpdateOutput,
    TrackerPredictionState,
)


Covariance3x3 = Tuple[float, float, float, float, float, float, float, float, float]


@dataclass(frozen=True)
class ConstantAccelerationKalmanPredictorConfig:
    """Config for a scalar Kalman filter tracking value, velocity, acceleration."""

    process_noise_position: float = 1.0
    process_noise_velocity: float = 0.5
    process_noise_acceleration: float = 0.25
    measurement_noise_position: float = 4.0
    default_covariance: Covariance3x3 = (10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0)
    adaptive_measurement_noise: bool = True
    min_measurement_noise_scale: float = 0.25
    max_measurement_noise_scale: float = 3.0
    max_position_velocity: Optional[float] = None
    max_position_acceleration: Optional[float] = None
    reject_velocity_damping: float = 0.85
    reject_acceleration_damping: float = 0.65
    reject_uncertainty_growth: float = 1.5
    reject_covariance_growth: float = 1.15
    uncertainty_accept_decay: float = 0.90
    min_uncertainty: float = 0.0
    max_uncertainty: Optional[float] = 100.0


class ConstantAccelerationKalmanPredictorModel(PredictorModel):
    """Kalman predictor with value, velocity, and acceleration per position dimension."""

    _META_PREFIX = "constant_accel_kalman"

    def __init__(self, config: Optional[ConstantAccelerationKalmanPredictorConfig] = None) -> None:
        self.config = config or ConstantAccelerationKalmanPredictorConfig()
        self._state: TrackerPredictionState | None = None

    def initialize(self, request: PredictorInitializeInput) -> PredictorInitializeOutput:
        self._state = request.predictor_state
        return PredictorInitializeOutput(ok=True, metadata=request.metadata)

    def predict(self, request: PredictorPredictInput) -> PredictorPredictOutput:
        prediction = self._require_state()
        dt = frame_dt(prediction.metadata, request.frame, prefix=self._META_PREFIX)
        accelerations = self._accelerations(prediction.metadata)
        covariances = self._covariances(prediction.metadata)
        ax, ay = self._clamp_acceleration_pair(
            accelerations,
            self.config.max_position_acceleration,
        )

        x, vx, ax, cov_x = self._predict_1d(
            prediction.target_pos[0],
            prediction.target_velocity[0],
            ax,
            covariances["x"],
            dt,
        )
        y, vy, ay, cov_y = self._predict_1d(
            prediction.target_pos[1],
            prediction.target_velocity[1],
            ay,
            covariances["y"],
            dt,
        )
        velocity = clamp_pair_abs((vx, vy), self.config.max_position_velocity)
        metadata = self._metadata(
            prediction.metadata,
            cov_x,
            cov_y,
            acceleration=(ax, ay),
            stage="predict",
        )
        metadata[f"{self._META_PREFIX}_last_dt"] = dt
        metadata[f"{self._META_PREFIX}_last_frame_idx"] = request.frame.idx
        metadata[f"{self._META_PREFIX}_last_timestamp"] = request.frame.timestamp

        self._state = TrackerPredictionState(
            target_pos=(x, y),
            target_velocity=velocity,
            uncertainty=self._uncertainty_from_covariances(metadata),
            metadata=metadata,
        )
        return PredictorPredictOutput(predictor_state=self._state, metadata=request.metadata)

    def update(self, request: PredictorUpdateInput) -> PredictorUpdateOutput:
        prediction = request.predictor_state
        metadata = dict(prediction.metadata)

        if request.accepted:
            previous = self._state or prediction
            score = clamp01(float(request.metadata.get("score", request.metadata.get("confidence", 1.0))))
            noise_scale = self._measurement_noise_scale(score)
            noise = self.config.measurement_noise_position * noise_scale
            accelerations = self._accelerations(previous.metadata)
            covariances = self._covariances(previous.metadata)
            x, vx, ax, cov_x = self._update_1d(
                previous.target_pos[0],
                previous.target_velocity[0],
                accelerations[0],
                covariances["x"],
                prediction.target_pos[0],
                noise,
            )
            y, vy, ay, cov_y = self._update_1d(
                previous.target_pos[1],
                previous.target_velocity[1],
                accelerations[1],
                covariances["y"],
                prediction.target_pos[1],
                noise,
            )
            ax, ay = self._clamp_acceleration_pair((ax, ay), self.config.max_position_acceleration)
            decay = self._accept_decay(score)
            metadata = self._metadata(
                metadata,
                self._scale_covariance(cov_x, decay),
                self._scale_covariance(cov_y, decay),
                acceleration=(ax, ay),
                stage="accept",
                reject_count=0,
            )
            metadata[f"{self._META_PREFIX}_last_score"] = score
            metadata[f"{self._META_PREFIX}_measurement_noise_scale"] = noise_scale
            target_pos = (x, y)
            target_velocity = clamp_pair_abs((vx, vy), self.config.max_position_velocity)
            uncertainty = self._uncertainty_from_covariances(metadata)
        else:
            covariances = self._covariances(metadata)
            accelerations = self._accelerations(metadata)
            covariance_growth = max(1.0, float(self.config.reject_covariance_growth))
            acceleration_damping = clamp(self.config.reject_acceleration_damping, 0.0, 1.0)
            metadata = self._metadata(
                metadata,
                self._scale_covariance(covariances["x"], covariance_growth),
                self._scale_covariance(covariances["y"], covariance_growth),
                acceleration=(
                    accelerations[0] * acceleration_damping,
                    accelerations[1] * acceleration_damping,
                ),
                stage="reject",
                reject_count=int(metadata.get(f"{self._META_PREFIX}_reject_count", 0)) + 1,
            )
            target_pos = prediction.target_pos
            target_velocity = clamp_pair_abs(
                damp_pair(prediction.target_velocity, self.config.reject_velocity_damping),
                self.config.max_position_velocity,
            )
            uncertainty = self._uncertainty_from_covariances(metadata) + max(
                0.0,
                float(self.config.reject_uncertainty_growth),
            )

        self._state = TrackerPredictionState(
            target_pos=target_pos,
            target_velocity=target_velocity,
            uncertainty=clamp_uncertainty(
                uncertainty,
                min_uncertainty=self.config.min_uncertainty,
                max_uncertainty=self.config.max_uncertainty,
            ),
            metadata=metadata,
        )
        return PredictorUpdateOutput(ok=True, metadata=request.metadata)

    def reset(self) -> None:
        self._state = None

    def close(self) -> None:
        self.reset()

    def _require_state(self) -> TrackerPredictionState:
        if self._state is None:
            raise RuntimeError("ConstantAccelerationKalmanPredictorModel is not initialized.")
        return self._state

    def _predict_1d(
        self,
        value: float,
        velocity: float,
        acceleration: float,
        covariance: Covariance3x3,
        dt: float,
    ) -> tuple[float, float, float, Covariance3x3]:
        next_value = value + velocity * dt + 0.5 * acceleration * dt * dt
        next_velocity = velocity + acceleration * dt
        return next_value, next_velocity, acceleration, self._predict_covariance(covariance, dt)

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
        return (
            value + gain_value * residual,
            velocity + gain_velocity * residual,
            acceleration + gain_acceleration * residual,
            (
                (1.0 - gain_value) * p00,
                (1.0 - gain_value) * p01,
                (1.0 - gain_value) * p02,
                p10 - gain_velocity * p00,
                p11 - gain_velocity * p01,
                p12 - gain_velocity * p02,
                p20 - gain_acceleration * p00,
                p21 - gain_acceleration * p01,
                p22 - gain_acceleration * p02,
            ),
        )

    def _predict_covariance(self, covariance: Covariance3x3, dt: float) -> Covariance3x3:
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
            fpf_t[0][0] + self.config.process_noise_position,
            fpf_t[0][1],
            fpf_t[0][2],
            fpf_t[1][0],
            fpf_t[1][1] + self.config.process_noise_velocity,
            fpf_t[1][2],
            fpf_t[2][0],
            fpf_t[2][1],
            fpf_t[2][2] + self.config.process_noise_acceleration,
        )

    def _covariances(self, metadata: Mapping[str, Any]) -> Dict[str, Covariance3x3]:
        stored = metadata.get(f"{self._META_PREFIX}_covariance", {})
        return {
            "x": tuple(stored.get("x", self.config.default_covariance)),
            "y": tuple(stored.get("y", self.config.default_covariance)),
        }

    def _accelerations(self, metadata: Mapping[str, Any]) -> Point:
        acceleration = metadata.get(f"{self._META_PREFIX}_acceleration", (0.0, 0.0))
        return float(acceleration[0]), float(acceleration[1])

    def _metadata(
        self,
        metadata: Mapping[str, Any],
        cov_x: Covariance3x3,
        cov_y: Covariance3x3,
        *,
        acceleration: Point,
        stage: str,
        reject_count: Optional[int] = None,
    ) -> dict[str, Any]:
        updated = dict(metadata)
        updated[f"{self._META_PREFIX}_covariance"] = {"x": cov_x, "y": cov_y}
        updated[f"{self._META_PREFIX}_acceleration"] = acceleration
        updated[f"{self._META_PREFIX}_last_stage"] = stage
        if reject_count is not None:
            updated[f"{self._META_PREFIX}_reject_count"] = int(reject_count)
        return updated

    def _uncertainty_from_covariances(self, metadata: Mapping[str, Any]) -> float:
        covariances = self._covariances(metadata)
        diagonal_sum = sum(covariance[0] + covariance[4] + covariance[8] for covariance in covariances.values())
        return clamp_uncertainty(
            max(0.0, diagonal_sum / 150.0),
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
