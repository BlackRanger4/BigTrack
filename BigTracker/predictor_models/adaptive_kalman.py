from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from BigTracker.predictor import PredictorModel
from BigTracker.predictor_models._motion import clamp, clamp01, clamp_pair_abs, clamp_uncertainty, damp_pair, frame_dt
from BigTracker.types import (
    PredictorInitializeInput,
    PredictorInitializeOutput,
    PredictorPredictInput,
    PredictorPredictOutput,
    PredictorUpdateInput,
    PredictorUpdateOutput,
    TrackerPredictionState,
)


Covariance2x2 = Tuple[float, float, float, float]


@dataclass(frozen=True)
class AdaptiveKalmanPredictorConfig:
    """Config for a constant-velocity Kalman predictor with score feedback."""

    process_noise_position: float = 1.0
    measurement_noise_position: float = 4.0
    default_covariance: Covariance2x2 = (10.0, 0.0, 0.0, 10.0)
    adaptive_measurement_noise: bool = True
    min_measurement_noise_scale: float = 0.25
    max_measurement_noise_scale: float = 3.0
    min_uncertainty: float = 0.0
    max_uncertainty: Optional[float] = 100.0
    reject_uncertainty_growth: float = 1.5
    reject_covariance_growth: float = 1.15
    reject_velocity_damping: float = 0.85
    max_position_velocity: Optional[float] = None
    uncertainty_accept_decay: float = 0.90


class AdaptiveKalmanPredictorModel(PredictorModel):
    """Constant-velocity Kalman predictor with score-aware accept/reject behavior."""

    _META_PREFIX = "adaptive_kalman"

    def __init__(self, config: Optional[AdaptiveKalmanPredictorConfig] = None) -> None:
        self.config = config or AdaptiveKalmanPredictorConfig()
        self._state: TrackerPredictionState | None = None

    def initialize(self, request: PredictorInitializeInput) -> PredictorInitializeOutput:
        self._state = request.predictor_state
        return PredictorInitializeOutput(ok=True, metadata=request.metadata)

    def predict(self, request: PredictorPredictInput) -> PredictorPredictOutput:
        prediction = self._require_state()
        dt = frame_dt(prediction.metadata, request.frame, prefix=self._META_PREFIX)
        covariances = self._covariances(prediction.metadata)
        vx0, vy0 = clamp_pair_abs(prediction.target_velocity, self.config.max_position_velocity)
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
        metadata = self._metadata(prediction.metadata, cov_x, cov_y, stage="predict")
        metadata[f"{self._META_PREFIX}_last_dt"] = dt
        metadata[f"{self._META_PREFIX}_last_frame_idx"] = request.frame.idx
        metadata[f"{self._META_PREFIX}_last_timestamp"] = request.frame.timestamp

        self._state = TrackerPredictionState(
            target_pos=(x, y),
            target_velocity=clamp_pair_abs((vx, vy), self.config.max_position_velocity),
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
            covariances = self._covariances(previous.metadata)
            x, vx, cov_x = self._update_1d(
                previous.target_pos[0],
                previous.target_velocity[0],
                covariances["x"],
                prediction.target_pos[0],
                noise,
            )
            y, vy, cov_y = self._update_1d(
                previous.target_pos[1],
                previous.target_velocity[1],
                covariances["y"],
                prediction.target_pos[1],
                noise,
            )
            decay = self._accept_decay(score)
            metadata = self._metadata(
                metadata,
                self._scale_covariance(cov_x, decay),
                self._scale_covariance(cov_y, decay),
                stage="accept",
                reject_count=0,
            )
            metadata[f"{self._META_PREFIX}_last_score"] = score
            metadata[f"{self._META_PREFIX}_measurement_noise_scale"] = noise_scale
            metadata[f"{self._META_PREFIX}_accept_decay"] = decay
            target_pos = (x, y)
            target_velocity = clamp_pair_abs((vx, vy), self.config.max_position_velocity)
            uncertainty = self._uncertainty_from_covariances(metadata)
        else:
            covariances = self._covariances(metadata)
            covariance_growth = max(1.0, float(self.config.reject_covariance_growth))
            metadata = self._metadata(
                metadata,
                self._scale_covariance(covariances["x"], covariance_growth),
                self._scale_covariance(covariances["y"], covariance_growth),
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
        return PredictorUpdateOutput(ok=True, predictor_state=self._state, metadata=request.metadata)

    def reset(self) -> None:
        self._state = None

    def close(self) -> None:
        self.reset()

    def _require_state(self) -> TrackerPredictionState:
        if self._state is None:
            raise RuntimeError("AdaptiveKalmanPredictorModel is not initialized.")
        return self._state

    def _predict_1d(
        self,
        value: float,
        velocity: float,
        covariance: Covariance2x2,
        dt: float,
        process_noise: float,
    ) -> Tuple[float, float, Covariance2x2]:
        p00, p01, p10, p11 = covariance
        return (
            value + velocity * dt,
            velocity,
            (
                p00 + dt * (p10 + p01) + dt * dt * p11 + process_noise,
                p01 + dt * p11,
                p10 + dt * p11,
                p11 + process_noise,
            ),
        )

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
        return (
            value + gain_value * residual,
            velocity + gain_velocity * residual,
            (
                (1.0 - gain_value) * p00,
                (1.0 - gain_value) * p01,
                p10 - gain_velocity * p00,
                p11 - gain_velocity * p01,
            ),
        )

    def _covariances(self, metadata: Mapping[str, Any]) -> Dict[str, Covariance2x2]:
        stored = metadata.get(f"{self._META_PREFIX}_covariance", {})
        return {
            "x": tuple(stored.get("x", self.config.default_covariance)),
            "y": tuple(stored.get("y", self.config.default_covariance)),
        }

    def _metadata(
        self,
        metadata: Mapping[str, Any],
        cov_x: Covariance2x2,
        cov_y: Covariance2x2,
        *,
        stage: str,
        reject_count: Optional[int] = None,
    ) -> dict[str, Any]:
        updated = dict(metadata)
        updated[f"{self._META_PREFIX}_covariance"] = {"x": cov_x, "y": cov_y}
        updated[f"{self._META_PREFIX}_last_stage"] = stage
        if reject_count is not None:
            updated[f"{self._META_PREFIX}_reject_count"] = int(reject_count)
        return updated

    def _uncertainty_from_covariances(self, metadata: Mapping[str, Any]) -> float:
        covariances = self._covariances(metadata)
        diagonal_sum = sum(covariance[0] + covariance[3] for covariance in covariances.values())
        return clamp_uncertainty(
            max(0.0, diagonal_sum / 100.0),
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
        base_decay = clamp(float(self.config.uncertainty_accept_decay), 0.0, 1.0)
        return 1.0 - (1.0 - base_decay) * clamp01(score)

    def _scale_covariance(self, covariance: Covariance2x2, scale: float) -> Covariance2x2:
        scale = max(0.0, float(scale))
        return tuple(float(value) * scale for value in covariance)  # type: ignore[return-value]
