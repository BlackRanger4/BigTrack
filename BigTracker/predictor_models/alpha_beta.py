from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from BigTracker.predictor import PredictorModel
from BigTracker.predictor_models._motion import (
    clamp01,
    clamp_pair_abs,
    clamp_pair_acceleration,
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


@dataclass(frozen=True)
class AlphaBetaPredictorConfig:
    """Config for a dependency-free alpha-beta motion predictor."""

    alpha_position: float = 0.85
    beta_position: float = 0.20
    max_position_velocity: Optional[float] = None
    max_position_acceleration: Optional[float] = None
    reject_velocity_damping: float = 0.85
    reject_uncertainty_growth: float = 1.0
    accept_uncertainty_decay: float = 0.85
    min_uncertainty: float = 0.0
    max_uncertainty: Optional[float] = 100.0


class AlphaBetaPredictorModel(PredictorModel):
    """Fast alpha-beta predictor for target center motion."""

    _META_PREFIX = "alpha_beta"

    def __init__(self, config: Optional[AlphaBetaPredictorConfig] = None) -> None:
        self.config = config or AlphaBetaPredictorConfig()
        self._state: TrackerPredictionState | None = None

    def initialize(self, request: PredictorInitializeInput) -> PredictorInitializeOutput:
        self._state = request.predictor_state
        return PredictorInitializeOutput(ok=True, metadata=request.metadata)

    def predict(self, request: PredictorPredictInput) -> PredictorPredictOutput:
        prediction = self._require_state()
        dt = frame_dt(prediction.metadata, request.frame, prefix=self._META_PREFIX)
        velocity = clamp_pair_abs(prediction.target_velocity, self.config.max_position_velocity)
        metadata = dict(prediction.metadata)
        metadata[f"{self._META_PREFIX}_last_dt"] = dt
        metadata[f"{self._META_PREFIX}_last_frame_idx"] = request.frame.idx
        metadata[f"{self._META_PREFIX}_last_timestamp"] = request.frame.timestamp
        metadata[f"{self._META_PREFIX}_last_stage"] = "predict"

        self._state = TrackerPredictionState(
            target_pos=(
                prediction.target_pos[0] + velocity[0] * dt,
                prediction.target_pos[1] + velocity[1] * dt,
            ),
            target_velocity=velocity,
            uncertainty=clamp_uncertainty(
                prediction.uncertainty,
                min_uncertainty=self.config.min_uncertainty,
                max_uncertainty=self.config.max_uncertainty,
            ),
            metadata=metadata,
        )
        return PredictorPredictOutput(predictor_state=self._state, metadata=request.metadata)

    def update(self, request: PredictorUpdateInput) -> PredictorUpdateOutput:
        prediction = request.predictor_state
        metadata = dict(prediction.metadata)
        score = clamp01(float(request.metadata.get("score", request.metadata.get("confidence", 1.0))))

        if request.accepted:
            previous = self._state or prediction
            dt = max(1e-6, float(metadata.get(f"{self._META_PREFIX}_last_dt", 1.0)))
            residual = (
                float(prediction.target_pos[0]) - float(previous.target_pos[0]),
                float(prediction.target_pos[1]) - float(previous.target_pos[1]),
            )
            alpha = self._score_weight(self.config.alpha_position, score)
            beta = self._score_weight(self.config.beta_position, score)
            target_pos = (
                previous.target_pos[0] + alpha * residual[0],
                previous.target_pos[1] + alpha * residual[1],
            )
            desired_velocity = (
                previous.target_velocity[0] + beta * residual[0] / dt,
                previous.target_velocity[1] + beta * residual[1] / dt,
            )
            velocity = clamp_pair_acceleration(
                previous.target_velocity,
                desired_velocity,
                max_acceleration=self.config.max_position_acceleration,
                dt=dt,
            )
            velocity = clamp_pair_abs(velocity, self.config.max_position_velocity)
            metadata[f"{self._META_PREFIX}_last_stage"] = "accept"
            metadata[f"{self._META_PREFIX}_last_score"] = score
            metadata[f"{self._META_PREFIX}_reject_count"] = 0
            metadata[f"{self._META_PREFIX}_alpha_position"] = alpha
            metadata[f"{self._META_PREFIX}_beta_position"] = beta
            uncertainty = prediction.uncertainty * self._accept_decay(score)
        else:
            target_pos = prediction.target_pos
            velocity = clamp_pair_abs(
                damp_pair(prediction.target_velocity, self.config.reject_velocity_damping),
                self.config.max_position_velocity,
            )
            metadata[f"{self._META_PREFIX}_last_stage"] = "reject"
            metadata[f"{self._META_PREFIX}_reject_count"] = int(
                metadata.get(f"{self._META_PREFIX}_reject_count", 0)
            ) + 1
            uncertainty = prediction.uncertainty + max(0.0, float(self.config.reject_uncertainty_growth))

        self._state = TrackerPredictionState(
            target_pos=target_pos,
            target_velocity=velocity,
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
            raise RuntimeError("AlphaBetaPredictorModel is not initialized.")
        return self._state

    def _score_weight(self, configured: float, score: float) -> float:
        return clamp01(configured) * (0.25 + 0.75 * clamp01(score))

    def _accept_decay(self, score: float) -> float:
        base_decay = clamp01(self.config.accept_uncertainty_decay)
        return 1.0 - (1.0 - base_decay) * clamp01(score)
