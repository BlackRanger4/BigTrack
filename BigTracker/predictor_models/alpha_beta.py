from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from BigTracker.predictor import PredictorModel
from BigTracker.predictor_models._motion import (
    clamp01,
    clamp_center_size,
    clamp_pair_abs,
    clamp_pair_acceleration,
    clamp_size_acceleration,
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


@dataclass(frozen=True)
class AlphaBetaPredictorConfig:
    """Config for a dependency-free alpha-beta motion predictor."""

    alpha_position: float = 0.85
    beta_position: float = 0.20
    alpha_size: float = 0.80
    beta_size: float = 0.15
    min_size: Size = (1.0, 1.0)
    max_position_velocity: Optional[float] = None
    max_size_velocity: Optional[float] = None
    max_position_acceleration: Optional[float] = None
    max_size_acceleration: Optional[float] = None
    reject_velocity_damping: float = 0.85
    reject_uncertainty_growth: float = 1.0
    accept_uncertainty_decay: float = 0.85
    min_uncertainty: float = 0.0
    max_uncertainty: Optional[float] = 100.0
    clamp_to_frame: bool = True


class AlphaBetaPredictorModel(PredictorModel):
    """Fast alpha-beta predictor for target center and size."""

    _META_PREFIX = "alpha_beta"

    def __init__(self, config: Optional[AlphaBetaPredictorConfig] = None) -> None:
        self.config = config or AlphaBetaPredictorConfig()

    def predict(self, state: BigTrackState, frame: FrameLike) -> TrackerPredictionState:
        prediction = state.prediction
        dt = frame_dt(state, frame)
        velocity = clamp_pair_abs(prediction.target_velocity, self.config.max_position_velocity)
        size_velocity = clamp_size_pair_abs(
            prediction.target_size_velocity,
            self.config.max_size_velocity,
        )
        target_pos = (
            prediction.target_pos[0] + velocity[0] * dt,
            prediction.target_pos[1] + velocity[1] * dt,
        )
        target_size = (
            prediction.target_size[0] + size_velocity[0] * dt,
            prediction.target_size[1] + size_velocity[1] * dt,
        )
        frame_shape = frame_shape_from_frame(frame)
        target_pos, target_size = clamp_center_size(
            target_pos,
            target_size,
            frame_shape=frame_shape,
            min_size=self.config.min_size,
            clamp_to_frame=self.config.clamp_to_frame,
        )
        metadata = dict(prediction.metadata)
        metadata[f"{self._META_PREFIX}_last_dt"] = dt
        metadata[f"{self._META_PREFIX}_last_stage"] = "predict"
        put_frame_shape(metadata, f"{self._META_PREFIX}_frame_shape", frame)

        return TrackerPredictionState(
            target_pos=target_pos,
            target_size=target_size,
            target_velocity=velocity,
            target_size_velocity=size_velocity,
            last_score=prediction.last_score,
            uncertainty=self._mode_uncertainty(prediction.uncertainty, state.mode),
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
        dt = max(1e-6, float(metadata.get(f"{self._META_PREFIX}_last_dt", 1.0)))
        score = clamp01(score)

        residual_pos = (
            float(accepted_pos[0]) - float(prediction.target_pos[0]),
            float(accepted_pos[1]) - float(prediction.target_pos[1]),
        )
        residual_size = (
            float(accepted_size[0]) - float(prediction.target_size[0]),
            float(accepted_size[1]) - float(prediction.target_size[1]),
        )
        alpha_pos = self._score_weight(self.config.alpha_position, score)
        beta_pos = self._score_weight(self.config.beta_position, score)
        alpha_size = self._score_weight(self.config.alpha_size, score)
        beta_size = self._score_weight(self.config.beta_size, score)

        target_pos = (
            prediction.target_pos[0] + alpha_pos * residual_pos[0],
            prediction.target_pos[1] + alpha_pos * residual_pos[1],
        )
        target_size = (
            prediction.target_size[0] + alpha_size * residual_size[0],
            prediction.target_size[1] + alpha_size * residual_size[1],
        )
        desired_velocity = (
            prediction.target_velocity[0] + beta_pos * residual_pos[0] / dt,
            prediction.target_velocity[1] + beta_pos * residual_pos[1] / dt,
        )
        desired_size_velocity = (
            prediction.target_size_velocity[0] + beta_size * residual_size[0] / dt,
            prediction.target_size_velocity[1] + beta_size * residual_size[1] / dt,
        )
        velocity = clamp_pair_acceleration(
            prediction.target_velocity,
            desired_velocity,
            max_acceleration=self.config.max_position_acceleration,
            dt=dt,
        )
        velocity = clamp_pair_abs(velocity, self.config.max_position_velocity)
        size_velocity = clamp_size_acceleration(
            prediction.target_size_velocity,
            desired_size_velocity,
            max_acceleration=self.config.max_size_acceleration,
            dt=dt,
        )
        size_velocity = clamp_size_pair_abs(size_velocity, self.config.max_size_velocity)
        target_pos, target_size = self._clamp_from_metadata(metadata, target_pos, target_size)

        metadata[f"{self._META_PREFIX}_last_stage"] = "accept"
        metadata[f"{self._META_PREFIX}_last_score"] = score
        metadata[f"{self._META_PREFIX}_reject_count"] = 0
        metadata[f"{self._META_PREFIX}_alpha_position"] = alpha_pos
        metadata[f"{self._META_PREFIX}_beta_position"] = beta_pos

        return TrackerPredictionState(
            target_pos=target_pos,
            target_size=target_size,
            target_velocity=velocity,
            target_size_velocity=size_velocity,
            last_score=score,
            uncertainty=clamp_uncertainty(
                prediction.uncertainty * self._accept_decay(score),
                min_uncertainty=self.config.min_uncertainty,
                max_uncertainty=self.config.max_uncertainty,
            ),
            metadata=metadata,
        )

    def update_from_reject(self, state: BigTrackState) -> TrackerPredictionState:
        prediction = state.prediction
        metadata = dict(prediction.metadata)
        metadata[f"{self._META_PREFIX}_last_stage"] = "reject"
        metadata[f"{self._META_PREFIX}_reject_count"] = int(
            metadata.get(f"{self._META_PREFIX}_reject_count", 0)
        ) + 1

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
                prediction.uncertainty + max(0.0, float(self.config.reject_uncertainty_growth)),
                min_uncertainty=self.config.min_uncertainty,
                max_uncertainty=self.config.max_uncertainty,
            ),
            metadata=metadata,
        )

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

    def _score_weight(self, configured: float, score: float) -> float:
        return clamp01(configured) * (0.25 + 0.75 * clamp01(score))

    def _accept_decay(self, score: float) -> float:
        base_decay = clamp01(self.config.accept_uncertainty_decay)
        return 1.0 - (1.0 - base_decay) * clamp01(score)

    def _mode_uncertainty(self, uncertainty: float, mode: TrackerMode) -> float:
        if mode in (TrackerMode.RECOVERY, TrackerMode.LOST):
            uncertainty *= 2.0
        elif mode in (TrackerMode.UNCERTAIN, TrackerMode.OCCLUDED):
            uncertainty *= 1.5
        return clamp_uncertainty(
            uncertainty,
            min_uncertainty=self.config.min_uncertainty,
            max_uncertainty=self.config.max_uncertainty,
        )
