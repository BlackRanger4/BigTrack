from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

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


HistoryRecord = dict[str, Any]


@dataclass(frozen=True)
class HistoryPredictorConfig:
    """Config for a bounded-history predictor with smoothed velocity."""

    history_length: int = 8
    velocity_window: int = 4
    velocity_smoothing: float = 0.60
    max_position_velocity: Optional[float] = None
    max_position_acceleration: Optional[float] = None
    reject_velocity_damping: float = 0.85
    reject_uncertainty_growth: float = 1.0
    accept_uncertainty_decay: float = 0.90
    min_uncertainty: float = 0.0
    max_uncertainty: Optional[float] = 100.0


class HistoryPredictorModel(PredictorModel):
    """Predict target center from recent accepted target centers."""

    _META_PREFIX = "history_predictor"

    def __init__(self, config: Optional[HistoryPredictorConfig] = None) -> None:
        self.config = config or HistoryPredictorConfig()
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
            history = self._append_history(
                self._history(metadata),
                {
                    "pos": prediction.target_pos,
                    "frame_idx": metadata.get(f"{self._META_PREFIX}_last_frame_idx"),
                    "timestamp": metadata.get(f"{self._META_PREFIX}_last_timestamp"),
                    "score": score,
                },
            )
            desired_velocity = self._velocity_from_history(history)
            dt = max(1e-6, float(metadata.get(f"{self._META_PREFIX}_last_dt", 1.0)))
            velocity = self._smooth_velocity(prediction.target_velocity, desired_velocity)
            velocity = clamp_pair_acceleration(
                prediction.target_velocity,
                velocity,
                max_acceleration=self.config.max_position_acceleration,
                dt=dt,
            )
            velocity = clamp_pair_abs(velocity, self.config.max_position_velocity)
            metadata[f"{self._META_PREFIX}_history"] = history
            metadata[f"{self._META_PREFIX}_last_stage"] = "accept"
            metadata[f"{self._META_PREFIX}_last_score"] = score
            metadata[f"{self._META_PREFIX}_reject_count"] = 0
            uncertainty = prediction.uncertainty * self._accept_decay(score)
        else:
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
            target_pos=prediction.target_pos,
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
            raise RuntimeError("HistoryPredictorModel is not initialized.")
        return self._state

    def _history(self, metadata: Mapping[str, Any]) -> tuple[HistoryRecord, ...]:
        records = metadata.get(f"{self._META_PREFIX}_history", ())
        return tuple(dict(record) for record in records)

    def _append_history(
        self,
        history: Sequence[HistoryRecord],
        record: HistoryRecord,
    ) -> tuple[HistoryRecord, ...]:
        max_length = max(2, int(self.config.history_length))
        return tuple(history[-(max_length - 1) :]) + (record,)

    def _velocity_from_history(self, history: Sequence[HistoryRecord]) -> Point:
        if len(history) < 2:
            return (0.0, 0.0)

        window = max(2, int(self.config.velocity_window))
        selected = tuple(history[-window:])
        first = selected[0]
        last = selected[-1]
        dt = float(last.get("timestamp") or 0.0) - float(first.get("timestamp") or 0.0)
        if dt <= 1e-6:
            dt = max(1.0, float(last.get("frame_idx") or 0.0) - float(first.get("frame_idx") or 0.0))

        first_pos = first["pos"]
        last_pos = last["pos"]
        return (
            (float(last_pos[0]) - float(first_pos[0])) / dt,
            (float(last_pos[1]) - float(first_pos[1])) / dt,
        )

    def _smooth_velocity(self, previous: Point, desired: Point) -> Point:
        weight = clamp01(self.config.velocity_smoothing)
        return (
            previous[0] * (1.0 - weight) + desired[0] * weight,
            previous[1] * (1.0 - weight) + desired[1] * weight,
        )

    def _accept_decay(self, score: float) -> float:
        base_decay = clamp01(self.config.accept_uncertainty_decay)
        return 1.0 - (1.0 - base_decay) * clamp01(score)
