from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

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


HistoryRecord = dict[str, Any]


@dataclass(frozen=True)
class HistoryPredictorConfig:
    """Config for a bounded-history predictor with smoothed velocity."""

    history_length: int = 8
    velocity_window: int = 4
    velocity_smoothing: float = 0.60
    size_velocity_smoothing: float = 0.60
    min_size: Size = (1.0, 1.0)
    max_position_velocity: Optional[float] = None
    max_size_velocity: Optional[float] = None
    max_position_acceleration: Optional[float] = None
    max_size_acceleration: Optional[float] = None
    reject_velocity_damping: float = 0.85
    reject_uncertainty_growth: float = 1.0
    accept_uncertainty_decay: float = 0.90
    min_uncertainty: float = 0.0
    max_uncertainty: Optional[float] = 100.0
    clamp_to_frame: bool = True


class HistoryPredictorModel(PredictorModel):
    """Predict from recent accepted target centers and sizes."""

    _META_PREFIX = "history_predictor"

    def __init__(self, config: Optional[HistoryPredictorConfig] = None) -> None:
        self.config = config or HistoryPredictorConfig()

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
        metadata[f"{self._META_PREFIX}_last_frame_idx"] = frame.idx
        metadata[f"{self._META_PREFIX}_last_timestamp"] = frame.timestamp
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
        score = clamp01(score)
        frame_idx = int(metadata.get(f"{self._META_PREFIX}_last_frame_idx", state.output.frame_idx + 1))
        timestamp = float(
            metadata.get(
                f"{self._META_PREFIX}_last_timestamp",
                state.output.timestamp + max(1e-6, float(metadata.get(f"{self._META_PREFIX}_last_dt", 1.0))),
            )
        )
        target_pos, target_size = self._clamp_from_metadata(metadata, accepted_pos, accepted_size)
        history = self._append_history(
            self._history(metadata),
            {
                "pos": target_pos,
                "size": target_size,
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "score": score,
            },
        )
        desired_velocity, desired_size_velocity = self._velocity_from_history(history)
        velocity = self._smooth_velocity(prediction.target_velocity, desired_velocity)
        velocity = clamp_pair_acceleration(
            prediction.target_velocity,
            velocity,
            max_acceleration=self.config.max_position_acceleration,
            dt=max(1e-6, float(metadata.get(f"{self._META_PREFIX}_last_dt", 1.0))),
        )
        velocity = clamp_pair_abs(velocity, self.config.max_position_velocity)
        size_velocity = self._smooth_size_velocity(
            prediction.target_size_velocity,
            desired_size_velocity,
        )
        size_velocity = clamp_size_acceleration(
            prediction.target_size_velocity,
            size_velocity,
            max_acceleration=self.config.max_size_acceleration,
            dt=max(1e-6, float(metadata.get(f"{self._META_PREFIX}_last_dt", 1.0))),
        )
        size_velocity = clamp_size_pair_abs(size_velocity, self.config.max_size_velocity)

        metadata[f"{self._META_PREFIX}_history"] = history
        metadata[f"{self._META_PREFIX}_last_stage"] = "accept"
        metadata[f"{self._META_PREFIX}_last_score"] = score
        metadata[f"{self._META_PREFIX}_reject_count"] = 0

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

    def _velocity_from_history(self, history: Sequence[HistoryRecord]) -> tuple[Point, Size]:
        if len(history) < 2:
            return (0.0, 0.0), (0.0, 0.0)

        window = max(2, int(self.config.velocity_window))
        selected = tuple(history[-window:])
        first = selected[0]
        last = selected[-1]
        dt = float(last["timestamp"]) - float(first["timestamp"])
        if dt <= 1e-6:
            dt = max(1.0, float(last["frame_idx"]) - float(first["frame_idx"]))

        first_pos = first["pos"]
        last_pos = last["pos"]
        first_size = first["size"]
        last_size = last["size"]
        return (
            ((float(last_pos[0]) - float(first_pos[0])) / dt, (float(last_pos[1]) - float(first_pos[1])) / dt),
            (
                (float(last_size[0]) - float(first_size[0])) / dt,
                (float(last_size[1]) - float(first_size[1])) / dt,
            ),
        )

    def _smooth_velocity(self, previous: Point, desired: Point) -> Point:
        weight = clamp01(self.config.velocity_smoothing)
        return (
            previous[0] * (1.0 - weight) + desired[0] * weight,
            previous[1] * (1.0 - weight) + desired[1] * weight,
        )

    def _smooth_size_velocity(self, previous: Size, desired: Size) -> Size:
        weight = clamp01(self.config.size_velocity_smoothing)
        return (
            previous[0] * (1.0 - weight) + desired[0] * weight,
            previous[1] * (1.0 - weight) + desired[1] * weight,
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
