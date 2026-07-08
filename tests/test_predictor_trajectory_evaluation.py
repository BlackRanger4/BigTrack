from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
import random
import statistics
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from BigTracker.predictor_models import (
    AdaptiveKalmanPredictorConfig,
    AdaptiveKalmanPredictorModel,
    AlphaBetaPredictorConfig,
    AlphaBetaPredictorModel,
    ConstantAccelerationKalmanPredictorConfig,
    ConstantAccelerationKalmanPredictorModel,
    HistoryPredictorConfig,
    HistoryPredictorModel,
    KalmanPredictorConfig,
    KalmanPredictorModel,
)
from BigTracker.state import (
    BigTrackState,
    MatcherState,
    TrackerPredictionState,
    TrackingOutput,
)
from BigTracker.types import OutputStatus, TrackerMode


FRAME_SIZE = (640.0, 360.0)
FRAME_COUNT = 180
SEEDS = tuple(range(1337, 1347))
SCENARIOS = (
    "mixed_accel",
    "constant_velocity",
    "constant_acceleration",
    "zig_zag",
    "sine_wave",
    "center_direct",
    "center_orbit",
    "random_maneuver",
)


@dataclass(frozen=True)
class _Image:
    shape: tuple[int, int, int]


@dataclass(frozen=True)
class _Frame:
    image: object
    idx: int
    timestamp: float


@dataclass(frozen=True)
class _Truth:
    frame_idx: int
    timestamp: float
    pos: tuple[float, float]
    size: tuple[float, float]
    velocity: tuple[float, float]
    size_velocity: tuple[float, float]


@dataclass(frozen=True)
class _Measurement:
    available: bool
    pos: tuple[float, float]
    size: tuple[float, float]
    score: float
    kind: str


@dataclass(frozen=True)
class _Metrics:
    name: str
    position_rmse: float
    size_rmse: float
    prediction_box_rmse: float
    accept_position_rmse: float
    max_position_error: float
    reject_position_rmse: float
    reject_frames: int
    accepted_frames: int
    final_position_error: float
    avg_uncertainty: float


@dataclass(frozen=True)
class _MetricSummary:
    average: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class _AggregateMetrics:
    name: str
    position_rmse: _MetricSummary
    size_rmse: _MetricSummary
    prediction_box_rmse: _MetricSummary
    accept_position_rmse: _MetricSummary
    max_position_error: _MetricSummary
    reject_position_rmse: _MetricSummary
    final_position_error: _MetricSummary
    avg_uncertainty: _MetricSummary


class PredictorTrajectoryEvaluationTest(unittest.TestCase):
    def test_predictors_handle_complex_noisy_trajectory(self) -> None:
        results = run_evaluation()

        self.assertEqual(set(results), set(_predictors()))
        for name, metrics in results.items():
            with self.subTest(predictor=name):
                self.assertLess(metrics.position_rmse.maximum, 70.0)
                self.assertLess(metrics.size_rmse.maximum, 35.0)
                self.assertLess(metrics.prediction_box_rmse.maximum, 80.0)
                self.assertLessEqual(metrics.position_rmse.minimum, metrics.position_rmse.average)
                self.assertLessEqual(metrics.position_rmse.average, metrics.position_rmse.maximum)


def run_evaluation() -> dict[str, _AggregateMetrics]:
    runs, _ = _collect_runs()
    return {name: _aggregate_metrics(name, metrics) for name, metrics in runs.items()}


def render_report() -> str:
    all_runs, scenario_runs = _collect_runs()
    results = {name: _aggregate_metrics(name, metrics) for name, metrics in all_runs.items()}
    scenario_results = {
        scenario: {
            name: _aggregate_metrics(name, metrics)
            for name, metrics in predictor_runs.items()
        }
        for scenario, predictor_runs in scenario_runs.items()
    }
    ordered = sorted(results.values(), key=lambda item: item.position_rmse.average)
    accepted = FRAME_COUNT - 1 - 24
    rejected = 24
    run_count = len(SEEDS) * len(SCENARIOS)

    lines = [
        "# Predictor Trajectory Evaluation Report",
        "",
        "Deterministic synthetic evaluation for the BigTracker predictor models.",
        "",
        "## Scenario",
        "",
        f"- Runs: `{run_count}`",
        f"- Seeds: `{SEEDS[0]}-{SEEDS[-1]}`",
        f"- Motion scenarios: `{len(SCENARIOS)}`",
        f"- Frames: `{FRAME_COUNT}`",
        f"- Frame size: `{int(FRAME_SIZE[0])}x{int(FRAME_SIZE[1])}`",
        "- Motion: mixed acceleration, constant speed, constant acceleration, zig-zag, sine wave, direct center return, circular center return, and random maneuvers.",
        "- Measurements: Gaussian noise, structured occlusion windows, and deterministic outliers.",
        f"- Accepted measurements per run: `{accepted}`",
        f"- Rejected measurements per run: `{rejected}`",
        "",
        "## Metrics",
        "",
        "Each cell is `average (minimum-maximum)` across all scenario and seed runs.",
        "",
        "| Predictor | Position RMSE | Size RMSE | Box RMSE | Accept Pos RMSE | Reject Pos RMSE | Max Pos Err | Final Pos Err | Avg Uncertainty |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metrics in ordered:
        lines.append(
            "| "
            f"{metrics.name} | "
            f"{_format_summary(metrics.position_rmse)} | "
            f"{_format_summary(metrics.size_rmse)} | "
            f"{_format_summary(metrics.prediction_box_rmse)} | "
            f"{_format_summary(metrics.accept_position_rmse)} | "
            f"{_format_summary(metrics.reject_position_rmse)} | "
            f"{_format_summary(metrics.max_position_error)} | "
            f"{_format_summary(metrics.final_position_error)} | "
            f"{_format_summary(metrics.avg_uncertainty)} |"
        )

    predictor_order = [metrics.name for metrics in ordered]
    lines.extend(
        [
            "",
            "## Position RMSE By Scenario",
            "",
            "Each cell is average position RMSE across the ten seeds for that motion scenario.",
            "",
            "| Scenario | " + " | ".join(predictor_order) + " |",
            "|---|" + "---:|" * len(predictor_order),
        ]
    )
    for scenario in SCENARIOS:
        row = [scenario]
        for name in predictor_order:
            row.append(f"{scenario_results[scenario][name].position_rmse.average:.3f}")
        lines.append("| " + " | ".join(row) + " |")

    best = ordered[0]
    lines.extend(
        [
            "",
            "## Notes",
            "",
            f"- Best average position RMSE: `{best.name}` at `{best.position_rmse.average:.3f}` pixels.",
            "- This is a synthetic stress test, not a final predictor ranking.",
            "- All scenario and seed combinations are fixed, so the aggregate report is reproducible.",
            "- `Reject Pos RMSE` measures prediction during occlusion/rejected measurement frames.",
        ]
    )
    return "\n".join(lines) + "\n"


def _collect_runs() -> tuple[dict[str, list[_Metrics]], dict[str, dict[str, list[_Metrics]]]]:
    all_runs: dict[str, list[_Metrics]] = {name: [] for name in _predictors()}
    scenario_runs: dict[str, dict[str, list[_Metrics]]] = {
        scenario: {name: [] for name in _predictors()} for scenario in SCENARIOS
    }
    for scenario in SCENARIOS:
        for seed in SEEDS:
            truth = _make_truth(seed, scenario)
            measurements = _make_measurements(truth, seed)
            for name, predictor in _predictors().items():
                metrics = _run_predictor(name, predictor, truth, measurements)
                all_runs[name].append(metrics)
                scenario_runs[scenario][name].append(metrics)
    return all_runs, scenario_runs


def _aggregate_metrics(name: str, runs: list[_Metrics]) -> _AggregateMetrics:
    def summarize(field_name: str) -> _MetricSummary:
        values = [float(getattr(run, field_name)) for run in runs]
        return _MetricSummary(
            average=statistics.fmean(values),
            minimum=min(values),
            maximum=max(values),
        )

    return _AggregateMetrics(
        name=name,
        position_rmse=summarize("position_rmse"),
        size_rmse=summarize("size_rmse"),
        prediction_box_rmse=summarize("prediction_box_rmse"),
        accept_position_rmse=summarize("accept_position_rmse"),
        max_position_error=summarize("max_position_error"),
        reject_position_rmse=summarize("reject_position_rmse"),
        final_position_error=summarize("final_position_error"),
        avg_uncertainty=summarize("avg_uncertainty"),
    )


def _format_summary(summary: _MetricSummary) -> str:
    return f"{summary.average:.3f} ({summary.minimum:.3f}-{summary.maximum:.3f})"


def _predictors():
    return {
        "kalman": KalmanPredictorModel(
            KalmanPredictorConfig(
                process_noise_position=1.2,
                process_noise_size=0.7,
                measurement_noise_position=7.0,
                measurement_noise_size=4.0,
                reject_uncertainty_growth=1.2,
                clamp_to_frame=True,
            )
        ),
        "adaptive_kalman": AdaptiveKalmanPredictorModel(
            AdaptiveKalmanPredictorConfig(
                process_noise_position=1.2,
                process_noise_size=0.7,
                measurement_noise_position=7.0,
                measurement_noise_size=4.0,
                adaptive_measurement_noise=True,
                reject_velocity_damping=0.90,
                max_position_velocity=30.0,
                max_size_velocity=6.0,
                clamp_to_frame=True,
            )
        ),
        "alpha_beta": AlphaBetaPredictorModel(
            AlphaBetaPredictorConfig(
                alpha_position=0.72,
                beta_position=0.18,
                alpha_size=0.65,
                beta_size=0.12,
                max_position_velocity=30.0,
                max_size_velocity=6.0,
                max_position_acceleration=8.0,
                max_size_acceleration=2.5,
                reject_velocity_damping=0.90,
                clamp_to_frame=True,
            )
        ),
        "history": HistoryPredictorModel(
            HistoryPredictorConfig(
                history_length=8,
                velocity_window=4,
                velocity_smoothing=0.70,
                size_velocity_smoothing=0.65,
                max_position_velocity=30.0,
                max_size_velocity=6.0,
                max_position_acceleration=8.0,
                max_size_acceleration=2.5,
                reject_velocity_damping=0.90,
                clamp_to_frame=True,
            )
        ),
        "constant_accel_kalman": ConstantAccelerationKalmanPredictorModel(
            ConstantAccelerationKalmanPredictorConfig(
                process_noise_position=1.2,
                process_noise_size=0.7,
                process_noise_velocity=0.7,
                process_noise_size_velocity=0.25,
                process_noise_acceleration=0.35,
                process_noise_size_acceleration=0.12,
                measurement_noise_position=8.0,
                measurement_noise_size=4.0,
                adaptive_measurement_noise=True,
                max_position_velocity=30.0,
                max_size_velocity=6.0,
                max_position_acceleration=8.0,
                max_size_acceleration=2.5,
                reject_velocity_damping=0.90,
                reject_acceleration_damping=0.70,
                clamp_to_frame=True,
            )
        ),
    }


def _run_predictor(
    name: str,
    predictor,
    truth: list[_Truth],
    measurements: list[_Measurement],
) -> _Metrics:
    first = truth[0]
    state = _state_from_prediction(
        TrackerPredictionState(
            target_pos=first.pos,
            target_size=first.size,
            target_velocity=first.velocity,
            target_size_velocity=first.size_velocity,
            last_score=1.0,
            uncertainty=0.0,
        ),
        frame_idx=first.frame_idx,
        timestamp=first.timestamp,
    )
    position_errors: list[float] = []
    size_errors: list[float] = []
    box_errors: list[float] = []
    accept_position_errors: list[float] = []
    reject_position_errors: list[float] = []
    uncertainties: list[float] = []
    accepted_frames = 0
    reject_frames = 0
    final_position_error = 0.0

    for current_truth, measurement in zip(truth[1:], measurements[1:]):
        frame = _frame(current_truth.frame_idx, current_truth.timestamp)
        predicted = predictor.predict(state, frame)
        position_error = _point_error(predicted.target_pos, current_truth.pos)
        size_error = _size_error(predicted.target_size, current_truth.size)
        box_error = _box_error(predicted.target_pos, predicted.target_size, current_truth)
        position_errors.append(position_error)
        size_errors.append(size_error)
        box_errors.append(box_error)
        uncertainties.append(predicted.uncertainty)
        predicted_state = replace(state, prediction=predicted)

        if measurement.available:
            next_prediction = predictor.update_from_accept(
                predicted_state,
                accepted_pos=measurement.pos,
                accepted_size=measurement.size,
                score=measurement.score,
            )
            accepted_frames += 1
            accept_position_errors.append(position_error)
            output_status = OutputStatus.ACTIVE
            mode = TrackerMode.TRACKING
        else:
            next_prediction = predictor.update_from_reject(predicted_state)
            reject_frames += 1
            reject_position_errors.append(position_error)
            output_status = OutputStatus.OCCLUDED
            mode = TrackerMode.OCCLUDED

        state = _state_from_prediction(
            next_prediction,
            frame_idx=current_truth.frame_idx,
            timestamp=current_truth.timestamp,
            status=output_status,
            mode=mode,
        )
        final_position_error = _point_error(next_prediction.target_pos, current_truth.pos)

    return _Metrics(
        name=name,
        position_rmse=_rmse(position_errors),
        size_rmse=_rmse(size_errors),
        prediction_box_rmse=_rmse(box_errors),
        accept_position_rmse=_rmse(accept_position_errors),
        max_position_error=max(position_errors),
        reject_position_rmse=_rmse(reject_position_errors),
        reject_frames=reject_frames,
        accepted_frames=accepted_frames,
        final_position_error=final_position_error,
        avg_uncertainty=sum(uncertainties) / len(uncertainties),
    )


def _make_truth(seed: int, scenario: str) -> list[_Truth]:
    rng = random.Random(seed)
    pos = (125.0 + rng.uniform(-8.0, 8.0), 95.0 + rng.uniform(-6.0, 6.0))
    velocity = (4.5 + rng.uniform(-0.4, 0.4), 2.0 + rng.uniform(-0.3, 0.3))
    size = (48.0, 58.0)
    size_velocity = (0.25, -0.05)
    truth = [_Truth(0, 0.0, pos, size, velocity, size_velocity)]

    for idx in range(1, FRAME_COUNT):
        acceleration = _scenario_acceleration(scenario, idx, pos, velocity, rng)
        velocity = (
            max(-13.0, min(13.0, velocity[0] + acceleration[0])),
            max(-10.0, min(10.0, velocity[1] + acceleration[1])),
        )
        pos = (pos[0] + velocity[0], pos[1] + velocity[1])
        size_acceleration = (
            0.05 * math.sin(idx / 10.0) + rng.gauss(0.0, 0.03),
            0.04 * math.cos(idx / 12.0) + rng.gauss(0.0, 0.03),
        )
        size_velocity = (
            max(-1.8, min(1.8, size_velocity[0] + size_acceleration[0])),
            max(-1.5, min(1.5, size_velocity[1] + size_acceleration[1])),
        )
        size = (
            max(18.0, min(95.0, size[0] + size_velocity[0])),
            max(22.0, min(105.0, size[1] + size_velocity[1])),
        )
        pos, velocity = _bounce_position(pos, velocity, size)
        truth.append(_Truth(idx, float(idx), pos, size, velocity, size_velocity))

    return truth


def _scenario_acceleration(
    scenario: str,
    idx: int,
    pos: tuple[float, float],
    velocity: tuple[float, float],
    rng: random.Random,
) -> tuple[float, float]:
    if scenario == "mixed_accel":
        turn_x = 1.8 * math.sin(idx / 14.0) + 0.9 * math.sin(idx / 5.5)
        turn_y = 1.4 * math.cos(idx / 18.0) - 0.7 * math.sin(idx / 7.0)
        if 45 <= idx < 70:
            turn_x -= 2.4
            turn_y += 1.1
        if 105 <= idx < 130:
            turn_x += 2.8
            turn_y -= 1.6
        return (
            0.18 * turn_x + rng.gauss(0.0, 0.18),
            0.18 * turn_y + rng.gauss(0.0, 0.18),
        )

    if scenario == "constant_velocity":
        return (rng.gauss(0.0, 0.025), rng.gauss(0.0, 0.025))

    if scenario == "constant_acceleration":
        direction = 1.0 if idx < 95 else -0.65
        return (
            direction * 0.11 + rng.gauss(0.0, 0.035),
            -0.055 + rng.gauss(0.0, 0.03),
        )

    if scenario == "zig_zag":
        target_vx = 8.5 if (idx // 18) % 2 == 0 else -8.5
        target_vy = 4.8 if (idx // 30) % 2 == 0 else -4.8
        return (
            0.42 * (target_vx - velocity[0]) + rng.gauss(0.0, 0.08),
            0.36 * (target_vy - velocity[1]) + rng.gauss(0.0, 0.08),
        )

    if scenario == "sine_wave":
        target_vx = 5.2
        target_vy = 6.0 * math.cos(idx / 11.0)
        return (
            0.18 * (target_vx - velocity[0]) + rng.gauss(0.0, 0.04),
            0.28 * (target_vy - velocity[1]) + rng.gauss(0.0, 0.06),
        )

    if scenario == "center_direct":
        center = (FRAME_SIZE[0] * 0.5, FRAME_SIZE[1] * 0.5)
        return (
            0.012 * (center[0] - pos[0]) - 0.28 * velocity[0] + rng.gauss(0.0, 0.04),
            0.012 * (center[1] - pos[1]) - 0.28 * velocity[1] + rng.gauss(0.0, 0.04),
        )

    if scenario == "center_orbit":
        center = (FRAME_SIZE[0] * 0.5, FRAME_SIZE[1] * 0.5)
        dx = pos[0] - center[0]
        dy = pos[1] - center[1]
        swirl = (-0.018 * dy, 0.018 * dx)
        spring = (-0.009 * dx, -0.009 * dy)
        damping = (-0.10 * velocity[0], -0.10 * velocity[1])
        return (
            spring[0] + swirl[0] + damping[0] + rng.gauss(0.0, 0.04),
            spring[1] + swirl[1] + damping[1] + rng.gauss(0.0, 0.04),
        )

    if scenario == "random_maneuver":
        phase = idx // 24
        target_vx = 8.0 * math.sin(phase * 1.7 + 0.3)
        target_vy = 6.0 * math.cos(phase * 1.2 - 0.4)
        burst_x = rng.gauss(0.0, 0.55) if idx % 23 == 0 else rng.gauss(0.0, 0.10)
        burst_y = rng.gauss(0.0, 0.45) if idx % 31 == 0 else rng.gauss(0.0, 0.10)
        return (
            0.22 * (target_vx - velocity[0]) + burst_x,
            0.22 * (target_vy - velocity[1]) + burst_y,
        )

    raise ValueError(f"Unknown trajectory scenario: {scenario}")


def _make_measurements(truth: list[_Truth], seed: int) -> list[_Measurement]:
    rng = random.Random(seed + 100)
    measurements: list[_Measurement] = []
    for item in truth:
        if item.frame_idx == 0:
            measurements.append(_Measurement(True, item.pos, item.size, 1.0, "init"))
            continue

        occluded = 54 <= item.frame_idx <= 65 or 132 <= item.frame_idx <= 143
        outlier = item.frame_idx in {38, 91, 117, 156}
        if occluded:
            measurements.append(_Measurement(False, item.pos, item.size, 0.0, "occluded"))
            continue

        if outlier:
            pos_noise = (rng.gauss(35.0, 8.0), rng.gauss(-28.0, 7.0))
            size_noise = (rng.gauss(18.0, 4.0), rng.gauss(-14.0, 4.0))
            score = 0.28
            kind = "outlier"
        else:
            pos_noise = (rng.gauss(0.0, 4.0), rng.gauss(0.0, 4.0))
            size_noise = (rng.gauss(0.0, 2.0), rng.gauss(0.0, 2.0))
            score = max(0.45, min(1.0, 0.92 - 0.015 * math.hypot(*pos_noise)))
            kind = "noisy"

        measured_size = (
            max(1.0, item.size[0] + size_noise[0]),
            max(1.0, item.size[1] + size_noise[1]),
        )
        measured_pos = (item.pos[0] + pos_noise[0], item.pos[1] + pos_noise[1])
        measured_pos, measured_size = _clamp_center_size(measured_pos, measured_size)
        measurements.append(_Measurement(True, measured_pos, measured_size, score, kind))

    return measurements


def _bounce_position(
    pos: tuple[float, float],
    velocity: tuple[float, float],
    size: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    x, y = pos
    vx, vy = velocity
    half_w = size[0] / 2.0
    half_h = size[1] / 2.0
    if x < half_w:
        x = half_w
        vx = abs(vx) * 0.85
    elif x > FRAME_SIZE[0] - half_w:
        x = FRAME_SIZE[0] - half_w
        vx = -abs(vx) * 0.85
    if y < half_h:
        y = half_h
        vy = abs(vy) * 0.85
    elif y > FRAME_SIZE[1] - half_h:
        y = FRAME_SIZE[1] - half_h
        vy = -abs(vy) * 0.85
    return (x, y), (vx, vy)


def _clamp_center_size(
    pos: tuple[float, float],
    size: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    width = max(1.0, min(FRAME_SIZE[0], size[0]))
    height = max(1.0, min(FRAME_SIZE[1], size[1]))
    x = max(width / 2.0, min(FRAME_SIZE[0] - width / 2.0, pos[0]))
    y = max(height / 2.0, min(FRAME_SIZE[1] - height / 2.0, pos[1]))
    return (x, y), (width, height)


def _frame(idx: int, timestamp: float) -> _Frame:
    return _Frame(image=_Image((int(FRAME_SIZE[1]), int(FRAME_SIZE[0]), 3)), idx=idx, timestamp=timestamp)


def _state_from_prediction(
    prediction: TrackerPredictionState,
    *,
    frame_idx: int,
    timestamp: float,
    status: OutputStatus = OutputStatus.ACTIVE,
    mode: TrackerMode = TrackerMode.TRACKING,
) -> BigTrackState:
    pos = prediction.target_pos
    size = prediction.target_size
    return BigTrackState(
        prediction=prediction,
        matcher=MatcherState(init_template=object()),
        output=TrackingOutput(
            box=(pos[0] - size[0] / 2.0, pos[1] - size[1] / 2.0, size[0], size[1]),
            frame_idx=frame_idx,
            timestamp=timestamp,
            status=status,
            confidence=prediction.last_score,
        ),
        mode=mode,
        last_seen_frame=frame_idx,
    )


def _point_error(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _size_error(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _box_error(
    predicted_pos: tuple[float, float],
    predicted_size: tuple[float, float],
    truth: _Truth,
) -> float:
    pred_box = (
        predicted_pos[0] - predicted_size[0] / 2.0,
        predicted_pos[1] - predicted_size[1] / 2.0,
        predicted_size[0],
        predicted_size[1],
    )
    truth_box = (
        truth.pos[0] - truth.size[0] / 2.0,
        truth.pos[1] - truth.size[1] / 2.0,
        truth.size[0],
        truth.size[1],
    )
    return math.sqrt(sum((pred_box[idx] - truth_box[idx]) ** 2 for idx in range(4)) / 4.0)


def _rmse(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(value * value for value in values) / len(values))


if __name__ == "__main__":
    print(render_report())
