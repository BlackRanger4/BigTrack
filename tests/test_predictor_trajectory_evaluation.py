from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import random
import statistics
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from BigTracker.predictor_models import (  # noqa: E402
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
    MatcherTargetPredictorConfig,
    MatcherTargetPredictorModel,
)
from BigTracker.types import (  # noqa: E402
    PredictorInitializeInput,
    PredictorPredictInput,
    PredictorUpdateInput,
    TrackerPredictionState,
)


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
class _Frame:
    image: object
    idx: int
    timestamp: float


@dataclass(frozen=True)
class _Truth:
    frame_idx: int
    timestamp: float
    pos: tuple[float, float]
    velocity: tuple[float, float]


@dataclass(frozen=True)
class _Measurement:
    available: bool
    pos: tuple[float, float]
    score: float
    kind: str


@dataclass(frozen=True)
class _Metrics:
    name: str
    position_rmse: float
    accept_position_rmse: float
    reject_position_rmse: float
    reject_frames: int
    accepted_frames: int
    max_position_error: float
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
    accept_position_rmse: _MetricSummary
    reject_position_rmse: _MetricSummary
    max_position_error: _MetricSummary
    final_position_error: _MetricSummary
    avg_uncertainty: _MetricSummary


class PredictorTrajectoryEvaluationTest(unittest.TestCase):
    def test_predictors_handle_complex_noisy_trajectory(self) -> None:
        results = run_evaluation()

        self.assertEqual(set(results), set(_predictors()))
        for name, metrics in results.items():
            with self.subTest(predictor=name):
                self.assertLess(metrics.position_rmse.maximum, 80.0)
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
        "- Measurements: Gaussian noise, structured occlusion windows, and deterministic outliers.",
        f"- Accepted measurements per run: `{accepted}`",
        f"- Rejected measurements per run: `{rejected}`",
        "",
        "## Metrics",
        "",
        "Each cell is `average (minimum-maximum)` across all scenario and seed runs.",
        "",
        "| Predictor | Position RMSE | Accept Pos RMSE | Reject Pos RMSE | Max Pos Err | Final Pos Err | Avg Uncertainty |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for metrics in ordered:
        lines.append(
            "| "
            f"{metrics.name} | "
            f"{_format_summary(metrics.position_rmse)} | "
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
        accept_position_rmse=summarize("accept_position_rmse"),
        reject_position_rmse=summarize("reject_position_rmse"),
        max_position_error=summarize("max_position_error"),
        final_position_error=summarize("final_position_error"),
        avg_uncertainty=summarize("avg_uncertainty"),
    )


def _format_summary(summary: _MetricSummary) -> str:
    return f"{summary.average:.3f} ({summary.minimum:.3f}-{summary.maximum:.3f})"


def _predictors():
    return {
        "matcher_target": MatcherTargetPredictorModel(MatcherTargetPredictorConfig()),
        "kalman": KalmanPredictorModel(
            KalmanPredictorConfig(
                process_noise_position=1.2,
                measurement_noise_position=7.0,
                reject_uncertainty_growth=1.2,
            )
        ),
        "adaptive_kalman": AdaptiveKalmanPredictorModel(
            AdaptiveKalmanPredictorConfig(
                process_noise_position=1.2,
                measurement_noise_position=7.0,
                adaptive_measurement_noise=True,
                reject_velocity_damping=0.90,
                max_position_velocity=30.0,
            )
        ),
        "alpha_beta": AlphaBetaPredictorModel(
            AlphaBetaPredictorConfig(
                alpha_position=0.72,
                beta_position=0.18,
                max_position_velocity=30.0,
                max_position_acceleration=8.0,
                reject_velocity_damping=0.90,
            )
        ),
        "history": HistoryPredictorModel(
            HistoryPredictorConfig(
                history_length=8,
                velocity_window=4,
                velocity_smoothing=0.70,
                max_position_velocity=30.0,
                max_position_acceleration=8.0,
                reject_velocity_damping=0.90,
            )
        ),
        "constant_accel_kalman": ConstantAccelerationKalmanPredictorModel(
            ConstantAccelerationKalmanPredictorConfig(
                process_noise_position=1.2,
                process_noise_velocity=0.7,
                process_noise_acceleration=0.35,
                measurement_noise_position=8.0,
                adaptive_measurement_noise=True,
                max_position_velocity=30.0,
                max_position_acceleration=8.0,
                reject_velocity_damping=0.90,
                reject_acceleration_damping=0.70,
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
    predictor.initialize(
        PredictorInitializeInput(
            predictor_state=TrackerPredictionState(
                target_pos=first.pos,
                target_velocity=first.velocity,
                uncertainty=0.0,
            )
        )
    )
    position_errors: list[float] = []
    accept_position_errors: list[float] = []
    reject_position_errors: list[float] = []
    uncertainties: list[float] = []
    accepted_frames = 0
    reject_frames = 0
    final_position_error = 0.0

    for current_truth, measurement in zip(truth[1:], measurements[1:]):
        predicted = predictor.predict(
            PredictorPredictInput(frame=_frame(current_truth.frame_idx, current_truth.timestamp))
        ).predictor_state
        position_error = _point_error(predicted.target_pos, current_truth.pos)
        position_errors.append(position_error)
        uncertainties.append(predicted.uncertainty)

        if measurement.available:
            predictor.update(
                PredictorUpdateInput(
                    accepted=True,
                    predictor_state=TrackerPredictionState(
                        target_pos=measurement.pos,
                        target_velocity=predicted.target_velocity,
                        uncertainty=predicted.uncertainty,
                        metadata=predicted.metadata,
                    ),
                    metadata={"score": measurement.score},
                )
            )
            accepted_frames += 1
            accept_position_errors.append(position_error)
        else:
            predictor.update(PredictorUpdateInput(accepted=False, predictor_state=predicted))
            reject_frames += 1
            reject_position_errors.append(position_error)

        updated = _current_state(predictor)
        final_position_error = _point_error(updated.target_pos, current_truth.pos)

    return _Metrics(
        name=name,
        position_rmse=_rmse(position_errors),
        accept_position_rmse=_rmse(accept_position_errors),
        reject_position_rmse=_rmse(reject_position_errors),
        reject_frames=reject_frames,
        accepted_frames=accepted_frames,
        max_position_error=max(position_errors),
        final_position_error=final_position_error,
        avg_uncertainty=sum(uncertainties) / len(uncertainties),
    )


def _make_truth(seed: int, scenario: str) -> list[_Truth]:
    rng = random.Random(seed)
    pos = (125.0 + rng.uniform(-8.0, 8.0), 95.0 + rng.uniform(-6.0, 6.0))
    velocity = (4.5 + rng.uniform(-0.4, 0.4), 2.0 + rng.uniform(-0.3, 0.3))
    truth = [_Truth(0, 0.0, pos, velocity)]

    for idx in range(1, FRAME_COUNT):
        acceleration = _scenario_acceleration(scenario, idx, pos, velocity, rng)
        velocity = (
            max(-13.0, min(13.0, velocity[0] + acceleration[0])),
            max(-10.0, min(10.0, velocity[1] + acceleration[1])),
        )
        pos = (pos[0] + velocity[0], pos[1] + velocity[1])
        pos, velocity = _bounce_position(pos, velocity)
        truth.append(_Truth(idx, float(idx), pos, velocity))

    return truth


def _scenario_acceleration(
    scenario: str,
    idx: int,
    pos: tuple[float, float],
    velocity: tuple[float, float],
    rng: random.Random,
) -> tuple[float, float]:
    if scenario == "mixed_accel":
        return (
            0.18 * (1.8 * math.sin(idx / 14.0) + 0.9 * math.sin(idx / 5.5)) + rng.gauss(0.0, 0.18),
            0.18 * (1.4 * math.cos(idx / 18.0) - 0.7 * math.sin(idx / 7.0)) + rng.gauss(0.0, 0.18),
        )
    if scenario == "constant_velocity":
        return (rng.gauss(0.0, 0.025), rng.gauss(0.0, 0.025))
    if scenario == "constant_acceleration":
        direction = 1.0 if idx < 95 else -0.65
        return (direction * 0.11 + rng.gauss(0.0, 0.035), -0.055 + rng.gauss(0.0, 0.03))
    if scenario == "zig_zag":
        target_vx = 8.5 if (idx // 18) % 2 == 0 else -8.5
        target_vy = 4.8 if (idx // 30) % 2 == 0 else -4.8
        return (
            0.42 * (target_vx - velocity[0]) + rng.gauss(0.0, 0.08),
            0.36 * (target_vy - velocity[1]) + rng.gauss(0.0, 0.08),
        )
    if scenario == "sine_wave":
        return (
            0.18 * (5.2 - velocity[0]) + rng.gauss(0.0, 0.04),
            0.28 * (6.0 * math.cos(idx / 11.0) - velocity[1]) + rng.gauss(0.0, 0.06),
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
        return (
            -0.009 * dx - 0.018 * dy - 0.10 * velocity[0] + rng.gauss(0.0, 0.04),
            0.018 * dx - 0.009 * dy - 0.10 * velocity[1] + rng.gauss(0.0, 0.04),
        )
    if scenario == "random_maneuver":
        phase = idx // 24
        target_vx = 8.0 * math.sin(phase * 1.7 + 0.3)
        target_vy = 6.0 * math.cos(phase * 1.2 - 0.4)
        burst_x = rng.gauss(0.0, 0.55) if idx % 23 == 0 else rng.gauss(0.0, 0.10)
        burst_y = rng.gauss(0.0, 0.45) if idx % 31 == 0 else rng.gauss(0.0, 0.10)
        return (0.22 * (target_vx - velocity[0]) + burst_x, 0.22 * (target_vy - velocity[1]) + burst_y)
    raise ValueError(f"Unknown trajectory scenario: {scenario}")


def _make_measurements(truth: list[_Truth], seed: int) -> list[_Measurement]:
    rng = random.Random(seed + 100)
    measurements: list[_Measurement] = []
    for item in truth:
        if item.frame_idx == 0:
            measurements.append(_Measurement(True, item.pos, 1.0, "init"))
            continue

        occluded = 54 <= item.frame_idx <= 65 or 132 <= item.frame_idx <= 143
        outlier = item.frame_idx in {38, 91, 117, 156}
        if occluded:
            measurements.append(_Measurement(False, item.pos, 0.0, "occluded"))
            continue

        if outlier:
            pos_noise = (rng.gauss(35.0, 8.0), rng.gauss(-28.0, 7.0))
            score = 0.28
            kind = "outlier"
        else:
            pos_noise = (rng.gauss(0.0, 4.0), rng.gauss(0.0, 4.0))
            score = max(0.45, min(1.0, 0.92 - 0.015 * math.hypot(*pos_noise)))
            kind = "noisy"

        measured_pos = _clamp_point((item.pos[0] + pos_noise[0], item.pos[1] + pos_noise[1]))
        measurements.append(_Measurement(True, measured_pos, score, kind))

    return measurements


def _bounce_position(
    pos: tuple[float, float],
    velocity: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    x, y = pos
    vx, vy = velocity
    if x < 0.0:
        x = 0.0
        vx = abs(vx) * 0.85
    elif x > FRAME_SIZE[0]:
        x = FRAME_SIZE[0]
        vx = -abs(vx) * 0.85
    if y < 0.0:
        y = 0.0
        vy = abs(vy) * 0.85
    elif y > FRAME_SIZE[1]:
        y = FRAME_SIZE[1]
        vy = -abs(vy) * 0.85
    return (x, y), (vx, vy)


def _clamp_point(pos: tuple[float, float]) -> tuple[float, float]:
    return (
        max(0.0, min(FRAME_SIZE[0], pos[0])),
        max(0.0, min(FRAME_SIZE[1], pos[1])),
    )


def _frame(idx: int, timestamp: float) -> _Frame:
    return _Frame(image=None, idx=idx, timestamp=timestamp)


def _current_state(predictor) -> TrackerPredictionState:
    state = predictor._state
    if state is None:
        raise AssertionError("predictor state was not initialized")
    return state


def _point_error(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _rmse(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(value * value for value in values) / len(values))


if __name__ == "__main__":
    print(render_report())
