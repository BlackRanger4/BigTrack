from __future__ import annotations

from dataclasses import dataclass
import math
import random
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Callable


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


FRAME_SIZE = (1920.0, 1080.0)
DEFAULT_M_VALUES = "1,3,5,10"


@dataclass(frozen=True)
class Frame:
    image: object
    idx: int
    timestamp: float


@dataclass(frozen=True)
class TrajectoryConfig:
    frame_count: int
    seed: int
    observation_sigma: float
    acceleration_noise: float
    spring: float
    swirl: float
    damping: float


@dataclass(frozen=True)
class TrajectoryPoint:
    frame_idx: int
    timestamp: float
    pos: tuple[float, float]
    velocity: tuple[float, float]
    observation: tuple[float, float]
    observation_score: float


@dataclass(frozen=True)
class PredictorSpec:
    name: str
    color: str
    factory: Callable[[], object]


@dataclass(frozen=True)
class CachedPrediction:
    predictor_name: str
    color: str
    source_frame: int
    target_frame: int
    horizon: int
    pos: tuple[float, float]
    uncertainty: float
    error: float


@dataclass(frozen=True)
class CalculationResult:
    config: TrajectoryConfig
    m_values: tuple[int, ...]
    trajectory: tuple[TrajectoryPoint, ...]
    cache: dict[tuple[str, int, int], CachedPrediction]


PREDICTORS = (
    PredictorSpec(
        "matcher_target",
        "#7f7f7f",
        lambda: MatcherTargetPredictorModel(MatcherTargetPredictorConfig()),
    ),
    PredictorSpec(
        "kalman",
        "#d62728",
        lambda: KalmanPredictorModel(
            KalmanPredictorConfig(
                process_noise_position=1.0,
                measurement_noise_position=6.0,
                reject_uncertainty_growth=1.2,
            )
        ),
    ),
    PredictorSpec(
        "adaptive_kalman",
        "#9467bd",
        lambda: AdaptiveKalmanPredictorModel(
            AdaptiveKalmanPredictorConfig(
                process_noise_position=1.0,
                measurement_noise_position=6.0,
                adaptive_measurement_noise=True,
                reject_velocity_damping=0.9,
                max_position_velocity=40.0,
            )
        ),
    ),
    PredictorSpec(
        "alpha_beta",
        "#2ca02c",
        lambda: AlphaBetaPredictorModel(
            AlphaBetaPredictorConfig(
                alpha_position=0.75,
                beta_position=0.20,
                max_position_velocity=40.0,
                max_position_acceleration=10.0,
                reject_velocity_damping=0.9,
            )
        ),
    ),
    PredictorSpec(
        "history",
        "#ff7f0e",
        lambda: HistoryPredictorModel(
            HistoryPredictorConfig(
                history_length=10,
                velocity_window=4,
                velocity_smoothing=0.7,
                max_position_velocity=40.0,
                max_position_acceleration=10.0,
                reject_velocity_damping=0.9,
            )
        ),
    ),
    PredictorSpec(
        "constant_accel",
        "#17becf",
        lambda: ConstantAccelerationKalmanPredictorModel(
            ConstantAccelerationKalmanPredictorConfig(
                process_noise_position=1.0,
                process_noise_velocity=0.6,
                process_noise_acceleration=0.25,
                measurement_noise_position=7.0,
                adaptive_measurement_noise=True,
                max_position_velocity=40.0,
                max_position_acceleration=10.0,
                reject_velocity_damping=0.9,
                reject_acceleration_damping=0.7,
            )
        ),
    ),
)


class TrajectoryPredictorUI:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("BigTracker Predictor Trajectory UI")

        self.frame_count_var = tk.IntVar(value=180)
        self.seed_var = tk.IntVar(value=1337)
        self.observation_sigma_var = tk.DoubleVar(value=4.0)
        self.accel_noise_var = tk.DoubleVar(value=0.08)
        self.spring_var = tk.DoubleVar(value=0.006)
        self.swirl_var = tk.DoubleVar(value=0.012)
        self.damping_var = tk.DoubleVar(value=0.035)
        self.m_values_var = tk.StringVar(value=DEFAULT_M_VALUES)
        self.status_var = tk.StringVar(value="push Calculate to build cache")
        self.frame_var = tk.IntVar(value=60)
        self.predictor_vars = {spec.name: tk.BooleanVar(value=True) for spec in PREDICTORS}
        self.m_vars: dict[int, tk.BooleanVar] = {}

        self.result: CalculationResult | None = None
        self.canvas_width = 980
        self.canvas_height = 560
        self.pad = 42

        self._build_layout()

    def run(self) -> None:
        self.root.mainloop()

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        self.config_tab = ttk.Frame(self.notebook, padding=12)
        self.show_tab = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(self.config_tab, text="Config")
        self.notebook.add(self.show_tab, text="Show")

        self._build_config_tab()
        self._build_show_tab()

    def _build_config_tab(self) -> None:
        self.config_tab.columnconfigure(1, weight=1)

        rows = (
            ("frames", self.frame_count_var, 30, 2000, 1),
            ("seed", self.seed_var, 1, 999999, 1),
            ("observation sigma", self.observation_sigma_var, 0.0, 50.0, 0.5),
            ("acceleration noise", self.accel_noise_var, 0.0, 2.0, 0.01),
            ("spring", self.spring_var, 0.0, 0.05, 0.001),
            ("swirl", self.swirl_var, 0.0, 0.05, 0.001),
            ("damping", self.damping_var, 0.0, 0.2, 0.001),
        )
        for row, (label, variable, from_value, to_value, increment) in enumerate(rows):
            ttk.Label(self.config_tab, text=label).grid(row=row, column=0, sticky="w", pady=4)
            ttk.Spinbox(
                self.config_tab,
                from_=from_value,
                to=to_value,
                increment=increment,
                textvariable=variable,
                width=14,
            ).grid(row=row, column=1, sticky="w", pady=4)

        ttk.Label(self.config_tab, text="M list").grid(row=len(rows), column=0, sticky="w", pady=4)
        ttk.Entry(self.config_tab, textvariable=self.m_values_var, width=32).grid(
            row=len(rows),
            column=1,
            sticky="w",
            pady=4,
        )
        ttk.Button(self.config_tab, text="Calculate", command=self._calculate).grid(
            row=len(rows) + 1,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(16, 4),
        )
        ttk.Label(self.config_tab, textvariable=self.status_var).grid(
            row=len(rows) + 2,
            column=0,
            columnspan=2,
            sticky="w",
            pady=8,
        )

    def _build_show_tab(self) -> None:
        self.show_tab.columnconfigure(0, weight=1)
        self.show_tab.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            self.show_tab,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="#fafafa",
            highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")

        side = ttk.Frame(self.show_tab, padding=(10, 0, 0, 0))
        side.grid(row=0, column=1, sticky="ns")

        ttk.Label(side, text="Frame").grid(row=0, column=0, sticky="w")
        self.frame_scale = ttk.Scale(
            side,
            from_=0,
            to=179,
            variable=self.frame_var,
            command=self._on_frame_change,
            orient="horizontal",
            length=250,
        )
        self.frame_scale.grid(row=1, column=0, sticky="ew", pady=(4, 4))

        frame_buttons = ttk.Frame(side)
        frame_buttons.grid(row=2, column=0, sticky="w", pady=(0, 12))
        ttk.Button(frame_buttons, text="-1", command=lambda: self._step_frame(-1)).pack(side="left")
        ttk.Button(frame_buttons, text="+1", command=lambda: self._step_frame(1)).pack(side="left", padx=6)

        ttk.Label(side, text="Predictors").grid(row=3, column=0, sticky="w")
        for row, spec in enumerate(PREDICTORS, start=4):
            ttk.Checkbutton(
                side,
                text=spec.name,
                variable=self.predictor_vars[spec.name],
                command=self._render,
            ).grid(row=row, column=0, sticky="w")

        ttk.Label(side, text="M").grid(row=10, column=0, sticky="w", pady=(12, 0))
        self.m_checks_frame = ttk.Frame(side)
        self.m_checks_frame.grid(row=11, column=0, sticky="w")

        ttk.Label(side, text="Details").grid(row=12, column=0, sticky="w", pady=(12, 0))
        self.details = ttk.Treeview(
            side,
            columns=("predictor", "m", "source", "dx", "dy", "err", "unc"),
            show="headings",
            height=14,
        )
        for column, width in (
            ("predictor", 110),
            ("m", 42),
            ("source", 55),
            ("dx", 62),
            ("dy", 62),
            ("err", 62),
            ("unc", 62),
        ):
            self.details.heading(column, text=column)
            self.details.column(column, width=width, anchor="e" if column != "predictor" else "w")
        self.details.grid(row=13, column=0, sticky="nsew", pady=(4, 0))

    def _calculate(self) -> None:
        config = TrajectoryConfig(
            frame_count=max(2, int(self.frame_count_var.get())),
            seed=int(self.seed_var.get()),
            observation_sigma=max(0.0, float(self.observation_sigma_var.get())),
            acceleration_noise=max(0.0, float(self.accel_noise_var.get())),
            spring=max(0.0, float(self.spring_var.get())),
            swirl=max(0.0, float(self.swirl_var.get())),
            damping=max(0.0, float(self.damping_var.get())),
        )
        m_values = tuple(value for value in parse_m_values(self.m_values_var.get()) if value < config.frame_count)
        if not m_values:
            m_values = (1,)
        trajectory = tuple(generate_trajectory(config))
        cache = precompute_predictions(trajectory, m_values)
        self.result = CalculationResult(config=config, m_values=m_values, trajectory=trajectory, cache=cache)

        self.frame_scale.configure(to=config.frame_count - 1)
        self.frame_var.set(min(max(0, self.frame_var.get()), config.frame_count - 1))
        self._rebuild_m_checks(m_values)
        self.status_var.set(
            f"cached {len(cache)} predictions for {len(PREDICTORS)} predictors, "
            f"{len(m_values)} horizons, {config.frame_count} frames"
        )
        self.notebook.select(self.show_tab)
        self._render()

    def _rebuild_m_checks(self, m_values: tuple[int, ...]) -> None:
        for child in self.m_checks_frame.winfo_children():
            child.destroy()
        self.m_vars = {m: tk.BooleanVar(value=True) for m in m_values}
        for row, m in enumerate(m_values):
            ttk.Checkbutton(
                self.m_checks_frame,
                text=str(m),
                variable=self.m_vars[m],
                command=self._render,
            ).grid(row=row // 4, column=row % 4, sticky="w", padx=(0, 8))

    def _on_frame_change(self, _value: str) -> None:
        self.frame_var.set(int(float(self.frame_var.get())))
        self._render()

    def _step_frame(self, delta: int) -> None:
        if self.result is None:
            return
        frame_idx = max(0, min(self.result.config.frame_count - 1, self.frame_var.get() + delta))
        self.frame_var.set(frame_idx)
        self._render()

    def _render(self) -> None:
        self.canvas.delete("all")
        self._clear_details()
        if self.result is None:
            self.canvas.create_text(
                self.canvas_width / 2,
                self.canvas_height / 2,
                text="Open Config tab and press Calculate.",
                fill="#444444",
                font=("Segoe UI", 14),
            )
            return

        frame_idx = max(0, min(self.result.config.frame_count - 1, int(self.frame_var.get())))
        self.frame_var.set(frame_idx)
        selected = self._selected_predictions(frame_idx)

        self._draw_axes()
        self._draw_truth_path()
        self._draw_observations(frame_idx)
        self._draw_current_frame(frame_idx)
        self._draw_predictions(selected)
        self._draw_legend(selected)
        self._fill_details(selected, frame_idx)

    def _selected_predictions(self, frame_idx: int) -> list[CachedPrediction]:
        if self.result is None:
            return []
        rows: list[CachedPrediction] = []
        for spec in PREDICTORS:
            if not self.predictor_vars[spec.name].get():
                continue
            for m in self.result.m_values:
                if not self.m_vars.get(m, tk.BooleanVar(value=False)).get():
                    continue
                prediction = self.result.cache.get((spec.name, m, frame_idx))
                if prediction is not None:
                    rows.append(prediction)
        return rows

    def _draw_axes(self) -> None:
        left, top = self._to_canvas((0.0, FRAME_SIZE[1]))
        right, bottom = self._to_canvas((FRAME_SIZE[0], 0.0))
        self.canvas.create_rectangle(left, top, right, bottom, outline="#d0d0d0")
        for i in range(0, int(FRAME_SIZE[0]) + 1, 80):
            x, _ = self._to_canvas((float(i), 0.0))
            self.canvas.create_line(x, top, x, bottom, fill="#eeeeee")
        for i in range(0, int(FRAME_SIZE[1]) + 1, 60):
            _, y = self._to_canvas((0.0, float(i)))
            self.canvas.create_line(left, y, right, y, fill="#eeeeee")

    def _draw_truth_path(self) -> None:
        if self.result is None:
            return
        coords: list[float] = []
        for point in self.result.trajectory:
            coords.extend(self._to_canvas(point.pos))
        self.canvas.create_line(*coords, fill="#303030", width=2)

    def _draw_observations(self, frame_idx: int) -> None:
        if self.result is None:
            return
        for point in self.result.trajectory[: frame_idx + 1]:
            x, y = self._to_canvas(point.observation)
            self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, outline="", fill="#1f77b4")

    def _draw_current_frame(self, frame_idx: int) -> None:
        if self.result is None:
            return
        point = self.result.trajectory[frame_idx]
        truth_x, truth_y = self._to_canvas(point.pos)
        obs_x, obs_y = self._to_canvas(point.observation)
        self.canvas.create_oval(truth_x - 8, truth_y - 8, truth_x + 8, truth_y + 8, outline="#000000", width=3)
        self.canvas.create_oval(obs_x - 6, obs_y - 6, obs_x + 6, obs_y + 6, outline="#1f77b4", width=2)
        self.canvas.create_text(
            truth_x + 10,
            truth_y + 10,
            text=f"frame {frame_idx}",
            anchor="w",
            fill="#000000",
        )

    def _draw_predictions(self, rows: list[CachedPrediction]) -> None:
        if self.result is None:
            return
        for row in rows:
            source = self.result.trajectory[row.source_frame]
            source_x, source_y = self._to_canvas(source.pos)
            pred_x, pred_y = self._to_canvas(row.pos)
            self.canvas.create_line(source_x, source_y, pred_x, pred_y, fill=row.color, width=1, dash=(3, 5))
            self.canvas.create_oval(pred_x - 5, pred_y - 5, pred_x + 5, pred_y + 5, outline=row.color, width=3)
            self.canvas.create_text(
                pred_x + 8,
                pred_y - 8,
                text=f"{row.predictor_name}:{row.horizon}",
                anchor="w",
                fill=row.color,
                font=("Segoe UI", 8, "bold"),
            )

    def _draw_legend(self, rows: list[CachedPrediction]) -> None:
        self.canvas.create_text(
            52,
            24,
            text="black: true target   blue: observations   colored points: cached future predictions",
            anchor="w",
            fill="#333333",
        )
        by_key: dict[tuple[str, int], CachedPrediction] = {}
        for row in rows:
            by_key[(row.predictor_name, row.horizon)] = row
        for index, row in enumerate(by_key.values()):
            y = 48 + index * 18
            self.canvas.create_line(52, y, 76, y, fill=row.color, width=3)
            self.canvas.create_text(
                84,
                y,
                text=f"{row.predictor_name} M={row.horizon} err={row.error:.2f}",
                anchor="w",
                fill=row.color,
            )

    def _fill_details(self, rows: list[CachedPrediction], frame_idx: int) -> None:
        if self.result is None:
            return
        truth = self.result.trajectory[frame_idx].pos
        for row in sorted(rows, key=lambda item: (item.predictor_name, item.horizon)):
            dx = row.pos[0] - truth[0]
            dy = row.pos[1] - truth[1]
            self.details.insert(
                "",
                "end",
                values=(
                    row.predictor_name,
                    row.horizon,
                    row.source_frame,
                    f"{dx:.2f}",
                    f"{dy:.2f}",
                    f"{row.error:.2f}",
                    f"{row.uncertainty:.2f}",
                ),
            )

    def _clear_details(self) -> None:
        for item in self.details.get_children():
            self.details.delete(item)

    def _to_canvas(self, pos: tuple[float, float]) -> tuple[float, float]:
        width = self.canvas_width - 2 * self.pad
        height = self.canvas_height - 2 * self.pad
        x = self.pad + pos[0] / FRAME_SIZE[0] * width
        y = self.pad + (1.0 - pos[1] / FRAME_SIZE[1]) * height
        return x, y


def parse_m_values(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.replace(";", ",").split(","):
        stripped = part.strip()
        if not stripped:
            continue
        value = max(1, int(stripped))
        if value not in values:
            values.append(value)
    return values or [1]


def generate_trajectory(config: TrajectoryConfig) -> list[TrajectoryPoint]:
    rng = random.Random(config.seed)
    pos = (130.0 + rng.uniform(-20.0, 20.0), 90.0 + rng.uniform(-18.0, 18.0))
    velocity = (5.0 + rng.uniform(-1.0, 1.0), 2.5 + rng.uniform(-0.8, 0.8))
    points: list[TrajectoryPoint] = []

    for idx in range(config.frame_count):
        if idx > 0:
            acceleration = _acceleration(idx, pos, velocity, rng, config)
            velocity = (
                max(-14.0, min(14.0, velocity[0] + acceleration[0])),
                max(-11.0, min(11.0, velocity[1] + acceleration[1])),
            )
            pos = (pos[0] + velocity[0], pos[1] + velocity[1])
            pos, velocity = _bounce_position(pos, velocity)

        noise = (
            rng.gauss(0.0, config.observation_sigma),
            rng.gauss(0.0, config.observation_sigma),
        )
        observation = _clamp_point((pos[0] + noise[0], pos[1] + noise[1]))
        score = max(0.2, min(1.0, 1.0 - 0.03 * math.hypot(*noise)))
        points.append(
            TrajectoryPoint(
                frame_idx=idx,
                timestamp=float(idx),
                pos=pos,
                velocity=velocity,
                observation=observation,
                observation_score=score,
            )
        )
    return points


def precompute_predictions(
    trajectory: tuple[TrajectoryPoint, ...],
    m_values: tuple[int, ...],
) -> dict[tuple[str, int, int], CachedPrediction]:
    cache: dict[tuple[str, int, int], CachedPrediction] = {}
    for spec in PREDICTORS:
        predictor = spec.factory()
        first = trajectory[0]
        predictor.initialize(
            PredictorInitializeInput(
                predictor_state=TrackerPredictionState(
                    target_pos=first.observation,
                    target_velocity=first.velocity,
                    uncertainty=0.0,
                )
            )
        )

        for source_idx, point in enumerate(trajectory):
            # The cached row is defined as a forecast *from* source_idx.
            # Bring the live model through that frame before taking its state,
            # so an M=1 row predicts source_idx + 1 with one frame of dt.
            if source_idx > 0:
                predicted_current = predictor.predict(PredictorPredictInput(frame=_frame(point))).predictor_state
                predictor.update(
                    PredictorUpdateInput(
                        accepted=True,
                        predictor_state=TrackerPredictionState(
                            target_pos=point.observation,
                            target_velocity=predicted_current.target_velocity,
                            uncertainty=predicted_current.uncertainty,
                            metadata=predicted_current.metadata,
                        ),
                        metadata={"score": point.observation_score},
                    )
                )

            source_state = _current_state(predictor)
            for m in m_values:
                target_idx = source_idx + m
                if target_idx >= len(trajectory):
                    continue
                prediction = predict_future_from_state(spec, source_state, trajectory, source_idx, target_idx)
                truth = trajectory[target_idx].pos
                error = _point_error(prediction.target_pos, truth)
                cache[(spec.name, m, target_idx)] = CachedPrediction(
                    predictor_name=spec.name,
                    color=spec.color,
                    source_frame=source_idx,
                    target_frame=target_idx,
                    horizon=m,
                    pos=prediction.target_pos,
                    uncertainty=prediction.uncertainty,
                    error=error,
                )

    return cache


def predict_future_from_state(
    spec: PredictorSpec,
    state: TrackerPredictionState,
    trajectory: tuple[TrajectoryPoint, ...],
    source_idx: int,
    target_idx: int,
) -> TrackerPredictionState:
    predictor = spec.factory()
    predictor.initialize(PredictorInitializeInput(predictor_state=state))
    predicted = state
    for idx in range(source_idx + 1, target_idx + 1):
        predicted = predictor.predict(PredictorPredictInput(frame=_frame(trajectory[idx]))).predictor_state
    return predicted


def _frame(point: TrajectoryPoint) -> Frame:
    return Frame(image=None, idx=point.frame_idx, timestamp=point.timestamp)


def _current_state(predictor: object) -> TrackerPredictionState:
    state = getattr(predictor, "_state", None)
    if state is None:
        raise RuntimeError("predictor state is not initialized")
    return state


def _acceleration(
    idx: int,
    pos: tuple[float, float],
    velocity: tuple[float, float],
    rng: random.Random,
    config: TrajectoryConfig,
) -> tuple[float, float]:
    center = (FRAME_SIZE[0] * 0.5, FRAME_SIZE[1] * 0.5)
    dx = pos[0] - center[0]
    dy = pos[1] - center[1]
    spring = (-config.spring * dx, -config.spring * dy)
    swirl = (-config.swirl * dy, config.swirl * dx)
    wave = (0.75 * math.sin(idx / 13.0), 0.55 * math.cos(idx / 17.0))
    damping = (-config.damping * velocity[0], -config.damping * velocity[1])
    random_push = (
        rng.gauss(0.0, config.acceleration_noise),
        rng.gauss(0.0, config.acceleration_noise),
    )
    return (
        spring[0] + swirl[0] + wave[0] + damping[0] + random_push[0],
        spring[1] + swirl[1] + wave[1] + damping[1] + random_push[1],
    )


def _bounce_position(
    pos: tuple[float, float],
    velocity: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    x, y = pos
    vx, vy = velocity
    if x < 0.0:
        x = 0.0
        vx = abs(vx) * 0.82
    elif x > FRAME_SIZE[0]:
        x = FRAME_SIZE[0]
        vx = -abs(vx) * 0.82
    if y < 0.0:
        y = 0.0
        vy = abs(vy) * 0.82
    elif y > FRAME_SIZE[1]:
        y = FRAME_SIZE[1]
        vy = -abs(vy) * 0.82
    return (x, y), (vx, vy)


def _clamp_point(pos: tuple[float, float]) -> tuple[float, float]:
    return (
        max(0.0, min(FRAME_SIZE[0], pos[0])),
        max(0.0, min(FRAME_SIZE[1], pos[1])),
    )


def _point_error(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def main() -> None:
    TrajectoryPredictorUI().run()


if __name__ == "__main__":
    main()
