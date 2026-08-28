from __future__ import annotations

import ast
import dataclasses
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from BigTracker.big_trackers.score_gated import ScoreGatedBigTrack, ScoreGatedBigTrackConfig  # noqa: E402
from BigTracker.big_trackers.simple import SimpleBigTrack  # noqa: E402
from BigTracker.matcher_models.fft import FftMatcherConfig, FftMatcherModel  # noqa: E402
from BigTracker.matcher_models.litetrack import LiteTrackMatcherConfig, LiteTrackMatcherModel  # noqa: E402
from BigTracker.matcher_models.mixformerv2 import MixFormerV2MatcherConfig, MixFormerV2MatcherModel  # noqa: E402
from BigTracker.matcher_models.nanotrack import NanoTrackMatcherConfig, NanoTrackMatcherModel  # noqa: E402
from BigTracker.matcher_models.ostrack import OSTrackMatcherConfig, OSTrackMatcherModel  # noqa: E402
from BigTracker.predictor_models import (  # noqa: E402
    AdaptiveKalmanPredictorConfig,
    AdaptiveKalmanPredictorModel,
    AlphaBetaPredictorConfig,
    AlphaBetaPredictorModel,
    ConstantAccelerationKalmanPredictorConfig,
    ConstantAccelerationKalmanPredictorModel,
    HistoryPredictorConfig,
    HistoryPredictorModel,
    MatcherTargetPredictorConfig,
    MatcherTargetPredictorModel,
    KalmanPredictorConfig,
    KalmanPredictorModel,
)
from tools.bigtrack_fulltest.frame_source import build_frame_source  # noqa: E402
from tools.bigtrack_fulltest.runner import FullTestRunner, RunnerConfig  # noqa: E402


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    config_type: type | None
    factory: Callable[[object | None], object]


PREDICTORS: dict[str, ComponentSpec] = {
    "adaptive_kalman": ComponentSpec(
        "adaptive_kalman",
        AdaptiveKalmanPredictorConfig,
        lambda config: AdaptiveKalmanPredictorModel(config),
    ),
    "kalman": ComponentSpec("kalman", KalmanPredictorConfig, lambda config: KalmanPredictorModel(config)),
    "alpha_beta": ComponentSpec(
        "alpha_beta",
        AlphaBetaPredictorConfig,
        lambda config: AlphaBetaPredictorModel(config),
    ),
    "history": ComponentSpec("history", HistoryPredictorConfig, lambda config: HistoryPredictorModel(config)),
    "matcher_target": ComponentSpec(
        "matcher_target",
        MatcherTargetPredictorConfig,
        lambda config: MatcherTargetPredictorModel(config),
    ),
    "constant_accel_kalman": ComponentSpec(
        "constant_accel_kalman",
        ConstantAccelerationKalmanPredictorConfig,
        lambda config: ConstantAccelerationKalmanPredictorModel(config),
    ),
}

MATCHERS: dict[str, ComponentSpec] = {
    "fft": ComponentSpec("fft", FftMatcherConfig, lambda config: FftMatcherModel(config)),
    "nanotrack": ComponentSpec("nanotrack", NanoTrackMatcherConfig, lambda config: NanoTrackMatcherModel(config)),
    "ostrack": ComponentSpec("ostrack", OSTrackMatcherConfig, lambda config: OSTrackMatcherModel(config)),
    "litetrack": ComponentSpec("litetrack", LiteTrackMatcherConfig, lambda config: LiteTrackMatcherModel(config)),
    "mixformerv2": ComponentSpec(
        "mixformerv2",
        MixFormerV2MatcherConfig,
        lambda config: MixFormerV2MatcherModel(config),
    ),
}

POLICIES: dict[str, ComponentSpec] = {
    "score_gated": ComponentSpec(
        "score_gated",
        ScoreGatedBigTrackConfig,
        lambda config, predictor=None, matcher=None: ScoreGatedBigTrack(
            predictor=predictor,
            matcher=matcher,
            config=config,
        ),
    ),
    "simple": ComponentSpec(
        "simple",
        None,
        lambda config, predictor=None, matcher=None: SimpleBigTrack(predictor=predictor, matcher=matcher),
    ),
}

DEFAULT_OVERRIDES = {
    ("nanotrack", "backend"): "onnx",
    ("nanotrack", "config_path"): r"ignores\Models\nanotrack\config\configv3.yaml",
    ("nanotrack", "checkpoint_path"): r"ignores\Models\nanotrack\pretrained\nanotrackv3.pth",
    ("nanotrack", "backbone_path"): r"ignores\Models\nanotrack\nanotrackv3\nanotrack_backbone.onnx",
    ("nanotrack", "head_path"): r"ignores\Models\nanotrack\nanotrackv3\nanotrack_head.onnx",
    ("nanotrack", "onnx_provider"): "cpu",
    ("ostrack", "config_path"): r"ignores\Models\Ostrack\config\vitb_256_mae_32x4_ep300.yaml",
    ("ostrack", "checkpoint_path"): (
        r"ignores\Models\Ostrack\models\vitb_256_mae_32x4_ep300\OSTrack_ep0300.pth.tar"
    ),
    ("litetrack", "config_path"): r"ignores\Models\litetrack\config\B6_cae_center_got10k_ep100.yaml",
    ("litetrack", "checkpoint_path"): (
        r"ignores\Models\litetrack\B6_cae_center_got10k_ep100\LiteTrack_ep0100.pth.tar"
    ),
    ("litetrack", "device"): "cuda",
    ("mixformerv2", "config_path"): r"ignores\Models\mixformerv2\config\288_depth8_score.yaml",
    ("mixformerv2", "checkpoint_path"): r"ignores\Models\mixformerv2\models\mixformerv2_base.pth.tar",
    ("mixformerv2", "variant"): "online",
}


class FullTestSetupApp:
    """Tkinter setup flow for the BigTrack full-test tool."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("BigTracker fulltest setup")
        self.root.geometry("940x720")

        self.predictor_name = tk.StringVar(value="adaptive_kalman")
        self.matcher_name = tk.StringVar(value="fft")
        self.policy_name = tk.StringVar(value="score_gated")
        self.input_kind = tk.StringVar(value="video")
        self.input_path = tk.StringVar(value=r"ignores\girl_dance.mp4")
        self.folder_fps = tk.StringVar(value="30.0")
        self.log_jsonl = tk.BooleanVar(value=False)
        self.log_path = tk.StringVar(value="logs/bigtrack_fulltest.jsonl")
        self.config_vars: dict[str, dict[str, tk.StringVar]] = {}

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        self.select_tab = ttk.Frame(self.notebook)
        self.config_tab = ttk.Frame(self.notebook)
        self.source_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.select_tab, text="1 Select")
        self.notebook.add(self.config_tab, text="2 Config")
        self.notebook.add(self.source_tab, text="3 Source")

        self._build_select_tab()
        self._build_config_tab()
        self._build_source_tab()

    def run(self) -> None:
        self.root.mainloop()

    def _build_select_tab(self) -> None:
        outer = ttk.Frame(self.select_tab, padding=18)
        outer.pack(fill="both", expand=True)
        self._combo_group(outer, "Predictor", self.predictor_name, tuple(PREDICTORS))
        self._combo_group(outer, "Matcher", self.matcher_name, tuple(MATCHERS))
        self._combo_group(outer, "BigTrack", self.policy_name, tuple(POLICIES))
        ttk.Button(outer, text="Next", command=lambda: self.notebook.select(self.config_tab)).pack(anchor="e", pady=24)

    def _build_config_tab(self) -> None:
        outer = ttk.Frame(self.config_tab)
        outer.pack(fill="both", expand=True)
        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)
        canvas = tk.Canvas(body, highlightthickness=0)
        scrollbar = ttk.Scrollbar(body, orient="vertical", command=canvas.yview)
        self.config_fields_frame = ttk.Frame(canvas, padding=16)
        self.config_fields_frame.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.config_fields_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self._refresh_config_fields()

        buttons = ttk.Frame(self.config_tab, padding=(16, 8))
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Back", command=lambda: self.notebook.select(self.select_tab)).pack(side="left")
        ttk.Button(buttons, text="Reload selected configs", command=self._refresh_config_fields).pack(side="left", padx=8)
        ttk.Button(buttons, text="Next", command=lambda: self.notebook.select(self.source_tab)).pack(side="right")

    def _build_source_tab(self) -> None:
        outer = ttk.Frame(self.source_tab, padding=18)
        outer.pack(fill="both", expand=True)
        source_box = ttk.LabelFrame(outer, text="Source", padding=12)
        source_box.pack(fill="x")
        ttk.Radiobutton(source_box, text="Video", variable=self.input_kind, value="video").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(source_box, text="Image folder", variable=self.input_kind, value="folder").grid(
            row=0,
            column=1,
            sticky="w",
            padx=12,
        )
        ttk.Label(source_box, text="Path").grid(row=1, column=0, sticky="w", pady=8)
        ttk.Entry(source_box, textvariable=self.input_path, width=88).grid(row=1, column=1, sticky="ew", pady=8)
        ttk.Button(source_box, text="Browse", command=self._browse_source).grid(row=1, column=2, padx=8)
        ttk.Label(source_box, text="Folder FPS").grid(row=2, column=0, sticky="w")
        ttk.Entry(source_box, textvariable=self.folder_fps, width=12).grid(row=2, column=1, sticky="w")
        source_box.columnconfigure(1, weight=1)

        log_box = ttk.LabelFrame(outer, text="Logging", padding=12)
        log_box.pack(fill="x", pady=12)
        ttk.Label(log_box, text="JSONL path").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(log_box, textvariable=self.log_path, width=88).grid(
            row=0,
            column=1,
            sticky="ew",
            pady=4,
        )
        ttk.Button(log_box, text="Browse", command=self._browse_log_path).grid(row=0, column=2, padx=8)
        ttk.Checkbutton(log_box, text="Enable JSONL logging", variable=self.log_jsonl).grid(
            row=1,
            column=1,
            sticky="w",
            pady=4,
        )
        log_box.columnconfigure(1, weight=1)

        buttons = ttk.Frame(outer)
        buttons.pack(fill="x", pady=24)
        ttk.Button(buttons, text="Back", command=lambda: self.notebook.select(self.config_tab)).pack(side="left")
        ttk.Button(buttons, text="Run", command=self._run_tracker).pack(side="right")

    def _combo_group(self, parent, label: str, variable: tk.StringVar, values: tuple[str, ...]) -> None:
        frame = ttk.LabelFrame(parent, text=label, padding=12)
        frame.pack(fill="x", pady=8)
        combo = ttk.Combobox(frame, textvariable=variable, values=values, state="readonly")
        combo.pack(fill="x")

    def _refresh_config_fields(self) -> None:
        for child in self.config_fields_frame.winfo_children():
            child.destroy()
        self.config_vars.clear()
        self._config_section("Predictor", self.predictor_name.get(), PREDICTORS[self.predictor_name.get()].config_type)
        self._config_section("Matcher", self.matcher_name.get(), MATCHERS[self.matcher_name.get()].config_type)
        self._config_section("BigTrack", self.policy_name.get(), POLICIES[self.policy_name.get()].config_type)
        self._config_section("Runner", "runner", RunnerConfig)

    def _config_section(self, title: str, key: str, config_type: type | None) -> None:
        box = ttk.LabelFrame(self.config_fields_frame, text=f"{title}: {key}", padding=12)
        box.pack(fill="x", pady=8)
        self.config_vars[key] = {}
        if config_type is None:
            ttk.Label(box, text="No config fields.").pack(anchor="w")
            return

        defaults = config_type()
        for row, field in enumerate(dataclasses.fields(config_type)):
            if key == "runner" and field.name in {"log_jsonl", "log_path"}:
                continue
            value = DEFAULT_OVERRIDES.get((key, field.name), getattr(defaults, field.name))
            variable = tk.StringVar(value="" if value is None else str(value))
            self.config_vars[key][field.name] = variable
            ttk.Label(box, text=field.name).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=3)
            ttk.Entry(box, textvariable=variable, width=90).grid(row=row, column=1, sticky="ew", pady=3)
        box.columnconfigure(1, weight=1)

    def _browse_source(self) -> None:
        if self.input_kind.get() == "folder":
            path = filedialog.askdirectory(title="Choose image folder")
        else:
            path = filedialog.askopenfilename(
                title="Choose video",
                filetypes=(("Video", "*.mp4 *.mkv *.avi *.mov"), ("All files", "*.*")),
            )
        if path:
            self.input_path.set(path)

    def _browse_log_path(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Choose JSONL log path",
            defaultextension=".jsonl",
            filetypes=(("JSON Lines", "*.jsonl"), ("All files", "*.*")),
        )
        if path:
            self.log_path.set(path)

    def _run_tracker(self) -> None:
        try:
            predictor = self._build_component(PREDICTORS, self.predictor_name.get())
            matcher = self._build_component(MATCHERS, self.matcher_name.get())
            policy_spec = POLICIES[self.policy_name.get()]
            policy_config = self._build_config(self.policy_name.get(), policy_spec.config_type)
            tracker = policy_spec.factory(policy_config, predictor=predictor, matcher=matcher)
            runner_config = dataclasses.replace(
                self._build_config("runner", RunnerConfig),
                log_jsonl=bool(self.log_jsonl.get()),
                log_path=self.log_path.get(),
            )
            source = build_frame_source(
                self.input_kind.get(),
                self.input_path.get(),
                float(self.folder_fps.get()),
            )
        except Exception as error:
            messagebox.showerror("BigTracker setup error", str(error))
            return

        self.root.destroy()
        FullTestRunner(tracker=tracker, source=source, config=runner_config).run()

    def _build_component(self, specs: dict[str, ComponentSpec], name: str):
        spec = specs[name]
        return spec.factory(self._build_config(name, spec.config_type))

    def _build_config(self, key: str, config_type: type | None):
        if config_type is None:
            return None
        values = {}
        for field in dataclasses.fields(config_type):
            if key == "runner" and field.name in {"log_jsonl", "log_path"}:
                continue
            raw = self.config_vars.get(key, {}).get(field.name)
            values[field.name] = _parse_value(raw.get() if raw is not None else "")
        return config_type(**values)


def _parse_value(raw: str):
    text = raw.strip()
    if text == "":
        return None
    if text.lower() == "none":
        return None
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text


def main() -> None:
    FullTestSetupApp().run()


if __name__ == "__main__":
    main()
