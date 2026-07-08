from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from BigTracker import (  # noqa: E402
    AdaptiveKalmanPredictorConfig,
    AdaptiveKalmanPredictorModel,
    AlphaBetaPredictorConfig,
    AlphaBetaPredictorModel,
    ConstantAccelerationKalmanPredictorConfig,
    ConstantAccelerationKalmanPredictorModel,
    FftMatcherConfig,
    FftMatcherModel,
    HistoryPredictorConfig,
    HistoryPredictorModel,
    KalmanPredictorConfig,
    KalmanPredictorModel,
    LiteTrackMatcherConfig,
    LiteTrackMatcherModel,
    MixFormerV2MatcherConfig,
    MixFormerV2MatcherModel,
    NanoTrackMatcherConfig,
    NanoTrackMatcherModel,
    OSTrackMatcherConfig,
    OSTrackMatcherModel,
    ScoreGatedBigTrack,
    ScoreGatedBigTrackConfig,
    SimpleBigTrack,
)
from tests.fulltest.frame_source import build_frame_source  # noqa: E402
from tests.fulltest.runner import FullTestRunner, RunnerConfig  # noqa: E402


# -----------------------------------------------------------------------------
# Input config
# -----------------------------------------------------------------------------

# Choose "video" for mp4/mkv/avi files, or "folder" for sorted image folders.
INPUT_KIND = "video"

# Put your video file path or image folder path here.
INPUT_PATH = r"ignores\girl_dance.mp4"

# Used only when INPUT_KIND == "folder"; timestamps become frame_idx / FOLDER_FPS.
FOLDER_FPS = 30.0


# -----------------------------------------------------------------------------
# Predictor config
# -----------------------------------------------------------------------------

# Current supported values:
# "kalman", "adaptive_kalman", "alpha_beta", "history", "constant_accel_kalman".
PREDICTOR_KIND = "adaptive_kalman"

# Kalman process noise: higher means prediction follows motion changes faster.
# Kalman measurement noise: higher means matcher measurements are trusted less.
# default_covariance controls initial uncertainty for x/y/w/h scalar filters.
KALMAN_CONFIG = KalmanPredictorConfig(
    process_noise_position=1.0,
    process_noise_size=0.5,
    measurement_noise_position=4.0,
    measurement_noise_size=2.0,
    default_covariance=(10.0, 0.0, 0.0, 10.0),
    min_size=(1.0, 1.0),
    reject_uncertainty_growth=1.5,
    clamp_to_frame=True,
)

ADAPTIVE_KALMAN_CONFIG = AdaptiveKalmanPredictorConfig(
    process_noise_position=1.0,
    process_noise_size=0.5,
    measurement_noise_position=4.0,
    measurement_noise_size=2.0,
    default_covariance=(10.0, 0.0, 0.0, 10.0),
    min_size=(1.0, 1.0),
    adaptive_measurement_noise=True,
    min_measurement_noise_scale=0.25,
    max_measurement_noise_scale=3.0,
    min_uncertainty=0.0,
    max_uncertainty=100.0,
    reject_uncertainty_growth=1.5,
    reject_covariance_growth=1.15,
    reject_velocity_damping=0.85,
    max_position_velocity=80.0,
    max_size_velocity=20.0,
    uncertainty_accept_decay=0.90,
    clamp_to_frame=True,
)

ALPHA_BETA_CONFIG = AlphaBetaPredictorConfig(
    alpha_position=0.85,
    beta_position=0.20,
    alpha_size=0.80,
    beta_size=0.15,
    min_size=(1.0, 1.0),
    max_position_velocity=80.0,
    max_size_velocity=20.0,
    max_position_acceleration=30.0,
    max_size_acceleration=10.0,
    reject_velocity_damping=0.85,
    reject_uncertainty_growth=1.0,
    accept_uncertainty_decay=0.85,
    clamp_to_frame=True,
)

HISTORY_CONFIG = HistoryPredictorConfig(
    history_length=8,
    velocity_window=4,
    velocity_smoothing=0.60,
    size_velocity_smoothing=0.60,
    min_size=(1.0, 1.0),
    max_position_velocity=80.0,
    max_size_velocity=20.0,
    max_position_acceleration=30.0,
    max_size_acceleration=10.0,
    reject_velocity_damping=0.85,
    reject_uncertainty_growth=1.0,
    accept_uncertainty_decay=0.90,
    clamp_to_frame=True,
)

CONSTANT_ACCEL_KALMAN_CONFIG = ConstantAccelerationKalmanPredictorConfig(
    process_noise_position=1.0,
    process_noise_size=0.5,
    process_noise_velocity=0.5,
    process_noise_size_velocity=0.25,
    process_noise_acceleration=0.25,
    process_noise_size_acceleration=0.10,
    measurement_noise_position=4.0,
    measurement_noise_size=2.0,
    min_size=(1.0, 1.0),
    adaptive_measurement_noise=True,
    max_position_velocity=80.0,
    max_size_velocity=20.0,
    max_position_acceleration=30.0,
    max_size_acceleration=10.0,
    reject_velocity_damping=0.85,
    reject_acceleration_damping=0.65,
    reject_uncertainty_growth=1.5,
    reject_covariance_growth=1.15,
    uncertainty_accept_decay=0.90,
    clamp_to_frame=True,
)


# -----------------------------------------------------------------------------
# Matcher config
# -----------------------------------------------------------------------------

# Current supported values: "fft", "nanotrack", "ostrack", "litetrack", "mixformerv2".
MATCHER_KIND = "mixformerv2"

# template_area_factor controls template crop size around the initialized target.
# search_area_factor controls search crop size around the predicted target center.
# uncertain/recovery factors are used when a real policy enters those modes.
# max_best_templates exists, but SimpleBigTrack disables template updates.
FFT_CONFIG = FftMatcherConfig(
    template_area_factor=2.0,
    search_area_factor=2.5,
    uncertain_search_area_factor=3.25,
    recovery_search_area_factor=4.0,
    min_crop_size=16,
    max_best_templates=5,
    peak_exclusion_radius=8,
)

# NanoTrack source lives under ignores\Trackers, while model assets live under
# ignores\Models.
NANOTRACK_VARIANT = "nanotrackv3"
NANOTRACK_VARIANTS = {
    "nanotrackv1": {
        "config_path": r"ignores\Models\nanotrack\config\configv1.yaml",
        "checkpoint_path": r"ignores\Models\nanotrack\pretrained\nanotrackv1.pth",
        "backbone_path": r"ignores\Models\nanotrack\nanotrackv1\nanotrack_backbone_sim.onnx",
        "head_path": r"ignores\Models\nanotrack\nanotrackv1\nanotrack_head_sim.onnx",
    },
    "nanotrackv2": {
        "config_path": r"ignores\Models\nanotrack\config\configv2.yaml",
        "checkpoint_path": r"ignores\Models\nanotrack\pretrained\nanotrackv2.pth",
        "backbone_path": r"ignores\Models\nanotrack\nanotrackv2\nanotrack_backbone_sim.onnx",
        "head_path": r"ignores\Models\nanotrack\nanotrackv2\nanotrack_head_sim.onnx",
    },
    "nanotrackv3": {
        "config_path": r"ignores\Models\nanotrack\config\configv3.yaml",
        "checkpoint_path": r"ignores\Models\nanotrack\pretrained\nanotrackv3.pth",
        "backbone_path": r"ignores\Models\nanotrack\nanotrackv3\nanotrack_backbone.onnx",
        "head_path": r"ignores\Models\nanotrack\nanotrackv3\nanotrack_head.onnx",
    },
}
NANOTRACK_CONFIG = NanoTrackMatcherConfig(
    backend="onnx",  # "torch" for .pth, or "onnx" for ONNX Runtime.
    source_root=r"ignores\Trackers\NanoTrack",
    **NANOTRACK_VARIANTS[NANOTRACK_VARIANT],
    onnx_provider="cpu",  # Use "cuda" only with onnxruntime-gpu installed.
    device=None,
    max_best_templates=5,
)

# OSTrack source lives under ignores\Trackers. Model config/checkpoint assets
# live under ignores\Models.
OSTRACK_VARIANT = "vitb_256_mae_32x4_ep300"
OSTRACK_VARIANTS = {
    "vitb_256_mae_32x4_ep300": {
        "config_path": r"ignores\Models\Ostrack\config\vitb_256_mae_32x4_ep300.yaml",
        "checkpoint_path": r"ignores\Models\Ostrack\models\vitb_256_mae_32x4_ep300\OSTrack_ep0300.pth.tar",
    },
    "vitb_256_mae_ce_32x4_got10k_ep100": {
        "config_path": r"ignores\Models\Ostrack\config\vitb_256_mae_ce_32x4_got10k_ep100.yaml",
        "checkpoint_path": r"ignores\Models\Ostrack\models\vitb_256_mae_ce_32x4_got10k_ep100\OSTrack_ep0100.pth.tar",
    },
    "vitb_384_mae_32x4_ep300": {
        "config_path": r"ignores\Models\Ostrack\config\vitb_384_mae_32x4_ep300.yaml",
        "checkpoint_path": r"ignores\Models\Ostrack\models\vitb_384_mae_32x4_ep300\OSTrack_ep0300.pth.tar",
    },
    "vitb_384_mae_ce_32x4_ep300": {
        "config_path": r"ignores\Models\Ostrack\config\vitb_384_mae_ce_32x4_ep300.yaml",
        "checkpoint_path": r"ignores\Models\Ostrack\models\vitb_384_mae_ce_32x4_ep300\OSTrack_ep0300.pth.tar",
    },
    "vitb_384_mae_ce_32x4_got10k_ep100": {
        "config_path": r"ignores\Models\Ostrack\config\vitb_384_mae_ce_32x4_got10k_ep100.yaml",
        "checkpoint_path": r"ignores\Models\Ostrack\models\vitb_384_mae_ce_32x4_got10k_ep100\OSTrack_ep0100.pth.tar",
    },
}
OSTRACK_CONFIG = OSTrackMatcherConfig(
    source_root=r"ignores\Trackers\OSTrack",
    **OSTRACK_VARIANTS[OSTRACK_VARIANT],
    device=None,
    max_best_templates=5,
)

# LiteTrack source lives under ignores\Trackers. Model config/checkpoint assets
# live under ignores\Models.
LITETRACK_VARIANT = "B6_cae_center_got10k_ep100"
LITETRACK_VARIANTS = {
    "B6_cae_center_got10k_ep100": {
        "config_path": r"ignores\Models\litetrack\config\B6_cae_center_got10k_ep100.yaml",
        "checkpoint_path": r"ignores\Models\litetrack\B6_cae_center_got10k_ep100\LiteTrack_ep0100.pth.tar",
    },
    "B8_cae_center_all_ep300": {
        "config_path": r"ignores\Models\litetrack\config\B8_cae_center_all_ep300.yaml",
        "checkpoint_path": r"ignores\Models\litetrack\B8_cae_center_all_ep300\LiteTrack_ep0300.pth.tar",
    },
    "B8_cae_center_got10k_ep100": {
        "config_path": r"ignores\Models\litetrack\config\B8_cae_center_got10k_ep100.yaml",
        "checkpoint_path": r"ignores\Models\litetrack\B8_cae_center_got10k_ep100\LiteTrack_ep0100.pth.tar",
    },
    "B9_cae_center_all_ep300": {
        "config_path": r"ignores\Models\litetrack\config\B9_cae_center_all_ep300.yaml",
        "checkpoint_path": r"ignores\Models\litetrack\B9_cae_center_all_ep300\LiteTrack_ep0300.pth.tar",
    },
    "B9_cae_center_got10k_ep100": {
        "config_path": r"ignores\Models\litetrack\config\B9_cae_center_got10k_ep100.yaml",
        "checkpoint_path": r"ignores\Models\litetrack\B9_cae_center_got10k_ep100\LiteTrack_ep0100.pth.tar",
    },
}
LITETRACK_CONFIG = LiteTrackMatcherConfig(
    source_root=r"ignores\Trackers\LiteTrack",
    **LITETRACK_VARIANTS[LITETRACK_VARIANT],
    device="cuda",
    max_best_templates=5,
)

# MixFormerV2 source lives under ignores\Trackers. Model config/checkpoint assets
# live under ignores\Models.
MIXFORMERV2_VARIANT = "base"
MIXFORMERV2_VARIANTS = {
    "base": {
        "config_path": r"ignores\Models\mixformerv2\config\288_depth8_score.yaml",
        "checkpoint_path": r"ignores\Models\mixformerv2\models\mixformerv2_base.pth.tar",
    },
    "small": {
        "config_path": r"ignores\Models\mixformerv2\config\224_depth4_mlp1_score.yaml",
        "checkpoint_path": r"ignores\Models\mixformerv2\models\mixformerv2_small.pth.tar",
    },
}
MIXFORMERV2_CONFIG = MixFormerV2MatcherConfig(
    source_root=r"ignores\Trackers\MixFormerV2",
    **MIXFORMERV2_VARIANTS[MIXFORMERV2_VARIANT],
    device=None,
    variant="online",
    max_best_templates=5,
)


# -----------------------------------------------------------------------------
# BigTrack policy config
# -----------------------------------------------------------------------------

# Current supported values: "simple", "score_gated".
# SimpleBigTrack creates one candidate, accepts the first matcher result, and
# never updates templates. It is intentionally dumb for first integration tests.
POLICY_KIND = "score_gated"

SCORE_GATED_CONFIG = ScoreGatedBigTrackConfig(
    th_good=0.90,
    th_bad=0.75,
    max_center_error=0.35,
    max_size_error=0.50,
    predictor_uncertainty_scale=10.0,
    recovery_after=3,
    lost_after=10,
    template_update_interval=5,
)


# -----------------------------------------------------------------------------
# UI/runtime config
# -----------------------------------------------------------------------------

RUNNER_CONFIG = RunnerConfig(
    window_name="BigTracker fulltest",
    # The frame is always rendered to this size, so the OpenCV window stays fixed.
    window_width=1280,
    window_height=720,
    start_paused=True,
    continuous=False,
    frame_delay_ms=1,
    print_every_n_frames=1,
    draw_tracker_box=True,
    # Draw timing, playback state, and key controls directly on the window.
    draw_key_help=True,
    max_timing_samples=300,
)


def main() -> None:
    """Build configured tracker and start the interactive full test."""

    source = build_frame_source(
        input_kind=INPUT_KIND,
        input_path=INPUT_PATH,
        folder_fps=FOLDER_FPS,
    )
    predictor = _build_predictor()
    matcher = _build_matcher()
    tracker = _build_policy(predictor, matcher)
    runner = FullTestRunner(
        tracker=tracker,
        source=source,
        config=RUNNER_CONFIG,
    )
    runner.run()


def _build_predictor():
    """Create the predictor selected by PREDICTOR_KIND."""

    if PREDICTOR_KIND == "kalman":
        return KalmanPredictorModel(KALMAN_CONFIG)
    if PREDICTOR_KIND == "adaptive_kalman":
        return AdaptiveKalmanPredictorModel(ADAPTIVE_KALMAN_CONFIG)
    if PREDICTOR_KIND == "alpha_beta":
        return AlphaBetaPredictorModel(ALPHA_BETA_CONFIG)
    if PREDICTOR_KIND == "history":
        return HistoryPredictorModel(HISTORY_CONFIG)
    if PREDICTOR_KIND == "constant_accel_kalman":
        return ConstantAccelerationKalmanPredictorModel(CONSTANT_ACCEL_KALMAN_CONFIG)
    raise ValueError(f"Unknown PREDICTOR_KIND: {PREDICTOR_KIND!r}")


def _build_matcher():
    """Create the matcher selected by MATCHER_KIND."""

    if MATCHER_KIND == "fft":
        return FftMatcherModel(FFT_CONFIG)
    if MATCHER_KIND == "nanotrack":
        return NanoTrackMatcherModel(NANOTRACK_CONFIG)
    if MATCHER_KIND == "ostrack":
        return OSTrackMatcherModel(OSTRACK_CONFIG)
    if MATCHER_KIND == "litetrack":
        return LiteTrackMatcherModel(LITETRACK_CONFIG)
    if MATCHER_KIND == "mixformerv2":
        return MixFormerV2MatcherModel(MIXFORMERV2_CONFIG)
    raise ValueError(f"Unknown MATCHER_KIND: {MATCHER_KIND!r}")


def _build_policy(predictor, matcher):
    """Create the BigTrack policy selected by POLICY_KIND."""

    if POLICY_KIND == "simple":
        return SimpleBigTrack(predictor=predictor, matcher=matcher)
    if POLICY_KIND == "score_gated":
        return ScoreGatedBigTrack(
            predictor=predictor,
            matcher=matcher,
            config=SCORE_GATED_CONFIG,
        )
    raise ValueError(f"Unknown POLICY_KIND: {POLICY_KIND!r}")


if __name__ == "__main__":
    main()
