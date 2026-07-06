from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from BigTracker import (  # noqa: E402
    FftMatcherConfig,
    FftMatcherModel,
    KalmanPredictorConfig,
    KalmanPredictorModel,
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

# Current supported value: "kalman".
PREDICTOR_KIND = "kalman"

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
)


# -----------------------------------------------------------------------------
# Matcher config
# -----------------------------------------------------------------------------

# Current supported value: "fft".
MATCHER_KIND = "fft"

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


# -----------------------------------------------------------------------------
# BigTrack policy config
# -----------------------------------------------------------------------------

# Current supported value: "simple".
# SimpleBigTrack creates one candidate, accepts the first matcher result, and
# never updates templates. It is intentionally dumb for first integration tests.
POLICY_KIND = "simple"


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
    raise ValueError(f"Unknown PREDICTOR_KIND: {PREDICTOR_KIND!r}")


def _build_matcher():
    """Create the matcher selected by MATCHER_KIND."""

    if MATCHER_KIND == "fft":
        return FftMatcherModel(FFT_CONFIG)
    raise ValueError(f"Unknown MATCHER_KIND: {MATCHER_KIND!r}")


def _build_policy(predictor, matcher):
    """Create the BigTrack policy selected by POLICY_KIND."""

    if POLICY_KIND == "simple":
        return SimpleBigTrack(predictor=predictor, matcher=matcher)
    raise ValueError(f"Unknown POLICY_KIND: {POLICY_KIND!r}")


if __name__ == "__main__":
    main()
