from __future__ import annotations

import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple

from BigTracker.big_track import BigTrack
from BigTracker.state import BigTrackState, TrackingOutput
from BigTracker.types import Box

from tests.fulltest.frame_source import Frame, FrameSource


@dataclass
class RunnerConfig:
    """Interactive full-test runtime settings."""

    window_name: str = "BigTracker fulltest"
    start_paused: bool = True
    continuous: bool = False
    frame_delay_ms: int = 1
    print_every_n_frames: int = 1
    draw_tracker_box: bool = True
    max_timing_samples: int = 300


@dataclass
class TimingStats:
    """Rolling runtime timing metrics shown on the OpenCV window."""

    max_samples: int
    update_ms: Deque[float] = field(default_factory=deque)
    frame_ms: Deque[float] = field(default_factory=deque)
    last_frame_time: Optional[float] = None

    def record_frame_start(self) -> None:
        """Record wall-clock frame spacing for fps and jitter."""

        now = time.perf_counter()
        if self.last_frame_time is not None:
            self._append(self.frame_ms, (now - self.last_frame_time) * 1000.0)
        self.last_frame_time = now

    def record_update_ms(self, value: float) -> None:
        """Record one tracker update duration in milliseconds."""

        self._append(self.update_ms, value)

    def summary(self) -> str:
        """Return compact timing text for the display overlay."""

        frame_avg = _mean(self.frame_ms)
        update_avg = _mean(self.update_ms)
        fps = 1000.0 / frame_avg if frame_avg > 0.0 else 0.0
        jitter = statistics.pstdev(self.frame_ms) if len(self.frame_ms) > 1 else 0.0
        p99 = _percentile(tuple(self.update_ms), 99.0)
        return f"fps={fps:5.1f} jitter={jitter:5.1f}ms update={update_avg:5.1f}ms p99={p99:5.1f}ms"

    def _append(self, values: Deque[float], value: float) -> None:
        """Append one timing sample with a fixed rolling limit."""

        values.append(float(value))
        while len(values) > self.max_samples:
            values.popleft()


@dataclass
class ViewState:
    """Zoom state for displaying frames without changing tracker coordinates."""

    zoom: float = 1.0
    center: Optional[Tuple[float, float]] = None
    last_view: Tuple[float, float, float] = (0.0, 0.0, 1.0)

    def zoom_at(self, x: int, y: int, factor: float) -> None:
        """Zoom around the mouse cursor in the latest displayed view."""

        old_zoom = self.zoom
        new_zoom = max(1.0, min(8.0, old_zoom * factor))
        if abs(new_zoom - old_zoom) < 1e-6:
            return

        view_x, view_y, view_scale = self.last_view
        image_x = view_x + float(x) / max(view_scale, 1e-6)
        image_y = view_y + float(y) / max(view_scale, 1e-6)
        self.zoom = new_zoom
        self.center = (image_x, image_y)

    def reset(self) -> None:
        """Reset display zoom to the full image."""

        self.zoom = 1.0
        self.center = None
        self.last_view = (0.0, 0.0, 1.0)


class FullTestRunner:
    """Interactive OpenCV runner for feeding frames into a BigTrack instance."""

    def __init__(self, tracker: BigTrack, source: FrameSource, config: RunnerConfig) -> None:
        """Store tracker, frame source, and runtime UI configuration."""

        self.tracker = tracker
        self.source = source
        self.config = config
        self.timing = TimingStats(max_samples=config.max_timing_samples)
        self.view = ViewState()
        self.paused = config.start_paused
        self.continuous = config.continuous
        self.current_frame: Optional[Frame] = None
        self.latest_output: Optional[TrackingOutput] = None
        self.frame_step_requested = False
        self.quit_requested = False

    def run(self) -> None:
        """Start the interactive loop."""

        cv2 = _require_cv2()
        cv2.namedWindow(self.config.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.config.window_name, self._on_mouse)
        self._print_controls()

        try:
            while not self.quit_requested:
                should_advance = self.continuous and not self.paused
                should_advance = should_advance or self.frame_step_requested
                if should_advance:
                    self.frame_step_requested = False
                    if not self._advance_one_frame():
                        self.paused = True

                if self.current_frame is not None:
                    display = self._render_frame(self.current_frame)
                    cv2.imshow(self.config.window_name, display)

                key = cv2.waitKey(self.config.frame_delay_ms) & 0xFF
                self._handle_key(key)
        finally:
            self.source.close()
            cv2.destroyWindow(self.config.window_name)

    def _advance_one_frame(self) -> bool:
        """Read one frame, run tracker if initialized, and print state."""

        frame = self.source.read()
        if frame is None:
            print("End of source.")
            return False

        self.current_frame = frame
        self.timing.record_frame_start()

        if self.tracker.get_state() is not None:
            start = time.perf_counter()
            self.latest_output = self.tracker.update(frame)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.timing.record_update_ms(elapsed_ms)

        self._print_state(frame)
        return True

    def _handle_key(self, key: int) -> None:
        """Handle keyboard controls for playback and tracker commands."""

        if key in (255, -1):
            return
        if key in (27, ord("q")):
            self.quit_requested = True
        elif key == ord(" "):
            self.paused = not self.paused
        elif key == ord("c"):
            self.continuous = not self.continuous
            self.paused = not self.continuous
        elif key == ord("n"):
            self.frame_step_requested = True
            self.paused = True
        elif key == ord("i"):
            self._initialize_tracker_from_roi()
        elif key == ord("r"):
            self.tracker.reset()
            self.latest_output = None
            print("Tracker reset.")
        elif key in (ord("+"), ord("=")):
            self._keyboard_zoom(1.25)
        elif key in (ord("-"), ord("_")):
            self._keyboard_zoom(0.8)
        elif key == ord("0"):
            self.view.reset()

    def _initialize_tracker_from_roi(self) -> None:
        """Let the user draw an ROI and initialize BigTrack from that box."""

        if self.current_frame is None:
            print("Cannot initialize: no frame loaded. Press 'n' first.")
            return

        cv2 = _require_cv2()
        self.paused = True
        roi_window = f"{self.config.window_name} init"
        roi = cv2.selectROI(roi_window, self.current_frame.image, showCrosshair=True)
        cv2.destroyWindow(roi_window)

        x, y, width, height = roi
        if width <= 0 or height <= 0:
            print("Initialization cancelled.")
            return

        box = (float(x), float(y), float(width), float(height))
        self.tracker.initialize(self.current_frame, box)
        self.latest_output = self.tracker.get_output()
        print(f"Tracker initialized: box={_fmt_box(box)}")
        self._print_state(self.current_frame, force=True)

    def _render_frame(self, frame: Frame):
        """Render image, tracker box, and timing overlay for the OpenCV window."""

        cv2 = _require_cv2()
        image = frame.image.copy()
        if self.config.draw_tracker_box and self.latest_output and self.latest_output.box:
            _draw_box(image, self.latest_output.box, color=(0, 255, 0), thickness=2)

        display = _apply_zoom(image, self.view)
        cv2.rectangle(display, (8, 8), (690, 38), (0, 0, 0), thickness=-1)
        cv2.putText(
            display,
            self.timing.summary(),
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return display

    def _print_state(self, frame: Frame, force: bool = False) -> None:
        """Print internal tracker state to the command line."""

        state = self.tracker.get_state()
        if state is None:
            if force:
                print(f"frame={frame.idx} tracker=not_initialized")
            return
        if not force and frame.idx % max(self.config.print_every_n_frames, 1) != 0:
            return

        pred = state.prediction
        print(
            "frame={frame_idx} ts={timestamp:.3f} mode={mode} "
            "pos={pos} vel={vel} size={size} size_vel={size_vel} "
            "score={score:.3f} uncertainty={uncertainty:.3f} output={box}".format(
                frame_idx=frame.idx,
                timestamp=frame.timestamp,
                mode=state.mode.value,
                pos=_fmt_pair(pred.target_pos),
                vel=_fmt_pair(pred.target_velocity),
                size=_fmt_pair(pred.target_size),
                size_vel=_fmt_pair(pred.target_size_velocity),
                score=pred.last_score,
                uncertainty=pred.uncertainty,
                box=_fmt_box(state.output.box),
            )
        )

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        """Handle mouse-wheel zoom."""

        cv2 = _require_cv2()
        if event == cv2.EVENT_MOUSEWHEEL:
            self.view.zoom_at(x, y, 1.25 if flags > 0 else 0.8)

    def _keyboard_zoom(self, factor: float) -> None:
        """Zoom around the center of the current image."""

        if self.current_frame is None:
            return
        height, width = self.current_frame.image.shape[:2]
        self.view.zoom_at(width // 2, height // 2, factor)

    def _print_controls(self) -> None:
        """Print available UI controls once at startup."""

        print("Controls:")
        print("  space: pause/resume")
        print("  c: toggle continuous playback")
        print("  n: next frame")
        print("  i: initialize tracker from drawn ROI")
        print("  r: reset tracker")
        print("  mouse wheel / +/-: zoom")
        print("  0: reset zoom")
        print("  q or esc: quit")


def _apply_zoom(image, view: ViewState):
    """Return a zoomed display image without changing tracker coordinates."""

    cv2 = _require_cv2()
    height, width = image.shape[:2]
    if view.zoom <= 1.0:
        view.last_view = (0.0, 0.0, 1.0)
        return image

    crop_width = max(1, int(round(width / view.zoom)))
    crop_height = max(1, int(round(height / view.zoom)))
    center_x, center_y = view.center or (width / 2.0, height / 2.0)
    left = int(round(center_x - crop_width / 2.0))
    top = int(round(center_y - crop_height / 2.0))
    left = max(0, min(width - crop_width, left))
    top = max(0, min(height - crop_height, top))
    crop = image[top : top + crop_height, left : left + crop_width]
    view.center = (left + crop_width / 2.0, top + crop_height / 2.0)
    view.last_view = (float(left), float(top), width / float(crop_width))
    return cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)


def _draw_box(image, box: Box, color: Tuple[int, int, int], thickness: int) -> None:
    """Draw one x, y, width, height box."""

    cv2 = _require_cv2()
    x, y, width, height = box
    p1 = (int(round(x)), int(round(y)))
    p2 = (int(round(x + width)), int(round(y + height)))
    cv2.rectangle(image, p1, p2, color, thickness)


def _mean(values: Deque[float]) -> float:
    """Return mean for a possibly empty deque."""

    if not values:
        return 0.0
    return sum(values) / len(values)


def _percentile(values: Tuple[float, ...], percentile: float) -> float:
    """Return nearest-rank percentile for timing samples."""

    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * percentile / 100.0))
    return ordered[max(0, min(len(ordered) - 1, index))]


def _fmt_pair(pair) -> str:
    """Format a 2D pair for terminal output."""

    return f"({float(pair[0]):.2f}, {float(pair[1]):.2f})"


def _fmt_box(box) -> str:
    """Format an optional x, y, width, height box for terminal output."""

    if box is None:
        return "None"
    return "({:.2f}, {:.2f}, {:.2f}, {:.2f})".format(
        float(box[0]),
        float(box[1]),
        float(box[2]),
        float(box[3]),
    )


def _require_cv2():
    """Import OpenCV only when the full test harness is used."""

    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("tests/fulltest requires opencv-python") from error
    return cv2
