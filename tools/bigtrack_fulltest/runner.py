from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple

from BigTracker.big_track import BigTrack
from BigTracker.types import (
    BigTrackInitializeInput,
    BigTrackState,
    BigTrackUpdateInput,
    BigTrackUpdateOutput,
    Box,
    MatcherInitializeInput,
    PredictorInitializeInput,
)

from tools.bigtrack_fulltest.frame_source import Frame, FrameSource


@dataclass
class RunnerConfig:
    """Interactive full-test runtime settings."""

    window_name: str = "BigTracker"
    window_width: int = 1280
    window_height: int = 720
    start_paused: bool = True
    continuous: bool = False
    frame_delay_ms: int = 1
    draw_tracker_box: bool = True
    draw_key_help: bool = True
    max_timing_samples: int = 300


@dataclass
class TimingStats:
    max_samples: int
    update_ms: Deque[float] = field(default_factory=deque)
    frame_ms: Deque[float] = field(default_factory=deque)
    last_frame_time: Optional[float] = None

    def record_frame_start(self) -> None:
        now = time.perf_counter()
        if self.last_frame_time is not None:
            self._append(self.frame_ms, (now - self.last_frame_time) * 1000.0)
        self.last_frame_time = now

    def reset_frame_clock(self, *, clear_samples: bool = False) -> None:
        self.last_frame_time = None
        if clear_samples:
            self.frame_ms.clear()

    def record_update_ms(self, value: float) -> None:
        self._append(self.update_ms, value)

    def summary(self) -> str:
        frame_avg = _mean(self.frame_ms)
        update_avg = _mean(self.update_ms)
        tracker_fps_text = f"{1000.0 / update_avg:5.1f}" if update_avg > 0.0 else "  n/a"
        display_fps_text = f"{1000.0 / frame_avg:5.1f}" if frame_avg > 0.0 else "  n/a"
        p99 = _percentile(tuple(self.update_ms), 99.0)
        return f"tracker_fps={tracker_fps_text} display_fps={display_fps_text} update={update_avg:5.1f}ms p99={p99:5.1f}ms"

    def _append(self, values: Deque[float], value: float) -> None:
        values.append(float(value))
        while len(values) > self.max_samples:
            values.popleft()


@dataclass
class ViewState:
    zoom: float = 1.0
    center: Optional[Tuple[float, float]] = None
    last_view: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    drag_start: Optional[Tuple[int, int, Tuple[float, float]]] = None

    def zoom_at(self, x: int, y: int, factor: float) -> None:
        old_zoom = self.zoom
        new_zoom = max(1.0, min(12.0, old_zoom * factor))
        if abs(new_zoom - old_zoom) < 1e-6:
            return

        view_x, view_y, scale_x, scale_y = self.last_view
        image_x = view_x + float(x) / max(scale_x, 1e-6)
        image_y = view_y + float(y) / max(scale_y, 1e-6)
        self.zoom = new_zoom
        self.center = (image_x, image_y)

    def start_drag(self, x: int, y: int) -> None:
        center = self.center or self._current_center()
        self.drag_start = (x, y, center)

    def drag_to(self, x: int, y: int) -> None:
        if self.drag_start is None:
            return
        start_x, start_y, start_center = self.drag_start
        _, _, scale_x, scale_y = self.last_view
        dx = (float(x) - float(start_x)) / max(scale_x, 1e-6)
        dy = (float(y) - float(start_y)) / max(scale_y, 1e-6)
        self.center = (start_center[0] - dx, start_center[1] - dy)

    def end_drag(self) -> None:
        self.drag_start = None

    def reset(self) -> None:
        self.zoom = 1.0
        self.center = None
        self.last_view = (0.0, 0.0, 1.0, 1.0)
        self.drag_start = None

    def _current_center(self) -> Tuple[float, float]:
        view_x, view_y, scale_x, scale_y = self.last_view
        return (view_x + 0.5 / max(scale_x, 1e-6), view_y + 0.5 / max(scale_y, 1e-6))


class FullTestRunner:
    """Interactive OpenCV runner for feeding frames into one BigTrack instance."""

    def __init__(self, tracker: BigTrack, source: FrameSource, config: RunnerConfig) -> None:
        self.tracker = tracker
        self.source = source
        self.config = config
        self.timing = TimingStats(max_samples=config.max_timing_samples)
        self.frame_view = ViewState()
        self.paused = config.start_paused
        self.continuous = config.continuous
        self.current_frame: Optional[Frame] = None
        self.latest_output: Optional[BigTrackUpdateOutput] = None
        self.saved_state: Optional[BigTrackState] = None
        self.frame_step_requested = False
        self.quit_requested = False

    def run(self) -> None:
        cv2 = _require_cv2()
        cv2.namedWindow(self.config.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.config.window_name, self.config.window_width, self.config.window_height)
        cv2.setMouseCallback(self.config.window_name, self._on_frame_mouse)
        self._print_controls()

        try:
            while not self.quit_requested:
                if (self.continuous and not self.paused) or self.frame_step_requested:
                    self.frame_step_requested = False
                    if not self._advance_one_frame():
                        self.paused = True

                if self.current_frame is not None:
                    cv2.imshow(self.config.window_name, self._render_frame(self.current_frame))

                key = cv2.waitKey(self.config.frame_delay_ms) & 0xFF
                self._handle_key(key)
        finally:
            self.source.close()
            cv2.destroyWindow(self.config.window_name)

    def _advance_one_frame(self) -> bool:
        frame = self.source.read()
        if frame is None:
            print("End of source.")
            return False

        self.current_frame = frame
        self.timing.record_frame_start()
        if self._current_state() is not None:
            start = time.perf_counter()
            self.latest_output = self.tracker.update(BigTrackUpdateInput(frame=frame))
            self.timing.record_update_ms((time.perf_counter() - start) * 1000.0)
        return True

    def _handle_key(self, key: int) -> None:
        if key in (255, -1):
            return
        if key in (27, ord("q")):
            self.quit_requested = True
        elif key == ord(" "):
            self.paused = not self.paused
            self.timing.reset_frame_clock(clear_samples=True)
        elif key == ord("c"):
            self.continuous = not self.continuous
            self.paused = not self.continuous
            self.timing.reset_frame_clock(clear_samples=True)
        elif key == ord("n"):
            self.frame_step_requested = True
            self.paused = True
            self.timing.reset_frame_clock(clear_samples=True)
        elif key == ord("i"):
            self._initialize_tracker_from_roi()
        elif key == ord("r"):
            state = self._current_state()
            if state is not None:
                self.saved_state = state
            self.tracker.reset()
            self.latest_output = None
            self.timing.reset_frame_clock(clear_samples=True)
        elif key == ord("b"):
            self._initialize_tracker_from_saved_state()
        elif key in (ord("+"), ord("=")):
            self.frame_view.zoom_at(self.config.window_width // 2, self.config.window_height // 2, 1.25)
        elif key in (ord("-"), ord("_")):
            self.frame_view.zoom_at(self.config.window_width // 2, self.config.window_height // 2, 0.8)
        elif key == ord("0"):
            self.frame_view.reset()

    def _initialize_tracker_from_roi(self) -> None:
        if self.current_frame is None:
            print("Cannot initialize: no frame loaded. Press 'n' first.")
            return

        cv2 = _require_cv2()
        self.paused = True
        display = self._render_frame(self.current_frame, draw_overlay=False)
        cv2.imshow(self.config.window_name, display)
        roi = cv2.selectROI(self.config.window_name, display, showCrosshair=True)
        cv2.setMouseCallback(self.config.window_name, self._on_frame_mouse)

        x, y, width, height = roi
        if width <= 0 or height <= 0:
            return

        box = _display_box_to_image_box((float(x), float(y), float(width), float(height)), self.frame_view)
        self.tracker.initialize(BigTrackInitializeInput(frame=self.current_frame, box=box))
        self.latest_output = self.tracker.get_output()
        self.timing.reset_frame_clock(clear_samples=True)

    def _initialize_tracker_from_saved_state(self) -> None:
        if self.saved_state is None or self.current_frame is None:
            return

        state = self.saved_state
        if state.output is None or state.output.box is None:
            return

        self.tracker.initialize_from_state(
            BigTrackInitializeInput(
                frame=self.current_frame,
                box=state.output.box,
                predictor=PredictorInitializeInput(predictor_state=state.predictor_state),
                matcher=MatcherInitializeInput(
                    frame=self.current_frame,
                    box=state.output.box,
                    matcher_state=state.matcher_state,
                ),
                metadata=state.metadata,
            )
        )
        self.latest_output = self.tracker.get_output()
        self.timing.reset_frame_clock(clear_samples=True)

    def _render_frame(self, frame: Frame, draw_overlay: bool = True):
        image = frame.image.copy()
        if self.config.draw_tracker_box and self.latest_output and self.latest_output.box:
            _draw_box(image, self.latest_output.box, color=(0, 255, 0), thickness=2)

        display = _apply_zoom(
            image,
            self.frame_view,
            display_size=(self.config.window_width, self.config.window_height),
        )
        if draw_overlay:
            _draw_overlay(
                display=display,
                lines=self._general_lines(frame),
            )
        return display

    def _general_lines(self, frame: Frame) -> list[str]:
        state = self._current_state()
        output = self.latest_output
        lines = [
            f"{self.timing.summary()}   frame={frame.idx} ts={frame.timestamp:.3f}",
            f"{'PAUSED' if self.paused else 'RUNNING'} {'CONT' if self.continuous else 'STEP'} "
            f"{'INIT' if state is not None else 'NO INIT'} {'SAVED' if self.saved_state else 'NO SAVE'}",
        ]
        if state is not None:
            pred = state.predictor_state
            target_size = state.metadata.get("target_size")
            lines.append(
                "mode={mode} pos={pos} vel={vel} size={size} out={box}".format(
                    mode=state.mode.value,
                    pos=_fmt_pair(pred.target_pos),
                    vel=_fmt_pair(pred.target_velocity),
                    size=_fmt_pair(target_size) if target_size is not None else "None",
                    box=_fmt_box(output.box if output else None),
                )
            )
        if self.config.draw_key_help:
            lines.append("space pause | c continuous | n next | i init ROI | r save+reset | b restore | q quit")
        return lines

    def _on_frame_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        self._handle_view_mouse(self.frame_view, event, x, y, flags)

    def _handle_view_mouse(self, view: ViewState, event: int, x: int, y: int, flags: int) -> None:
        cv2 = _require_cv2()
        if event == cv2.EVENT_MOUSEWHEEL:
            view.zoom_at(x, y, 1.25 if flags > 0 else 0.8)
        elif event == cv2.EVENT_RBUTTONDOWN:
            view.start_drag(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_RBUTTON:
            view.drag_to(x, y)
        elif event == cv2.EVENT_RBUTTONUP:
            view.end_drag()

    def _current_state(self) -> Optional[BigTrackState]:
        try:
            return self.tracker.get_state()
        except RuntimeError:
            return None

    def _print_controls(self) -> None:
        print("Controls:")
        print("  space: pause/resume")
        print("  c: toggle continuous playback")
        print("  n: next frame")
        print("  i: initialize tracker from drawn ROI")
        print("  r: save current state and reset tracker")
        print("  b: restore saved tracker state")
        print("  mouse wheel: zoom")
        print("  right drag: pan")
        print("  0: reset zoom")
        print("  q or esc: quit")


def _apply_zoom(image, view: ViewState, display_size: Tuple[int, int]):
    cv2 = _require_cv2()
    height, width = image.shape[:2]
    display_width = max(1, int(display_size[0]))
    display_height = max(1, int(display_size[1]))
    if view.zoom <= 1.0:
        view.last_view = (0.0, 0.0, display_width / float(width), display_height / float(height))
        return cv2.resize(image, (display_width, display_height), interpolation=cv2.INTER_LINEAR)

    crop_width = max(1, min(width, int(round(width / view.zoom))))
    crop_height = max(1, min(height, int(round(height / view.zoom))))
    center_x, center_y = view.center or (width / 2.0, height / 2.0)
    left = int(round(center_x - crop_width / 2.0))
    top = int(round(center_y - crop_height / 2.0))
    left = max(0, min(width - crop_width, left))
    top = max(0, min(height - crop_height, top))
    crop = image[top : top + crop_height, left : left + crop_width]
    view.center = (left + crop_width / 2.0, top + crop_height / 2.0)
    view.last_view = (
        float(left),
        float(top),
        display_width / float(crop_width),
        display_height / float(crop_height),
    )
    return cv2.resize(crop, (display_width, display_height), interpolation=cv2.INTER_LINEAR)


def _display_box_to_image_box(display_box: Box, view: ViewState) -> Box:
    view_x, view_y, scale_x, scale_y = view.last_view
    x, y, width, height = display_box
    return (
        view_x + x / max(scale_x, 1e-6),
        view_y + y / max(scale_y, 1e-6),
        width / max(scale_x, 1e-6),
        height / max(scale_y, 1e-6),
    )


def _draw_overlay(display, lines: list[str]) -> None:
    cv2 = _require_cv2()
    line_height = 24
    panel_height = 16 + line_height * len(lines)
    cv2.rectangle(display, (8, 8), (display.shape[1] - 8, panel_height), (0, 0, 0), thickness=-1)
    for index, line in enumerate(lines):
        cv2.putText(
            display,
            line,
            (16, 30 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _draw_box(image, box: Box, color: Tuple[int, int, int], thickness: int) -> None:
    cv2 = _require_cv2()
    x, y, width, height = box
    p1 = (int(round(x)), int(round(y)))
    p2 = (int(round(x + width)), int(round(y + height)))
    cv2.rectangle(image, p1, p2, color, thickness)


def _mean(values: Deque[float]) -> float:
    return 0.0 if not values else sum(values) / len(values)


def _percentile(values: Tuple[float, ...], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * percentile / 100.0))
    return ordered[max(0, min(len(ordered) - 1, index))]


def _fmt_pair(pair) -> str:
    return f"({float(pair[0]):.2f}, {float(pair[1]):.2f})"


def _fmt_box(box) -> str:
    if box is None:
        return "None"
    return "({:.2f}, {:.2f}, {:.2f}, {:.2f})".format(
        float(box[0]),
        float(box[1]),
        float(box[2]),
        float(box[3]),
    )


def _require_cv2():
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("bigtrack_fulltest requires opencv-python") from error
    return cv2
