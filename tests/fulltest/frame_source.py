from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Frame:
    """Concrete test frame that satisfies BigTracker.types.FrameLike."""

    image: object
    idx: int
    timestamp: float
    source: str


class FrameSource:
    """Base API for frame readers used by the full test harness."""

    def read(self) -> Optional[Frame]:
        """Return the next frame, or None when the source is exhausted."""

        raise NotImplementedError

    def reset(self) -> None:
        """Rewind the source to the first frame."""

        raise NotImplementedError

    def close(self) -> None:
        """Release source resources."""


class FolderFrameSource(FrameSource):
    """Read image files from a folder in sorted filename order."""

    def __init__(self, folder_path: str, fps: float = 30.0) -> None:
        """Create a folder source from jpg/png-style image files."""

        self.folder_path = Path(folder_path)
        self.fps = max(float(fps), 1e-6)
        self.image_paths = _find_images(self.folder_path)
        self.index = 0

        if not self.image_paths:
            raise ValueError(f"No images found in folder: {self.folder_path}")

    def read(self) -> Optional[Frame]:
        """Read the next image as a BGR OpenCV array."""

        if self.index >= len(self.image_paths):
            return None

        cv2 = _require_cv2()
        path = self.image_paths[self.index]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV could not read image: {path}")

        frame = Frame(
            image=image,
            idx=self.index,
            timestamp=self.index / self.fps,
            source=str(path),
        )
        self.index += 1
        return frame

    def reset(self) -> None:
        """Rewind to the first image."""

        self.index = 0


class VideoFrameSource(FrameSource):
    """Read frames from a video file through OpenCV."""

    def __init__(self, video_path: str) -> None:
        """Open an mp4/mkv/avi-style video source."""

        cv2 = _require_cv2()
        self.video_path = Path(video_path)
        self.capture = cv2.VideoCapture(str(self.video_path))
        if not self.capture.isOpened():
            raise ValueError(f"OpenCV could not open video: {self.video_path}")

        fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps = fps if fps > 0.0 else 30.0
        self.index = 0

    def read(self) -> Optional[Frame]:
        """Read the next video frame as a BGR OpenCV array."""

        cv2 = _require_cv2()
        ok, image = self.capture.read()
        if not ok:
            return None

        timestamp_ms = float(self.capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        timestamp = timestamp_ms / 1000.0 if timestamp_ms > 0.0 else self.index / self.fps
        frame = Frame(
            image=image,
            idx=self.index,
            timestamp=timestamp,
            source=str(self.video_path),
        )
        self.index += 1
        return frame

    def reset(self) -> None:
        """Rewind the video to the first frame."""

        cv2 = _require_cv2()
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.index = 0

    def close(self) -> None:
        """Release the OpenCV capture handle."""

        self.capture.release()


def build_frame_source(input_kind: str, input_path: str, folder_fps: float) -> FrameSource:
    """Build a video or folder frame source from hardcoded test config."""

    kind = input_kind.strip().lower()
    if kind == "video":
        return VideoFrameSource(input_path)
    if kind == "folder":
        return FolderFrameSource(input_path, fps=folder_fps)
    raise ValueError(f"Unknown INPUT_KIND: {input_kind!r}. Use 'video' or 'folder'.")


def _find_images(folder_path: Path) -> Sequence[Path]:
    """Return sorted image paths from one folder."""

    if not folder_path.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    if not folder_path.is_dir():
        raise ValueError(f"Path is not a folder: {folder_path}")

    return tuple(
        path
        for path in sorted(folder_path.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _require_cv2():
    """Import OpenCV only when the full test harness is used."""

    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("tests/fulltest requires opencv-python") from error
    return cv2
