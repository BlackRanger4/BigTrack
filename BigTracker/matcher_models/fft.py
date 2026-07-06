from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Optional, Sequence, Tuple

from BigTracker.matcher import MatcherModel
from BigTracker.state import MatchEvidence, MatcherState, SearchCandidate, TemplateCandidate
from BigTracker.types import Box, FrameLike, Point, Size, TrackerMode


@dataclass(frozen=True)
class FftMatcherConfig:
    """Configuration for the simple FFT correlation matcher."""

    template_area_factor: float = 2.0
    search_area_factor: float = 2.5
    uncertain_search_area_factor: float = 3.25
    recovery_search_area_factor: float = 4.0
    min_crop_size: int = 16
    max_best_templates: int = 5
    peak_exclusion_radius: int = 8


@dataclass(frozen=True)
class FftTemplate:
    """Template object owned by FftMatcherModel."""

    patch: Any
    spectrum: Any
    target_size: Size
    crop_size: Size
    source_frame_idx: int
    source_box: Box
    clipped: bool


@dataclass(frozen=True)
class _FftMatchResult:
    """Internal result for one template against one search candidate."""

    box: Box
    match_score: float
    ambiguity_score: float
    localization_score: float
    peak_value: float
    second_peak_value: float
    selected_template_index: int
    is_clipped: bool
    search_box: Box


class FftMatcherModel(MatcherModel):
    """Simple FFT cross-correlation matcher.

    This model is useful as a first working tracker backend. It crops its own
    template/search regions, normalizes them, computes `M(f) * conj(G(f))`,
    and returns a peak-based `MatchEvidence`.
    """

    def __init__(self, config: Optional[FftMatcherConfig] = None) -> None:
        """Create an FFT matcher with crop sizes and template queue limits."""

        self.config = config or FftMatcherConfig()

    def initialize_template(
        self,
        frame: FrameLike,
        target_pos: Point,
        target_size: Size,
    ) -> MatcherState:
        """Create matcher state with the first template as protected identity anchor."""

        template = self._build_template(frame, target_pos, target_size)
        return MatcherState(init_template=template)

    def extract_template(
        self,
        frame: FrameLike,
        target_pos: Point,
        target_size: Size,
        previous_state: MatcherState,
    ) -> TemplateCandidate:
        """Create a candidate template from a BigTrack-approved target region."""

        template = self._build_template(frame, target_pos, target_size)
        return TemplateCandidate(
            template=template,
            source_frame_idx=frame.idx,
            source_box=_center_size_to_box(target_pos, target_size),
            quality_score=1.0,
            identity_score=1.0,
            metadata={
                "matcher": "fft",
                "template_crop_size": template.crop_size,
                "was_clipped": template.clipped,
                "previous_best_template_count": len(previous_state.best_templates),
            },
        )

    def update_templates(
        self,
        state: MatcherState,
        template: TemplateCandidate,
    ) -> MatcherState:
        """Insert an approved template while keeping the initial template unchanged."""

        best_templates = tuple(state.best_templates) + (template.template,)
        if len(best_templates) > self.config.max_best_templates:
            best_templates = best_templates[-self.config.max_best_templates :]

        return replace(
            state,
            best_templates=best_templates,
            adaptive_template=template.template,
        )

    def match(
        self,
        frame: FrameLike,
        matcher_state: MatcherState,
        candidate: SearchCandidate,
        mode: TrackerMode,
    ) -> MatchEvidence:
        """Run FFT matching for one search candidate and return evidence only."""

        templates = self._collect_templates(matcher_state)
        if not templates:
            raise ValueError("FftMatcherModel.match requires at least one template")

        results = [
            self._match_one_template(frame, template, candidate, mode, index)
            for index, template in enumerate(templates)
        ]
        best = max(results, key=lambda result: result.match_score)
        scale_score = self._score_scale(candidate.predicted_target_size, best.box)
        identity_score = best.match_score

        return MatchEvidence(
            candidate_id=candidate.candidate_id,
            box=best.box,
            match_score=best.match_score,
            identity_score=identity_score,
            appearance_score=best.match_score,
            localization_score=best.localization_score,
            ambiguity_score=best.ambiguity_score,
            scale_score=scale_score,
            occlusion_score=_clamp01(1.0 - identity_score),
            is_clipped=best.is_clipped,
            metadata={
                "matcher": "fft",
                "mode": mode.value,
                "template_count": len(templates),
                "selected_template_index": best.selected_template_index,
                "peak_value": best.peak_value,
                "second_peak_value": best.second_peak_value,
                "search_box": best.search_box,
            },
        )

    def _build_template(self, frame: FrameLike, target_pos: Point, target_size: Size) -> FftTemplate:
        """Crop, normalize, and encode one FFT template."""

        np = _require_numpy()
        image = _as_gray_float(frame.image)
        crop_size = self._template_crop_size(target_size)
        patch, _, clipped = _crop_centered(image, target_pos, crop_size)
        patch = _normalize_patch(patch)
        return FftTemplate(
            patch=patch,
            spectrum=np.fft.fft2(patch),
            target_size=target_size,
            crop_size=(float(patch.shape[1]), float(patch.shape[0])),
            source_frame_idx=frame.idx,
            source_box=_center_size_to_box(target_pos, target_size),
            clipped=clipped,
        )

    def _match_one_template(
        self,
        frame: FrameLike,
        template: FftTemplate,
        candidate: SearchCandidate,
        mode: TrackerMode,
        template_index: int,
    ) -> _FftMatchResult:
        """Match one encoded template against one search crop."""

        np = _require_numpy()
        image = _as_gray_float(frame.image)
        search_size = self._search_crop_size(candidate.predicted_target_size, mode)
        search_patch, search_box, search_clipped = _crop_centered(
            image,
            candidate.search_center,
            search_size,
        )
        search_patch = _normalize_patch(search_patch)

        template_canvas = np.zeros_like(search_patch)
        _paste_center(template_canvas, template.patch)
        response = np.fft.ifft2(
            np.fft.fft2(search_patch) * np.conj(np.fft.fft2(template_canvas))
        ).real
        response = np.fft.fftshift(response)

        peak_y, peak_x = _peak_index(response)
        center_y = (response.shape[0] - 1) / 2.0
        center_x = (response.shape[1] - 1) / 2.0
        offset_x = float(peak_x - center_x)
        offset_y = float(peak_y - center_y)
        matched_center = (
            candidate.search_center[0] + offset_x,
            candidate.search_center[1] + offset_y,
        )
        matched_box = _center_size_to_box(matched_center, candidate.predicted_target_size)

        peak_value, second_peak_value, match_score, ambiguity_score = self._score_response(
            response,
            peak_y,
            peak_x,
        )
        distance = math.hypot(offset_x, offset_y)
        max_distance = max(search_patch.shape) / 2.0
        localization_score = _clamp01(1.0 - distance / max(max_distance, 1.0))

        return _FftMatchResult(
            box=matched_box,
            match_score=match_score,
            ambiguity_score=ambiguity_score,
            localization_score=localization_score,
            peak_value=peak_value,
            second_peak_value=second_peak_value,
            selected_template_index=template_index,
            is_clipped=search_clipped,
            search_box=search_box,
        )

    def _collect_templates(self, matcher_state: MatcherState) -> Sequence[FftTemplate]:
        """Return all templates that should vote during matching."""

        templates = [matcher_state.init_template]
        templates.extend(matcher_state.best_templates)
        if matcher_state.adaptive_template is not None:
            templates.append(matcher_state.adaptive_template)
        return tuple(template for template in templates if isinstance(template, FftTemplate))

    def _template_crop_size(self, target_size: Size) -> Size:
        """Choose a square template crop from the target size."""

        side = self.config.template_area_factor * _target_side(target_size)
        side = max(side, float(self.config.min_crop_size))
        return (side, side)

    def _search_crop_size(self, target_size: Size, mode: TrackerMode) -> Size:
        """Choose a square search crop from target size and tracker mode."""

        factor = self.config.search_area_factor
        if mode in {TrackerMode.UNCERTAIN, TrackerMode.OCCLUDED}:
            factor = self.config.uncertain_search_area_factor
        elif mode is TrackerMode.RECOVERY:
            factor = self.config.recovery_search_area_factor

        side = factor * _target_side(target_size)
        side = max(side, float(self.config.min_crop_size))
        return (side, side)

    def _score_response(
        self,
        response: Any,
        peak_y: int,
        peak_x: int,
    ) -> Tuple[float, float, float, float]:
        """Convert a raw FFT response map into confidence and ambiguity scores."""

        np = _require_numpy()
        peak_value = float(response[peak_y, peak_x])
        masked = response.copy()
        radius = max(1, self.config.peak_exclusion_radius)
        y0 = max(0, peak_y - radius)
        y1 = min(masked.shape[0], peak_y + radius + 1)
        x0 = max(0, peak_x - radius)
        x1 = min(masked.shape[1], peak_x + radius + 1)
        masked[y0:y1, x0:x1] = -np.inf

        second_peak_value = float(np.max(masked))
        if not math.isfinite(second_peak_value):
            second_peak_value = 0.0

        response_mean = float(np.mean(response))
        response_std = float(np.std(response)) + 1e-6
        peak_z = max(0.0, (peak_value - response_mean) / response_std)
        match_score = _clamp01(peak_z / (peak_z + 8.0))

        if abs(peak_value) < 1e-6:
            ambiguity_score = 1.0
        else:
            ambiguity_score = _clamp01(second_peak_value / peak_value)

        return peak_value, second_peak_value, match_score, ambiguity_score

    def _score_scale(self, expected_size: Size, box: Box) -> float:
        """Score whether the returned box scale is compatible with prediction."""

        expected_area = max(expected_size[0] * expected_size[1], 1e-6)
        found_area = max(box[2] * box[3], 1e-6)
        return _clamp01(math.exp(-abs(math.log(found_area / expected_area))))


def _require_numpy() -> Any:
    """Import numpy only when this matcher model is actually used."""

    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("FftMatcherModel requires numpy") from error
    return np


def _as_gray_float(image: Any) -> Any:
    """Convert HxW or HxWxC image data into a float grayscale array."""

    np = _require_numpy()
    array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., :3].mean(axis=2)
    elif array.ndim != 2:
        raise ValueError("FftMatcherModel expects frame.image with shape HxW or HxWxC")
    return array.astype("float64", copy=False)


def _normalize_patch(patch: Any) -> Any:
    """Normalize a patch so raw brightness changes affect correlation less."""

    np = _require_numpy()
    patch = patch.astype("float64", copy=False)
    mean = float(np.mean(patch))
    std = float(np.std(patch))
    if std < 1e-6:
        return patch * 0.0
    return (patch - mean) / std


def _crop_centered(image: Any, center: Point, size: Size) -> Tuple[Any, Box, bool]:
    """Crop a centered region and zero-pad when it crosses frame boundaries."""

    np = _require_numpy()
    height, width = image.shape[:2]
    crop_width = max(1, int(round(size[0])))
    crop_height = max(1, int(round(size[1])))
    left = int(round(center[0] - crop_width / 2.0))
    top = int(round(center[1] - crop_height / 2.0))
    right = left + crop_width
    bottom = top + crop_height

    src_left = max(0, left)
    src_top = max(0, top)
    src_right = min(width, right)
    src_bottom = min(height, bottom)

    crop = np.zeros((crop_height, crop_width), dtype=image.dtype)
    if src_right > src_left and src_bottom > src_top:
        dst_left = src_left - left
        dst_top = src_top - top
        dst_right = dst_left + (src_right - src_left)
        dst_bottom = dst_top + (src_bottom - src_top)
        crop[dst_top:dst_bottom, dst_left:dst_right] = image[src_top:src_bottom, src_left:src_right]

    clipped = src_left != left or src_top != top or src_right != right or src_bottom != bottom
    return crop, (float(left), float(top), float(crop_width), float(crop_height)), clipped


def _paste_center(canvas: Any, patch: Any) -> None:
    """Paste a patch into the center of a same-or-larger canvas."""

    crop_height = min(canvas.shape[0], patch.shape[0])
    crop_width = min(canvas.shape[1], patch.shape[1])
    canvas_top = (canvas.shape[0] - crop_height) // 2
    canvas_left = (canvas.shape[1] - crop_width) // 2
    patch_top = (patch.shape[0] - crop_height) // 2
    patch_left = (patch.shape[1] - crop_width) // 2
    canvas[
        canvas_top : canvas_top + crop_height,
        canvas_left : canvas_left + crop_width,
    ] = patch[
        patch_top : patch_top + crop_height,
        patch_left : patch_left + crop_width,
    ]


def _peak_index(response: Any) -> Tuple[int, int]:
    """Return row and column of the highest response value."""

    np = _require_numpy()
    flat_index = int(np.argmax(response))
    peak_y, peak_x = np.unravel_index(flat_index, response.shape)
    return int(peak_y), int(peak_x)


def _target_side(target_size: Size) -> float:
    """Return a robust square side length from target width and height."""

    width = max(float(target_size[0]), 1.0)
    height = max(float(target_size[1]), 1.0)
    return math.sqrt(width * height)


def _center_size_to_box(center: Point, size: Size) -> Box:
    """Convert center and size into frame-coordinate x, y, width, height."""

    return (
        float(center[0] - size[0] / 2.0),
        float(center[1] - size[1] / 2.0),
        float(size[0]),
        float(size[1]),
    )


def _clamp01(value: float) -> float:
    """Clamp a float into the confidence-score range."""

    return max(0.0, min(1.0, float(value)))
