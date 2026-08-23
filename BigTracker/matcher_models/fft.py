from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

from BigTracker.matcher import MatcherModel
from BigTracker.matcher_models._boxes import box_to_center_size, center_size_to_box
from BigTracker.matcher_models._crop import crop_centered
from BigTracker.matcher_models._templates import update_template_bank
from BigTracker.types import (
    Box,
    FrameLike,
    MatcherInitializeInput,
    MatcherInitializeOutput,
    MatcherMatchInput,
    MatcherMatchOutput,
    MatcherState,
    MatcherTemplateInput,
    MatcherTemplateOutput,
    MatcherUpdateInput,
    MatcherUpdateOutput,
    Point,
    Size,
)


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
    and returns peak-based boxes and scores.
    """

    def __init__(self, config: Optional[FftMatcherConfig] = None) -> None:
        """Create an FFT matcher with crop sizes and template queue limits."""

        self.config = config or FftMatcherConfig()
        self._state: Optional[MatcherState] = None

    def initialize_template(self, request: MatcherInitializeInput) -> MatcherInitializeOutput:
        """Create matcher state with the first template as protected identity anchor."""

        if request.matcher_state is not None:
            self._state = request.matcher_state
            return MatcherInitializeOutput(ok=True, metadata={"matcher": "fft", "restored": True})

        target_pos, target_size = box_to_center_size(request.box)
        template = self._build_template(request.frame, target_pos, target_size)
        self._state = MatcherState(
            init_template=template,
            best_templates=(),
            adaptive_template=template,
            metadata={"matcher": "fft"},
        )
        return MatcherInitializeOutput(
            ok=True,
            metadata={
                "matcher": "fft",
                "template_crop_size": template.crop_size,
                "was_clipped": template.clipped,
            },
        )

    def extract_template(self, request: MatcherTemplateInput) -> MatcherTemplateOutput:
        """Create a candidate template from a BigTrack-approved target region."""

        target_pos, target_size = box_to_center_size(request.box)
        template = self._build_template(request.frame, target_pos, target_size)
        return MatcherTemplateOutput(
            template=template,
            score=1.0,
            metadata={
                "matcher": "fft",
                "template_crop_size": template.crop_size,
                "was_clipped": template.clipped,
                "source_frame_idx": request.frame.idx,
                "source_box": template.source_box,
                "previous_best_template_count": len(self._require_state().best_templates),
            },
        )

    def update_templates(self, request: MatcherUpdateInput) -> MatcherUpdateOutput:
        """Insert an approved template while keeping the initial template unchanged."""

        self._state = update_template_bank(
            self._require_state(),
            request.template,
            request.score,
            self.config.max_best_templates,
        )
        return MatcherUpdateOutput(ok=True, metadata={"matcher": "fft"})

    def match(self, request: MatcherMatchInput) -> MatcherMatchOutput:
        """Run FFT matching for each requested target position."""

        matcher_state = self._require_state()
        templates = self._collect_templates(matcher_state)
        if not templates:
            raise ValueError("FftMatcherModel.match requires at least one template")

        target_size = self._active_target_size(matcher_state)
        bboxes: list[Box] = []
        scores: list[float] = []
        details: list[dict[str, Any]] = []
        for target_index, target_pos in enumerate(request.target_poses):
            results = [
                self._match_one_template(request.frame, template, target_pos, target_size, index)
                for index, template in enumerate(templates)
            ]
            best = max(results, key=lambda result: result.match_score)
            bboxes.append(best.box)
            scores.append(best.match_score)
            details.append(
                {
                    "target_index": target_index,
                    "selected_template_index": best.selected_template_index,
                    "localization_score": best.localization_score,
                    "ambiguity_score": best.ambiguity_score,
                    "peak_value": best.peak_value,
                    "second_peak_value": best.second_peak_value,
                    "search_box": best.search_box,
                    "is_clipped": best.is_clipped,
                }
            )

        return MatcherMatchOutput(
            bboxes=bboxes,
            scores=scores,
            metadata={
                "matcher": "fft",
                "template_count": len(templates),
                "target_size": target_size,
                "details": details,
            },
        )

    def reset(self) -> None:
        """Clear matcher runtime state."""

        self._state = None

    def close(self) -> None:
        """Release matcher runtime state."""

        self.reset()

    def _build_template(self, frame: FrameLike, target_pos: Point, target_size: Size) -> FftTemplate:
        """Crop, normalize, and encode one FFT template."""

        np = _require_numpy()
        image = _as_gray_float(frame.image)
        crop_size = self._template_crop_size(target_size)
        crop = crop_centered(image, target_pos, crop_size)
        patch = _normalize_patch(crop.image)
        return FftTemplate(
            patch=patch,
            spectrum=np.fft.fft2(patch),
            target_size=target_size,
            crop_size=(float(patch.shape[1]), float(patch.shape[0])),
            source_frame_idx=frame.idx,
            source_box=center_size_to_box(target_pos, target_size),
            clipped=crop.is_clipped,
        )

    def _match_one_template(
        self,
        frame: FrameLike,
        template: FftTemplate,
        search_center: Point,
        target_size: Size,
        template_index: int,
    ) -> _FftMatchResult:
        """Match one encoded template against one search crop."""

        np = _require_numpy()
        image = _as_gray_float(frame.image)
        search_size = self._search_crop_size(target_size)
        search_crop = crop_centered(
            image,
            search_center,
            search_size,
        )
        search_patch = _normalize_patch(search_crop.image)

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
            search_center[0] + offset_x,
            search_center[1] + offset_y,
        )
        matched_box = center_size_to_box(matched_center, target_size)

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
            is_clipped=search_crop.is_clipped,
            search_box=search_crop.crop_box,
        )

    def _collect_templates(self, matcher_state: MatcherState) -> Sequence[FftTemplate]:
        """Return all templates that should vote during matching."""

        templates = [matcher_state.init_template]
        templates.extend(template_state.template for template_state in matcher_state.best_templates)
        if matcher_state.adaptive_template is not None:
            templates.append(matcher_state.adaptive_template)
        return tuple(template for template in templates if isinstance(template, FftTemplate))

    def _template_crop_size(self, target_size: Size) -> Size:
        """Choose a square template crop from the target size."""

        side = self.config.template_area_factor * _target_side(target_size)
        side = max(side, float(self.config.min_crop_size))
        return (side, side)

    def _search_crop_size(self, target_size: Size) -> Size:
        """Choose a square search crop from target size."""

        factor = self.config.search_area_factor
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

    def _active_target_size(self, matcher_state: MatcherState) -> Size:
        """Return target size from the currently active FFT template."""

        template = matcher_state.adaptive_template or matcher_state.init_template
        if not isinstance(template, FftTemplate):
            raise TypeError("FftMatcherModel requires FftTemplate state")
        return template.target_size

    def _require_state(self) -> MatcherState:
        """Return initialized matcher state."""

        if self._state is None:
            raise RuntimeError("FftMatcherModel must be initialized before use")
        return self._state


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


def _clamp01(value: float) -> float:
    """Clamp a float into the confidence-score range."""

    return max(0.0, min(1.0, float(value)))
