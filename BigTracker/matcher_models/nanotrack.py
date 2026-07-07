from __future__ import annotations

import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from BigTracker.matcher import MatcherModel
from BigTracker.matcher_models._boxes import center_size_to_box
from BigTracker.matcher_models._crop import nanotrack_subwindow
from BigTracker.matcher_models._torch import inference_context, move_to_device, resolve_device
from BigTracker.state import MatchEvidence, MatcherState, SearchCandidate, TemplateCandidate
from BigTracker.types import Box, FrameLike, Point, Size, TrackerMode


class NanoTrackBackend(Protocol):
    """Minimal model protocol required by NanoTrackMatcherModel."""

    def eval(self) -> Any:
        """Switch backend to inference mode."""
        ...

    def template(self, z: Any) -> Any:
        """Encode a template crop and optionally return the encoded feature."""
        ...

    def track(self, x: Any) -> Mapping[str, Any]:
        """Run one search crop against the active template."""
        ...


BackendFactory = Callable[["NanoTrackMatcherConfig"], NanoTrackBackend]


@dataclass(frozen=True)
class NanoTrackMatcherConfig:
    """Configuration for the NanoTrack matcher wrapper."""

    source_root: str = "ignores/Trackers/NanoTrack"
    config_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    device: Optional[str] = None
    max_best_templates: int = 5
    context_amount: float = 0.5
    exemplar_size: int = 127
    instance_size: int = 255
    output_size: int = 16
    point_stride: int = 8
    cls_out_channels: int = 2
    penalty_k: float = 0.16
    window_influence: float = 0.46
    lr: float = 0.34
    min_box_size: float = 10.0


@dataclass(frozen=True)
class NanoTrackTemplate:
    """Template object owned by NanoTrackMatcherModel."""

    feature_state: Any
    channel_average: Any
    target_size: Size
    source_frame_idx: int
    source_box: Box
    crop_box: Box
    crop_resize_factor: float
    was_clipped: bool
    metadata: Mapping[str, Any]


class NanoTrackMatcherModel(MatcherModel):
    """NanoTrack matcher adapter implementing the BigTracker MatcherModel API."""

    def __init__(
        self,
        config: Optional[NanoTrackMatcherConfig] = None,
        backend: Optional[NanoTrackBackend] = None,
        backend_factory: Optional[BackendFactory] = None,
    ) -> None:
        """Load NanoTrack model/config/checkpoint once and precompute static arrays."""

        self.config = config or NanoTrackMatcherConfig()
        self.device = resolve_device(self.config.device) if backend is None else None
        if backend is not None:
            self.backend = backend
        elif backend_factory is not None:
            self.backend = backend_factory(self.config)
        else:
            self.backend, self.config = _load_real_nanotrack_backend(self.config, self.device)
        self.backend.eval()
        self.points = self._generate_points(self.config.point_stride, self.config.output_size)
        self.window = self._generate_window(self.config.output_size)

    def initialize_template(
        self,
        frame: FrameLike,
        target_pos: Point,
        target_size: Size,
    ) -> MatcherState:
        """Create protected initial and adaptive templates for one object."""

        template = self._build_template(frame, target_pos, target_size)
        return MatcherState(
            init_template=template,
            best_templates=(),
            adaptive_template=template,
            cached_features={"matcher": "nanotrack"},
        )

    def extract_template(
        self,
        frame: FrameLike,
        target_pos: Point,
        target_size: Size,
        previous_state: MatcherState,
    ) -> TemplateCandidate:
        """Build a template candidate from a BigTrack-approved target."""

        template = self._build_template(frame, target_pos, target_size)
        return TemplateCandidate(
            template=template,
            source_frame_idx=frame.idx,
            source_box=center_size_to_box(target_pos, target_size),
            quality_score=1.0,
            identity_score=1.0,
            metadata={
                "matcher": "nanotrack",
                "previous_best_template_count": len(previous_state.best_templates),
                "was_clipped": template.was_clipped,
            },
        )

    def update_templates(
        self,
        state: MatcherState,
        template: TemplateCandidate,
    ) -> MatcherState:
        """Insert an approved template into the latest-good bank."""

        best_templates = tuple(state.best_templates) + (template.template,)
        max_templates = max(0, int(self.config.max_best_templates))
        if max_templates == 0:
            best_templates = ()
        elif len(best_templates) > max_templates:
            best_templates = best_templates[-max_templates:]

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
        """Run NanoTrack search for one candidate and return evidence only."""

        template = self._select_template(matcher_state)
        self._activate_template(template)

        search_crop, scale_z = self._build_search_crop(
            frame=frame,
            search_center=candidate.search_center,
            predicted_target_size=candidate.predicted_target_size,
            channel_average=template.channel_average,
        )
        outputs = self._track_backend(search_crop.image)
        score = self._convert_score(outputs["cls"])
        pred_bbox = self._convert_bbox(outputs["loc"], self.points)
        result = self._decode_prediction(
            score=score,
            pred_bbox=pred_bbox,
            scale_z=scale_z,
            search_center=candidate.search_center,
            predicted_target_size=candidate.predicted_target_size,
            image_shape=frame.image.shape,
        )

        return MatchEvidence(
            candidate_id=candidate.candidate_id,
            box=result["box"],
            match_score=result["best_score"],
            identity_score=result["best_score"],
            appearance_score=result["best_score"],
            localization_score=result["localization_score"],
            ambiguity_score=result["ambiguity_score"],
            scale_score=result["scale_score"],
            occlusion_score=max(0.0, min(1.0, 1.0 - result["best_score"])),
            is_clipped=search_crop.is_clipped or result["was_clipped"],
            metadata={
                "matcher": "nanotrack",
                "mode": mode.value,
                "best_idx": result["best_idx"],
                "penalty": result["penalty"],
                "pscore": result["pscore"],
                "scale_z": scale_z,
                "search_crop_box": search_crop.crop_box,
                "template_source_frame_idx": template.source_frame_idx,
            },
        )

    def _build_template(
        self,
        frame: FrameLike,
        target_pos: Point,
        target_size: Size,
    ) -> NanoTrackTemplate:
        """Crop and encode one NanoTrack template."""

        np = _require_numpy()
        image = np.asarray(frame.image)
        channel_average = np.mean(image, axis=(0, 1))
        crop_size = self._template_crop_size(target_size)
        crop = nanotrack_subwindow(
            image=image,
            center=target_pos,
            model_size=self.config.exemplar_size,
            original_size=crop_size,
            channel_average=channel_average,
        )
        feature_state = self._encode_template(crop.image)
        return NanoTrackTemplate(
            feature_state=feature_state,
            channel_average=channel_average,
            target_size=(float(target_size[0]), float(target_size[1])),
            source_frame_idx=frame.idx,
            source_box=center_size_to_box(target_pos, target_size),
            crop_box=crop.crop_box,
            crop_resize_factor=crop.resize_factor,
            was_clipped=crop.is_clipped,
            metadata={
                "matcher": "nanotrack",
                "exemplar_size": self.config.exemplar_size,
                "context_amount": self.config.context_amount,
            },
        )

    def _encode_template(self, crop: Any) -> Any:
        """Run backend.template and snapshot its encoded template feature."""

        tensor = self._to_backend_input(crop)
        with inference_context():
            returned = self.backend.template(tensor)
        feature_state = returned if returned is not None else getattr(self.backend, "zf", None)
        return _snapshot_feature_state(feature_state)

    def _track_backend(self, crop: Any) -> Mapping[str, Any]:
        """Run backend.track under inference mode."""

        tensor = self._to_backend_input(crop)
        with inference_context():
            return self.backend.track(tensor)

    def _to_backend_input(self, value: Any) -> Any:
        """Convert numpy crops to torch tensors and move them to the backend device."""

        torch = _require_torch()
        if not torch.is_tensor(value):
            value = torch.from_numpy(value)
        return move_to_device(value.float(), self.device) if self.device is not None else value.float()

    def _activate_template(self, template: NanoTrackTemplate) -> None:
        """Install a stored template feature before running search."""

        if template.feature_state is not None:
            setattr(self.backend, "zf", _snapshot_feature_state(template.feature_state))

    def _select_template(self, matcher_state: MatcherState) -> NanoTrackTemplate:
        """Select the active template for this match call."""

        template = matcher_state.adaptive_template or matcher_state.init_template
        if not isinstance(template, NanoTrackTemplate):
            raise TypeError("NanoTrackMatcherModel requires NanoTrackTemplate state")
        return template

    def _build_search_crop(
        self,
        frame: FrameLike,
        search_center: Point,
        predicted_target_size: Size,
        channel_average: Any,
    ) -> tuple[Any, float]:
        """Build NanoTrack search crop and return crop plus template scale."""

        width, height = predicted_target_size
        w_z = float(width) + self.config.context_amount * (float(width) + float(height))
        h_z = float(height) + self.config.context_amount * (float(width) + float(height))
        s_z = math.sqrt(max(w_z * h_z, 1.0))
        scale_z = float(self.config.exemplar_size) / s_z
        s_x = s_z * (float(self.config.instance_size) / float(self.config.exemplar_size))
        crop = nanotrack_subwindow(
            image=frame.image,
            center=search_center,
            model_size=self.config.instance_size,
            original_size=round(s_x),
            channel_average=channel_average,
        )
        return crop, scale_z

    def _template_crop_size(self, target_size: Size) -> float:
        """Compute NanoTrack exemplar crop size from target size."""

        width, height = target_size
        w_z = float(width) + self.config.context_amount * (float(width) + float(height))
        h_z = float(height) + self.config.context_amount * (float(width) + float(height))
        return float(round(math.sqrt(max(w_z * h_z, 1.0))))

    def _convert_score(self, score: Any) -> Any:
        """Convert raw NanoTrack classification output to positive-class scores."""

        np = _require_numpy()
        torch = _require_torch()
        if torch.is_tensor(score):
            if self.config.cls_out_channels == 1:
                return score.permute(1, 2, 3, 0).contiguous().view(-1).sigmoid().detach().cpu().numpy()
            score = score.permute(1, 2, 3, 0).contiguous().view(self.config.cls_out_channels, -1)
            score = score.permute(1, 0).softmax(1).detach()[:, 1].cpu().numpy()
            return score

        array = np.asarray(score)
        if array.ndim == 1:
            return array.astype("float64", copy=False)
        if array.shape[0] == self.config.cls_out_channels:
            array = array.reshape(self.config.cls_out_channels, -1).transpose(1, 0)
        if array.shape[-1] == 2:
            exp = np.exp(array - np.max(array, axis=1, keepdims=True))
            return exp[:, 1] / np.sum(exp, axis=1)
        return array.reshape(-1).astype("float64", copy=False)

    def _convert_bbox(self, delta: Any, points: Any) -> Any:
        """Convert raw NanoTrack location output to center-size predictions."""

        np = _require_numpy()
        torch = _require_torch()
        if torch.is_tensor(delta):
            delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).detach().cpu().numpy()
        else:
            delta = np.asarray(delta)
            if delta.shape[0] != 4:
                delta = delta.reshape(4, -1)

        points = self._points_for_prediction_count(delta.shape[1])
        converted = np.empty_like(delta, dtype="float64")
        converted[0, :] = points[:, 0] - delta[0, :]
        converted[1, :] = points[:, 1] - delta[1, :]
        converted[2, :] = points[:, 0] + delta[2, :]
        converted[3, :] = points[:, 1] + delta[3, :]

        x1, y1, x2, y2 = converted
        return np.vstack(((x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1))

    def _decode_prediction(
        self,
        score: Any,
        pred_bbox: Any,
        scale_z: float,
        search_center: Point,
        predicted_target_size: Size,
        image_shape: Sequence[int],
    ) -> Mapping[str, Any]:
        """Apply NanoTrack penalty/window decode and return frame-coordinate evidence."""

        np = _require_numpy()
        target_width = max(float(predicted_target_size[0]), 1.0)
        target_height = max(float(predicted_target_size[1]), 1.0)
        score = np.asarray(score, dtype="float64").reshape(-1)
        pred_bbox = np.asarray(pred_bbox, dtype="float64")
        window = self._window_for_prediction_count(score.shape[0])

        s_c = _change(
            _sz(pred_bbox[2, :], pred_bbox[3, :])
            / _sz(target_width * scale_z, target_height * scale_z)
        )
        r_c = _change((target_width / target_height) / (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1.0) * self.config.penalty_k)
        pscore = penalty * score
        pscore = pscore * (1.0 - self.config.window_influence) + window * self.config.window_influence
        best_idx = int(np.argmax(pscore))

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * self.config.lr
        cx = float(bbox[0]) + float(search_center[0])
        cy = float(bbox[1]) + float(search_center[1])
        width = target_width * (1.0 - lr) + float(bbox[2]) * lr
        height = target_height * (1.0 - lr) + float(bbox[3]) * lr
        cx, cy, width, height, was_clipped = self._clip_center_size(
            cx,
            cy,
            width,
            height,
            image_shape,
        )

        best_score = max(0.0, min(1.0, float(score[best_idx])))
        ordered = np.sort(score)
        second_score = float(ordered[-2]) if len(ordered) > 1 else 0.0
        ambiguity = 1.0 if best_score <= 1e-9 else max(0.0, min(1.0, second_score / best_score))
        distance = math.hypot(float(bbox[0]), float(bbox[1]))
        max_distance = max(float(self.config.instance_size) / max(scale_z, 1e-6), 1.0) / 2.0
        localization = max(0.0, min(1.0, 1.0 - distance / max_distance))
        scale_score = _scale_score(predicted_target_size, (width, height))

        return {
            "box": (cx - width / 2.0, cy - height / 2.0, width, height),
            "best_score": best_score,
            "best_idx": best_idx,
            "penalty": float(penalty[best_idx]),
            "pscore": float(pscore[best_idx]),
            "ambiguity_score": ambiguity,
            "localization_score": localization,
            "scale_score": scale_score,
            "was_clipped": was_clipped,
        }

    def _clip_center_size(
        self,
        cx: float,
        cy: float,
        width: float,
        height: float,
        image_shape: Sequence[int],
    ) -> tuple[float, float, float, float, bool]:
        """Clip center-size geometry using NanoTrack's minimum box-size rule."""

        image_height = float(image_shape[0])
        image_width = float(image_shape[1])
        next_cx = max(0.0, min(cx, image_width))
        next_cy = max(0.0, min(cy, image_height))
        next_width = max(self.config.min_box_size, min(width, image_width))
        next_height = max(self.config.min_box_size, min(height, image_height))
        return (
            next_cx,
            next_cy,
            next_width,
            next_height,
            (
                abs(next_cx - cx) > 1e-9
                or abs(next_cy - cy) > 1e-9
                or abs(next_width - width) > 1e-9
                or abs(next_height - height) > 1e-9
            ),
        )

    def _generate_points(self, stride: int, size: int) -> Any:
        """Generate NanoTrack search-grid points."""

        np = _require_numpy()
        origin = -(int(size) // 2) * int(stride)
        values = [origin + int(stride) * dx for dx in range(int(size))]
        x, y = np.meshgrid(values, values)
        points = np.zeros((int(size) * int(size), 2), dtype=np.float32)
        points[:, 0] = x.astype(np.float32).flatten()
        points[:, 1] = y.astype(np.float32).flatten()
        return points

    def _generate_window(self, size: int) -> Any:
        """Generate NanoTrack Hann penalty window."""

        np = _require_numpy()
        hanning = np.hanning(int(size))
        return np.outer(hanning, hanning).flatten()

    def _points_for_prediction_count(self, count: int) -> Any:
        """Return a point grid compatible with a flattened model output."""

        if len(self.points) == int(count):
            return self.points
        size = _square_size_from_count(count, "NanoTrack location output")
        return self._generate_points(self.config.point_stride, size)

    def _window_for_prediction_count(self, count: int) -> Any:
        """Return a Hann window compatible with a flattened score output."""

        if len(self.window) == int(count):
            return self.window
        size = _square_size_from_count(count, "NanoTrack score output")
        return self._generate_window(size)


def _load_real_nanotrack_backend(
    config: NanoTrackMatcherConfig,
    device: Any,
) -> tuple[NanoTrackBackend, NanoTrackMatcherConfig]:
    """Load NanoTrack source modules, config, model, and checkpoint."""

    source_root = Path(config.source_root)
    config_path = Path(config.config_path) if config.config_path else source_root / "models/config/configv3.yaml"
    checkpoint_path = (
        Path(config.checkpoint_path)
        if config.checkpoint_path
        else source_root / "models/pretrained/nanotrackv3.pth"
    )
    if not config_path.exists():
        raise FileNotFoundError(f"NanoTrack config does not exist: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"NanoTrack checkpoint does not exist: {checkpoint_path}")

    inserted = False
    source_root_str = str(source_root.resolve())
    if source_root_str not in sys.path:
        sys.path.insert(0, source_root_str)
        inserted = True
    try:
        from nanotrack.core.config import cfg
        from nanotrack.models.model_builder import ModelBuilder

        cfg.merge_from_file(str(config_path))
        effective_config = _config_from_loaded_nanotrack_cfg(config, cfg)
        model = ModelBuilder()
        _load_nanotrack_checkpoint(model, str(checkpoint_path))
        model = model.to(device)
        model.eval()
        return model, effective_config
    finally:
        if inserted:
            try:
                sys.path.remove(source_root_str)
            except ValueError:
                pass


def _snapshot_feature_state(value: Any) -> Any:
    """Detach/clone feature tensors so template state is independent."""

    if value is None:
        return None
    if hasattr(value, "detach") and hasattr(value, "clone"):
        return value.detach().clone()
    if isinstance(value, (tuple, list)):
        return type(value)(_snapshot_feature_state(item) for item in value)
    if isinstance(value, dict):
        return {key: _snapshot_feature_state(item) for key, item in value.items()}
    return value


def _load_nanotrack_checkpoint(model: Any, checkpoint_path: str) -> None:
    """Load NanoTrack weights with PyTorch 2.6-compatible checkpoint handling."""

    torch = _require_torch()
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"NanoTrack checkpoint does not contain a state dict: {checkpoint_path}")

    state_dict = {
        key.split("module.", 1)[-1] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=False)


def _config_from_loaded_nanotrack_cfg(
    config: NanoTrackMatcherConfig,
    cfg: Any,
) -> NanoTrackMatcherConfig:
    """Return matcher config synchronized with the loaded NanoTrack YAML."""

    return replace(
        config,
        context_amount=float(cfg.TRACK.CONTEXT_AMOUNT),
        exemplar_size=int(cfg.TRACK.EXEMPLAR_SIZE),
        instance_size=int(cfg.TRACK.INSTANCE_SIZE),
        output_size=int(cfg.TRACK.OUTPUT_SIZE),
        point_stride=int(cfg.POINT.STRIDE),
        penalty_k=float(cfg.TRACK.PENALTY_K),
        window_influence=float(cfg.TRACK.WINDOW_INFLUENCE),
        lr=float(cfg.TRACK.LR),
    )


def _square_size_from_count(count: int, label: str) -> int:
    """Infer square output size from a flattened prediction count."""

    size = int(round(math.sqrt(float(count))))
    if size * size != int(count):
        raise ValueError(f"{label} has non-square prediction count: {count}")
    return size


def _change(value: Any) -> Any:
    """NanoTrack scale/aspect change function."""

    np = _require_numpy()
    return np.maximum(value, 1.0 / value)


def _sz(width: Any, height: Any) -> Any:
    """NanoTrack padded size function."""

    np = _require_numpy()
    pad = (width + height) * 0.5
    return np.sqrt((width + pad) * (height + pad))


def _scale_score(expected_size: Size, found_size: Size) -> float:
    """Score whether found area is compatible with predicted size."""

    expected_area = max(float(expected_size[0]) * float(expected_size[1]), 1e-6)
    found_area = max(float(found_size[0]) * float(found_size[1]), 1e-6)
    return max(0.0, min(1.0, math.exp(-abs(math.log(found_area / expected_area)))))


def _require_numpy() -> Any:
    """Import numpy only when NanoTrack matcher is used."""

    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("NanoTrackMatcherModel requires numpy") from error
    return np


def _require_torch() -> Any:
    """Import torch only when NanoTrack matcher is used."""

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("NanoTrackMatcherModel requires torch") from error
    return torch
