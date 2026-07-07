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

    backend: str = "torch"
    source_root: str = "ignores/Trackers/NanoTrack"
    config_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    backbone_path: Optional[str] = None
    template_backbone_path: Optional[str] = None
    search_backbone_path: Optional[str] = None
    head_path: Optional[str] = None
    onnx_provider: str = "cpu"
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
        needs_torch_device = (
            backend is None
            and backend_factory is None
            and _normalized_backend_name(self.config.backend) == "torch"
        )
        self.device = resolve_device(self.config.device) if needs_torch_device else None
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
        if getattr(self.backend, "expects_numpy", False):
            returned = self.backend.template(tensor)
        else:
            with inference_context():
                returned = self.backend.template(tensor)
        feature_state = returned if returned is not None else getattr(self.backend, "zf", None)
        return _snapshot_feature_state(feature_state)

    def _track_backend(self, crop: Any) -> Mapping[str, Any]:
        """Run backend.track under inference mode."""

        tensor = self._to_backend_input(crop)
        if getattr(self.backend, "expects_numpy", False):
            return self.backend.track(tensor)
        with inference_context():
            return self.backend.track(tensor)

    def _to_backend_input(self, value: Any) -> Any:
        """Convert numpy crops to torch tensors and move them to the backend device."""

        if getattr(self.backend, "expects_numpy", False):
            np = _require_numpy()
            return np.asarray(value, dtype=np.float32)
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
        torch = _optional_torch()
        if torch is not None and torch.is_tensor(score):
            if self.config.cls_out_channels == 1:
                return score.permute(1, 2, 3, 0).contiguous().view(-1).sigmoid().detach().cpu().numpy()
            score = score.permute(1, 2, 3, 0).contiguous().view(self.config.cls_out_channels, -1)
            score = score.permute(1, 0).softmax(1).detach()[:, 1].cpu().numpy()
            return score

        array = np.asarray(score)
        if array.ndim == 1:
            return array.astype("float64", copy=False)
        if array.ndim == 4 and array.shape[1] == self.config.cls_out_channels:
            array = array.transpose(0, 2, 3, 1).reshape(-1, self.config.cls_out_channels)
        if array.shape[0] == self.config.cls_out_channels:
            array = array.reshape(self.config.cls_out_channels, -1).transpose(1, 0)
        if array.shape[-1] == 2:
            exp = np.exp(array - np.max(array, axis=1, keepdims=True))
            return exp[:, 1] / np.sum(exp, axis=1)
        return array.reshape(-1).astype("float64", copy=False)

    def _convert_bbox(self, delta: Any, points: Any) -> Any:
        """Convert raw NanoTrack location output to center-size predictions."""

        np = _require_numpy()
        torch = _optional_torch()
        if torch is not None and torch.is_tensor(delta):
            delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).detach().cpu().numpy()
        else:
            delta = np.asarray(delta)
            if delta.ndim == 4 and delta.shape[1] == 4:
                delta = delta.transpose(1, 2, 3, 0).reshape(4, -1)
            elif delta.shape[0] != 4:
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


class _SplitNanoTrackTorchBackend:
    """Use separate NanoTrack model instances for template and search paths."""

    def __init__(self, template_model: Any, search_model: Any) -> None:
        self.template_model = template_model
        self.search_model = search_model
        self.zf = None

    def eval(self) -> "_SplitNanoTrackTorchBackend":
        self.template_model.eval()
        self.search_model.eval()
        return self

    def template(self, z: Any) -> Any:
        returned = self.template_model.template(z)
        self.zf = returned if returned is not None else getattr(self.template_model, "zf", None)
        return self.zf

    def track(self, x: Any) -> Mapping[str, Any]:
        setattr(self.search_model, "zf", self.zf)
        return self.search_model.track(x)


class _SplitNanoTrackOnnxBackend:
    """Use separate ONNX Runtime backbone sessions for template and search paths."""

    expects_numpy = True

    def __init__(
        self,
        template_backbone_session: Any,
        search_backbone_session: Any,
        head_session: Any,
    ) -> None:
        self.template_backbone_session = template_backbone_session
        self.search_backbone_session = search_backbone_session
        self.head_session = head_session
        self.zf = None

    def eval(self) -> "_SplitNanoTrackOnnxBackend":
        return self

    def template(self, z: Any) -> Any:
        template_feature = self._run_backbone(self.template_backbone_session, z)
        target_hw = _session_input_hw(self.head_session, 0)
        self.zf = _center_crop_feature(template_feature, target_hw)
        return self.zf

    def track(self, x: Any) -> Mapping[str, Any]:
        if self.zf is None:
            raise RuntimeError("NanoTrack ONNX template features are not initialized")

        search_feature = self._run_backbone(self.search_backbone_session, x)
        head_inputs = self.head_session.get_inputs()
        outputs = self.head_session.run(
            None,
            {
                head_inputs[0].name: self.zf,
                head_inputs[1].name: search_feature,
            },
        )
        return {"cls": outputs[0], "loc": outputs[1]}

    def _run_backbone(self, session: Any, crop: Any) -> Any:
        session_input = session.get_inputs()[0]
        array = _fit_onnx_backbone_input(crop, _input_hw(session_input))
        return session.run(None, {session_input.name: array})[0]


def _load_real_nanotrack_backend(
    config: NanoTrackMatcherConfig,
    device: Any,
) -> tuple[NanoTrackBackend, NanoTrackMatcherConfig]:
    """Load NanoTrack source modules/config and the requested backend."""

    backend = _normalized_backend_name(config.backend)
    if backend == "torch":
        return _load_real_nanotrack_torch_backend(config, device)
    if backend == "onnx":
        return _load_real_nanotrack_onnx_backend(config)
    raise ValueError("NanoTrack backend must be 'torch'/'pth' or 'onnx'")


def _load_real_nanotrack_torch_backend(
    config: NanoTrackMatcherConfig,
    device: Any,
) -> tuple[NanoTrackBackend, NanoTrackMatcherConfig]:
    """Load NanoTrack source modules, config, split torch models, and checkpoint."""

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
        template_model = ModelBuilder()
        search_model = ModelBuilder()
        _load_nanotrack_checkpoint(template_model, str(checkpoint_path))
        _load_nanotrack_checkpoint(search_model, str(checkpoint_path))
        template_model = template_model.to(device)
        search_model = search_model.to(device)
        return _SplitNanoTrackTorchBackend(template_model, search_model), effective_config
    finally:
        if inserted:
            try:
                sys.path.remove(source_root_str)
            except ValueError:
                pass


def _load_real_nanotrack_onnx_backend(
    config: NanoTrackMatcherConfig,
) -> tuple[NanoTrackBackend, NanoTrackMatcherConfig]:
    """Load split NanoTrack ONNX Runtime sessions."""

    config_path = _resolve_nanotrack_config_path(config)
    if not config_path.exists():
        raise FileNotFoundError(f"NanoTrack config does not exist: {config_path}")

    template_backbone_path, search_backbone_path, head_path = _resolve_onnx_paths(config, config_path)
    for label, path in (
        ("NanoTrack ONNX template backbone", template_backbone_path),
        ("NanoTrack ONNX search backbone", search_backbone_path),
        ("NanoTrack ONNX head", head_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    ort = _require_onnxruntime()
    providers = _onnx_providers(config.onnx_provider, ort)
    template_session = ort.InferenceSession(str(template_backbone_path), providers=providers)
    search_session = ort.InferenceSession(str(search_backbone_path), providers=providers)
    head_session = ort.InferenceSession(str(head_path), providers=providers)
    effective_config = _config_from_nanotrack_yaml(config, config_path)
    return _SplitNanoTrackOnnxBackend(template_session, search_session, head_session), effective_config


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


def _resolve_nanotrack_config_path(config: NanoTrackMatcherConfig) -> Path:
    source_root = Path(config.source_root)
    return Path(config.config_path) if config.config_path else source_root / "models/config/configv3.yaml"


def _resolve_onnx_paths(config: NanoTrackMatcherConfig, config_path: Path) -> tuple[Path, Path, Path]:
    base_backbone_path = Path(config.backbone_path) if config.backbone_path else None
    template_backbone_path = (
        Path(config.template_backbone_path)
        if config.template_backbone_path
        else base_backbone_path
    )
    search_backbone_path = (
        Path(config.search_backbone_path)
        if config.search_backbone_path
        else base_backbone_path
    )
    head_path = Path(config.head_path) if config.head_path else None

    if template_backbone_path is not None and search_backbone_path is not None and head_path is not None:
        return template_backbone_path, search_backbone_path, head_path

    default_backbone_path, default_head_path = _default_onnx_paths(config, config_path)
    return (
        template_backbone_path or default_backbone_path,
        search_backbone_path or default_backbone_path,
        head_path or default_head_path,
    )


def _default_onnx_paths(config: NanoTrackMatcherConfig, config_path: Path) -> tuple[Path, Path]:
    model_root = config_path.parent.parent if config.config_path else Path(config.source_root) / "models"
    version_name = config_path.stem.replace("config", "nanotrack", 1)
    version_dir = model_root / version_name
    backbone_candidates = (
        version_dir / "nanotrack_backbone.onnx",
        version_dir / "nanotrack_backbone_sim.onnx",
        model_root / "onnx/nanotrack_backbone.onnx",
        model_root / "onnx/nanotrack_backbone_sim.onnx",
    )
    head_candidates = (
        version_dir / "nanotrack_head.onnx",
        version_dir / "nanotrack_head_sim.onnx",
        model_root / "onnx/nanotrack_head.onnx",
        model_root / "onnx/nanotrack_head_sim.onnx",
    )
    return _first_existing_or_first(backbone_candidates), _first_existing_or_first(head_candidates)


def _first_existing_or_first(paths: Sequence[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


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


def _config_from_nanotrack_yaml(
    config: NanoTrackMatcherConfig,
    config_path: Path,
) -> NanoTrackMatcherConfig:
    """Return matcher config synchronized with a NanoTrack YAML file."""

    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("NanoTrack ONNX backend requires PyYAML to read config files") from error

    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}
    track_config = raw_config.get("TRACK", {})
    point_config = raw_config.get("POINT", {})

    return replace(
        config,
        context_amount=float(track_config.get("CONTEXT_AMOUNT", config.context_amount)),
        exemplar_size=int(track_config.get("EXEMPLAR_SIZE", config.exemplar_size)),
        instance_size=int(track_config.get("INSTANCE_SIZE", config.instance_size)),
        output_size=int(track_config.get("OUTPUT_SIZE", config.output_size)),
        point_stride=int(point_config.get("STRIDE", config.point_stride)),
        penalty_k=float(track_config.get("PENALTY_K", config.penalty_k)),
        window_influence=float(track_config.get("WINDOW_INFLUENCE", config.window_influence)),
        lr=float(track_config.get("LR", config.lr)),
    )


def _normalized_backend_name(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"torch", "pth", "pytorch"}:
        return "torch"
    if normalized == "onnx":
        return "onnx"
    return normalized


def _require_onnxruntime() -> Any:
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("NanoTrack ONNX backend requires onnxruntime") from error
    return ort


def _onnx_providers(provider: str, ort: Any) -> list[str]:
    normalized = str(provider).strip().lower()
    if normalized in {"cpu", "default"}:
        return ["CPUExecutionProvider"]
    if normalized in {"cuda", "gpu"}:
        available = list(ort.get_available_providers())
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "NanoTrack ONNX CUDA requested, but CUDAExecutionProvider is not available. "
                f"Installed ONNX Runtime providers: {available}"
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    raise ValueError("NanoTrack ONNX provider must be 'cpu' or 'cuda'")


def _input_hw(session_input: Any) -> Optional[tuple[int, int]]:
    shape = getattr(session_input, "shape", None)
    if shape is None or len(shape) < 4:
        return None
    height, width = shape[-2], shape[-1]
    if isinstance(height, int) and isinstance(width, int):
        return int(height), int(width)
    return None


def _session_input_hw(session: Any, input_index: int) -> Optional[tuple[int, int]]:
    inputs = session.get_inputs()
    if input_index >= len(inputs):
        return None
    return _input_hw(inputs[input_index])


def _fit_onnx_backbone_input(value: Any, expected_hw: Optional[tuple[int, int]]) -> Any:
    np = _require_numpy()
    array = np.asarray(value, dtype=np.float32)
    if expected_hw is None:
        return array

    expected_height, expected_width = expected_hw
    current_height, current_width = array.shape[-2], array.shape[-1]
    if (current_height, current_width) == (expected_height, expected_width):
        return array

    if current_height > expected_height or current_width > expected_width:
        top = max(0, (current_height - expected_height) // 2)
        left = max(0, (current_width - expected_width) // 2)
        return array[..., top : top + expected_height, left : left + expected_width]

    pad_shape = array.shape[:-2] + (expected_height, expected_width)
    padded = np.empty(pad_shape, dtype=np.float32)
    fill_value = array.mean(axis=(-2, -1), keepdims=True)
    padded[...] = fill_value
    top = (expected_height - current_height) // 2
    left = (expected_width - current_width) // 2
    padded[..., top : top + current_height, left : left + current_width] = array
    return padded


def _center_crop_feature(value: Any, target_hw: Optional[tuple[int, int]]) -> Any:
    np = _require_numpy()
    array = np.asarray(value, dtype=np.float32)
    if target_hw is None:
        return array
    target_height, target_width = target_hw
    current_height, current_width = array.shape[-2], array.shape[-1]
    if (current_height, current_width) == (target_height, target_width):
        return array
    if current_height < target_height or current_width < target_width:
        raise ValueError(
            "NanoTrack ONNX backbone feature is smaller than head input: "
            f"feature={(current_height, current_width)} head_input={target_hw}"
        )
    top = (current_height - target_height) // 2
    left = (current_width - target_width) // 2
    return array[..., top : top + target_height, left : left + target_width]


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


def _optional_torch() -> Any:
    try:
        import torch
    except ImportError:
        return None
    return torch
