from __future__ import annotations

import copy
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from BigTracker.matcher import MatcherModel
from BigTracker.matcher_models._boxes import center_size_to_box, clip_box, map_crop_box_back
from BigTracker.matcher_models._crop import sample_target
from BigTracker.matcher_models._torch import inference_context, resolve_device
from BigTracker.state import MatchEvidence, MatcherState, SearchCandidate, TemplateCandidate
from BigTracker.types import Box, FrameLike, Point, Size, TrackerMode


class OSTrackBackend(Protocol):
    """Minimal runtime protocol required by OSTrackMatcherModel."""

    output_window: Any

    def eval(self) -> Any:
        """Switch backend to inference mode."""
        ...

    def preprocess(self, image: Any, attention_mask: Any) -> Any:
        """Normalize one cropped image and attention mask for the backend."""
        ...

    def build_box_mask(self, source_box: Box, resize_factor: float) -> Any:
        """Build candidate-elimination template mask when the model needs it."""
        ...

    def forward(self, template: Any, search: Any, ce_template_mask: Any) -> Mapping[str, Any]:
        """Run one OSTrack forward pass."""
        ...

    def cal_bbox(self, response: Any, size_map: Any, offset_map: Any) -> Any:
        """Decode center-head maps into normalized cx, cy, w, h boxes."""
        ...


BackendFactory = Callable[["OSTrackMatcherConfig"], OSTrackBackend]


@dataclass(frozen=True)
class OSTrackMatcherConfig:
    """Configuration for the OSTrack matcher wrapper."""

    source_root: str = "ignores/Trackers/OSTrack"
    config_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    device: Optional[str] = None
    max_best_templates: int = 5
    template_factor: float = 2.0
    template_size: int = 192
    search_factor: float = 5.0
    search_size: int = 384
    backbone_stride: int = 16
    head_type: str = "CENTER"
    score_floor: float = 1e-6
    clip_margin: float = 10.0


@dataclass(frozen=True)
class OSTrackTemplate:
    """Template object owned by OSTrackMatcherModel."""

    template_tensor: Any
    box_mask_z: Any
    target_size: Size
    source_frame_idx: int
    source_box: Box
    crop_box: Box
    crop_resize_factor: float
    was_clipped: bool
    metadata: Mapping[str, Any]


class OSTrackMatcherModel(MatcherModel):
    """OSTrack adapter implementing the BigTracker MatcherModel API."""

    def __init__(
        self,
        config: Optional[OSTrackMatcherConfig] = None,
        backend: Optional[OSTrackBackend] = None,
        backend_factory: Optional[BackendFactory] = None,
    ) -> None:
        """Load OSTrack model/config/checkpoint once and keep inference helpers."""

        self.config = config or OSTrackMatcherConfig()
        if backend is not None:
            self.backend = backend
        elif backend_factory is not None:
            self.backend = backend_factory(self.config)
        else:
            self.backend, self.config = _load_real_ostrack_backend(self.config)
        self.backend.eval()
        self.feat_sz = int(self.config.search_size) // int(self.config.backbone_stride)

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
            cached_features={"matcher": "ostrack"},
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
                "matcher": "ostrack",
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
        """Run OSTrack search for one candidate and return evidence only."""

        template = self._select_template(matcher_state)
        search_crop = self._build_search_crop(
            frame=frame,
            search_center=candidate.search_center,
            predicted_target_size=candidate.predicted_target_size,
        )
        search_tensor = _processed_tensors(
            self.backend.preprocess(search_crop.image, search_crop.attention_mask)
        )

        with inference_context():
            outputs = self.backend.forward(template.template_tensor, search_tensor, template.box_mask_z)
            response = _multiply(self.backend.output_window, outputs["score_map"])
            pred_boxes = self.backend.cal_bbox(response, outputs["size_map"], outputs["offset_map"])

        pred_box = self._prediction_to_crop_box(pred_boxes, search_crop.resize_factor)
        mapped_box = map_crop_box_back(
            pred_box_cxcywh=pred_box,
            crop_center=candidate.search_center,
            search_size=float(self.config.search_size),
            resize_factor=search_crop.resize_factor,
        )
        clipped_box = clip_box(mapped_box, frame.image.shape, margin=self.config.clip_margin)
        stats = self._score_stats(response)

        return MatchEvidence(
            candidate_id=candidate.candidate_id,
            box=clipped_box,
            match_score=stats["best_score"],
            identity_score=stats["best_score"],
            appearance_score=stats["best_score"],
            localization_score=stats["localization_score"],
            ambiguity_score=stats["ambiguity_score"],
            scale_score=_scale_score(candidate.predicted_target_size, (clipped_box[2], clipped_box[3])),
            occlusion_score=max(0.0, min(1.0, 1.0 - stats["best_score"])),
            is_clipped=search_crop.is_clipped or _box_changed(mapped_box, clipped_box),
            metadata={
                "matcher": "ostrack",
                "mode": mode.value,
                "best_idx": stats["best_idx"],
                "second_score": stats["second_score"],
                "search_crop_box": search_crop.crop_box,
                "search_resize_factor": search_crop.resize_factor,
                "template_source_frame_idx": template.source_frame_idx,
            },
        )

    def _build_template(
        self,
        frame: FrameLike,
        target_pos: Point,
        target_size: Size,
    ) -> OSTrackTemplate:
        """Crop, preprocess, and store one OSTrack template."""

        source_box = center_size_to_box(target_pos, target_size)
        crop = sample_target(
            image=frame.image,
            target_box=source_box,
            search_area_factor=self.config.template_factor,
            output_size=self.config.template_size,
            pad_value=0,
        )
        processed = self.backend.preprocess(crop.image, crop.attention_mask)
        template_tensor = _processed_tensors(processed)
        box_mask_z = self.backend.build_box_mask(source_box, crop.resize_factor)
        return OSTrackTemplate(
            template_tensor=template_tensor,
            box_mask_z=box_mask_z,
            target_size=(float(target_size[0]), float(target_size[1])),
            source_frame_idx=frame.idx,
            source_box=source_box,
            crop_box=crop.crop_box,
            crop_resize_factor=crop.resize_factor,
            was_clipped=crop.is_clipped,
            metadata={
                "matcher": "ostrack",
                "template_factor": self.config.template_factor,
                "template_size": self.config.template_size,
            },
        )

    def _build_search_crop(
        self,
        frame: FrameLike,
        search_center: Point,
        predicted_target_size: Size,
    ) -> Any:
        """Build an OSTrack search crop around the BigTrack candidate."""

        candidate_box = center_size_to_box(search_center, predicted_target_size)
        return sample_target(
            image=frame.image,
            target_box=candidate_box,
            search_area_factor=self.config.search_factor,
            output_size=self.config.search_size,
            pad_value=0,
        )

    def _select_template(self, matcher_state: MatcherState) -> OSTrackTemplate:
        """Select the active template for this match call."""

        template = matcher_state.adaptive_template or matcher_state.init_template
        if not isinstance(template, OSTrackTemplate):
            raise TypeError("OSTrackMatcherModel requires OSTrackTemplate state")
        return template

    def _prediction_to_crop_box(self, pred_boxes: Any, resize_factor: float) -> Box:
        """Convert normalized OSTrack predictions to crop-local cx, cy, w, h."""

        array = _to_numpy(pred_boxes).reshape(-1, 4)
        pred_box = array.mean(axis=0)
        scale = float(self.config.search_size) / float(resize_factor)
        return tuple(float(value) * scale for value in pred_box)  # type: ignore[return-value]

    def _score_stats(self, response: Any) -> Mapping[str, Any]:
        """Derive confidence, ambiguity, and localization from score response."""

        np = _require_numpy()
        array = _to_numpy(response).astype("float64", copy=False).reshape(-1)
        if array.size == 0:
            return {
                "best_score": 0.0,
                "second_score": 0.0,
                "best_idx": -1,
                "ambiguity_score": 1.0,
                "localization_score": 0.0,
            }

        best_idx = int(np.argmax(array))
        best_score = max(0.0, min(1.0, float(array[best_idx])))
        ordered = np.sort(array)
        second_score = float(ordered[-2]) if ordered.size > 1 else 0.0
        ambiguity = 1.0 if best_score <= self.config.score_floor else max(
            0.0,
            min(1.0, second_score / max(best_score, self.config.score_floor)),
        )
        feat_size = _square_size_from_count(array.size)
        row = best_idx // feat_size
        col = best_idx % feat_size
        center = (feat_size - 1.0) / 2.0
        max_distance = max(math.hypot(center, center), 1.0)
        localization = max(0.0, min(1.0, 1.0 - math.hypot(row - center, col - center) / max_distance))
        return {
            "best_score": best_score,
            "second_score": second_score,
            "best_idx": best_idx,
            "ambiguity_score": ambiguity,
            "localization_score": localization,
        }


class _TorchOSTrackBackend:
    """Torch OSTrack backend with device-aware preprocessing."""

    def __init__(self, network: Any, cfg: Any, device: Any) -> None:
        self.network = network
        self.cfg = cfg
        self.device = device
        torch = _require_torch()
        self.mean = torch.tensor(cfg.DATA.MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg.DATA.STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        feat_sz = int(cfg.TEST.SEARCH_SIZE) // int(cfg.MODEL.BACKBONE.STRIDE)
        self.output_window = _hann2d(feat_sz, torch, device)
        self.NestedTensor = _load_nested_tensor_class()
        self.generate_mask_cond = _load_generate_mask_cond()

    def eval(self) -> "_TorchOSTrackBackend":
        self.network.eval()
        return self

    def preprocess(self, image: Any, attention_mask: Any) -> Any:
        np = _require_numpy()
        torch = _require_torch()
        image_array = np.asarray(image)
        mask_array = np.asarray(attention_mask, dtype=bool)
        image_tensor = torch.as_tensor(image_array, device=self.device).float().permute(2, 0, 1).unsqueeze(0)
        image_tensor = ((image_tensor / 255.0) - self.mean) / self.std
        mask_tensor = torch.as_tensor(mask_array, dtype=torch.bool, device=self.device).unsqueeze(0)
        return self.NestedTensor(image_tensor, mask_tensor)

    def build_box_mask(self, source_box: Box, resize_factor: float) -> Any:
        if not getattr(self.cfg.MODEL.BACKBONE, "CE_LOC", []):
            return None

        torch = _require_torch()
        width = float(source_box[2]) * float(resize_factor)
        height = float(source_box[3]) * float(resize_factor)
        crop_size = float(self.cfg.TEST.TEMPLATE_SIZE)
        template_bbox = torch.tensor(
            [
                [
                    [
                        ((crop_size - 1.0) / 2.0 - width / 2.0) / crop_size,
                        ((crop_size - 1.0) / 2.0 - height / 2.0) / crop_size,
                        width / crop_size,
                        height / crop_size,
                    ]
                ]
            ],
            dtype=torch.float32,
            device=self.device,
        )
        return self.generate_mask_cond(self.cfg, 1, self.device, template_bbox.squeeze(1))

    def forward(self, template: Any, search: Any, ce_template_mask: Any) -> Mapping[str, Any]:
        return self.network.forward(template=template, search=search, ce_template_mask=ce_template_mask)

    def cal_bbox(self, response: Any, size_map: Any, offset_map: Any) -> Any:
        return self.network.box_head.cal_bbox(response, size_map, offset_map)


def _load_real_ostrack_backend(
    config: OSTrackMatcherConfig,
) -> tuple[OSTrackBackend, OSTrackMatcherConfig]:
    """Load OSTrack source modules, YAML config, network, and checkpoint."""

    source_root = Path(config.source_root)
    config_path = Path(config.config_path) if config.config_path else (
        Path("ignores/Models/Ostrack/config/vitb_384_mae_ce_32x4_ep300.yaml")
    )
    checkpoint_path = Path(config.checkpoint_path) if config.checkpoint_path else (
        Path("ignores/Models/Ostrack/models/vitb_384_mae_ce_32x4_ep300/OSTrack_ep0300.pth.tar")
    )
    if not config_path.exists():
        raise FileNotFoundError(f"OSTrack config does not exist: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"OSTrack checkpoint does not exist: {checkpoint_path}")

    device = resolve_device(config.device)
    _prepare_ostrack_source(source_root)
    from lib.config.ostrack.config import cfg as default_cfg
    from lib.config.ostrack.config import update_config_from_file
    from lib.models.ostrack import build_ostrack

    loaded_cfg = copy.deepcopy(default_cfg)
    update_config_from_file(str(config_path), base_cfg=loaded_cfg)
    effective_config = _config_from_loaded_ostrack_cfg(config, loaded_cfg)
    network = build_ostrack(loaded_cfg, training=False)
    _load_ostrack_checkpoint(network, checkpoint_path)
    network = network.to(device)
    return _TorchOSTrackBackend(network, loaded_cfg, device), effective_config


def _prepare_ostrack_source(source_root: Path) -> None:
    """Expose OSTrack's top-level lib package, failing on package conflicts."""

    source_root_str = str(source_root.resolve())
    existing_lib = sys.modules.get("lib")
    if existing_lib is not None:
        existing_file = getattr(existing_lib, "__file__", "")
        if existing_file and source_root_str not in str(Path(existing_file).resolve()):
            raise RuntimeError(
                "OSTrack uses a top-level 'lib' package, but another 'lib' package is already loaded: "
                f"{existing_file}"
            )

    if source_root_str not in sys.path:
        sys.path.insert(0, source_root_str)


def _load_ostrack_checkpoint(network: Any, checkpoint_path: Path) -> None:
    torch = _require_torch()
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint.get("net", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"OSTrack checkpoint does not contain a state dict: {checkpoint_path}")
    network.load_state_dict(state_dict, strict=True)


def _config_from_loaded_ostrack_cfg(
    config: OSTrackMatcherConfig,
    cfg: Any,
) -> OSTrackMatcherConfig:
    """Return matcher config synchronized with the loaded OSTrack YAML."""

    return replace(
        config,
        template_factor=float(cfg.TEST.TEMPLATE_FACTOR),
        template_size=int(cfg.TEST.TEMPLATE_SIZE),
        search_factor=float(cfg.TEST.SEARCH_FACTOR),
        search_size=int(cfg.TEST.SEARCH_SIZE),
        backbone_stride=int(cfg.MODEL.BACKBONE.STRIDE),
        head_type=str(cfg.MODEL.HEAD.TYPE),
    )


def _processed_tensors(value: Any) -> Any:
    return getattr(value, "tensors", value)


def _multiply(left: Any, right: Any) -> Any:
    return left * right


def _to_numpy(value: Any) -> Any:
    np = _require_numpy()
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _hann2d(size: int, torch: Any, device: Any) -> Any:
    values = 0.5 * (1.0 - torch.cos((2.0 * math.pi / (int(size) + 1)) * torch.arange(1, int(size) + 1).float()))
    values = values.to(device)
    return values.reshape(1, 1, -1, 1) * values.reshape(1, 1, 1, -1)


def _scale_score(expected_size: Size, found_size: Size) -> float:
    expected_area = max(float(expected_size[0]) * float(expected_size[1]), 1e-6)
    found_area = max(float(found_size[0]) * float(found_size[1]), 1e-6)
    return max(0.0, min(1.0, math.exp(-abs(math.log(found_area / expected_area)))))


def _box_changed(left: Box, right: Box) -> bool:
    return any(abs(float(a) - float(b)) > 1e-9 for a, b in zip(left, right))


def _square_size_from_count(count: int) -> int:
    size = int(round(math.sqrt(float(count))))
    if size * size != int(count):
        raise ValueError(f"OSTrack score output has non-square prediction count: {count}")
    return size


def _load_nested_tensor_class() -> Any:
    from lib.utils.misc import NestedTensor

    return NestedTensor


def _load_generate_mask_cond() -> Any:
    from lib.utils.ce_utils import generate_mask_cond

    return generate_mask_cond


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("OSTrackMatcherModel requires numpy") from error
    return np


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("OSTrackMatcherModel requires torch") from error
    return torch
