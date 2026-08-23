from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol

from BigTracker.matcher import MatcherModel
from BigTracker.matcher_models._boxes import box_to_center_size, center_size_to_box, clip_box, map_crop_box_back
from BigTracker.matcher_models._crop import sample_target
from BigTracker.matcher_models._templates import update_template_bank
from BigTracker.matcher_models._torch import inference_context, resolve_device
from BigTracker.thirdparty.litetrack import build_litetrack_network, load_litetrack_config
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


class LiteTrackBackend(Protocol):
    """Minimal runtime protocol required by LiteTrackMatcherModel."""

    output_window: Any

    def eval(self) -> Any:
        """Switch backend to inference mode."""
        ...

    def preprocess(self, image: Any, attention_mask: Any) -> Any:
        """Normalize one cropped image and attention mask for the backend."""
        ...

    def encode_template(self, template: Any, source_box: Box, resize_factor: float) -> Any:
        """Encode a template crop into LiteTrack template features."""
        ...

    def forward(self, template_features: Any, search: Any) -> Mapping[str, Any]:
        """Run one LiteTrack search forward pass."""
        ...

    def cal_bbox(self, response: Any, size_map: Any, offset_map: Any) -> Any:
        """Decode center-head maps into normalized cx, cy, w, h boxes."""
        ...


BackendFactory = Callable[["LiteTrackMatcherConfig"], LiteTrackBackend]


@dataclass(frozen=True)
class LiteTrackMatcherConfig:
    """Configuration for the LiteTrack matcher wrapper."""

    config_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    device: Optional[str] = None
    max_best_templates: int = 5
    template_factor: float = 2.0
    template_size: int = 128
    search_factor: float = 4.0
    search_size: int = 256
    backbone_stride: int = 16
    head_type: str = "CENTER"
    score_floor: float = 1e-6
    clip_margin: float = 10.0


@dataclass(frozen=True)
class LiteTrackTemplate:
    """Template object owned by LiteTrackMatcherModel."""

    template_features: Any
    template_tensor: Any
    target_size: Size
    source_frame_idx: int
    source_box: Box
    crop_box: Box
    crop_resize_factor: float
    was_clipped: bool
    template_score: float
    metadata: Mapping[str, Any]


class LiteTrackMatcherModel(MatcherModel):
    """LiteTrack adapter implementing the BigTracker MatcherModel API."""

    def __init__(
        self,
        config: Optional[LiteTrackMatcherConfig] = None,
        backend: Optional[LiteTrackBackend] = None,
        backend_factory: Optional[BackendFactory] = None,
    ) -> None:
        """Load LiteTrack model/config/checkpoint once and keep inference helpers."""

        self.config = config or LiteTrackMatcherConfig()
        if backend is not None:
            self.backend = backend
        elif backend_factory is not None:
            self.backend = backend_factory(self.config)
        else:
            self.backend, self.config = _load_real_litetrack_backend(self.config)
        self.backend.eval()
        self.feat_sz = int(self.config.search_size) // int(self.config.backbone_stride)
        self._state: Optional[MatcherState] = None

    def initialize_template(self, request: MatcherInitializeInput) -> MatcherInitializeOutput:
        """Create protected initial and adaptive templates for one object."""

        if request.matcher_state is not None:
            self._state = request.matcher_state
            return MatcherInitializeOutput(ok=True, metadata={"matcher": "litetrack", "restored": True})

        target_pos, target_size = box_to_center_size(request.box)
        template = self._build_template(request.frame, target_pos, target_size)
        self._state = MatcherState(
            init_template=template,
            best_templates=(),
            adaptive_template=template,
            metadata={"matcher": "litetrack"},
        )
        return MatcherInitializeOutput(
            ok=True,
            metadata={
                "matcher": "litetrack",
                "template_source_frame_idx": template.source_frame_idx,
                "was_clipped": template.was_clipped,
            },
        )

    def extract_template(self, request: MatcherTemplateInput) -> MatcherTemplateOutput:
        """Build a template candidate from a BigTrack-approved target."""

        target_pos, target_size = box_to_center_size(request.box)
        template = self._build_template(request.frame, target_pos, target_size)
        return MatcherTemplateOutput(
            template=template,
            score=1.0,
            metadata={
                "matcher": "litetrack",
                "previous_best_template_count": len(self._require_state().best_templates),
                "source_frame_idx": request.frame.idx,
                "source_box": template.source_box,
                "was_clipped": template.was_clipped,
            },
        )

    def update_templates(self, request: MatcherUpdateInput) -> MatcherUpdateOutput:
        """Insert an approved template and select the best one in the window."""

        self._state = update_template_bank(
            self._require_state(),
            request.template,
            request.score,
            self.config.max_best_templates,
        )
        return MatcherUpdateOutput(ok=True, metadata={"matcher": "litetrack"})

    def match(self, request: MatcherMatchInput) -> MatcherMatchOutput:
        """Run LiteTrack search for each requested target position."""

        matcher_state = self._require_state()
        template = self._select_template(matcher_state)
        target_size = template.target_size
        bboxes: list[Box] = []
        scores: list[float] = []
        details: list[dict[str, Any]] = []

        for target_index, target_pos in enumerate(request.target_poses):
            search_crop = self._build_search_crop(
                frame=request.frame,
                search_center=target_pos,
                predicted_target_size=target_size,
            )
            search_tensor = _processed_tensors(
                self.backend.preprocess(search_crop.image, search_crop.attention_mask)
            )

            with inference_context():
                outputs = self.backend.forward(template.template_features, search_tensor)
                response = self.backend.output_window * outputs["score_map"]
                pred_boxes = self.backend.cal_bbox(response, outputs["size_map"], outputs["offset_map"])

            pred_box = self._prediction_to_crop_box(pred_boxes, search_crop.resize_factor)
            mapped_box = map_crop_box_back(
                pred_box_cxcywh=pred_box,
                crop_center=target_pos,
                search_size=float(self.config.search_size),
                resize_factor=search_crop.resize_factor,
            )
            clipped_box = clip_box(mapped_box, request.frame.image.shape, margin=self.config.clip_margin)
            stats = self._score_stats(response)

            bboxes.append(clipped_box)
            scores.append(stats["best_score"])
            details.append(
                {
                    "target_index": target_index,
                    "best_idx": stats["best_idx"],
                    "second_score": stats["second_score"],
                    "localization_score": stats["localization_score"],
                    "ambiguity_score": stats["ambiguity_score"],
                    "search_crop_box": search_crop.crop_box,
                    "search_resize_factor": search_crop.resize_factor,
                    "template_source_frame_idx": template.source_frame_idx,
                    "is_clipped": search_crop.is_clipped or _box_changed(mapped_box, clipped_box),
                }
            )

        return MatcherMatchOutput(
            bboxes=bboxes,
            scores=scores,
            metadata={
                "matcher": "litetrack",
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

    def _build_template(
        self,
        frame: FrameLike,
        target_pos: Point,
        target_size: Size,
    ) -> LiteTrackTemplate:
        """Crop, preprocess, and encode one LiteTrack template."""

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
        with inference_context():
            template_features = self.backend.encode_template(template_tensor, source_box, crop.resize_factor)

        return LiteTrackTemplate(
            template_features=_snapshot_feature_state(template_features),
            template_tensor=_snapshot_feature_state(template_tensor),
            target_size=(float(target_size[0]), float(target_size[1])),
            source_frame_idx=frame.idx,
            source_box=source_box,
            crop_box=crop.crop_box,
            crop_resize_factor=crop.resize_factor,
            was_clipped=crop.is_clipped,
            template_score=1.0,
            metadata={
                "matcher": "litetrack",
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
        """Build a LiteTrack search crop around the BigTrack candidate."""

        candidate_box = center_size_to_box(search_center, predicted_target_size)
        return sample_target(
            image=frame.image,
            target_box=candidate_box,
            search_area_factor=self.config.search_factor,
            output_size=self.config.search_size,
            pad_value=0,
        )

    def _select_template(self, matcher_state: MatcherState) -> LiteTrackTemplate:
        """Select the active template for this match call."""

        template = matcher_state.adaptive_template or matcher_state.init_template
        if not isinstance(template, LiteTrackTemplate):
            raise TypeError("LiteTrackMatcherModel requires LiteTrackTemplate state")
        return template

    def _require_state(self) -> MatcherState:
        """Return initialized matcher state."""

        if self._state is None:
            raise RuntimeError("LiteTrackMatcherModel must be initialized before use")
        return self._state

    def _prediction_to_crop_box(self, pred_boxes: Any, resize_factor: float) -> Box:
        """Convert normalized LiteTrack predictions to crop-local cx, cy, w, h."""

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


class _TorchLiteTrackBackend:
    """Torch LiteTrack backend with device-aware preprocessing."""

    def __init__(self, network: Any, cfg: Any, device: Any) -> None:
        self.network = network
        self.cfg = cfg
        self.device = device
        torch = _require_torch()
        self.mean = torch.tensor(cfg.DATA.MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg.DATA.STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.feat_sz = int(cfg.TEST.SEARCH_SIZE) // int(cfg.MODEL.BACKBONE.STRIDE)
        self.output_window = _hann2d(self.feat_sz, torch, device)
        self.NestedTensor = _load_nested_tensor_class()

    def eval(self) -> "_TorchLiteTrackBackend":
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

    def encode_template(self, template: Any, source_box: Box, resize_factor: float) -> Any:
        template_bbox = _template_bbox_xyxy(
            source_box=source_box,
            resize_factor=resize_factor,
            template_size=float(self.cfg.TEST.TEMPLATE_SIZE),
            torch=_require_torch(),
            device=self.device,
        )
        return self.network.forward_z(template, template_bb=template_bbox)

    def forward(self, template_features: Any, search: Any) -> Mapping[str, Any]:
        return self.network(template_feats=template_features, search=search)

    def cal_bbox(self, response: Any, size_map: Any, offset_map: Any) -> Any:
        return self.network.box_head.cal_bbox(response, self.feat_sz, size_map, offset_map)


def _load_real_litetrack_backend(
    config: LiteTrackMatcherConfig,
) -> tuple[LiteTrackBackend, LiteTrackMatcherConfig]:
    """Load LiteTrack source modules, YAML config, network, and checkpoint."""

    config_path = Path(config.config_path) if config.config_path else (
        Path("ignores/Models/litetrack/config/B8_cae_center_all_ep300.yaml")
    )
    checkpoint_path = Path(config.checkpoint_path) if config.checkpoint_path else (
        Path("ignores/Models/litetrack/B8_cae_center_all_ep300/LiteTrack_ep0300.pth.tar")
    )
    if not config_path.exists():
        raise FileNotFoundError(f"LiteTrack config does not exist: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"LiteTrack checkpoint does not exist: {checkpoint_path}")

    device = resolve_device(config.device)
    loaded_cfg = copy.deepcopy(load_litetrack_config(config_path))
    effective_config = _config_from_loaded_litetrack_cfg(config, loaded_cfg)
    network = build_litetrack_network(loaded_cfg, training=False)
    _load_litetrack_checkpoint(network, checkpoint_path)
    network = network.to(device)
    return _TorchLiteTrackBackend(network, loaded_cfg, device), effective_config


def _load_litetrack_checkpoint(network: Any, checkpoint_path: Path) -> None:
    torch = _require_torch()
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint.get("net", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"LiteTrack checkpoint does not contain a state dict: {checkpoint_path}")
    network.load_state_dict(state_dict, strict=False)


def _config_from_loaded_litetrack_cfg(
    config: LiteTrackMatcherConfig,
    cfg: Any,
) -> LiteTrackMatcherConfig:
    """Return matcher config synchronized with the loaded LiteTrack YAML."""

    return replace(
        config,
        template_factor=float(cfg.TEST.TEMPLATE_FACTOR),
        template_size=int(cfg.TEST.TEMPLATE_SIZE),
        search_factor=float(cfg.TEST.SEARCH_FACTOR),
        search_size=int(cfg.TEST.SEARCH_SIZE),
        backbone_stride=int(cfg.MODEL.BACKBONE.STRIDE),
        head_type=str(cfg.MODEL.HEAD.TYPE),
    )


def _template_bbox_xyxy(
    source_box: Box,
    resize_factor: float,
    template_size: float,
    torch: Any,
    device: Any,
) -> Any:
    """Build normalized xyxy target token box for a centered template crop."""

    width = float(source_box[2]) * float(resize_factor)
    height = float(source_box[3]) * float(resize_factor)
    cx = (template_size - 1.0) / 2.0
    cy = (template_size - 1.0) / 2.0
    xywh = torch.tensor(
        [
            [
                (cx - width / 2.0) / template_size,
                (cy - height / 2.0) / template_size,
                width / template_size,
                height / template_size,
            ]
        ],
        dtype=torch.float32,
        device=device,
    )
    x, y, w, h = xywh.unbind(-1)
    return torch.stack((x, y, x + w, y + h), dim=-1).float()


def _processed_tensors(value: Any) -> Any:
    return getattr(value, "tensors", value)


def _snapshot_feature_state(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "detach") and hasattr(value, "clone"):
        return value.detach().clone()
    if isinstance(value, (tuple, list)):
        return type(value)(_snapshot_feature_state(item) for item in value)
    if isinstance(value, dict):
        return {key: _snapshot_feature_state(item) for key, item in value.items()}
    return value


def _to_numpy(value: Any) -> Any:
    np = _require_numpy()
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _hann2d(size: int, torch: Any, device: Any) -> Any:
    values = 0.5 * (1.0 - torch.cos((2.0 * math.pi / (int(size) + 1)) * torch.arange(1, int(size) + 1).float()))
    values = values.to(device)
    return values.reshape(1, 1, -1, 1) * values.reshape(1, 1, 1, -1)


def _box_changed(left: Box, right: Box) -> bool:
    return any(abs(float(a) - float(b)) > 1e-9 for a, b in zip(left, right))


def _square_size_from_count(count: int) -> int:
    size = int(round(math.sqrt(float(count))))
    if size * size != int(count):
        raise ValueError(f"LiteTrack score output has non-square prediction count: {count}")
    return size


def _load_nested_tensor_class() -> Any:
    from BigTracker.thirdparty.litetrack.lib.utils.misc import NestedTensor

    return NestedTensor


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("LiteTrackMatcherModel requires numpy") from error
    return np


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("LiteTrackMatcherModel requires torch") from error
    return torch
