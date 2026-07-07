from __future__ import annotations

import copy
import math
import sys
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol

from BigTracker.matcher import MatcherModel
from BigTracker.matcher_models._boxes import center_size_to_box, clip_box, map_crop_box_back
from BigTracker.matcher_models._crop import sample_target
from BigTracker.matcher_models._torch import inference_context, resolve_device
from BigTracker.state import MatchEvidence, MatcherState, SearchCandidate, TemplateCandidate
from BigTracker.types import Box, FrameLike, Point, Size, TrackerMode


class MixFormerV2Backend(Protocol):
    """Minimal runtime protocol required by MixFormerV2MatcherModel."""

    def eval(self) -> Any:
        """Switch backend to inference mode."""
        ...

    def preprocess(self, image: Any) -> Any:
        """Normalize one cropped image for the backend."""
        ...

    def forward(self, template: Any, online_template: Any, search: Any) -> Mapping[str, Any]:
        """Run one MixFormerV2 forward pass."""
        ...


BackendFactory = Callable[["MixFormerV2MatcherConfig"], MixFormerV2Backend]


@dataclass(frozen=True)
class MixFormerV2MatcherConfig:
    """Configuration for the MixFormerV2 matcher wrapper."""

    source_root: str = "ignores/Trackers/MixFormerV2"
    config_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    device: Optional[str] = None
    variant: str = "online"
    max_best_templates: int = 5
    template_factor: float = 2.0
    template_size: int = 128
    search_factor: float = 4.5
    search_size: int = 288
    score_floor: float = 1e-6
    fallback_match_score: float = 0.5
    pred_scores_are_logits: bool = True
    clip_margin: float = 10.0


@dataclass(frozen=True)
class MixFormerV2Template:
    """Template object owned by MixFormerV2MatcherModel."""

    template_tensor: Any
    target_size: Size
    source_frame_idx: int
    source_box: Box
    crop_box: Box
    crop_resize_factor: float
    was_clipped: bool
    metadata: Mapping[str, Any]


class MixFormerV2MatcherModel(MatcherModel):
    """MixFormerV2 adapter implementing the BigTracker MatcherModel API."""

    def __init__(
        self,
        config: Optional[MixFormerV2MatcherConfig] = None,
        backend: Optional[MixFormerV2Backend] = None,
        backend_factory: Optional[BackendFactory] = None,
    ) -> None:
        """Load MixFormerV2 model/config/checkpoint once and keep inference helpers."""

        self.config = config or MixFormerV2MatcherConfig()
        if backend is not None:
            self.backend = backend
        elif backend_factory is not None:
            self.backend = backend_factory(self.config)
        else:
            self.backend, self.config = _load_real_mixformerv2_backend(self.config)
        self.backend.eval()

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
            cached_features={"matcher": "mixformerv2", "variant": self.config.variant},
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
                "matcher": "mixformerv2",
                "variant": self.config.variant,
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
        """Run MixFormerV2 search for one candidate and return evidence only."""

        init_template = self._select_init_template(matcher_state)
        online_template = self._select_online_template(matcher_state)
        search_crop = self._build_search_crop(
            frame=frame,
            search_center=candidate.search_center,
            predicted_target_size=candidate.predicted_target_size,
        )
        search_tensor = self.backend.preprocess(search_crop.image)

        with inference_context():
            outputs = self.backend.forward(
                init_template.template_tensor,
                online_template.template_tensor,
                search_tensor,
            )

        pred_boxes = outputs["pred_boxes"]
        pred_box = self._prediction_to_crop_box(pred_boxes, search_crop.resize_factor)
        mapped_box = map_crop_box_back(
            pred_box_cxcywh=pred_box,
            crop_center=candidate.search_center,
            search_size=float(self.config.search_size),
            resize_factor=search_crop.resize_factor,
        )
        clipped_box = clip_box(mapped_box, frame.image.shape, margin=self.config.clip_margin)
        stats = self._score_stats(outputs)

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
                "matcher": "mixformerv2",
                "variant": self.config.variant,
                "mode": mode.value,
                "score_source": stats["score_source"],
                "search_crop_box": search_crop.crop_box,
                "search_resize_factor": search_crop.resize_factor,
                "init_template_source_frame_idx": init_template.source_frame_idx,
                "online_template_source_frame_idx": online_template.source_frame_idx,
            },
        )

    def _build_template(
        self,
        frame: FrameLike,
        target_pos: Point,
        target_size: Size,
    ) -> MixFormerV2Template:
        """Crop, preprocess, and store one MixFormerV2 template."""

        source_box = center_size_to_box(target_pos, target_size)
        crop = sample_target(
            image=frame.image,
            target_box=source_box,
            search_area_factor=self.config.template_factor,
            output_size=self.config.template_size,
            pad_value=0,
        )
        template_tensor = self.backend.preprocess(crop.image)
        return MixFormerV2Template(
            template_tensor=_snapshot_feature_state(template_tensor),
            target_size=(float(target_size[0]), float(target_size[1])),
            source_frame_idx=frame.idx,
            source_box=source_box,
            crop_box=crop.crop_box,
            crop_resize_factor=crop.resize_factor,
            was_clipped=crop.is_clipped,
            metadata={
                "matcher": "mixformerv2",
                "variant": self.config.variant,
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
        """Build a MixFormerV2 search crop around the BigTrack candidate."""

        candidate_box = center_size_to_box(search_center, predicted_target_size)
        return sample_target(
            image=frame.image,
            target_box=candidate_box,
            search_area_factor=self.config.search_factor,
            output_size=self.config.search_size,
            pad_value=0,
        )

    def _select_init_template(self, matcher_state: MatcherState) -> MixFormerV2Template:
        """Select the fixed identity template for this match call."""

        template = matcher_state.init_template
        if not isinstance(template, MixFormerV2Template):
            raise TypeError("MixFormerV2MatcherModel requires MixFormerV2Template state")
        return template

    def _select_online_template(self, matcher_state: MatcherState) -> MixFormerV2Template:
        """Select the current BigTrack-approved online template."""

        template = matcher_state.adaptive_template or matcher_state.init_template
        if not isinstance(template, MixFormerV2Template):
            raise TypeError("MixFormerV2MatcherModel requires MixFormerV2Template state")
        return template

    def _prediction_to_crop_box(self, pred_boxes: Any, resize_factor: float) -> Box:
        """Convert normalized MixFormerV2 predictions to crop-local cx, cy, w, h."""

        array = _to_numpy(pred_boxes).reshape(-1, 4)
        pred_box = array.mean(axis=0)
        scale = float(self.config.search_size) / float(resize_factor)
        return tuple(float(value) * scale for value in pred_box)  # type: ignore[return-value]

    def _score_stats(self, outputs: Mapping[str, Any]) -> Mapping[str, Any]:
        """Derive confidence and ambiguity from score head or box agreement."""

        np = _require_numpy()
        if "pred_scores" in outputs:
            score_values = _to_numpy(outputs["pred_scores"]).astype("float64", copy=False).reshape(-1)
            if score_values.size == 0:
                score = 0.0
            elif self.config.pred_scores_are_logits:
                score = float(1.0 / (1.0 + math.exp(-float(np.max(score_values)))))
            else:
                score = float(np.max(score_values))
            score = max(0.0, min(1.0, score))
            return {
                "best_score": score,
                "ambiguity_score": max(0.0, min(1.0, 1.0 - score)),
                "localization_score": score,
                "score_source": "pred_scores",
            }

        boxes = _to_numpy(outputs["pred_boxes"]).astype("float64", copy=False).reshape(-1, 4)
        if boxes.size == 0:
            score = 0.0
        elif boxes.shape[0] == 1:
            score = float(self.config.fallback_match_score)
        else:
            mean_box = boxes.mean(axis=0, keepdims=True)
            mean_distance = float(np.linalg.norm(boxes - mean_box, axis=1).mean())
            score = max(0.0, min(1.0, math.exp(-mean_distance)))
        ambiguity = max(0.0, min(1.0, 1.0 - score))
        return {
            "best_score": score,
            "ambiguity_score": ambiguity,
            "localization_score": score,
            "score_source": "box_agreement",
        }


class _TorchMixFormerV2Backend:
    """Torch MixFormerV2 backend with device-aware preprocessing."""

    def __init__(self, network: Any, cfg: Any, device: Any) -> None:
        self.network = network
        self.cfg = cfg
        self.device = device
        torch = _require_torch()
        self.mean = torch.tensor(cfg.DATA.MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg.DATA.STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    def eval(self) -> "_TorchMixFormerV2Backend":
        self.network.eval()
        return self

    def preprocess(self, image: Any) -> Any:
        np = _require_numpy()
        torch = _require_torch()
        image_array = np.asarray(image)
        image_tensor = torch.as_tensor(image_array, device=self.device).float().permute(2, 0, 1).unsqueeze(0)
        return ((image_tensor / 255.0) - self.mean) / self.std

    def forward(self, template: Any, online_template: Any, search: Any) -> Mapping[str, Any]:
        return self.network(template, online_template, search, softmax=True, run_score_head=True)


def _load_real_mixformerv2_backend(
    config: MixFormerV2MatcherConfig,
) -> tuple[MixFormerV2Backend, MixFormerV2MatcherConfig]:
    """Load MixFormerV2 source modules, YAML config, network, and checkpoint."""

    source_root = Path(config.source_root)
    config_path = Path(config.config_path) if config.config_path else (
        Path("ignores/Models/mixformerv2/config/288_depth8_score.yaml")
    )
    checkpoint_path = Path(config.checkpoint_path) if config.checkpoint_path else (
        Path("ignores/Models/mixformerv2/models/mixformerv2_base.pth.tar")
    )
    if not config_path.exists():
        raise FileNotFoundError(f"MixFormerV2 config does not exist: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"MixFormerV2 checkpoint does not exist: {checkpoint_path}")

    device = resolve_device(config.device)
    variant = str(config.variant).lower()
    _prepare_mixformerv2_source(source_root)
    torch = _require_torch()
    if variant == "online":
        from lib.config.mixformer2_vit_online.config import cfg as default_cfg
        from lib.config.mixformer2_vit_online.config import update_new_config_from_file
        from lib.models.mixformer2_vit import build_mixformer2_vit_online

        loaded_cfg = update_new_config_from_file(str(config_path))
        with _source_cuda_build_guard(torch, device):
            network = build_mixformer2_vit_online(loaded_cfg, train=False)
    elif variant == "offline":
        from lib.config.mixformer2_vit.config import cfg as default_cfg
        from lib.config.mixformer2_vit.config import update_new_config_from_file
        from lib.models.mixformer2_vit import build_mixformer2_vit

        loaded_cfg = update_new_config_from_file(str(config_path))
        with _source_cuda_build_guard(torch, device):
            network = build_mixformer2_vit(loaded_cfg)
    else:
        raise ValueError(f"Unknown MixFormerV2 variant: {config.variant!r}")

    if loaded_cfg is default_cfg:
        loaded_cfg = copy.deepcopy(default_cfg)
    effective_config = _config_from_loaded_mixformerv2_cfg(config, loaded_cfg, variant)
    _load_mixformerv2_checkpoint(network, checkpoint_path)
    network = network.to(device)
    _move_mixformerv2_internal_tensors(network, device)
    return _TorchMixFormerV2Backend(network, loaded_cfg, device), effective_config


def _prepare_mixformerv2_source(source_root: Path) -> None:
    """Expose MixFormerV2's top-level lib package, failing on package conflicts."""

    source_root_str = str(source_root.resolve())
    existing_lib = sys.modules.get("lib")
    if existing_lib is not None:
        existing_file = getattr(existing_lib, "__file__", "")
        if existing_file and source_root_str not in str(Path(existing_file).resolve()):
            raise RuntimeError(
                "MixFormerV2 uses a top-level 'lib' package, but another 'lib' package is already loaded: "
                f"{existing_file}"
            )

    if source_root_str not in sys.path:
        sys.path.insert(0, source_root_str)


def _load_mixformerv2_checkpoint(network: Any, checkpoint_path: Path) -> None:
    torch = _require_torch()
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint.get("net", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"MixFormerV2 checkpoint does not contain a state dict: {checkpoint_path}")
    network.load_state_dict(state_dict, strict=True)


@contextmanager
def _source_cuda_build_guard(torch: Any, device: Any) -> Any:
    """Keep MixFormerV2 source tensor `.cuda()` calls on CPU when CPU is requested."""

    if str(device) != "cpu":
        yield
        return

    original_cuda = torch.Tensor.cuda

    def _cuda_to_cpu(tensor: Any, *args: Any, **kwargs: Any) -> Any:
        return tensor

    torch.Tensor.cuda = _cuda_to_cpu
    try:
        yield
    finally:
        torch.Tensor.cuda = original_cuda


def _move_mixformerv2_internal_tensors(network: Any, device: Any) -> None:
    """Move source tensors that are not registered buffers."""

    box_head = getattr(network, "box_head", None)
    indice = getattr(box_head, "indice", None)
    if indice is not None and hasattr(indice, "to"):
        box_head.indice = indice.to(device)


def _config_from_loaded_mixformerv2_cfg(
    config: MixFormerV2MatcherConfig,
    cfg: Any,
    variant: str,
) -> MixFormerV2MatcherConfig:
    """Return matcher config synchronized with the loaded MixFormerV2 YAML."""

    return replace(
        config,
        variant=variant,
        template_factor=float(cfg.TEST.TEMPLATE_FACTOR),
        template_size=int(cfg.TEST.TEMPLATE_SIZE),
        search_factor=float(cfg.TEST.SEARCH_FACTOR),
        search_size=int(cfg.TEST.SEARCH_SIZE),
    )


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


def _scale_score(expected_size: Size, found_size: Size) -> float:
    expected_area = max(float(expected_size[0]) * float(expected_size[1]), 1e-6)
    found_area = max(float(found_size[0]) * float(found_size[1]), 1e-6)
    return max(0.0, min(1.0, math.exp(-abs(math.log(found_area / expected_area)))))


def _box_changed(left: Box, right: Box) -> bool:
    return any(abs(float(a) - float(b)) > 1e-9 for a, b in zip(left, right))


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("MixFormerV2MatcherModel requires numpy") from error
    return np


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("MixFormerV2MatcherModel requires torch") from error
    return torch
