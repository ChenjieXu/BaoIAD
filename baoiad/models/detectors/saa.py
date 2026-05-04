"""SAA/SAA+ (Segment Any Anomaly) detector.

Zero-shot anomaly detection by cascading Grounding DINO (text-guided detection)
with SAM (Segment Anything Model) for segmentation.

SAA+: adds hybrid prompt regularization (language + property + saliency prompts).

Reference: arXiv 2305.10724
Official repo: https://github.com/caoyunkang/Segment-Any-Anomaly
"""

import copy
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS
from baoiad.structures import ADDataSample
from baoiad.models.base_ad_model import BaseADModel
from baoiad.utils.score_utils import minmax_normalize

from .saa_prompts import build_saa_prompts, parse_property_prompt

logger = logging.getLogger(__name__)

# Weight download URLs
_GDINO_URL = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
_SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
_WEIGHTS_DIR = "pretrained"


def _download_gdino_weights() -> str:
    """Download GroundingDINO weights if not present.

    Returns:
        Path to the GroundingDINO weights file.
    """
    os.makedirs(_WEIGHTS_DIR, exist_ok=True)
    save_path = os.path.join(_WEIGHTS_DIR, "groundingdino_swint_ogc.pth")

    if os.path.isfile(save_path):
        return save_path

    import urllib.request
    logger.info(f"Downloading GroundingDINO weights from {_GDINO_URL}...")
    logger.info("This may take a few minutes (file size: ~700MB)")
    urllib.request.urlretrieve(_GDINO_URL, save_path)
    logger.info(f"Saved to {save_path}")
    return save_path


def _download_sam_weights() -> str:
    """Download SAM weights if not present.

    Returns:
        Path to the SAM weights file.
    """
    os.makedirs(_WEIGHTS_DIR, exist_ok=True)
    save_path = os.path.join(_WEIGHTS_DIR, "sam_vit_h_4b8939.pth")

    if os.path.isfile(save_path):
        return save_path

    import urllib.request
    logger.info(f"Downloading SAM weights from {_SAM_URL}...")
    logger.info("This may take a few minutes (file size: ~2.5GB)")
    urllib.request.urlretrieve(_SAM_URL, save_path)
    logger.info(f"Saved to {save_path}")
    return save_path

try:
    from groundingdino.util.inference import load_model as load_gdino_model
    from groundingdino.datasets import transforms as gdino_transforms
    from groundingdino.util.utils import get_phrases_from_posmap
    HAS_GROUNDING_DINO = True
except ImportError:
    HAS_GROUNDING_DINO = False

try:
    from segment_anything import SamPredictor, sam_model_registry
    HAS_SAM = True
except ImportError:
    HAS_SAM = False

# ImageNet normalization constants used by NormalizeAD (on 0-255 scale)
_NORM_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_NORM_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


@MODELS.register_module(force=True)
class SAADetector(BaseADModel):
    """SAA/SAA+ anomaly detector.

    Cascades Grounding DINO + SAM for zero-shot anomaly segmentation.

    Args:
        mode: 'saa' (vanilla) or 'saa+' (hybrid prompts + saliency).
        class_name: Default category name for prompt generation.
        image_size: Expected input image size (H=W).
        grounding_dino_cfg: Dict with 'config_path' and 'checkpoint' for
            Grounding DINO model loading.
        sam_cfg: Dict with 'model_type' (e.g. 'vit_h') and 'checkpoint'
            for SAM model loading.
        saliency_backbone: Optional backbone config for SAA+ saliency scoring.
            Built via MODELS registry (e.g. TIMMBackbone with WRN-50-2).
        prompts: Custom per-category prompt override dict.
        box_threshold: Grounding DINO box confidence threshold.
        text_threshold: Grounding DINO text confidence threshold.
        nms_threshold: Optional IoU threshold for an extra cross-prompt NMS
            stage. The official repo does not apply this by default.
        k_mask: Max number of top-scoring masks to aggregate.
        image_score_aggregation: How to compute image-level score from the
            selected proposals / anomaly map.
        image_score_topk: Number of selected proposal scores to use for
            proposal-level image score modes.
        topk_rank_mode: Which score to use when selecting top-k proposals.
        image_score_rank_mode: Which score to use when choosing proposals for
            proposal-level image scoring modes.
        saliency_score_mode: How to combine saliency with detection scores for
            SAA+ anomaly scoring. One of ``multiply``, ``identity``, or
            ``clipped_multiply``.
        saliency_score_mode_overrides: Optional per-class overrides for
            ``saliency_score_mode``.
        saliency_score_clip_max: Optional global clip max used by
            ``clipped_multiply``.
        saliency_score_clip_max_overrides: Optional per-class clip-max
            overrides used by ``clipped_multiply``.
        image_score_area_range: Optional global area range
            ``(min_area_ratio, max_area_ratio)`` applied only to the
            image-score proposal pool.
        image_score_area_range_overrides: Optional per-class area-range
            overrides applied only to the image-score proposal pool.
        image_score_phrase_allowlist: Optional per-class substring allowlist
            applied only to the image-score proposal pool.
        image_score_phrase_blocklist: Optional per-class substring blocklist
            applied only to the image-score proposal pool.
        defect_area_threshold: Max fraction of image area for a valid defect
            mask (filters out overly large detections).
        data_preprocessor: Data preprocessor config dict.
        init_cfg: Initialization config.
    """

    def __init__(
        self,
        mode: str = 'saa',
        class_name: str = 'object',
        image_size: int = 256,
        grounding_dino_cfg: Optional[dict] = None,
        sam_cfg: Optional[dict] = None,
        saliency_backbone: Optional[dict] = None,
        prompts: Optional[dict] = None,
        image_score_phrase_allowlist: Optional[dict] = None,
        image_score_phrase_blocklist: Optional[dict] = None,
        box_threshold: float = 0.25,
        text_threshold: float = 0.2,
        nms_threshold: Optional[float] = None,
        k_mask: int = 5,
        k_mask_overrides: Optional[dict] = None,
        box_area_tolerance: float = 0.0,
        box_area_tolerance_overrides: Optional[dict] = None,
        image_score_aggregation: str = 'map_max',
        image_score_topk: int = 3,
        topk_rank_mode: str = 'combined',
        image_score_rank_mode: str = 'combined',
        saliency_score_mode: str = 'multiply',
        saliency_score_mode_overrides: Optional[dict] = None,
        saliency_score_clip_max: Optional[float] = None,
        saliency_score_clip_max_overrides: Optional[dict] = None,
        image_score_area_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
        image_score_area_range_overrides: Optional[dict] = None,
        image_score_rank_mode_overrides: Optional[dict] = None,
        sam_preconvert_rgb: bool = True,
        defect_area_threshold: float = 0.5,
        defect_area_threshold_overrides: Optional[dict] = None,
        data_preprocessor: Optional[dict] = None,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        assert mode in ('saa', 'saa+'), f"mode must be 'saa' or 'saa+', got '{mode}'"
        assert image_score_aggregation in (
            'map_max',
            'map_p99',
            'support_mean',
            'topk_combined_score_max',
            'topk_combined_score_mean',
        ), f'Unsupported image_score_aggregation: {image_score_aggregation}'
        assert image_score_topk > 0, 'image_score_topk must be positive'
        assert topk_rank_mode in (
            'combined',
            'det',
            'saliency',
        ), f'Unsupported topk_rank_mode: {topk_rank_mode}'
        assert image_score_rank_mode in (
            'combined',
            'det',
            'saliency',
        ), f'Unsupported image_score_rank_mode: {image_score_rank_mode}'
        assert saliency_score_mode in (
            'multiply',
            'identity',
            'clipped_multiply',
        ), f'Unsupported saliency_score_mode: {saliency_score_mode}'

        self.mode = mode
        self.class_name = class_name
        self.image_size = image_size
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_threshold = nms_threshold
        self.k_mask = k_mask
        self.k_mask_overrides = {
            key: self._normalize_k_mask(value)
            for key, value in copy.deepcopy(k_mask_overrides or {}).items()
        }
        self.box_area_tolerance = self._normalize_box_area_tolerance(box_area_tolerance)
        self.box_area_tolerance_overrides = {
            key: self._normalize_box_area_tolerance(value)
            for key, value in copy.deepcopy(box_area_tolerance_overrides or {}).items()
        }
        self.image_score_aggregation = image_score_aggregation
        self.image_score_topk = image_score_topk
        self.topk_rank_mode = topk_rank_mode
        self.image_score_rank_mode = image_score_rank_mode
        self.saliency_score_mode = saliency_score_mode
        self.saliency_score_clip_max = self._normalize_saliency_score_clip_max(saliency_score_clip_max)
        self.image_score_area_range = self._normalize_area_range(image_score_area_range)
        self.defect_area_threshold = defect_area_threshold
        self.defect_area_threshold_overrides = {
            key: self._normalize_defect_area_threshold(value)
            for key, value in copy.deepcopy(defect_area_threshold_overrides or {}).items()
        }
        self.custom_prompts = prompts  # optional per-category override
        self.image_score_phrase_allowlist = copy.deepcopy(image_score_phrase_allowlist or {})
        self.image_score_phrase_blocklist = copy.deepcopy(image_score_phrase_blocklist or {})
        self.saliency_score_mode_overrides = copy.deepcopy(saliency_score_mode_overrides or {})
        self.saliency_score_clip_max_overrides = {
            key: self._normalize_saliency_score_clip_max(value)
            for key, value in copy.deepcopy(saliency_score_clip_max_overrides or {}).items()
        }
        self.image_score_area_range_overrides = {
            key: self._normalize_area_range(value)
            for key, value in copy.deepcopy(image_score_area_range_overrides or {}).items()
        }
        self.image_score_rank_mode_overrides = copy.deepcopy(image_score_rank_mode_overrides or {})
        self.sam_preconvert_rgb = bool(sam_preconvert_rgb)

        # --- Grounding DINO ---
        if not HAS_GROUNDING_DINO:
            raise ImportError(
                'groundingdino is required for SAADetector. '
                'Install via: pip install groundingdino'
            )
        grounding_dino_cfg = grounding_dino_cfg or {}
        self._gdino_config_path = grounding_dino_cfg.get('config_path', '')
        self._gdino_checkpoint = grounding_dino_cfg.get('checkpoint', '')
        self.gdino_model = None  # lazy-loaded
        self._gdino_transform = None

        # --- SAM ---
        if not HAS_SAM:
            raise ImportError(
                'segment_anything is required for SAADetector. '
                'Install via: pip install segment-anything'
            )
        sam_cfg = sam_cfg or {}
        self._sam_model_type = sam_cfg.get('model_type', 'vit_h')
        self._sam_checkpoint = sam_cfg.get('checkpoint', '')
        self.sam_predictor = None  # lazy-loaded

        # --- Saliency backbone (SAA+ only) ---
        self.saliency_backbone = None
        if mode == 'saa+' and saliency_backbone is not None:
            self.saliency_backbone = MODELS.build(copy.deepcopy(saliency_backbone))
            self.saliency_backbone.eval()
            for p in self.saliency_backbone.parameters():
                p.requires_grad = False

        # Official evaluation normalizes anomaly maps across the full test set
        # before computing image/pixel metrics.
        self.requires_full_test_postprocess = True
        self._pending_samples: List[ADDataSample] = []
        self._pending_score_maps: List[Tensor] = []
        self._pending_raw_image_scores: List[float] = []

        # Dummy parameter so optimizer doesn't get empty param list
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=True)

    def _clear_pending_postprocess(self) -> None:
        """Clear deferred test-time normalization buffers."""
        self._pending_samples = []
        self._pending_score_maps = []
        self._pending_raw_image_scores = []

    def _resolve_runtime_image_score_aggregation(self) -> str:
        """Resolve the effective image-score aggregation for inference.

        Vanilla SAA uses anomaly-map max as the image score. Proposal-level
        image-score heuristics are reserved for the hybrid SAA+ path.
        """
        if self.mode == 'saa':
            return 'map_max'
        return self.image_score_aggregation

    def _resolve_runtime_topk_rank_mode(self) -> str:
        """Resolve the effective top-k ranking mode for inference.

        Vanilla SAA should follow the detection-score path without saliency-
        specific proposal ranking overrides.
        """
        if self.mode == 'saa':
            return 'combined'
        return self.topk_rank_mode

    @staticmethod
    def _normalize_k_mask(value: int) -> int:
        """Validate a runtime mask-count override."""
        value = int(value)
        if value <= 0:
            raise AssertionError('k_mask must be positive')
        return value

    @staticmethod
    def _normalize_defect_area_threshold(value: float) -> float:
        """Validate a runtime defect-area threshold override."""
        value = float(value)
        if value <= 0.0:
            raise AssertionError('defect_area_threshold must be positive')
        return value

    @staticmethod
    def _normalize_box_area_tolerance(value: float) -> float:
        """Validate a numeric tolerance applied to proposal area gating."""
        value = float(value)
        if value < 0.0:
            raise AssertionError('box_area_tolerance must be non-negative')
        return value

    @staticmethod
    def _resolve_class_override(mapping: Dict[str, Union[int, float]], cls_name: str):
        """Resolve a per-class override using canonicalized class names."""
        if cls_name in mapping:
            return mapping[cls_name]
        canonical_name = cls_name.lower().replace(' ', '_')
        if canonical_name in mapping:
            return mapping[canonical_name]
        return None

    def _resolve_runtime_k_mask(self, cls_name: str) -> int:
        """Resolve the effective mask count for inference."""
        if self.mode != 'saa':
            return self.k_mask
        override = self._resolve_class_override(self.k_mask_overrides, cls_name)
        return self.k_mask if override is None else override

    def _resolve_runtime_defect_area_threshold(self, cls_name: str) -> float:
        """Resolve the effective defect-area threshold for inference."""
        if self.mode != 'saa':
            return self.defect_area_threshold
        override = self._resolve_class_override(self.defect_area_threshold_overrides, cls_name)
        return self.defect_area_threshold if override is None else override

    def _resolve_runtime_box_area_tolerance(self, cls_name: str) -> float:
        """Resolve the effective bbox max-area tolerance for inference."""
        override = self._resolve_class_override(self.box_area_tolerance_overrides, cls_name)
        return self.box_area_tolerance if override is None else override

    def _resolve_gdino_config(self, path: str) -> str:
        """Resolve GroundingDINO config path, auto-finding from pip package."""
        if os.path.isfile(path):
            return path
        # Try resolving from the groundingdino package directory
        import groundingdino
        pkg_dir = os.path.dirname(groundingdino.__file__)
        basename = os.path.basename(path)
        pkg_path = os.path.join(pkg_dir, 'config', basename)
        if os.path.isfile(pkg_path):
            return pkg_path
        raise FileNotFoundError(
            f'GroundingDINO config not found at {path} or {pkg_path}. '
            f'Ensure groundingdino is installed correctly.'
        )

    def _ensure_gdino_loaded(self) -> None:
        """Lazy-load Grounding DINO on first use."""
        if self.gdino_model is None:
            config_path = self._resolve_gdino_config(self._gdino_config_path)

            # Handle auto-download for GroundingDINO weights
            gdino_ckpt = self._gdino_checkpoint
            if gdino_ckpt == 'auto' or not os.path.isfile(gdino_ckpt):
                try:
                    gdino_ckpt = _download_gdino_weights()
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to download GroundingDINO weights: {e}. "
                        "Please download manually from "
                        "https://github.com/IDEA-Research/GroundingDINO/releases"
                    )

            self.gdino_model = load_gdino_model(config_path, gdino_ckpt)
            device = next(self.parameters(), torch.tensor(0)).device
            self.gdino_model = self.gdino_model.to(device)
            self.gdino_model.eval()
            self._gdino_transform = gdino_transforms.Compose(
                [
                    gdino_transforms.RandomResize([800], max_size=1333),
                    gdino_transforms.ToTensor(),
                    gdino_transforms.Normalize(
                        [0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225],
                    ),
                ]
            )

    def _ensure_sam_loaded(self) -> None:
        """Lazy-load SAM on first use."""
        if self.sam_predictor is None:
            # Handle auto-download for SAM weights
            sam_ckpt = self._sam_checkpoint
            if sam_ckpt == 'auto' or not os.path.isfile(sam_ckpt):
                try:
                    sam_ckpt = _download_sam_weights()
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to download SAM weights: {e}. "
                        "Please download manually from "
                        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
                    )

            device = next(self.parameters(), torch.tensor(0)).device
            sam = sam_model_registry[self._sam_model_type](checkpoint=sam_ckpt)
            sam = sam.to(device)
            sam.eval()
            self.sam_predictor = SamPredictor(sam)

    def _ensure_models_loaded(self) -> None:
        """Lazy-load Grounding DINO and SAM on first use."""
        self._ensure_gdino_loaded()
        self._ensure_sam_loaded()

    @staticmethod
    def _denormalize_to_bgr_numpy(tensor: Tensor) -> np.ndarray:
        """Reverse ImageNet normalization and convert to BGR uint8 numpy.

        Args:
            tensor: (C, H, W) float tensor, ImageNet-normalized.

        Returns:
            (H, W, 3) uint8 BGR numpy array.
        """
        img = tensor.detach().cpu().float().numpy()  # (C, H, W)
        img = img.transpose(1, 2, 0)  # (H, W, C), RGB order
        img = img * _NORM_STD + _NORM_MEAN  # denormalize to 0-255
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = img[:, :, ::-1]  # RGB → BGR
        return np.ascontiguousarray(img)

    def _get_prompts(self, cls_name: str) -> Tuple[List[Tuple[str, str]], Optional[str]]:
        """Resolve prompts for a given category.

        Returns:
            prompts: List of (defect_prompt, filter_phrase) tuples.
            property_prompt: Property prompt string for SAA+ mode.
        """
        custom = None
        if self.custom_prompts and cls_name in self.custom_prompts:
            custom = self.custom_prompts[cls_name]
        return build_saa_prompts(cls_name, mode=self.mode, custom_prompts=custom)

    def _prepare_gdino_image(self, image_bgr: np.ndarray) -> Tensor:
        """Prepare a BGR image for official-style GroundingDINO inference."""
        import cv2
        from PIL import Image

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        dino_image, _ = self._gdino_transform(pil_image, None)
        return dino_image

    @torch.no_grad()
    def _get_grounding_output(
        self,
        dino_image: Tensor,
        caption: str,
    ) -> Tuple[Tensor, Tensor, str]:
        """Get raw GroundingDINO outputs following the official SAA path."""
        caption = caption.lower().strip()
        if not caption.endswith('.'):
            caption = caption + '.'

        device = next(self.gdino_model.parameters()).device
        with torch.no_grad():
            outputs = self.gdino_model(dino_image[None].to(device), captions=[caption])

        logits = outputs['pred_logits'].sigmoid()[0]
        boxes = outputs['pred_boxes'][0]
        return boxes, logits, caption

    def _bbox_suppression(
        self,
        boxes: Tensor,
        logits: Tensor,
        object_phrase: str,
        filtered_phrase: str,
        bbox_score_thr: float,
        text_score_thr: float,
        object_max_area: float,
        object_min_area: float = 0.0,
        box_area_tolerance: Optional[float] = None,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[List[str]]]:
        """Official SAA bbox filtering pipeline."""
        if box_area_tolerance is None:
            box_area_tolerance = self.box_area_tolerance
        else:
            box_area_tolerance = self._normalize_box_area_tolerance(box_area_tolerance)
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        boxes_area = boxes_filt[:, 2] * boxes_filt[:, 3]

        box_score_mask = logits_filt.max(dim=1)[0] > bbox_score_thr
        box_max_area_mask = boxes_area < (object_max_area + box_area_tolerance)
        box_min_area_mask = boxes_area > object_min_area

        filt_mask = torch.bitwise_and(box_score_mask, box_max_area_mask)
        filt_mask = torch.bitwise_and(filt_mask, box_min_area_mask)

        if torch.sum(filt_mask) == 0:
            return None, None, None

        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]

        tokenizer = self.gdino_model.tokenizer
        tokenized = tokenizer(object_phrase)

        pred_phrases = []
        boxes_filtered = []
        logits_filtered = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_score_thr, tokenized, tokenizer)
            if pred_phrase.count(filtered_phrase) > 0:
                continue

            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            boxes_filtered.append(box)
            logits_filtered.append(logit.max().item())

        if not boxes_filtered:
            return None, None, None

        return (
            torch.stack(boxes_filtered, dim=0),
            torch.tensor(logits_filtered, dtype=torch.float32),
            pred_phrases,
        )

    @torch.no_grad()
    def _detect_object(
        self,
        image_bgr: np.ndarray,
        object_prompt: str,
        object_max_area: float = 1.0,
        use_sam: bool = True,
    ) -> Tuple[float, Optional[Tensor]]:
        """Detect the object itself to determine max defect area.

        This is the Object TGMP step from the official implementation.
        First detects the object (e.g., "bottle") to calculate object_area,
        then defect_max_area = object_area * defect_area_threshold.

        Args:
            image_bgr: (H, W, 3) BGR uint8 numpy array.
            object_prompt: The object name to detect (e.g., "bottle").

        Returns:
            object_area: Maximum area of detected object (normalized 0-1).
                         Returns 1.0 if no object detected.
            object_masks: Masks of detected objects (N, H, W).
        """
        # Use a simple prompt to detect the object itself
        # Format: (defect_prompt, filter_phrase) - no filtering for object detection
        prompts = [(object_prompt, 'PlaceHolder')]  # filter_phrase won't match anything
        dino_image = self._prepare_gdino_image(image_bgr)

        all_boxes, all_scores = [], []
        for defect_prompt, filter_phrase in prompts:
            boxes, logits, object_phrase = self._get_grounding_output(
                dino_image,
                defect_prompt,
            )
            boxes_filtered, logits_filtered, _ = self._bbox_suppression(
                boxes,
                logits,
                object_phrase,
                filter_phrase,
                self.box_threshold,
                self.text_threshold,
                object_max_area,
                0.0,
            )
            if boxes_filtered is not None:
                all_boxes.append(boxes_filtered.cpu())
                all_scores.append(logits_filtered.cpu())

        H, W = image_bgr.shape[:2]
        if not all_boxes:
            # Official fallback keeps a zero mask and full-area prior.
            return 1.0, torch.zeros(1, H, W)

        boxes = torch.cat(all_boxes, dim=0)
        box_areas = boxes[:, 2] * boxes[:, 3]
        valid = box_areas < object_max_area

        if not valid.any():
            return 1.0, torch.zeros(1, H, W)

        boxes = boxes[valid]
        box_areas = box_areas[valid]
        max_box_area = box_areas.max().item()

        if not use_sam:
            return max_box_area, torch.zeros(len(boxes), H, W)

        # Get masks for objects using SAM
        masks, _ = self._segment_with_sam(image_bgr, boxes, H, W)

        return max_box_area, masks

    @torch.no_grad()
    def _detect_with_grounding_dino(
        self,
        image_bgr: np.ndarray,
        prompts: List[Tuple[str, str]],
        object_max_area: float = 1.0,
        object_min_area: float = 0.0,
        box_area_tolerance: Optional[float] = None,
    ) -> Tuple[Tensor, Tensor, List[str]]:
        """Run Grounding DINO on a single image with multiple prompts.

        Args:
            image_bgr: (H, W, 3) BGR uint8 numpy array.
            prompts: List of (defect_prompt, filter_phrase) tuples.
                The filter_phrase is used to filter out detections of the object itself.

        Returns:
            boxes: (N, 4) tensor in cxcywh format (normalized 0-1).
            scores: (N,) confidence scores.
            phrases: list of matched phrases.
        """
        dino_image = self._prepare_gdino_image(image_bgr)

        all_boxes, all_scores, all_phrases = [], [], []
        for defect_prompt, filter_phrase in prompts:
            boxes, logits, object_phrase = self._get_grounding_output(
                dino_image,
                defect_prompt,
            )
            boxes_filtered, logits_filtered, pred_phrases = self._bbox_suppression(
                boxes,
                logits,
                object_phrase,
                filter_phrase,
                self.box_threshold,
                self.text_threshold,
                object_max_area,
                object_min_area,
                box_area_tolerance=box_area_tolerance,
            )
            if boxes_filtered is not None:
                all_boxes.append(boxes_filtered.cpu())
                all_scores.append(logits_filtered.cpu())
                all_phrases.extend(pred_phrases)

        if not all_boxes:
            return (
                torch.zeros(0, 4),
                torch.zeros(0),
                [],
            )

        boxes = torch.cat(all_boxes, dim=0)
        scores = torch.cat(all_scores, dim=0)

        if self.nms_threshold is not None and len(boxes) > 1:
            boxes_xyxy = self._cxcywh_to_xyxy(boxes)
            keep = self._nms(boxes_xyxy, scores, self.nms_threshold)
            boxes = boxes[keep]
            scores = scores[keep]
            all_phrases = [all_phrases[i] for i in keep]

        return boxes, scores, all_phrases

    @staticmethod
    def _cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
        """Convert (cx, cy, w, h) to (x1, y1, x2, y2)."""
        cx, cy, w, h = boxes.unbind(-1)
        return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)

    @staticmethod
    def _nms(boxes_xyxy: Tensor, scores: Tensor, threshold: float) -> List[int]:
        """Simple NMS implementation."""
        from torchvision.ops import nms
        keep = nms(boxes_xyxy, scores, threshold)
        return keep.tolist()

    @torch.no_grad()
    def _segment_with_sam(
        self, image_bgr: np.ndarray, boxes: Tensor, H: int, W: int,
    ) -> Tuple[Tensor, Tensor]:
        """Run SAM on detected boxes.

        Args:
            image_bgr: (H, W, 3) BGR uint8 numpy array.
            boxes: (N, 4) normalized cxcywh boxes.
            H, W: original image dimensions.

        Returns:
            masks: (N, H, W) binary masks.
            iou_scores: (N,) IoU prediction scores from SAM.
        """
        import cv2
        if self.sam_preconvert_rgb:
            image_for_sam = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        else:
            # Official SAA repo forwards the cv2-loaded BGR image directly.
            image_for_sam = image_bgr
        self.sam_predictor.set_image(image_for_sam)

        # Convert normalized cxcywh to pixel xyxy
        boxes_xyxy = self._cxcywh_to_xyxy(boxes)
        boxes_pixel = boxes_xyxy.clone()
        boxes_pixel[:, [0, 2]] *= W
        boxes_pixel[:, [1, 3]] *= H

        device = self.sam_predictor.model.device
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
            boxes_pixel.to(device), (H, W)
        )

        masks, iou_preds, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        # masks: (N, 1, H_sam, W_sam) → (N, H, W)
        masks = masks[:, 0]
        if masks.shape[-2:] != (H, W):
            masks = F.interpolate(
                masks.unsqueeze(1).float(), size=(H, W),
                mode='bilinear', align_corners=False,
            )[:, 0]
        masks = (masks > 0.5).float()
        iou_preds = iou_preds[:, 0]

        return masks.cpu(), iou_preds.cpu()

    @torch.no_grad()
    def _extract_region_features(
        self,
        features: Tensor,
        one_object_mask: np.ndarray,
        other_object_masks: List[np.ndarray],
    ) -> Tuple[Tensor, np.ndarray, Tensor]:
        """Official multi-instance feature extraction helper for SAA+."""
        features_clone = features.clone()

        one_mask_features = []
        one_feature_locations = []
        for h in range(one_object_mask.shape[0]):
            for w in range(one_object_mask.shape[1]):
                if one_object_mask[h, w] > 0:
                    one_mask_features.append(features_clone[:, :, h, w].clone())
                    one_feature_locations.append(np.array((h, w)))
                    features_clone[:, :, h, w] = 0.0

        one_feature_locations_np = np.stack(one_feature_locations, axis=0)
        one_mask_feature = torch.cat(one_mask_features, dim=0)

        _, channels, _, _ = features_clone.shape
        features_clone_flat = features_clone.view(channels, -1)

        other_mask_features = []
        for other_mask in other_object_masks:
            other_mask_flat = other_mask.reshape(-1)
            mask_features = features_clone_flat[:, other_mask_flat > 0]
            other_mask_features.append(mask_features)

        other_mask_features = torch.cat(other_mask_features, dim=1).T
        return one_mask_feature, one_feature_locations_np, other_mask_features

    @torch.no_grad()
    def _compute_single_object_saliency_map(self, image_bgr: np.ndarray) -> np.ndarray:
        """Compute the official single-instance self-similarity map."""
        import cv2

        self.saliency_backbone.set_img_size(256)
        resize_image = cv2.resize(image_bgr, (256, 256))
        features, _, _ = self.saliency_backbone(resize_image)

        batch, channels, height, width = features.shape
        assert batch == 1

        features_flat = features.view(batch * channels, height * width)
        features_self_similarity = features_flat.T @ features_flat
        features_self_similarity = 0.5 * (1 - features_self_similarity)
        features_self_similarity = features_self_similarity.sort(dim=1, descending=True)[0]
        features_self_similarity = torch.mean(
            features_self_similarity[:, :min(400, features_self_similarity.shape[1])],
            dim=1,
        )
        heat_map = features_self_similarity.view(height, width).cpu().numpy()
        return cv2.resize(heat_map, (image_bgr.shape[1], image_bgr.shape[0]))

    @torch.no_grad()
    def _compute_multi_object_saliency_map(
        self,
        image_bgr: np.ndarray,
        object_masks: Tensor,
    ) -> np.ndarray:
        """Compute the official multi-instance self-similarity map."""
        import cv2

        self.saliency_backbone.set_img_size(1024)
        resize_image = cv2.resize(image_bgr, (1024, 1024))
        features, _, _ = self.saliency_backbone(resize_image)

        feature_size = features.shape[2:]
        object_masks_np = object_masks.cpu().numpy().astype(np.int32)
        resize_object_masks = [
            cv2.resize(mask, feature_size, interpolation=cv2.INTER_NEAREST)
            for mask in object_masks_np
        ]

        mask_anomaly_scores = []
        for index in range(len(resize_object_masks)):
            other_masks = resize_object_masks[:index] + resize_object_masks[index + 1:]
            if not other_masks:
                continue

            (
                one_mask_feature,
                one_feature_locations,
                other_mask_features,
            ) = self._extract_region_features(
                features,
                resize_object_masks[index],
                other_masks,
            )

            similarity = one_mask_feature @ other_mask_features.T
            similarity = similarity.max(dim=1)[0]
            anomaly_score = 0.5 * (1.0 - similarity)
            anomaly_score = anomaly_score.cpu().numpy()

            mask_anomaly_score = np.zeros(feature_size, dtype=np.float32)
            for location, score in zip(one_feature_locations, anomaly_score):
                mask_anomaly_score[location[0], location[1]] = score
            mask_anomaly_scores.append(mask_anomaly_score)

        if not mask_anomaly_scores:
            return np.zeros((image_bgr.shape[0], image_bgr.shape[1]), dtype=np.float32)

        mask_anomaly_scores = np.stack(mask_anomaly_scores, axis=0)
        mask_anomaly_scores = np.max(mask_anomaly_scores, axis=0)
        return cv2.resize(mask_anomaly_scores, (image_bgr.shape[1], image_bgr.shape[0]))

    @torch.no_grad()
    def _compute_saliency_map(
        self,
        image_bgr: np.ndarray,
        object_masks: Optional[Tensor],
        object_number: int,
    ) -> np.ndarray:
        """Dispatch saliency-map computation following the official SAA+ logic."""
        if self.saliency_backbone is None:
            return np.zeros((image_bgr.shape[0], image_bgr.shape[1]), dtype=np.float32)

        if object_masks is None or len(object_masks) == 0:
            return self._compute_single_object_saliency_map(image_bgr)

        if object_number <= 1:
            return self._compute_single_object_saliency_map(image_bgr)

        non_empty = [mask for mask in object_masks if float(mask.sum().item()) > 0]
        if len(non_empty) <= 1:
            return self._compute_single_object_saliency_map(image_bgr)

        return self._compute_multi_object_saliency_map(image_bgr, torch.stack(non_empty))

    @torch.no_grad()
    def _compute_saliency(
        self,
        image_bgr: np.ndarray,
        object_masks: Optional[Tensor],
        defect_masks: Tensor,
        object_number: int = 1,
    ) -> Tensor:
        """Compute saliency scores for SAA+ mode using feature self-similarity.

        Official implementation uses: 0.5 * (1 - cosine_similarity)
        with ImageNet pretrained features, then averages top-400 most similar features.

        Args:
            image_tensor: (C, H, W) normalized input tensor.
            object_masks: (N_obj, H, W) masks of detected objects, or None.
            defect_masks: (N, H, W) binary masks of detected defects.

        Returns:
            saliency_scores: (N,) saliency score per mask.
        """
        if self.saliency_backbone is None:
            return torch.ones(defect_masks.shape[0])

        saliency_map = self._compute_saliency_map(
            image_bgr,
            object_masks,
            object_number=object_number,
        )

        # Compute per-mask saliency scores
        saliency_scores = []
        for mask in defect_masks:
            mask_np = mask.cpu().numpy().astype(np.float32)
            mask_pixels = saliency_map[mask_np > 0.5]

            if len(mask_pixels) > 0:
                score = np.exp(3 * mask_pixels.mean())
            else:
                score = 1.0

            saliency_scores.append(score)

        return torch.tensor(saliency_scores, dtype=torch.float32, device=defect_masks.device)

    @staticmethod
    def _resolve_class_phrase_filter(
        phrase_filter: Optional[Dict[str, List[str]]],
        cls_name: str,
    ) -> List[str]:
        """Resolve class-specific phrase filter tokens."""
        if not phrase_filter:
            return []
        return phrase_filter.get(cls_name) or phrase_filter.get(cls_name.lower().replace(' ', '_')) or []

    @staticmethod
    def _normalize_area_range(
        area_range: Optional[Tuple[Optional[float], Optional[float]]],
    ) -> Optional[Tuple[Optional[float], Optional[float]]]:
        """Validate and normalize an optional area-range tuple."""
        if area_range is None:
            return None
        if not isinstance(area_range, (list, tuple)) or len(area_range) != 2:
            raise AssertionError('image_score_area_range must be a 2-item list/tuple or None')
        min_area, max_area = area_range
        if min_area is not None:
            min_area = float(min_area)
            if not 0.0 <= min_area <= 1.0:
                raise AssertionError('image_score_area_range min must be within [0, 1]')
        if max_area is not None:
            max_area = float(max_area)
            if not 0.0 <= max_area <= 1.0:
                raise AssertionError('image_score_area_range max must be within [0, 1]')
        if min_area is not None and max_area is not None and min_area > max_area:
            raise AssertionError('image_score_area_range min must not exceed max')
        return (min_area, max_area)

    @staticmethod
    def _normalize_saliency_score_clip_max(
        clip_max: Optional[float],
    ) -> Optional[float]:
        """Validate an optional saliency clipping upper bound."""
        if clip_max is None:
            return None
        clip_max = float(clip_max)
        if clip_max <= 0.0:
            raise AssertionError('saliency_score_clip_max must be positive')
        return clip_max

    @staticmethod
    def _match_phrase_tokens(
        phrases: List[str],
        tokens: List[str],
    ) -> List[int]:
        """Return proposal indices whose extracted phrases match any token."""
        if not tokens:
            return []
        normalized_tokens = [token.lower() for token in tokens]
        matched_indices = []
        for index, phrase in enumerate(phrases):
            phrase_lower = phrase.lower()
            if any(token in phrase_lower for token in normalized_tokens):
                matched_indices.append(index)
        return matched_indices

    def _resolve_image_score_rank_mode(self, cls_name: str) -> str:
        """Resolve per-class image-score ranking mode override."""
        return (
            self.image_score_rank_mode_overrides.get(cls_name)
            or self.image_score_rank_mode_overrides.get(cls_name.lower().replace(' ', '_'))
            or self.image_score_rank_mode
        )

    def _resolve_saliency_score_mode(self, cls_name: str) -> str:
        """Resolve per-class saliency-score combination mode override."""
        return (
            self.saliency_score_mode_overrides.get(cls_name)
            or self.saliency_score_mode_overrides.get(cls_name.lower().replace(' ', '_'))
            or self.saliency_score_mode
        )

    def _resolve_saliency_score_clip_max(self, cls_name: str) -> Optional[float]:
        """Resolve per-class saliency clipping upper bound override."""
        return (
            self.saliency_score_clip_max_overrides.get(cls_name)
            or self.saliency_score_clip_max_overrides.get(cls_name.lower().replace(' ', '_'))
            or self.saliency_score_clip_max
        )

    def _resolve_image_score_area_range(
        self,
        cls_name: str,
    ) -> Optional[Tuple[Optional[float], Optional[float]]]:
        """Resolve per-class image-score area-range override."""
        return (
            self.image_score_area_range_overrides.get(cls_name)
            or self.image_score_area_range_overrides.get(cls_name.lower().replace(' ', '_'))
            or self.image_score_area_range
        )

    @staticmethod
    def _compute_mask_area_ratios(masks: Tensor) -> Tensor:
        """Compute per-mask binary area ratios in the image plane."""
        if len(masks) == 0:
            return torch.zeros(0, dtype=torch.float32, device=masks.device)
        return (masks > 0.5).float().view(masks.shape[0], -1).mean(dim=1)

    @staticmethod
    def _select_scores_by_mode(
        det_scores: Tensor,
        saliency_scores: Tensor,
        combined_scores: Tensor,
        mode: str,
    ) -> Tensor:
        """Select the score tensor corresponding to a ranking mode."""
        if mode == 'combined':
            return combined_scores
        if mode == 'det':
            return det_scores
        if mode == 'saliency':
            return saliency_scores
        raise ValueError(f'Unsupported score selection mode: {mode}')

    def _combine_detection_saliency_scores(
        self,
        det_scores: Tensor,
        saliency_scores: Tensor,
        cls_name: str,
        saliency_score_mode: Optional[str] = None,
        saliency_score_clip_max: Optional[float] = None,
    ) -> Tuple[Tensor, str, Optional[float]]:
        """Combine detection and saliency scores using the resolved mode."""
        resolved_mode = saliency_score_mode or self._resolve_saliency_score_mode(cls_name)
        resolved_clip_max = (
            self._normalize_saliency_score_clip_max(saliency_score_clip_max)
            if saliency_score_clip_max is not None
            else self._resolve_saliency_score_clip_max(cls_name)
        )

        if resolved_mode == 'identity':
            return det_scores, resolved_mode, resolved_clip_max
        if resolved_mode == 'multiply':
            return det_scores * saliency_scores, resolved_mode, resolved_clip_max
        if resolved_mode == 'clipped_multiply':
            if resolved_clip_max is None:
                raise ValueError('clipped_multiply requires saliency_score_clip_max')
            clipped_saliency = torch.clamp(saliency_scores, max=resolved_clip_max)
            return det_scores * clipped_saliency, resolved_mode, resolved_clip_max
        raise ValueError(f'Unsupported saliency_score_mode: {resolved_mode}')

    def _select_image_score_indices(
        self,
        cls_name: str,
        phrases: List[str],
        masks: Tensor,
        det_scores: Tensor,
        saliency_scores: Tensor,
        combined_scores: Tensor,
    ) -> Tuple[List[int], Tensor, Optional[Tuple[Optional[float], Optional[float]]], str]:
        """Select proposals reserved for image-level scoring only."""
        image_score_indices = list(range(len(phrases)))
        mask_area_ratios = self._compute_mask_area_ratios(masks)
        runtime_image_score_aggregation = self._resolve_runtime_image_score_aggregation()
        resolved_area_range = None
        resolved_rank_mode = 'combined'

        if self.mode != 'saa+' or runtime_image_score_aggregation not in (
            'topk_combined_score_max',
            'topk_combined_score_mean',
        ):
            return image_score_indices, mask_area_ratios, resolved_area_range, resolved_rank_mode

        resolved_area_range = self._resolve_image_score_area_range(cls_name)
        resolved_rank_mode = self._resolve_image_score_rank_mode(cls_name)

        phrase_allowlist = self._resolve_class_phrase_filter(
            self.image_score_phrase_allowlist,
            cls_name,
        )
        if phrase_allowlist:
            allowed_indices = self._match_phrase_tokens(phrases, phrase_allowlist)
            if allowed_indices:
                image_score_indices = allowed_indices

        phrase_blocklist = self._resolve_class_phrase_filter(
            self.image_score_phrase_blocklist,
            cls_name,
        )
        if phrase_blocklist:
            blocked_indices = set(self._match_phrase_tokens(phrases, phrase_blocklist))
            if blocked_indices:
                kept_indices = [
                    index for index in image_score_indices
                    if index not in blocked_indices
                ]
                if kept_indices:
                    image_score_indices = kept_indices

        if resolved_area_range is not None:
            min_area, max_area = resolved_area_range
            kept_indices = []
            for index in image_score_indices:
                area_ratio = float(mask_area_ratios[index].item())
                if min_area is not None and area_ratio < min_area:
                    continue
                if max_area is not None and area_ratio > max_area:
                    continue
                kept_indices.append(index)
            if kept_indices:
                image_score_indices = kept_indices

        if resolved_rank_mode != 'combined':
            image_score_rank_scores = self._select_scores_by_mode(
                det_scores,
                saliency_scores,
                combined_scores,
                resolved_rank_mode,
            )
            rank_subset = image_score_rank_scores[image_score_indices]
            topk = min(self.image_score_topk, len(rank_subset))
            top_indices = rank_subset.topk(topk).indices.tolist()
            image_score_indices = [image_score_indices[index] for index in top_indices]

        return image_score_indices, mask_area_ratios, resolved_area_range, resolved_rank_mode

    def _aggregate_anomaly_map(
        self,
        masks: Tensor,
        scores: Tensor,
        H: int,
        W: int,
        k_mask: Optional[int] = None,
        rank_scores: Optional[Tensor] = None,
        image_score_aggregation: Optional[str] = None,
        image_score_scores: Optional[Tensor] = None,
        defect_max_area: Optional[float] = None,
    ) -> Tuple[float, Tensor]:
        """Aggregate masks into a single anomaly map.

        Args:
            masks: (N, H, W) binary masks.
            scores: (N,) confidence scores.
            H, W: target spatial dimensions.
            k_mask: Max number of top-scoring masks to aggregate.
                    Uses self.k_mask if None.
            rank_scores: Scores used to choose top-k proposals. Uses ``scores``
                when omitted.
            image_score_aggregation: How to convert the selected proposals /
                anomaly map into an image-level score.
            image_score_scores: Proposal scores reserved for image-level
                scoring when that should differ from the map aggregation path.
            defect_max_area: Max fraction of image area for a valid defect mask.
                            Uses self.defect_area_threshold if None.

        Returns:
            img_score: scalar anomaly score.
            anomaly_map: (1, H, W) anomaly map.
        """
        if len(masks) == 0:
            return 0.0, torch.zeros(1, H, W)

        if k_mask is None:
            k_mask = self.k_mask
        if image_score_aggregation is None:
            image_score_aggregation = self.image_score_aggregation
        if defect_max_area is None:
            defect_max_area = self.defect_area_threshold
        if rank_scores is None:
            rank_scores = scores
        if image_score_scores is None:
            image_score_scores = scores

        # Keep top-k
        if len(scores) > k_mask:
            topk_idx = rank_scores.topk(k_mask).indices
            masks = masks[topk_idx]
            scores = scores[topk_idx]
            rank_scores = rank_scores[topk_idx]

        # Official confidence prompting averages the selected masks with a
        # background prior term in the denominator.
        anomaly_map = torch.zeros(H, W)
        weight_map = torch.ones(H, W)
        for m, s in zip(masks, scores):
            anomaly_map = anomaly_map + m * s.item()
            weight_map = weight_map + m

        valid_pixels = weight_map > 0
        anomaly_map[valid_pixels] = anomaly_map[valid_pixels] / weight_map[valid_pixels]
        if image_score_aggregation == 'map_max':
            img_score = float(anomaly_map.max().item())
        elif image_score_aggregation == 'map_p99':
            img_score = float(torch.quantile(anomaly_map.view(-1), 0.99).item())
        elif image_score_aggregation == 'support_mean':
            support = anomaly_map > 0
            if bool(support.any().item()):
                img_score = float(anomaly_map[support].mean().item())
            else:
                img_score = 0.0
        elif image_score_aggregation == 'topk_combined_score_max':
            img_score = float(image_score_scores.max().item()) if len(image_score_scores) > 0 else 0.0
        elif image_score_aggregation == 'topk_combined_score_mean':
            if len(image_score_scores) > 0:
                topk = min(self.image_score_topk, len(image_score_scores))
                img_score = float(image_score_scores.topk(topk).values.mean().item())
            else:
                img_score = 0.0
        else:
            raise ValueError(f'Unsupported image_score_aggregation: {image_score_aggregation}')

        if anomaly_map.shape[-2:] != (self.image_size, self.image_size):
            anomaly_map = F.interpolate(
                anomaly_map.unsqueeze(0).unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode='bilinear', align_corners=False,
            )[0]
        else:
            anomaly_map = anomaly_map.unsqueeze(0)  # (1, H, W)

        return img_score, anomaly_map

    @torch.no_grad()
    def _predict_single(
        self,
        image_tensor: Tensor,
        cls_name: str,
        image_bgr: Optional[np.ndarray] = None,
    ) -> Tuple[float, Tensor]:
        """Full SAA pipeline for a single image.

        Pipeline:
        1. Object TGMP: Detect the object itself to calculate object_area
        2. Calculate defect_max_area = object_area * defect_area_threshold
        3. Defect TGMP: Detect defects with Grounding DINO
        4. Filter detections by defect_max_area
        5. Segment with SAM
        6. Saliency rescoring (SAA+ only)
        7. Aggregate top-k masks

        Args:
            image_tensor: (C, H, W) ImageNet-normalized tensor.
            cls_name: category name.

        Returns:
            img_score: scalar anomaly score.
            anomaly_map: (1, H, W) anomaly map.
        """
        if image_bgr is None:
            H, W = image_tensor.shape[1], image_tensor.shape[2]
            image_bgr = self._denormalize_to_bgr_numpy(image_tensor)
        else:
            H, W = image_bgr.shape[:2]

        prompts, property_prompt = self._get_prompts(cls_name)

        # Parse property prompt for SAA+ mode
        k_mask = self._resolve_runtime_k_mask(cls_name)
        defect_area_threshold = self._resolve_runtime_defect_area_threshold(cls_name)
        object_prompt = cls_name
        object_number = 1
        object_max_area = 1.0

        if self.mode == 'saa+' and property_prompt is not None:
            props = parse_property_prompt(property_prompt)
            k_mask = props['k_mask']
            defect_area_threshold = props['defect_area_threshold']
            object_prompt = props.get('object_prompt', cls_name)
            object_number = props.get('object_number', 1)
            object_max_area = props.get('object_max_area', 1.0)

        # Step 1: Object TGMP - detect the object itself first
        object_area, object_masks = self._detect_object(
            image_bgr,
            object_prompt,
            object_max_area=object_max_area,
        )

        # Step 2: Calculate defect_max_area based on object_area
        defect_max_area = object_area * defect_area_threshold

        # Step 3: Defect TGMP - detect defects
        boxes, det_scores, phrases = self._detect_with_grounding_dino(
            image_bgr,
            prompts,
            object_max_area=defect_max_area,
            object_min_area=0.0,
            box_area_tolerance=self._resolve_runtime_box_area_tolerance(cls_name),
        )

        if len(boxes) == 0:
            return 0.0, torch.zeros(1, self.image_size, self.image_size)

        # Step 5: Segment with SAM
        masks, _ = self._segment_with_sam(image_bgr, boxes, H, W)

        # Official SAA/SAA+ rescoring starts from GroundingDINO confidence.
        combined_scores = det_scores
        saliency = torch.ones_like(det_scores)
        runtime_image_score_aggregation = self._resolve_runtime_image_score_aggregation()

        # Step 6: Saliency rescoring (SAA+ only)
        if self.mode == 'saa+' and self.saliency_backbone is not None:
            saliency = self._compute_saliency(
                image_bgr,
                object_masks,
                masks,
                object_number=object_number,
            )
            combined_scores, _, _ = self._combine_detection_saliency_scores(
                det_scores=det_scores,
                saliency_scores=saliency,
                cls_name=cls_name,
            )

        rank_scores = self._select_scores_by_mode(
            det_scores,
            saliency,
            combined_scores,
            self._resolve_runtime_topk_rank_mode(),
        )

        image_score_indices, _, _, _ = self._select_image_score_indices(
            cls_name=cls_name,
            phrases=phrases,
            masks=masks,
            det_scores=det_scores,
            saliency_scores=saliency,
            combined_scores=combined_scores,
        )

        if runtime_image_score_aggregation in (
            'topk_combined_score_max',
            'topk_combined_score_mean',
        ):
            image_score_scores = combined_scores[image_score_indices]
        else:
            image_score_scores = combined_scores

        # Step 7: Aggregate top-k masks
        return self._aggregate_anomaly_map(
            masks, combined_scores, H, W,
            k_mask=k_mask,
            rank_scores=rank_scores,
            image_score_aggregation=runtime_image_score_aggregation,
            image_score_scores=image_score_scores,
        )

    def forward(
        self,
        inputs: Union[Tensor, List[Tensor]],
        data_samples: Optional[List[ADDataSample]] = None,
        mode: str = 'tensor',
    ) -> Union[Dict[str, Tensor], List[ADDataSample], Tuple[Tensor, ...]]:
        """Forward with three modes following MMEngine convention.

        Args:
            inputs: (B, C, H, W) batch tensor or list of (C, H, W).
            data_samples: List of ADDataSample.
            mode: 'loss', 'predict', or 'tensor'.
        """
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        if mode == 'loss':
            # Zero-shot: no training needed
            return {'loss': torch.tensor(0.0, device=inputs.device,
                                         requires_grad=True)}

        elif mode == 'predict':
            self._ensure_models_loaded()
            B = inputs.shape[0]
            img_scores = []
            score_maps = []
            for i in range(B):
                # Determine category name
                cls_name = self.class_name
                image_bgr = None
                if data_samples and i < len(data_samples):
                    sample_cls = getattr(data_samples[i], 'cls_name', None)
                    if sample_cls:
                        cls_name = sample_cls
                    image_bgr = getattr(data_samples[i], 'ori_img_bgr', None)

                score, amap = self._predict_single(inputs[i], cls_name, image_bgr=image_bgr)
                img_scores.append(score)
                score_maps.append(amap)

            score_maps = torch.stack(score_maps, dim=0)  # (B, 1, H, W)
            results = build_predict_results(data_samples, img_scores, score_maps)

            self._pending_samples.extend(results)
            self._pending_score_maps.append(score_maps.detach().cpu())
            self._pending_raw_image_scores.extend(float(score) for score in img_scores)
            return results

        elif mode == 'tensor':
            # Return empty features (no backbone in standard sense)
            return (inputs,)

        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def train(self, mode: bool = True):
        """Override to keep all sub-models in eval mode."""
        super().train(mode)
        if self.gdino_model is not None:
            self.gdino_model.eval()
        if self.sam_predictor is not None:
            self.sam_predictor.model.eval()
        if self.saliency_backbone is not None:
            self.saliency_backbone.eval()
        return self

    def score_all(self):
        """Apply official dataset-level min-max normalization to all maps."""
        if not self._pending_samples:
            return []

        score_maps = torch.cat(self._pending_score_maps, dim=0).float()
        runtime_image_score_aggregation = self._resolve_runtime_image_score_aggregation()
        score_maps = minmax_normalize(score_maps)

        if runtime_image_score_aggregation == 'map_max':
            img_scores = score_maps.view(score_maps.shape[0], -1).max(dim=1).values
        elif runtime_image_score_aggregation == 'map_p99':
            img_scores = torch.quantile(score_maps.view(score_maps.shape[0], -1), 0.99, dim=1)
        elif runtime_image_score_aggregation == 'support_mean':
            values = []
            for score_map in score_maps:
                flat = score_map.view(-1)
                support = flat > 0
                if bool(support.any().item()):
                    values.append(float(flat[support].mean().item()))
                else:
                    values.append(0.0)
            img_scores = torch.tensor(values, dtype=torch.float32)
        elif runtime_image_score_aggregation == 'topk_combined_score_max':
            raw_scores = torch.tensor(self._pending_raw_image_scores, dtype=torch.float32)
            raw_min = raw_scores.min()
            raw_max = raw_scores.max()
            if float(raw_max.item() - raw_min.item()) > 1e-12:
                img_scores = (raw_scores - raw_min) / (raw_max - raw_min)
            else:
                img_scores = torch.zeros_like(raw_scores)
        elif runtime_image_score_aggregation == 'topk_combined_score_mean':
            raw_scores = torch.tensor(self._pending_raw_image_scores, dtype=torch.float32)
            raw_min = raw_scores.min()
            raw_max = raw_scores.max()
            if float(raw_max.item() - raw_min.item()) > 1e-12:
                img_scores = (raw_scores - raw_min) / (raw_max - raw_min)
            else:
                img_scores = torch.zeros_like(raw_scores)
        else:
            raise ValueError(f'Unsupported image_score_aggregation: {runtime_image_score_aggregation}')

        for index, sample in enumerate(self._pending_samples):
            sample.pred_score = float(img_scores[index].item())
            sample.pred_anomaly_map = score_maps[index:index + 1].detach().cpu()[0]

        results = list(self._pending_samples)
        self._clear_pending_postprocess()
        return results
