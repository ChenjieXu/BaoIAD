"""UniVAD: Universal Visual Anomaly Detection (CVPR 2025).

Training-free, few-shot anomaly detection using three modules:
- C3: Component segmentation via GroundingDINO + SAM-HQ (offline preprocessing)
- CAPM: Patch matching combining CLIP + DINOv2 + VL scoring
- GECM: Graph-based logical anomaly detection for multi-component objects

Requires CLIP ViT-L/14-336 + DINOv2 as feature backbones.
"""

import enum
import importlib.util
import inspect
import logging
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS
from baoiad.utils.univad_assets import (
    heat_mask_path_candidates,
    mask_path_candidates,
    split_masks_from_one_mask,
    split_masks_from_one_mask_with_bg,
)
from baoiad.models.base_ad_model import BaseADModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Object type enum for gate logic
# ---------------------------------------------------------------------------

class ObjectType(enum.Enum):
    """Object type determined by C3 component analysis."""
    TEXTURE = 'texture'
    SINGLE = 'single'
    MULTI = 'multi'


# ---------------------------------------------------------------------------
# Gaussian blur utility (same pattern as AnomalyCLIP)
# ---------------------------------------------------------------------------

def _gaussian_blur_bchw(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply Gaussian blur to a (B, C, H, W) tensor."""
    if sigma <= 0:
        return x
    radius = max(int(round(4 * sigma)), 1)
    kernel_size = 2 * radius + 1
    coord = torch.arange(kernel_size, device=x.device, dtype=x.dtype) - radius
    kernel_1d = torch.exp(-(coord ** 2) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(x.shape[1], 1, 1, 1)
    return F.conv2d(x, kernel, padding=radius, groups=x.shape[1])


# ---------------------------------------------------------------------------
# CFA: parameter-free graph aggregation for GECM
# ---------------------------------------------------------------------------

class CFA(nn.Module):
    """Component Feature Aggregation via graph neural network.

    Parameter-free: builds cosine similarity adjacency matrix,
    row-normalizes, then propagates features via matmul.
    """

    def __init__(self, n_layers=2):
        super().__init__()
        self.n_layers = n_layers

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """Aggregate node features via graph convolution.

        Args:
            node_features: (N, D) per-component features.

        Returns:
            Aggregated features (N, D).
        """
        if node_features.shape[0] <= 1:
            return node_features

        h = node_features
        for _ in range(self.n_layers):
            adj = F.cosine_similarity(h.unsqueeze(1), h.unsqueeze(0), dim=-1)
            deg = adj.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            adj_norm = adj / deg
            h = adj_norm @ h
        return h


# ---------------------------------------------------------------------------
# Text prompt templates for VL scoring
# ---------------------------------------------------------------------------

NORMAL_STATES = [
    "{}", "flawless {}", "perfect {}", "unblemished {}",
    "{} without flaw", "{} without defect", "{} without damage",
]

ANOMALOUS_STATES = [
    "damaged {}", "broken {}", "{} with flaw", "{} with defect", "{} with damage",
]

TEMPLATES = [
    "a bad photo of a {}.",
    "a low resolution photo of the {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a bright photo of a {}.",
    "a dark photo of the {}.",
    "a photo of my {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a photo of one {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "a low resolution photo of a {}.",
    "a photo of a large {}.",
    "a blurry photo of a {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a photo of the small {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "a dark photo of a {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "there is a {} in the scene.",
    "there is the {} in the scene.",
    "this is a {} in the scene.",
    "this is the {} in the scene.",
    "this is one {} in the scene.",
]


from baoiad.utils.score_utils import normalize_class_name as _normalize_class_name


def _create_prompt_ensemble(class_name='object'):
    class_name = _normalize_class_name(class_name)
    normal_states = [s.format(class_name) for s in NORMAL_STATES]
    anomalous_states = [s.format(class_name) for s in ANOMALOUS_STATES]
    normal_prompts = [t.format(s) for s in normal_states for t in TEMPLATES]
    anomalous_prompts = [t.format(s) for s in anomalous_states for t in TEMPLATES]
    return normal_prompts, anomalous_prompts


def _supports_out_layers_api(encode_image) -> bool:
    """Check whether a CLIP encode_image method accepts `out_layers`."""
    try:
        signature = inspect.signature(encode_image)
    except (TypeError, ValueError):
        return False
    return 'out_layers' in signature.parameters


def _matching_clip_layer_indices(num_layers: int, requested_layers: int) -> List[int]:
    """Match UniVAD's local CLIP fork layer indexing semantics.

    The official local CLIP fork returns two tensors per requested layer
    (surgery output + original output), resulting in 2 * len(out_layers) tensors.
    In that case, we use odd indices to select the original outputs.

    When using standard OpenCLIP (fallback path), we get one tensor per layer,
    so we use all available layers for patch matching.
    """
    if num_layers > requested_layers:
        # Local CLIP fork returns 2 tensors per layer (surgery + original)
        return [idx for idx in range(num_layers) if idx % 2 == 1]
    # Standard OpenCLIP returns 1 tensor per layer
    return list(range(min(num_layers, requested_layers)))


def _reference_component_extractor_cls():
    """Load the official UniVAD component extractor class on demand."""
    ref_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        '.refs',
        'UniVAD_ref',
        'models',
        'component_feature_extractor.py',
    )
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f'Missing UniVAD reference component extractor: {ref_path}')
    spec = importlib.util.spec_from_file_location('univad_ref_component_feature_extractor', ref_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Failed to load UniVAD reference component extractor spec: {ref_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ComponentFeatureExtractor


def _vl_clip_layer_index(num_layers: int, requested_layers: int) -> int:
    """Select the CLIP token index used by UniVAD's VL branch.

    The official local CLIP fork returns two token tensors per requested layer,
    and UniVAD hard-codes `patch_tokens[6]` for VL scoring. When the fallback
    path only exposes one tensor per requested layer, use the last available
    layer as the closest approximation instead of silently falling back to the
    first layer.
    """
    if num_layers <= 0:
        return 0
    if num_layers > requested_layers:
        return min(6, num_layers - 1)
    return num_layers - 1


# ---------------------------------------------------------------------------
# UniVAD Detector
# ---------------------------------------------------------------------------

@MODELS.register_module(force=True)
class UniVADDetector(BaseADModel):
    """UniVAD: Universal Visual Anomaly Detection.

    Few-shot, training-free detector that combines:
    - CLIP multi-layer patch matching (CAPM)
    - DINOv2 patch matching (CAPM)
    - CLIP vision-language anomaly scoring (CAPM)
    - Graph-based component anomaly detection (GECM)

    The gate logic determines per-category object type (TEXTURE/SINGLE/MULTI)
    based on C3 component masks from offline preprocessing.

    Args:
        clip_model: OpenCLIP model name for CLIP backbone.
        clip_pretrained: Pretrained source for CLIP.
        dinov2_model: DINOv2 model name for DINOv2 backbone.
        clip_layers: 1-indexed intermediate CLIP transformer layers for
            multi-layer hook extraction.
        k_shot: Number of reference normal images per category.
        image_size: Input image resolution. Official few-shot benchmark uses 448.
        mask_dir: Directory containing C3 component masks (.npy or grounding_mask.png).
            If empty, all categories fall back to TEXTURE mode.
        object_ratio_threshold: Ratio threshold for texture vs. object gate.
        max_segment_for_texture: Max component count to consider texture.
        gaussian_sigma: Gaussian blur sigma for anomaly maps. Official code path
            does not smooth by default.
        gecm_enable: Whether to enable GECM for MULTI-type objects.
        gecm_layers: Number of CFA graph layers.
        clip_weight: Weight for CLIP patch score in CAPM fusion.
        dinov2_weight: Weight for DINOv2 patch score in CAPM fusion.
        vl_weight: Weight for VL anomaly score in CAPM fusion.
    """

    def __init__(
        self,
        clip_model='ViT-L-14-336',
        clip_pretrained='openai',
        dinov2_model='dinov2_vitg14',
        clip_layers=(6, 12, 18, 24),
        k_shot=1,
        image_size=448,
        mask_dir='',
        object_ratio_threshold=0.65,
        max_segment_for_texture=2,
        gate_overrides=None,
        gaussian_sigma=0.0,
        gecm_enable=True,
        gecm_layers=2,
        clip_weight=1.0,
        dinov2_weight=1.0,
        vl_weight=1.0,
        single_image_clip_weight=None,
        single_image_dinov2_weight=None,
        single_image_vl_weight=None,
        single_image_weight_overrides=None,
        single_image_score_overrides=None,
        single_image_mix_max_weight=1.0,
        single_image_mix_topk_weight=1.0,
        single_pixel_clip_weight=None,
        single_pixel_dinov2_weight=None,
        single_pixel_vl_weight=None,
        single_pixel_global_weight=1.0,
        single_pixel_part_weight=1.0,
        single_pixel_weight_overrides=None,
        multi_pixel_gecm_weight=1.0,
        multi_pixel_weight_overrides=None,
        multi_image_gecm_weight=1.0,
        multi_image_weight_overrides=None,
        multi_gecm_clip_weight=1.0,
        multi_gecm_dino_weight=1.0,
        multi_gecm_geo_weight=1.0,
        multi_gecm_feature_weight_overrides=None,
        multi_component_min_ratio=1e-4,
        multi_component_min_ratio_overrides=None,
        single_image_pooling='max',
        single_image_topk_ratio=0.01,
        official_scoring=False,
        strict_mode=False,
        require_mask_dir=False,
        require_heat_mask_dir=False,
        require_query_masks=False,
        heat_mask_dir='',
        ref_mask_dilation_kernel=20,
        query_mask_dilation_kernel=5,
        component_mask_dilation_kernel=5,
        component_rotations=4,
        data_preprocessor=None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.clip_layers = tuple(clip_layers)
        self.k_shot = k_shot
        self.image_size = image_size
        self.mask_dir = mask_dir
        self.object_ratio_threshold = object_ratio_threshold
        self.max_segment_for_texture = max_segment_for_texture
        self.gaussian_sigma = gaussian_sigma
        self.gecm_enable = gecm_enable
        self.clip_weight = clip_weight
        self.dinov2_weight = dinov2_weight
        self.vl_weight = vl_weight
        self.single_image_clip_weight = single_image_clip_weight
        self.single_image_dinov2_weight = single_image_dinov2_weight
        self.single_image_vl_weight = single_image_vl_weight
        self.single_image_weight_overrides = single_image_weight_overrides or {}
        self.single_image_score_overrides = single_image_score_overrides or {}
        self.single_image_mix_max_weight = single_image_mix_max_weight
        self.single_image_mix_topk_weight = single_image_mix_topk_weight
        self.single_pixel_clip_weight = single_pixel_clip_weight
        self.single_pixel_dinov2_weight = single_pixel_dinov2_weight
        self.single_pixel_vl_weight = single_pixel_vl_weight
        self.single_pixel_global_weight = single_pixel_global_weight
        self.single_pixel_part_weight = single_pixel_part_weight
        self.single_pixel_weight_overrides = single_pixel_weight_overrides or {}
        self.multi_pixel_gecm_weight = multi_pixel_gecm_weight
        self.multi_pixel_weight_overrides = multi_pixel_weight_overrides or {}
        self.multi_image_gecm_weight = multi_image_gecm_weight
        self.multi_image_weight_overrides = multi_image_weight_overrides or {}
        self.multi_gecm_clip_weight = multi_gecm_clip_weight
        self.multi_gecm_dino_weight = multi_gecm_dino_weight
        self.multi_gecm_geo_weight = multi_gecm_geo_weight
        self.multi_gecm_feature_weight_overrides = multi_gecm_feature_weight_overrides or {}
        self.multi_component_min_ratio = multi_component_min_ratio
        self.multi_component_min_ratio_overrides = multi_component_min_ratio_overrides or {}
        self.single_image_pooling = single_image_pooling
        self.single_image_topk_ratio = single_image_topk_ratio
        self.official_scoring = official_scoring
        self.strict_mode = strict_mode
        self.require_mask_dir = require_mask_dir
        self.require_heat_mask_dir = require_heat_mask_dir
        self.require_query_masks = require_query_masks
        self.heat_mask_dir = heat_mask_dir
        self.ref_mask_dilation_kernel = ref_mask_dilation_kernel
        self.query_mask_dilation_kernel = query_mask_dilation_kernel
        self.component_mask_dilation_kernel = component_mask_dilation_kernel
        self.component_rotations = component_rotations
        self.gate_overrides = gate_overrides or {}

        if self.require_mask_dir and not self.mask_dir:
            raise ValueError('UniVAD strict mode requires `mask_dir`.')
        if self.require_heat_mask_dir and not self.heat_mask_dir:
            raise ValueError('UniVAD strict mode requires `heat_mask_dir`.')

        # Build CLIP backbone via registry
        clip_backbone = MODELS.build(
            dict(type='OpenCLIPBackbone', model_name=clip_model,
                 pretrained=clip_pretrained, frozen=True, image_size=image_size,
                 prefer_local_reference=True,
                 force_quick_gelu=(clip_pretrained == 'openai')))
        self.clip = clip_backbone.model
        self._tokenize = clip_backbone.tokenize
        self.clip.visual.output_tokens = True

        # Build DINOv2 backbone via registry
        dinov2_backbone = MODELS.build(
            dict(type='DINOv2Backbone', model_name=dinov2_model,
                 frozen=True))
        self.dinov2 = dinov2_backbone

        # CLIP visual dimensions
        self._clip_patch_size = self.clip.visual.conv1.kernel_size[0] \
            if hasattr(self.clip.visual, 'conv1') else 14
        self._clip_grid_h = image_size // self._clip_patch_size
        self._clip_grid_w = image_size // self._clip_patch_size
        if hasattr(self.clip.visual, 'proj') and self.clip.visual.proj is not None:
            self._clip_proj_dim = self.clip.visual.proj.shape[1]
        else:
            self._clip_proj_dim = self.clip.visual.ln_post.normalized_shape[0] \
                if hasattr(self.clip.visual, 'ln_post') else 768

        # Adapt CLIP positional embedding for target image_size
        self._adapt_clip_pos_embed(image_size)

        # DINOv2 dimensions
        self._dinov2_patch_size = 14
        self._dinov2_resize = (image_size // self._dinov2_patch_size) * self._dinov2_patch_size
        self._dinov2_embed_dim = self.dinov2.encoder.embed_dim

        # ImageNet <-> CLIP normalization buffers
        self.register_buffer(
            '_imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False)
        self.register_buffer(
            '_imagenet_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False)
        self.register_buffer(
            '_clip_mean', torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
            persistent=False)
        self.register_buffer(
            '_clip_std', torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
            persistent=False)

        # GECM graph aggregation
        self.cfa = CFA(n_layers=gecm_layers) if gecm_enable else None

        # Hook storage for intermediate CLIP features
        self._clip_hook_features = {}
        self._clip_hooks = []

        # Reference data (populated during fit/build_memory_bank)
        self._ref_images = defaultdict(list)  # cls_name -> list of (img_tensor,)
        self._ref_data_built = False

        # Per-category memory bank after fit()
        self._gate = {}              # cls_name -> ObjectType
        self._ref_global_feats = {}  # cls_name -> (K, D)
        self._ref_clip_patches = {}  # cls_name -> (K, layers, N, D)
        self._ref_dinov2_patches = {}  # cls_name -> (K, N, D)
        self._ref_vl_text = {}       # cls_name -> (normal_emb, anomalous_emb)
        self._ref_masks = {}         # cls_name -> list of (H_mask, W_mask) int arrays
        self._ref_heat_masks = {}    # cls_name -> list of (H_mask, W_mask) int arrays
        self._ref_clip_part_patches = {}  # cls_name -> layer_idx -> part_idx -> tensor
        self._ref_dino_part_patches = {}  # cls_name -> part_idx -> tensor
        self._ref_component_bank = {}  # cls_name -> dict of component features
        self._official_component_extractor = None

    def _adapt_clip_pos_embed(self, input_size):
        """Interpolate CLIP positional embedding to match input resolution."""
        visual = self.clip.visual
        if not hasattr(visual, 'positional_embedding'):
            return
        pe = visual.positional_embedding
        patch_size = self._clip_patch_size
        old_grid = visual.grid_size if hasattr(visual, 'grid_size') else None
        if old_grid is None:
            return
        new_grid_h = input_size // patch_size
        new_grid_w = input_size // patch_size
        if old_grid[0] == new_grid_h and old_grid[1] == new_grid_w:
            return

        cls_pe = pe[:1]
        patch_pe = pe[1:]
        dim = pe.shape[1]
        patch_pe = patch_pe.reshape(1, old_grid[0], old_grid[1], dim).permute(0, 3, 1, 2)
        patch_pe = F.interpolate(
            patch_pe, size=(new_grid_h, new_grid_w),
            mode='bicubic', align_corners=False)
        patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(new_grid_h * new_grid_w, dim)
        visual.positional_embedding = nn.Parameter(torch.cat([cls_pe, patch_pe], dim=0))
        visual.grid_size = (new_grid_h, new_grid_w)

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _normalize_for_clip(self, x):
        """Re-normalize from ImageNet to CLIP mean/std."""
        x = x * self._imagenet_std + self._imagenet_mean
        return (x - self._clip_mean) / self._clip_std

    # ------------------------------------------------------------------
    # Feature extraction: CLIP multi-layer hooks
    # ------------------------------------------------------------------

    def _register_clip_hooks(self):
        """Register forward hooks on CLIP transformer layers."""
        self._remove_clip_hooks()
        self._clip_hook_features = {}
        blocks = self.clip.visual.transformer.resblocks

        for layer_idx in self.clip_layers:
            # clip_layers are 1-indexed; resblocks are 0-indexed
            block_idx = layer_idx - 1
            if block_idx >= len(blocks):
                continue

            def hook_fn(name):
                def _hook(_module, _input, output):
                    # output shape: (seq_len, batch, dim) for OpenCLIP ViT
                    self._clip_hook_features[name] = output.detach()
                return _hook

            handle = blocks[block_idx].register_forward_hook(hook_fn(f'layer_{layer_idx}'))
            self._clip_hooks.append(handle)

    def _remove_clip_hooks(self):
        for h in self._clip_hooks:
            h.remove()
        self._clip_hooks = []
        self._clip_hook_features = {}

    @torch.no_grad()
    def _extract_clip_multilayer(self, x):
        """Extract CLIP multi-layer patch features via hooks.

        Args:
            x: (B, 3, H, W) CLIP-normalized images.

        Returns:
            global_feats: (B, D_proj) global CLS features.
            layer_patch_feats: list of (B, N_patches, D_hidden) per layer.
        """
        # Official UniVAD's local CLIP fork exposes:
        #   encode_image(image, out_layers) -> (image_features, patch_tokens)
        # Prefer that path when available instead of relying on transformer hooks.
        encoded = None
        if _supports_out_layers_api(self.clip.encode_image):
            encoded = self.clip.encode_image(x, list(self.clip_layers))

        if encoded is not None:
            if isinstance(encoded, (tuple, list)) and len(encoded) == 2:
                global_feats, layer_patch_feats = encoded
                if global_feats.ndim == 3:
                    global_feats = global_feats[:, 0, :]
                processed_layers = []
                for feat in layer_patch_feats:
                    if feat.ndim == 3 and feat.shape[1] > 1:
                        feat = feat[:, 1:, :]
                    processed_layers.append(feat)
                return global_feats, processed_layers

        self._register_clip_hooks()
        try:
            encoded = self.clip.encode_image(x)
            if isinstance(encoded, (tuple, list)):
                global_feats = encoded[0]
            else:
                global_feats = encoded
        finally:
            hook_feats = dict(self._clip_hook_features)
            self._remove_clip_hooks()

        layer_patch_feats = []
        for layer_idx in self.clip_layers:
            key = f'layer_{layer_idx}'
            if key in hook_feats:
                feat = hook_feats[key]
                # OpenCLIP ViT: (seq_len, batch, dim) -> (batch, seq_len, dim)
                if feat.ndim == 3 and feat.shape[0] != x.shape[0]:
                    feat = feat.permute(1, 0, 2)
                # Strip CLS token -> patch tokens
                patch_feat = feat[:, 1:, :]  # (B, N_patches, D_hidden)
                layer_patch_feats.append(patch_feat)

        return global_feats, layer_patch_feats

    @torch.no_grad()
    def _extract_clip_component_layers(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract CLIP component layers for GECM feature averaging.

        Official UniVAD's component extractor averages the raw local-CLIP
        `patch_tokens[layers]` outputs directly, without the CAPM decoder-style
        CLS-token removal. Preserve that behavior on the strict official path.
        """
        if _supports_out_layers_api(self.clip.encode_image):
            encoded = self.clip.encode_image(x, list(self.clip_layers))
            if isinstance(encoded, (tuple, list)) and len(encoded) == 2:
                _global_feats, layer_patch_feats = encoded
                return list(layer_patch_feats)

        _global_feats, layer_patch_feats = self._extract_clip_multilayer(x)
        return layer_patch_feats

    # ------------------------------------------------------------------
    # Feature extraction: DINOv2
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _extract_dinov2_features(self, x):
        """Extract DINOv2 patch tokens.

        Args:
            x: (B, 3, H, W) ImageNet-normalized images.

        Returns:
            patch_tokens: (B, N_patches, D) normalized patch tokens.
        """
        if x.shape[-2] != self._dinov2_resize or x.shape[-1] != self._dinov2_resize:
            x = F.interpolate(x, size=(self._dinov2_resize, self._dinov2_resize),
                              mode='bilinear', align_corners=False)
        out = self.dinov2.encoder.forward_features(x)
        patch_tokens = out['x_norm_patchtokens']  # (B, N, D)
        return patch_tokens

    # ------------------------------------------------------------------
    # VL anomaly scoring
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_text_prompts(self, class_name='object'):
        """Encode normal/anomalous text prompts for a class.

        Returns:
            (normal_emb, anomalous_emb): each (D_proj,) normalized.
        """
        device = next(self.clip.parameters()).device
        normal_prompts, anomalous_prompts = _create_prompt_ensemble(class_name)

        normal_tokens = self._tokenize(normal_prompts).to(device)
        anomalous_tokens = self._tokenize(anomalous_prompts).to(device)

        normal_emb = F.normalize(self.clip.encode_text(normal_tokens), dim=-1).mean(dim=0)
        anomalous_emb = F.normalize(self.clip.encode_text(anomalous_tokens), dim=-1).mean(dim=0)

        return F.normalize(normal_emb, dim=-1), F.normalize(anomalous_emb, dim=-1)

    @torch.no_grad()
    def _vl_anomaly_map(self, clip_layer6_patches, normal_emb, anomalous_emb):
        """Compute VL anomaly map from CLIP layer-6 patch tokens.

        Projects patch tokens through visual.proj, then computes softmax
        similarity with normal/abnormal text embeddings.

        Args:
            clip_layer6_patches: (B, N, D_hidden) from layer 6.
            normal_emb: (D_proj,) normalized text embedding.
            anomalous_emb: (D_proj,) normalized text embedding.

        Returns:
            anomaly_map: (B, 1, H_grid, W_grid) anomaly scores.
        """
        B, N, D = clip_layer6_patches.shape
        patches = clip_layer6_patches
        if hasattr(self.clip.visual, 'proj') and self.clip.visual.proj is not None:
            patches = patches @ self.clip.visual.proj  # (B, N, D_proj)

        patches = F.normalize(patches, dim=-1)

        # Stack text embeddings: (2, D_proj)
        text_embs = torch.stack([normal_emb, anomalous_emb], dim=0)
        text_embs = F.normalize(text_embs, dim=-1)

        # Similarity: (B, N, 2)
        sim = 100.0 * torch.einsum('bnd,kd->bnk', patches, text_embs)
        # Official UniVAD formula: (P(abnormal) - P(normal) + 1) / 2
        # This yields values in [0, 1] where higher = more anomalous
        probs = sim.softmax(dim=-1)
        anomaly_scores = (probs[:, :, 1] - probs[:, :, 0] + 1) / 2  # (B, N)

        h, w = self._clip_grid_h, self._clip_grid_w
        if N == h * w:
            return anomaly_scores.reshape(B, 1, h, w)
        # Fallback: assume square
        side = int(math.sqrt(N))
        return anomaly_scores.reshape(B, 1, side, side)

    # ------------------------------------------------------------------
    # C3 mask loading
    # ------------------------------------------------------------------

    def _load_single_mask(self, cls_name, img_path):
        """Load one component mask in either `.npy` or grayscale PNG format."""
        for mask_path in mask_path_candidates(self.mask_dir, cls_name, img_path):
            if not os.path.exists(mask_path):
                continue
            if mask_path.endswith('.npy'):
                mask = np.load(mask_path).astype(np.int32)
            else:
                with Image.open(mask_path) as _img:
                    mask = np.array(_img.convert('L')).astype(np.int32)
            if mask.shape != (self.image_size, self.image_size):
                mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            return mask.astype(np.int32)
        return None

    def _load_heat_mask(self, cls_name, img_path):
        """Load one refined heat-mask in either `.png` or `.npy` format."""
        if not self.heat_mask_dir:
            return None
        for mask_path in heat_mask_path_candidates(self.heat_mask_dir, cls_name, img_path):
            if not os.path.exists(mask_path):
                continue
            if mask_path.endswith('.npy'):
                mask = np.load(mask_path).astype(np.int32)
            else:
                with Image.open(mask_path) as _img:
                    mask = np.array(_img.convert('L')).astype(np.int32)
            if mask.shape != (self.image_size, self.image_size):
                mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            return mask.astype(np.int32)
        return None

    def _load_masks_for_category(self, cls_name, img_paths):
        """Load precomputed C3 component masks for reference images.

        Returns:
            masks: list of np.ndarray (H, W) int arrays, or empty list.
        """
        if not self.mask_dir:
            return []

        masks = []
        for img_path in img_paths:
            mask = self._load_single_mask(cls_name, img_path)
            if mask is not None:
                masks.append(mask)
            else:
                logger.debug(f'No C3 mask found for {img_path}')
        return masks

    def _load_heat_masks_for_category(self, cls_name, img_paths):
        """Load precomputed MULTI heat-masks for a category."""
        if not self.heat_mask_dir:
            return []
        masks = []
        for img_path in img_paths:
            mask = self._load_heat_mask(cls_name, img_path)
            if mask is not None:
                masks.append(mask)
            else:
                logger.debug(f'No heat mask found for {img_path}')
        return masks

    def _require_asset(self, asset, *, img_path: str, description: str):
        """Enforce strict UniVAD asset requirements."""
        if asset is not None:
            return asset
        if self.strict_mode:
            raise FileNotFoundError(f'Missing UniVAD {description} for {img_path}')
        return None

    def _binary_masks(self, mask: np.ndarray, *, include_background: bool = False) -> Tuple[List[np.ndarray], List[int]]:
        """Split a labeled mask into binary masks."""
        if include_background:
            return split_masks_from_one_mask_with_bg(mask)
        return split_masks_from_one_mask(mask)

    def _dilate_mask(self, mask: np.ndarray, kernel_size: int) -> np.ndarray:
        """Dilate a binary mask."""
        if kernel_size <= 1:
            return mask
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.dilate(np.asarray(mask, dtype=np.uint8), kernel, iterations=1)

    # ------------------------------------------------------------------
    # Gate logic
    # ------------------------------------------------------------------

    def _determine_gate(self, cls_name, masks):
        """Determine object type for a category based on C3 masks.

        Gate logic follows the official implementation:
        - If gate_overrides specified for this category: use the override
        - If no masks: fall back to TEXTURE
        - Use the first reference mask for gate selection
        - Compute the ratio of the largest component area to image area
        - If largest_ratio > threshold and num_segments <= max_segment_for_texture:
          TEXTURE
        - If num_segments == 1: SINGLE
        - Otherwise: MULTI

        Returns:
            ObjectType enum value.
        """
        # Check gate overrides first (for strict alignment)
        if cls_name in self.gate_overrides:
            override = self.gate_overrides[cls_name].lower()
            if override == 'texture':
                gate = ObjectType.TEXTURE
            elif override == 'single':
                gate = ObjectType.SINGLE
            elif override == 'multi':
                gate = ObjectType.MULTI
            else:
                raise ValueError(f'Unknown gate override: {override}')
            logger.info(f'[{cls_name}] Gate OVERRIDE: {gate.value}')
            return gate

        if not masks:
            logger.info(f'[{cls_name}] No masks found, falling back to TEXTURE mode')
            return ObjectType.TEXTURE

        mask = masks[0]
        labels = sorted(int(label) for label in np.unique(mask) if int(label) != 0)
        if not labels:
            logger.info(f'[{cls_name}] Empty foreground mask, falling back to TEXTURE mode')
            return ObjectType.TEXTURE

        areas = [int(np.count_nonzero(mask == label)) for label in labels]
        largest_ratio = max(areas) / float(mask.size) if mask.size > 0 else 1.0
        num_segments = len(labels)

        if largest_ratio > self.object_ratio_threshold and num_segments <= self.max_segment_for_texture:
            gate = ObjectType.TEXTURE
        elif num_segments == 1:
            gate = ObjectType.SINGLE
        else:
            gate = ObjectType.MULTI

        logger.info(f'[{cls_name}] Gate: {gate.value} '
                     f'(largest_ratio={largest_ratio:.3f}, segments={num_segments})')
        return gate

    # ------------------------------------------------------------------
    # Component feature extraction for GECM
    # ------------------------------------------------------------------

    def _mask_to_patch_selector(
        self,
        mask: np.ndarray,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Project a binary mask to a patch grid and return a boolean selector."""
        mask_tensor = torch.from_numpy(np.asarray(mask)).float().unsqueeze(0).unsqueeze(0).to(device)
        selector = F.interpolate(
            mask_tensor,
            size=(height, width),
            mode='bilinear',
            align_corners=True,
        ).reshape(height * width)
        selector = selector > 0
        if not selector.any():
            selector = torch.ones(height * width, dtype=torch.bool, device=device)
        return selector

    def _crop_by_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Zero-out everything outside the component mask."""
        if image.shape[-1] == 3:
            mask_3c = cv2.merge([mask, mask, mask])
        else:
            mask_3c = mask
        return np.where(mask_3c != 0, image, 0)

    def _align_component_crops(
        self,
        image: np.ndarray,
        masks: Sequence[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[List[float]], List[float]]:
        """Align components to a square canvas, matching the reference extractor."""
        aligned_masks: List[np.ndarray] = []
        aligned_images: List[np.ndarray] = []
        center_positions: List[List[float]] = []
        scales: List[float] = []
        target_size = image.shape[0]

        for mask in masks:
            mask = np.asarray(mask, dtype=np.uint8)
            cropped_image = self._crop_by_mask(image, mask)
            contours, _ = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            mask = cv2.drawContours(np.zeros_like(mask), [contour], -1, (255, 255, 255), -1)

            x, y, width, height = cv2.boundingRect(contour)
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            diagonal = np.linalg.norm(box[0] - box[2], ord=2)
            scale = target_size / (diagonal + 1.0)

            crop_mask = mask[y:y + height, x:x + width]
            crop_image = cropped_image[y:y + height, x:x + width]
            resized_size = (np.array(crop_mask.shape) * scale).astype(np.int32)
            resized_size = np.where(resized_size > target_size, target_size, resized_size)
            resized_size = np.where(resized_size <= 0, 1, resized_size)

            crop_mask = cv2.resize(crop_mask, (resized_size[1], resized_size[0]), cv2.INTER_LINEAR)
            crop_image = cv2.resize(crop_image, (resized_size[1], resized_size[0]), cv2.INTER_LINEAR)
            crop_mask[crop_mask > 128] = 255
            crop_mask[crop_mask <= 128] = 0

            pad_width = int((target_size - crop_mask.shape[1]) // 2)
            pad_height = int((target_size - crop_mask.shape[0]) // 2)
            crop_mask = cv2.copyMakeBorder(
                crop_mask, pad_height, pad_height, pad_width, pad_width, cv2.BORDER_CONSTANT, value=0)
            crop_image = cv2.copyMakeBorder(
                crop_image, pad_height, pad_height, pad_width, pad_width, cv2.BORDER_CONSTANT, value=0)

            if crop_mask.shape[0] != target_size:
                crop_mask = cv2.copyMakeBorder(crop_mask, 1, 0, 0, 0, cv2.BORDER_CONSTANT, value=0)
                crop_image = cv2.copyMakeBorder(crop_image, 1, 0, 0, 0, cv2.BORDER_CONSTANT, value=0)
            if crop_mask.shape[1] != target_size:
                crop_mask = cv2.copyMakeBorder(crop_mask, 0, 0, 1, 0, cv2.BORDER_CONSTANT, value=0)
                crop_image = cv2.copyMakeBorder(crop_image, 0, 0, 1, 0, cv2.BORDER_CONSTANT, value=0)

            center_positions.append([(x + width / 2) / mask.shape[1], (y + height / 2) / mask.shape[0]])
            scales.append(scale)
            aligned_masks.append(crop_mask)
            aligned_images.append(crop_image)

        return aligned_masks, aligned_images, center_positions, scales

    def _rotate_component_images(self, images: Sequence[np.ndarray]) -> torch.Tensor:
        """Rotate aligned component crops and stack them into a tensor."""
        rotated = []
        angles = np.linspace(0, 360, self.component_rotations)
        for image in images:
            pil_image = Image.fromarray(image)
            rotated.extend(
                torch.from_numpy(np.array(pil_image.rotate(angle, Image.Resampling.BILINEAR))).permute(2, 0, 1).float() / 255.0
                for angle in angles
            )
        return torch.stack(rotated, dim=0)

    @torch.no_grad()
    def _extract_component_feature_bank(
        self,
        image: np.ndarray,
        masks: Sequence[np.ndarray],
        device: torch.device,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Extract component-level geo / CLIP / DINO features for GECM."""
        if not masks:
            return None
        if self.official_scoring:
            extractor = self._get_official_component_extractor()
            features = extractor.extract(image, list(masks))
            return {
                'area': features['area'].to(device),
                'color': features['color'].to(device),
                'position': features['position'].to(device),
                'clip_image': features['clip_image'].to(device),
                'dino_image': features['dino_image'].to(device),
            }

        aligned_masks, aligned_images, center_positions, _scales = self._align_component_crops(image, masks)
        if not aligned_images:
            return None

        rotated_images = self._rotate_component_images(aligned_images).to(device)
        clip_inputs = (rotated_images - self._clip_mean.to(device)) / self._clip_std.to(device)
        dino_inputs = (rotated_images - self._imagenet_mean.to(device)) / self._imagenet_std.to(device)

        clip_layers = self._extract_clip_component_layers(clip_inputs)
        dino_layers = self._extract_dinov2_features(dino_inputs)

        num_components = len(aligned_images)
        num_rotations = self.component_rotations
        clip_layer_indices = _matching_clip_layer_indices(len(clip_layers), len(self.clip_layers))
        clip_features = []
        for layer_idx in clip_layer_indices:
            layer_feat = clip_layers[layer_idx].reshape(num_components, num_rotations, -1, clip_layers[layer_idx].shape[-1])
            clip_features.append(layer_feat.mean(dim=2).mean(dim=1))
        clip_features = torch.stack(clip_features, dim=1) if clip_features else torch.empty(0, device=device)

        dino_features = dino_layers.reshape(num_components, num_rotations, -1, dino_layers.shape[-1])
        dino_features = dino_features.mean(dim=2).mean(dim=1)

        areas = []
        colors = []
        positions = []
        image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        for mask, center in zip(masks, center_positions):
            mask = np.asarray(mask, dtype=np.uint8)
            area = np.sum(mask != 0)
            areas.append(torch.tensor([area / float(mask.size)], device=device))

            color_sum_a = image_lab[:, :, 1].astype(np.float32)
            color_sum_b = image_lab[:, :, 2].astype(np.float32)
            color_div = (color_sum_b / (color_sum_a + 1e-7)) ** 2
            color_div = color_div * np.where(mask != 0, 1, 0)
            color_value = np.sum(color_div) / (area + 1e-7)
            colors.append(torch.tensor([color_value], device=device))

            positions.append(torch.tensor(center, device=device, dtype=torch.float32))

        area_tensor = torch.stack(areas, dim=0)
        color_tensor = torch.stack(colors, dim=0)
        position_tensor = torch.stack(positions, dim=0)

        return {
            'area': area_tensor,
            'color': color_tensor,
            'position': position_tensor,
            'clip_image': clip_features,
            'dino_image': dino_features,
        }

    def _get_official_component_extractor(self):
        """Instantiate the official UniVAD component extractor lazily."""
        if self._official_component_extractor is None:
            extractor_cls = _reference_component_extractor_cls()
            config = {
                'transform_clip': transforms.Compose([
                    transforms.Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711),
                    ),
                ]),
                'transform_dino': transforms.Compose([
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                ]),
            }
            self._official_component_extractor = extractor_cls(
                config,
                clip_model=self.clip,
                out_layers=list(self.clip_layers),
                dino_model=self.dinov2.encoder,
            )
        return self._official_component_extractor

    def _aggregate_reference_component_bank(
        self,
        component_features: Sequence[Dict[str, torch.Tensor]],
        device: torch.device,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Concatenate and graph-aggregate reference component features."""
        if not component_features:
            return None

        area = torch.cat([feat['area'] for feat in component_features], dim=0).to(device)
        color = torch.cat([feat['color'] for feat in component_features], dim=0).to(device)
        position = torch.cat([feat['position'] for feat in component_features], dim=0).to(device)
        dino_image = torch.cat([feat['dino_image'] for feat in component_features], dim=0).to(device)
        clip_image = torch.cat([feat['clip_image'] for feat in component_features], dim=0).to(device).transpose(0, 1)

        if self.cfa is not None and clip_image.numel() > 0:
            for layer_idx in range(clip_image.shape[0]):
                clip_image[layer_idx] = self.cfa(clip_image[layer_idx])
            dino_image = self.cfa(dino_image)

        geo = torch.cat([area, color, position], dim=1)
        return {
            'area': area,
            'color': color,
            'position': position,
            'clip_image': clip_image,
            'dino_image': dino_image,
            'geo': geo,
        }

    def _collect_part_feature_banks(
        self,
        clip_patches: torch.Tensor,
        dino_patches: torch.Tensor,
        masks: Sequence[np.ndarray],
        part_indices: Sequence[int],
        clip_banks: Sequence[Dict[int, List[torch.Tensor]]],
        dino_bank: Dict[int, List[torch.Tensor]],
        device: torch.device,
    ) -> None:
        """Collect patch-level reference banks for SINGLE/MULTI CAPM matching."""
        clip_layer_indices = _matching_clip_layer_indices(clip_patches.shape[0], len(self.clip_layers))
        h_clip, w_clip = self._clip_grid_h, self._clip_grid_w
        h_dino = self._dinov2_resize // self._dinov2_patch_size
        w_dino = h_dino

        for mask, part_idx in zip(masks, part_indices):
            selector_clip = self._mask_to_patch_selector(mask, h_clip, w_clip, device)
            selector_dino = self._mask_to_patch_selector(mask, h_dino, w_dino, device)
            dino_bank[part_idx].append(dino_patches[selector_dino])
            for bank_idx, layer_idx in enumerate(clip_layer_indices):
                clip_banks[bank_idx][part_idx].append(clip_patches[layer_idx][selector_clip])

    def _finalize_part_feature_banks(
        self,
        clip_banks: Sequence[Dict[int, List[torch.Tensor]]],
        dino_bank: Dict[int, List[torch.Tensor]],
        device: torch.device,
    ) -> Tuple[List[Dict[int, torch.Tensor]], Dict[int, torch.Tensor]]:
        """Concatenate part-level reference patch banks."""
        finalized_clip = []
        for layer_bank in clip_banks:
            finalized_clip.append({
                part_idx: torch.cat(tensors, dim=0).to(device)
                for part_idx, tensors in layer_bank.items() if tensors
            })
        finalized_dino = {
            part_idx: torch.cat(tensors, dim=0).to(device)
            for part_idx, tensors in dino_bank.items() if tensors
        }
        return finalized_clip, finalized_dino

    def _load_query_masks(
        self,
        cls_name: str,
        img_path: str,
        gate: ObjectType,
    ) -> Tuple[List[np.ndarray], List[int], List[np.ndarray]]:
        """Load query masks for SINGLE or MULTI runtime paths."""
        if gate == ObjectType.SINGLE:
            raw_mask = self._load_single_mask(cls_name, img_path)
            if raw_mask is None:
                self._require_asset(
                    None,
                    img_path=img_path,
                    description='query grounding mask',
                )
                full_mask = np.ones((self.image_size, self.image_size), dtype=np.uint8) * 255
                return [full_mask], [0], [full_mask]

            masks, _ = split_masks_from_one_mask(raw_mask)
            if not masks:
                masks = [np.ones_like(raw_mask, dtype=np.uint8) * 255]
            masks = [self._dilate_mask(mask, self.query_mask_dilation_kernel) for mask in masks]
            return masks, [0] * len(masks), masks

        if gate == ObjectType.MULTI:
            heat_mask = self._load_heat_mask(cls_name, img_path)
            if heat_mask is None:
                self._require_asset(
                    None,
                    img_path=img_path,
                    description='query refined heat mask',
                )
                return [], [], []

            raw_mask = self._load_single_mask(cls_name, img_path)
            capm_masks, capm_indices = split_masks_from_one_mask_with_bg(heat_mask)
            capm_masks = [self._dilate_mask(mask, self.query_mask_dilation_kernel) for mask in capm_masks]
            component_masks, _ = self._resolve_multi_component_masks(
                cls_name, raw_mask, heat_mask)
            return capm_masks, capm_indices, component_masks

        return [], [], []

    def _resolve_multi_component_masks(
        self,
        cls_name: str,
        raw_mask: Optional[np.ndarray],
        heat_mask: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Prefer refined MULTI components, but fall back to raw masks if they collapse."""
        min_ratio = self._resolve_multi_component_min_ratio(cls_name)
        component_masks, component_indices = split_masks_from_one_mask(
            heat_mask, min_ratio=min_ratio)
        if self.official_scoring:
            component_masks = [
                self._dilate_mask(mask, self.component_mask_dilation_kernel)
                for mask in component_masks
            ]
            return component_masks, component_indices
        if raw_mask is None:
            return component_masks, component_indices

        raw_component_masks, raw_component_indices = split_masks_from_one_mask(
            raw_mask, min_ratio=min_ratio)
        if len(component_masks) <= 1 < len(raw_component_masks):
            return raw_component_masks, raw_component_indices
        return component_masks, component_indices

    def _selected_clip_part_distance(
        self,
        q_clip_layers: Sequence[torch.Tensor],
        selector: torch.Tensor,
        ref_clip_parts: Sequence[Dict[int, torch.Tensor]],
        part_idx: int,
    ) -> Optional[torch.Tensor]:
        """Compute CLIP part distance for a selected patch subset."""
        distances = []
        layer_indices = _matching_clip_layer_indices(len(q_clip_layers), len(self.clip_layers))
        for bank_idx, layer_idx in enumerate(layer_indices):
            if bank_idx >= len(ref_clip_parts):
                break
            ref_tokens = ref_clip_parts[bank_idx].get(part_idx)
            if ref_tokens is None:
                continue
            q_tokens = q_clip_layers[layer_idx][0][selector]
            if q_tokens.numel() == 0:
                continue
            q_tokens = F.normalize(q_tokens, dim=-1)
            ref_tokens = F.normalize(ref_tokens, dim=-1)
            sim = q_tokens @ ref_tokens.T
            distances.append(1.0 - sim.max(dim=1).values)
        if not distances:
            return None
        return torch.stack(distances, dim=0).mean(dim=0)

    def _selected_dino_part_distance(
        self,
        q_dino: torch.Tensor,
        selector: torch.Tensor,
        ref_dino_parts: Dict[int, torch.Tensor],
        part_idx: int,
    ) -> Optional[torch.Tensor]:
        """Compute DINOv2 part distance for a selected patch subset."""
        ref_tokens = ref_dino_parts.get(part_idx)
        if ref_tokens is None:
            return None
        q_tokens = q_dino[0][selector]
        if q_tokens.numel() == 0:
            return None
        ref_tokens = F.normalize(ref_tokens, dim=-1)
        sim = q_tokens @ ref_tokens.T
        return 1.0 - sim.max(dim=1).values

    def _image_score_from_map(
        self,
        score_map: torch.Tensor,
        pooling: str,
        topk_ratio: float,
        *,
        max_weight: float = 1.0,
        topk_weight: float = 1.0,
    ) -> float:
        """Aggregate an anomaly map into an image-level score."""
        flat = score_map.reshape(-1)
        if flat.numel() == 0:
            return 0.0
        if pooling == 'max':
            return float(flat.max())
        if pooling == 'topk_mean':
            k = max(1, int(flat.numel() * topk_ratio))
            return float(torch.topk(flat, k).values.mean())
        if pooling == 'max_topk_mean':
            k = max(1, int(flat.numel() * topk_ratio))
            topk_mean = float(torch.topk(flat, k).values.mean())
            denom = max_weight + topk_weight
            if denom <= 0:
                denom = 1.0
            return (max_weight * float(flat.max()) + topk_weight * topk_mean) / denom
        raise ValueError(f'Unsupported UniVAD image pooling: {pooling}')

    def _resolve_single_branch_weights(
        self,
        cls_name: str,
        *,
        clip_weight: float,
        dinov2_weight: float,
        vl_weight: float,
        global_weight: float = 1.0,
        part_weight: float = 1.0,
        overrides: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, float]:
        """Resolve class-specific SINGLE-branch fusion weights."""
        resolved = {
            'clip_weight': clip_weight,
            'dinov2_weight': dinov2_weight,
            'vl_weight': vl_weight,
            'global_weight': global_weight,
            'part_weight': part_weight,
        }
        override = {} if overrides is None else overrides.get(cls_name, {})
        for key in ('clip_weight', 'dinov2_weight', 'vl_weight',
                    'global_weight', 'part_weight'):
            if key in override:
                resolved[key] = override[key]
        return resolved

    def _resolve_single_image_score_options(self, cls_name: str) -> Dict[str, float | str]:
        """Resolve class-specific SINGLE image score reduction options."""
        resolved: Dict[str, float | str] = {
            'pooling': self.single_image_pooling,
            'topk_ratio': self.single_image_topk_ratio,
            'max_weight': self.single_image_mix_max_weight,
            'topk_weight': self.single_image_mix_topk_weight,
        }
        override = self.single_image_score_overrides.get(cls_name, {})
        for key in ('pooling', 'topk_ratio', 'max_weight', 'topk_weight'):
            if key in override:
                resolved[key] = override[key]
        return resolved

    def _resolve_multi_pixel_gecm_weight(self, cls_name: str) -> float:
        """Resolve class-specific MULTI pixel GECM fusion weight."""
        override = self.multi_pixel_weight_overrides.get(cls_name, {})
        if 'gecm_weight' in override:
            return float(override['gecm_weight'])
        return float(self.multi_pixel_gecm_weight)

    def _resolve_multi_image_gecm_weight(self, cls_name: str) -> float:
        """Resolve class-specific MULTI image GECM fusion weight."""
        override = self.multi_image_weight_overrides.get(cls_name, {})
        if 'gecm_weight' in override:
            return float(override['gecm_weight'])
        return float(self.multi_image_gecm_weight)

    def _resolve_multi_component_min_ratio(self, cls_name: str) -> float:
        """Resolve class-specific MULTI component split threshold."""
        if cls_name in self.multi_component_min_ratio_overrides:
            return float(self.multi_component_min_ratio_overrides[cls_name])
        return float(self.multi_component_min_ratio)

    def _resolve_multi_gecm_feature_weights(self, cls_name: str) -> Dict[str, float]:
        """Resolve class-specific MULTI GECM feature weights."""
        resolved = {
            'clip_weight': float(self.multi_gecm_clip_weight),
            'dino_weight': float(self.multi_gecm_dino_weight),
            'geo_weight': float(self.multi_gecm_geo_weight),
        }
        override = self.multi_gecm_feature_weight_overrides.get(cls_name, {})
        for key in ('clip_weight', 'dino_weight', 'geo_weight'):
            if key in override:
                resolved[key] = float(override[key])
        return resolved

    def _combine_multi_gecm_distance(
        self,
        cls_name: str,
        *,
        dist_clip: float,
        dist_dino: float,
        dist_geo: float,
    ) -> Tuple[float, Dict[str, float]]:
        """Combine MULTI GECM component distances.

        Official UniVAD sums clip/dino/geo distances and applies the `/2`
        scaling only when adding the anomaly map back to the final score map.
        Keep that exact behavior on the strict official path, while preserving
        weighted experimental variants for targeted ablations.
        """
        if self.official_scoring:
            weights = {
                'clip_weight': 1.0,
                'dino_weight': 1.0,
                'geo_weight': 1.0,
            }
            return dist_clip + dist_dino + dist_geo, weights

        weights = self._resolve_multi_gecm_feature_weights(cls_name)
        dist_weight_sum = (
            weights['clip_weight']
            + weights['dino_weight']
            + weights['geo_weight']
        )
        if dist_weight_sum <= 0:
            dist_weight_sum = 1.0
        combined = (
            weights['clip_weight'] * dist_clip
            + weights['dino_weight'] * dist_dino
            + weights['geo_weight'] * dist_geo
        ) / dist_weight_sum
        return combined, weights

    def _score_map_align_corners(self) -> bool:
        """Match UniVAD's score-map interpolation behavior."""
        return bool(self.official_scoring)

    def _official_image_score_from_map(self, score_map: torch.Tensor, img_path: str) -> float:
        """Match UniVAD's official image-level reduction."""
        flat = score_map.reshape(-1)
        if flat.numel() == 0:
            return 0.0
        if 'HIS' in img_path:
            return float(flat.mean())
        return float(flat.max())

    def _extract_global_score(self, capm_map: torch.Tensor, capm_img: float, img_path: str) -> float:
        """Recover the additive global score from CAPM output."""
        if self.official_scoring:
            return capm_img - self._official_image_score_from_map(capm_map, img_path)
        return capm_img - float(capm_map.max())

    def _finalize_prediction_scores(
        self,
        capm_map: torch.Tensor,
        capm_img: float,
        gecm_map: torch.Tensor,
        img_path: str,
        *,
        image_gecm_weight: float,
        pixel_gecm_weight: float,
    ) -> Tuple[torch.Tensor, float, float, float, float]:
        """Combine CAPM and GECM into final pixel/image scores."""
        global_score = self._extract_global_score(capm_map, capm_img, img_path)
        if self.official_scoring:
            final_map = capm_map + gecm_map
            final_img = self._official_image_score_from_map(final_map, img_path) + global_score
            return final_map, final_img, 1.0, 1.0, global_score

        image_map = capm_map + image_gecm_weight * gecm_map
        final_map = capm_map + pixel_gecm_weight * gecm_map
        final_img = float(image_map.max()) + global_score
        return final_map, final_img, image_gecm_weight, pixel_gecm_weight, global_score

    # ------------------------------------------------------------------
    # Forward: loss mode (collect reference images)
    # ------------------------------------------------------------------

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        if mode == 'loss':
            return self._forward_loss(inputs, data_samples)
        elif mode == 'predict':
            return self._forward_predict(inputs, data_samples)
        else:
            # tensor mode: extract CLIP features
            x_clip = self._normalize_for_clip(inputs)
            if x_clip.shape[-2:] != (self.image_size, self.image_size):
                x_clip = F.interpolate(x_clip, size=(self.image_size, self.image_size),
                                       mode='bilinear', align_corners=False)
            global_feats, _ = self._extract_clip_multilayer(x_clip)
            return global_feats

    def _forward_loss(self, inputs, data_samples):
        """Collect up to k_shot reference images per category."""
        B = inputs.shape[0]
        for i in range(B):
            if data_samples is not None and i < len(data_samples):
                cls_name = getattr(data_samples[i], 'cls_name', 'object')
                img_path = getattr(data_samples[i], 'img_path', '')
            else:
                cls_name = 'object'
                img_path = ''

            if len(self._ref_images[cls_name]) < self.k_shot:
                self._ref_images[cls_name].append(
                    (inputs[i].detach().cpu(), img_path))

        return {'loss': torch.tensor(0.0, device=inputs.device, requires_grad=True)}

    # ------------------------------------------------------------------
    # fit / build_memory_bank (called by MemoryBankHook)
    # ------------------------------------------------------------------

    def fit(self):
        """Build memory bank from collected reference images.

        Called by MemoryBankHook after training epoch completes.
        """
        if self._ref_data_built:
            return
        if not self._ref_images:
            logger.warning('UniVAD fit() called but no reference images collected.')
            self._ref_data_built = True
            return

        device = next(self.clip.parameters()).device

        for cls_name, ref_list in self._ref_images.items():
            logger.info(f'[{cls_name}] Building memory bank from {len(ref_list)} refs')

            imgs = torch.stack([r[0] for r in ref_list]).to(device)
            img_paths = [r[1] for r in ref_list]

            # Extract CLIP features
            x_clip = self._normalize_for_clip(imgs)
            if x_clip.shape[-2:] != (self.image_size, self.image_size):
                x_clip = F.interpolate(x_clip, size=(self.image_size, self.image_size),
                                       mode='bilinear', align_corners=False)
            global_feats, layer_patches = self._extract_clip_multilayer(x_clip)
            self._ref_global_feats[cls_name] = F.normalize(global_feats, dim=-1).cpu()

            # Stack multi-layer patches: (K, n_layers, N, D)
            if layer_patches:
                stacked = torch.stack(layer_patches, dim=1)  # (K, n_layers, N, D)
                self._ref_clip_patches[cls_name] = stacked.cpu()
            else:
                self._ref_clip_patches[cls_name] = None

            # Extract DINOv2 features
            dinov2_patches = self._extract_dinov2_features(imgs)
            self._ref_dinov2_patches[cls_name] = F.normalize(dinov2_patches, dim=-1).cpu()

            # Build text embeddings
            normal_emb, anomalous_emb = self._encode_text_prompts(cls_name)
            self._ref_vl_text[cls_name] = (normal_emb.cpu(), anomalous_emb.cpu())

            # Load C3 masks and determine gate
            masks = self._load_masks_for_category(cls_name, img_paths)
            if self.require_mask_dir and len(masks) < len(img_paths):
                missing = [path for path in img_paths if self._load_single_mask(cls_name, path) is None]
                if missing:
                    raise FileNotFoundError(
                        f'Missing UniVAD reference masks for {cls_name}: {missing[0]}')
            self._ref_masks[cls_name] = masks
            gate = self._determine_gate(cls_name, masks)
            self._gate[cls_name] = gate

            num_clip_match_layers = len(_matching_clip_layer_indices(
                self._ref_clip_patches[cls_name].shape[1] if self._ref_clip_patches[cls_name] is not None else 0,
                len(self.clip_layers),
            ))
            clip_part_banks = [defaultdict(list) for _ in range(num_clip_match_layers)]
            dino_part_bank: Dict[int, List[torch.Tensor]] = defaultdict(list)
            self._ref_heat_masks[cls_name] = []

            if gate == ObjectType.SINGLE and masks:
                for k, mask in enumerate(masks[:len(ref_list)]):
                    comp_masks, _ = self._binary_masks(mask)
                    if not comp_masks:
                        comp_masks = [np.ones_like(mask, dtype=np.uint8) * 255]
                    ref_mask = self._dilate_mask(comp_masks[0], self.ref_mask_dilation_kernel)
                    self._collect_part_feature_banks(
                        self._ref_clip_patches[cls_name][k].to(device),
                        self._ref_dinov2_patches[cls_name][k].to(device),
                        [ref_mask],
                        [0],
                        clip_part_banks,
                        dino_part_bank,
                        device,
                    )

            component_features = []
            if gate == ObjectType.MULTI and masks:
                heat_masks = self._load_heat_masks_for_category(cls_name, img_paths)
                if self.require_heat_mask_dir and len(heat_masks) < len(img_paths):
                    missing = [path for path in img_paths if self._load_heat_mask(cls_name, path) is None]
                    if missing:
                        raise FileNotFoundError(
                            f'Missing UniVAD refined heat masks for {cls_name}: {missing[0]}')
                self._ref_heat_masks[cls_name] = heat_masks

                for k, img_path in enumerate(img_paths[:len(ref_list)]):
                    heat_mask = self._load_heat_mask(cls_name, img_path)
                    if heat_mask is None:
                        heat_mask = self._require_asset(
                            None,
                            img_path=img_path,
                            description='refined heat mask',
                        )
                    if heat_mask is None:
                        continue

                    capm_masks, capm_indices = split_masks_from_one_mask_with_bg(heat_mask)
                    capm_masks = [
                        self._dilate_mask(mask, self.ref_mask_dilation_kernel)
                        for mask in capm_masks
                    ]
                    self._collect_part_feature_banks(
                        self._ref_clip_patches[cls_name][k].to(device),
                        self._ref_dinov2_patches[cls_name][k].to(device),
                        capm_masks,
                        capm_indices,
                        clip_part_banks,
                        dino_part_bank,
                        device,
                    )

                    raw_mask = masks[k] if k < len(masks) else None
                    component_masks, _ = self._resolve_multi_component_masks(
                        cls_name, raw_mask, heat_mask)
                    with Image.open(img_path) as _img:
                        image = np.array(
                            _img.convert('RGB').resize((self.image_size, self.image_size))
                        )
                    comp_feature = self._extract_component_feature_bank(image, component_masks, device)
                    if comp_feature is not None:
                        component_features.append(comp_feature)

            clip_part_patches, dino_part_patches = self._finalize_part_feature_banks(
                clip_part_banks,
                dino_part_bank,
                device,
            )
            self._ref_clip_part_patches[cls_name] = clip_part_patches
            self._ref_dino_part_patches[cls_name] = dino_part_patches
            self._ref_component_bank[cls_name] = self._aggregate_reference_component_bank(
                component_features,
                device,
            )

        self._ref_data_built = True
        logger.info('UniVAD memory bank built successfully.')

    build_memory_bank = fit

    # ------------------------------------------------------------------
    # Forward: predict mode
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _forward_predict(self, inputs, data_samples):
        """Per-image prediction: CAPM + optional GECM scoring."""
        if not self._ref_data_built:
            self.fit()

        B = inputs.shape[0]
        device = inputs.device

        # Extract CLIP features for query batch
        x_clip = self._normalize_for_clip(inputs)
        if x_clip.shape[-2:] != (self.image_size, self.image_size):
            x_clip = F.interpolate(x_clip, size=(self.image_size, self.image_size),
                                   mode='bilinear', align_corners=False)
        query_global_feats, query_clip_layers = self._extract_clip_multilayer(x_clip)

        # Extract DINOv2 features for query batch
        query_dinov2 = self._extract_dinov2_features(inputs)
        query_dinov2 = F.normalize(query_dinov2, dim=-1)

        h_clip, w_clip = self._clip_grid_h, self._clip_grid_w
        h_dino = self._dinov2_resize // self._dinov2_patch_size
        w_dino = h_dino

        img_scores = []
        pixel_scores = []

        for i in range(B):
            cls_name = 'object'
            img_path = ''
            if data_samples is not None and i < len(data_samples):
                cls_name = getattr(data_samples[i], 'cls_name', 'object')
                img_path = getattr(data_samples[i], 'img_path', '')

            gate = self._gate.get(cls_name, ObjectType.TEXTURE)

            # Get per-image features
            q_global = query_global_feats[i:i+1]
            q_clip_layers = [layer[i:i+1] for layer in query_clip_layers]  # each (1, N, D)
            q_dinov2 = query_dinov2[i:i+1]  # (1, N_dino, D_dino)

            # CAPM scoring
            capm_map, capm_img = self._capm_score(
                cls_name, img_path, gate, q_global, q_clip_layers, q_dinov2,
                h_clip, w_clip, h_dino, w_dino, device)

            # GECM scoring for MULTI type
            gecm_map = torch.zeros_like(capm_map)
            if gate == ObjectType.MULTI and self.gecm_enable:
                _gecm_score, gecm_map = self._gecm_score(
                    cls_name, img_path, q_clip_layers, q_dinov2,
                    h_clip, w_clip, h_dino, w_dino, device)

            # Combine scores
            pixel_gecm_weight = 1.0
            image_gecm_weight = 1.0
            if gate == ObjectType.MULTI and self.gecm_enable:
                image_gecm_weight = self._resolve_multi_image_gecm_weight(cls_name)
                pixel_gecm_weight = self._resolve_multi_pixel_gecm_weight(cls_name)
            final_map, final_img, _image_gecm_weight, _pixel_gecm_weight, _global_score = (
                self._finalize_prediction_scores(
                    capm_map,
                    capm_img,
                    gecm_map,
                    img_path,
                    image_gecm_weight=image_gecm_weight,
                    pixel_gecm_weight=pixel_gecm_weight,
                )
            )

            img_scores.append(final_img)
            pixel_scores.append(final_map)

        img_scores = torch.tensor(img_scores, device=device, dtype=inputs.dtype)
        pixel_maps = torch.cat(pixel_scores, dim=0)  # (B, 1, H, W)

        return build_predict_results(data_samples, img_scores, pixel_maps)

    @torch.no_grad()
    def debug_predict(self, inputs, data_samples):
        """Return per-image debug breakdown for UniVAD scoring."""
        if not self._ref_data_built:
            self.fit()

        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        device = inputs.device
        x_clip = self._normalize_for_clip(inputs)
        if x_clip.shape[-2:] != (self.image_size, self.image_size):
            x_clip = F.interpolate(
                x_clip,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False,
            )
        query_global_feats, query_clip_layers = self._extract_clip_multilayer(x_clip)
        query_dinov2 = F.normalize(self._extract_dinov2_features(inputs), dim=-1)

        h_clip, w_clip = self._clip_grid_h, self._clip_grid_w
        h_dino = self._dinov2_resize // self._dinov2_patch_size
        w_dino = h_dino

        debug_rows = []
        for idx in range(inputs.shape[0]):
            sample = data_samples[idx]
            cls_name = getattr(sample, 'cls_name', 'object')
            img_path = getattr(sample, 'img_path', '')
            gate = self._gate.get(cls_name, ObjectType.TEXTURE)
            q_global = query_global_feats[idx:idx + 1]
            q_clip_layers = [layer[idx:idx + 1] for layer in query_clip_layers]
            q_dinov2 = query_dinov2[idx:idx + 1]

            capm_map, capm_img, capm_debug = self._capm_score(
                cls_name,
                img_path,
                gate,
                q_global,
                q_clip_layers,
                q_dinov2,
                h_clip,
                w_clip,
                h_dino,
                w_dino,
                device,
                return_debug=True,
            )

            gecm_score = 0.0
            gecm_map = torch.zeros_like(capm_map)
            gecm_debug = {}
            if gate == ObjectType.MULTI and self.gecm_enable:
                gecm_score, gecm_map, gecm_debug = self._gecm_score(
                    cls_name,
                    img_path,
                    q_clip_layers,
                    q_dinov2,
                    h_clip,
                    w_clip,
                    h_dino,
                    w_dino,
                    device,
                    return_debug=True,
                )
            pixel_gecm_weight = 1.0
            image_gecm_weight = 1.0
            if gate == ObjectType.MULTI and self.gecm_enable:
                image_gecm_weight = self._resolve_multi_image_gecm_weight(cls_name)
                pixel_gecm_weight = self._resolve_multi_pixel_gecm_weight(cls_name)
            final_map, final_img, image_gecm_weight, pixel_gecm_weight, global_score = (
                self._finalize_prediction_scores(
                    capm_map,
                    capm_img,
                    gecm_map,
                    img_path,
                    image_gecm_weight=image_gecm_weight,
                    pixel_gecm_weight=pixel_gecm_weight,
                )
            )
            capm_debug['global_score'] = float(global_score)

            debug_rows.append({
                'cls_name': cls_name,
                'img_path': img_path,
                'gate': gate.value,
                'pred_score': float(final_img),
                'capm_img_score': float(capm_img),
                'gecm_score': float(gecm_score),
                'multi_image_gecm_weight': float(image_gecm_weight),
                'multi_pixel_gecm_weight': float(pixel_gecm_weight),
                'capm': capm_debug,
                'gecm': gecm_debug,
                'map_max': float(final_map.max()),
                'map_mean': float(final_map.mean()),
            })
        return debug_rows

    # ------------------------------------------------------------------
    # CAPM: Combined Anomaly Patch Matching
    # ------------------------------------------------------------------

    def _capm_score(self, cls_name, img_path, gate, q_global, q_clip_layers, q_dinov2,
                    h_clip, w_clip, h_dino, w_dino, device, return_debug=False):
        """Compute CAPM anomaly scores.

        Three sub-scores:
        1. CLIP multi-layer patch cosine distance
        2. DINOv2 patch cosine distance
        3. CLIP VL anomaly map (text-based normal/abnormal scoring)

        For TEXTURE: global patch match (all patches)
        For SINGLE: masked match (foreground patches only)
        For MULTI: per-component match

        Returns:
            score_map: (1, 1, image_size, image_size) anomaly map.
            img_score: float image-level score.
        """
        ref_clip = self._ref_clip_patches.get(cls_name)
        ref_dinov2 = self._ref_dinov2_patches.get(cls_name)
        ref_global = self._ref_global_feats.get(cls_name)
        ref_vl = self._ref_vl_text.get(cls_name)
        ref_clip_parts = self._ref_clip_part_patches.get(cls_name, [])
        ref_dino_parts = self._ref_dino_part_patches.get(cls_name, {})
        debug = {
            'clip_map_max': 0.0,
            'clip_map_mean': 0.0,
            'dino_map_max': 0.0,
            'dino_map_mean': 0.0,
            'vl_map_max': 0.0,
            'vl_map_mean': 0.0,
            'part_clip_map_max': 0.0,
            'part_dino_map_max': 0.0,
            'global_score': 0.0,
            'image_score_map_max': 0.0,
            'image_score_map_mean': 0.0,
            'pixel_clip_weight': self.single_pixel_clip_weight if self.single_pixel_clip_weight is not None else self.clip_weight,
            'pixel_dino_weight': self.single_pixel_dinov2_weight if self.single_pixel_dinov2_weight is not None else self.dinov2_weight,
            'pixel_vl_weight': self.single_pixel_vl_weight if self.single_pixel_vl_weight is not None else self.vl_weight,
            'pixel_global_weight': self.single_pixel_global_weight,
            'pixel_part_weight': self.single_pixel_part_weight,
            'image_pooling': self.single_image_pooling,
            'image_topk_ratio': self.single_image_topk_ratio,
            'image_mix_max_weight': self.single_image_mix_max_weight,
            'image_mix_topk_weight': self.single_image_mix_topk_weight,
            'image_clip_weight': self.single_image_clip_weight if self.single_image_clip_weight is not None else self.clip_weight,
            'image_dino_weight': self.single_image_dinov2_weight if self.single_image_dinov2_weight is not None else self.dinov2_weight,
            'image_vl_weight': self.single_image_vl_weight if self.single_image_vl_weight is not None else self.vl_weight,
            'num_capm_masks': 0,
            'capm_part_indices': [],
            'official_scoring': bool(self.official_scoring),
        }

        # Fallback if no reference data for this class
        if ref_clip is None and ref_dinov2 is None:
            score_map = torch.zeros(1, 1, self.image_size, self.image_size, device=device)
            return (score_map, 0.0, debug) if return_debug else (score_map, 0.0)

        # ----- Sub-score 1: CLIP multi-layer patch distance -----
        clip_score_map = self._clip_patch_distance(
            q_clip_layers, ref_clip, h_clip, w_clip, device)
        debug['clip_map_max'] = float(clip_score_map.max())
        debug['clip_map_mean'] = float(clip_score_map.mean())

        # ----- Sub-score 2: DINOv2 patch distance -----
        dinov2_score_map = self._dinov2_patch_distance(
            q_dinov2, ref_dinov2, h_dino, w_dino, device)
        debug['dino_map_max'] = float(dinov2_score_map.max())
        debug['dino_map_mean'] = float(dinov2_score_map.mean())

        # ----- Sub-score 3: VL anomaly map -----
        vl_map = torch.zeros(1, 1, self.image_size, self.image_size, device=device)
        if ref_vl is not None and q_clip_layers:
            normal_emb, anomalous_emb = ref_vl
            normal_emb = normal_emb.to(device)
            anomalous_emb = anomalous_emb.to(device)
            vl_layer_idx = _vl_clip_layer_index(len(q_clip_layers), len(self.clip_layers))
            vl_raw = self._vl_anomaly_map(
                q_clip_layers[vl_layer_idx], normal_emb, anomalous_emb)
            vl_map = F.interpolate(vl_raw, size=(self.image_size, self.image_size),
                                   mode='bilinear', align_corners=self._score_map_align_corners())
        debug['vl_map_max'] = float(vl_map.max())
        debug['vl_map_mean'] = float(vl_map.mean())

        # Fuse sub-scores
        if self.official_scoring:
            combined_map = (clip_score_map + dinov2_score_map + vl_map) / 3.0
        else:
            total_weight = self.clip_weight + self.dinov2_weight + self.vl_weight
            if total_weight <= 0:
                total_weight = 1.0
            combined_map = (
                self.clip_weight * clip_score_map
                + self.dinov2_weight * dinov2_score_map
                + self.vl_weight * vl_map
            ) / total_weight

        if gate in {ObjectType.SINGLE, ObjectType.MULTI}:
            capm_masks, capm_indices, _component_masks = self._load_query_masks(cls_name, img_path, gate)
            debug['num_capm_masks'] = len(capm_masks)
            debug['capm_part_indices'] = [int(idx) for idx in capm_indices]
            part_clip_map = torch.zeros(1, 1, h_clip, w_clip, device=device)
            part_dino_map = torch.zeros(1, 1, h_dino, w_dino, device=device)
            if gate == ObjectType.MULTI:
                part_clip_map.fill_(100.0)
                part_dino_map.fill_(100.0)

            for query_mask, part_idx in zip(capm_masks, capm_indices):
                selector_clip = self._mask_to_patch_selector(query_mask, h_clip, w_clip, device)
                selector_dino = self._mask_to_patch_selector(query_mask, h_dino, w_dino, device)
                clip_dist = self._selected_clip_part_distance(
                    q_clip_layers,
                    selector_clip,
                    ref_clip_parts,
                    part_idx if gate == ObjectType.MULTI else 0,
                )
                dino_dist = self._selected_dino_part_distance(
                    q_dinov2,
                    selector_dino,
                    ref_dino_parts,
                    part_idx if gate == ObjectType.MULTI else 0,
                )

                clip_grid = selector_clip.reshape(1, 1, h_clip, w_clip)
                dino_grid = selector_dino.reshape(1, 1, h_dino, w_dino)
                if clip_dist is not None:
                    if gate == ObjectType.SINGLE:
                        part_clip_map[clip_grid] += clip_dist
                    else:
                        part_clip_map[clip_grid] = torch.minimum(part_clip_map[clip_grid], clip_dist)
                if dino_dist is not None:
                    if gate == ObjectType.SINGLE:
                        part_dino_map[dino_grid] += dino_dist
                    else:
                        part_dino_map[dino_grid] = torch.minimum(part_dino_map[dino_grid], dino_dist)

            if gate == ObjectType.MULTI:
                part_clip_map[part_clip_map == 100.0] = 0.0
                part_dino_map[part_dino_map == 100.0] = 0.0

            part_clip_map = F.interpolate(
                part_clip_map, size=(self.image_size, self.image_size),
                mode='bilinear', align_corners=True)
            part_dino_map = F.interpolate(
                part_dino_map, size=(self.image_size, self.image_size),
                mode='bilinear', align_corners=True)
            debug['part_clip_map_max'] = float(part_clip_map.max())
            debug['part_dino_map_max'] = float(part_dino_map.max())

            if self.official_scoring:
                global_pair = (clip_score_map + dinov2_score_map) / 2.0
                part_pair = (part_clip_map + part_dino_map) / 2.0
                combined_map = (global_pair + part_pair + vl_map) / 3.0
                image_score_map = combined_map
                debug['pixel_clip_weight'] = 1.0
                debug['pixel_dino_weight'] = 1.0
                debug['pixel_vl_weight'] = 1.0
                debug['pixel_global_weight'] = 1.0
                debug['pixel_part_weight'] = 1.0
                debug['image_pooling'] = 'official_max'
                debug['image_topk_ratio'] = 0.0
                debug['image_mix_max_weight'] = 1.0
                debug['image_mix_topk_weight'] = 0.0
                debug['image_clip_weight'] = 1.0
                debug['image_dino_weight'] = 1.0
                debug['image_vl_weight'] = 1.0
                debug['image_score_map_max'] = float(image_score_map.max())
                debug['image_score_map_mean'] = float(image_score_map.mean())
                image_score_options = None
            elif gate == ObjectType.SINGLE:
                pixel_weights = self._resolve_single_branch_weights(
                    cls_name,
                    clip_weight=self.clip_weight if self.single_pixel_clip_weight is None else self.single_pixel_clip_weight,
                    dinov2_weight=self.dinov2_weight if self.single_pixel_dinov2_weight is None else self.single_pixel_dinov2_weight,
                    vl_weight=self.vl_weight if self.single_pixel_vl_weight is None else self.single_pixel_vl_weight,
                    global_weight=self.single_pixel_global_weight,
                    part_weight=self.single_pixel_part_weight,
                    overrides=self.single_pixel_weight_overrides,
                )
                debug['pixel_clip_weight'] = pixel_weights['clip_weight']
                debug['pixel_dino_weight'] = pixel_weights['dinov2_weight']
                debug['pixel_vl_weight'] = pixel_weights['vl_weight']
                debug['pixel_global_weight'] = pixel_weights['global_weight']
                debug['pixel_part_weight'] = pixel_weights['part_weight']
                pair_weight = pixel_weights['clip_weight'] + pixel_weights['dinov2_weight']
                if pair_weight <= 0:
                    pair_weight = 1.0
                global_pair = (
                    pixel_weights['clip_weight'] * clip_score_map
                    + pixel_weights['dinov2_weight'] * dinov2_score_map
                ) / pair_weight
                part_pair = (
                    pixel_weights['clip_weight'] * part_clip_map
                    + pixel_weights['dinov2_weight'] * part_dino_map
                ) / pair_weight
                branch_weight = (
                    pixel_weights['global_weight']
                    + pixel_weights['part_weight']
                    + pixel_weights['vl_weight']
                )
                if branch_weight <= 0:
                    branch_weight = 1.0
                combined_map = (
                    pixel_weights['global_weight'] * global_pair
                    + pixel_weights['part_weight'] * part_pair
                    + pixel_weights['vl_weight'] * vl_map
                ) / branch_weight

                image_weights = self._resolve_single_branch_weights(
                    cls_name,
                    clip_weight=self.clip_weight if self.single_image_clip_weight is None else self.single_image_clip_weight,
                    dinov2_weight=self.dinov2_weight if self.single_image_dinov2_weight is None else self.single_image_dinov2_weight,
                    vl_weight=self.vl_weight if self.single_image_vl_weight is None else self.single_image_vl_weight,
                    overrides=self.single_image_weight_overrides,
                )
                debug['image_clip_weight'] = image_weights['clip_weight']
                debug['image_dino_weight'] = image_weights['dinov2_weight']
                debug['image_vl_weight'] = image_weights['vl_weight']
                image_score_options = self._resolve_single_image_score_options(cls_name)
                debug['image_pooling'] = str(image_score_options['pooling'])
                debug['image_topk_ratio'] = float(image_score_options['topk_ratio'])
                debug['image_mix_max_weight'] = float(image_score_options['max_weight'])
                debug['image_mix_topk_weight'] = float(image_score_options['topk_weight'])
                image_pair_weight = image_weights['clip_weight'] + image_weights['dinov2_weight']
                if image_pair_weight <= 0:
                    image_pair_weight = 1.0
                image_global_pair = (
                    image_weights['clip_weight'] * clip_score_map
                    + image_weights['dinov2_weight'] * dinov2_score_map
                ) / image_pair_weight
                image_part_pair = (
                    image_weights['clip_weight'] * part_clip_map
                    + image_weights['dinov2_weight'] * part_dino_map
                ) / image_pair_weight
                image_branch_weight = (
                    image_weights['global_weight']
                    + image_weights['part_weight']
                    + image_weights['vl_weight']
                )
                if image_branch_weight <= 0:
                    image_branch_weight = 1.0
                image_score_map = (
                    image_weights['global_weight'] * image_global_pair
                    + image_weights['part_weight'] * image_part_pair
                    + image_weights['vl_weight'] * vl_map
                ) / image_branch_weight
                debug['image_score_map_max'] = float(image_score_map.max())
                debug['image_score_map_mean'] = float(image_score_map.mean())
            else:
                pair_weight = self.clip_weight + self.dinov2_weight
                if pair_weight <= 0:
                    pair_weight = 1.0
                global_pair = (
                    self.clip_weight * clip_score_map
                    + self.dinov2_weight * dinov2_score_map
                ) / pair_weight
                part_pair = (
                    self.clip_weight * part_clip_map
                    + self.dinov2_weight * part_dino_map
                ) / pair_weight
                branch_weight = 2.0 + self.vl_weight
                combined_map = (
                    global_pair
                    + part_pair
                    + self.vl_weight * vl_map
                ) / branch_weight
                image_score_map = combined_map
                debug['image_score_map_max'] = float(image_score_map.max())
                debug['image_score_map_mean'] = float(image_score_map.mean())
                image_score_options = self._resolve_single_image_score_options(cls_name)
        else:
            image_score_map = combined_map
            debug['image_score_map_max'] = float(image_score_map.max())
            debug['image_score_map_mean'] = float(image_score_map.mean())
            image_score_options = None if self.official_scoring else self._resolve_single_image_score_options(cls_name)

        # Apply Gaussian blur
        if self.gaussian_sigma > 0:
            combined_map = _gaussian_blur_bchw(combined_map, self.gaussian_sigma)

        global_score = 0.0
        if gate != ObjectType.TEXTURE and ref_global is not None:
            q_global = F.normalize(q_global, dim=-1)
            ref_global = F.normalize(ref_global.to(device), dim=-1)
            sim = q_global @ ref_global.T
            global_score = float(1.0 - sim.max())
        debug['global_score'] = global_score

        if self.official_scoring:
            img_score = self._official_image_score_from_map(image_score_map, img_path) + global_score
        else:
            img_score = self._image_score_from_map(
                image_score_map,
                pooling=str(image_score_options['pooling']),
                topk_ratio=float(image_score_options['topk_ratio']),
                max_weight=float(image_score_options['max_weight']),
                topk_weight=float(image_score_options['topk_weight']),
            ) + global_score
        return (combined_map, img_score, debug) if return_debug else (combined_map, img_score)

    def _clip_patch_distance(self, q_layers, ref_clip, h, w, device):
        """Compute CLIP multi-layer patch cosine distance map.

        For each layer, compute min cosine distance to any reference patch.
        Average across layers.
        """
        if ref_clip is None or not q_layers:
            return torch.zeros(1, 1, self.image_size, self.image_size, device=device)

        ref_clip = ref_clip.to(device)  # (K, n_layers, N, D)
        K, n_layers, N_ref, D = ref_clip.shape

        layer_maps = []
        layer_indices = _matching_clip_layer_indices(len(q_layers), len(self.clip_layers))

        for l_idx in layer_indices:
            q_layer = q_layers[l_idx]
            if l_idx >= n_layers:
                break
            q = F.normalize(q_layer, dim=-1)  # (1, N_q, D)
            r = F.normalize(ref_clip[:, l_idx], dim=-1)  # (K, N_ref, D)
            r_flat = r.reshape(-1, D)  # (K*N_ref, D)

            # Cosine distance: 1 - max_similarity
            sim = q[0] @ r_flat.T  # (N_q, K*N_ref)
            max_sim, _ = sim.max(dim=1)  # (N_q,)
            dist = 1.0 - max_sim

            dist_map = dist.reshape(1, 1, h, w)
            dist_map = F.interpolate(dist_map, size=(self.image_size, self.image_size),
                                     mode='bilinear', align_corners=self._score_map_align_corners())
            layer_maps.append(dist_map)

        if not layer_maps:
            return torch.zeros(1, 1, self.image_size, self.image_size, device=device)

        return torch.stack(layer_maps, dim=0).mean(dim=0)

    def _dinov2_patch_distance(self, q_dinov2, ref_dinov2, h, w, device):
        """Compute DINOv2 patch cosine distance map."""
        if ref_dinov2 is None:
            return torch.zeros(1, 1, self.image_size, self.image_size, device=device)

        ref = ref_dinov2.to(device)  # (K, N, D)
        K, N_ref, D = ref.shape
        ref_flat = ref.reshape(-1, D)  # (K*N, D)

        q = q_dinov2[0]  # (N_q, D)
        sim = q @ ref_flat.T  # (N_q, K*N)
        max_sim, _ = sim.max(dim=1)
        dist = 1.0 - max_sim  # (N_q,)

        N_q = dist.shape[0]
        if N_q == h * w:
            dist_map = dist.reshape(1, 1, h, w)
        else:
            side = int(math.sqrt(N_q))
            dist_map = dist.reshape(1, 1, side, side)

        return F.interpolate(dist_map, size=(self.image_size, self.image_size),
                             mode='bilinear', align_corners=self._score_map_align_corners())

    # ------------------------------------------------------------------
    # GECM: Graph-Enhanced Component Matching
    # ------------------------------------------------------------------

    def _gecm_score(self, cls_name, img_path, q_clip_layers, q_dinov2,
                    h_clip, w_clip, h_dino, w_dino, device, return_debug=False):
        """Compute GECM image-level score for MULTI-type objects.

        For query image, extract component features, aggregate via CFA,
        then compute cosine distance to reference component graphs.

        Returns:
            score: float GECM anomaly score.
        """
        debug = {
            'query_component_count': 0,
            'ref_component_count': 0,
            'anomaly_map_dist_max': 0.0,
            'anomaly_map_dist_mean': 0.0,
            'component_min_ratio': self._resolve_multi_component_min_ratio(cls_name),
            'clip_weight': 0.0,
            'dino_weight': 0.0,
            'geo_weight': 0.0,
            'component_dist_clip': [],
            'component_dist_dino': [],
            'component_dist_geo': [],
        }
        empty_map = torch.zeros((1, 1, self.image_size, self.image_size), device=device)
        if self.cfa is None:
            return (0.0, empty_map, debug) if return_debug else (0.0, empty_map)

        ref_comp = self._ref_component_bank.get(cls_name)
        if ref_comp is None:
            return (0.0, empty_map, debug) if return_debug else (0.0, empty_map)

        _capm_masks, _capm_indices, component_masks = self._load_query_masks(
            cls_name, img_path, ObjectType.MULTI)
        if not component_masks:
            return (0.0, empty_map, debug) if return_debug else (0.0, empty_map)

        if not q_clip_layers:
            return (0.0, empty_map, debug) if return_debug else (0.0, empty_map)

        with Image.open(img_path) as _img:
            image = np.array(
                _img.convert('RGB').resize((self.image_size, self.image_size))
            )
        q_components = self._extract_component_feature_bank(image, component_masks, device)
        if q_components is None:
            return (0.0, empty_map, debug) if return_debug else (0.0, empty_map)
        debug['query_component_count'] = int(q_components['area'].shape[0])
        debug['ref_component_count'] = int(ref_comp['area'].shape[0])

        q_clip = q_components['clip_image'].transpose(0, 1)
        q_dino = q_components['dino_image']
        if self.cfa is not None and q_clip.numel() > 0:
            for layer_idx in range(q_clip.shape[0]):
                q_clip[layer_idx] = self.cfa(q_clip[layer_idx])
            q_dino = self.cfa(q_dino)
        q_geo = torch.cat(
            [q_components['area'], q_components['color'], q_components['position']],
            dim=1,
        )
        feature_weights = self._resolve_multi_gecm_feature_weights(cls_name)
        debug['clip_weight'] = feature_weights['clip_weight']
        debug['dino_weight'] = feature_weights['dino_weight']
        debug['geo_weight'] = feature_weights['geo_weight']

        anomaly_map_dist = torch.zeros((self.image_size, self.image_size), device=device)
        for mask_idx, component_mask in enumerate(component_masks):
            component_mask = torch.from_numpy(component_mask).to(device)
            if q_clip.numel() == 0:
                dist_clip = 0.0
            else:
                sim_clip = F.cosine_similarity(
                    q_clip[:, mask_idx, :].unsqueeze(1),
                    ref_comp['clip_image'],
                    dim=-1,
                )
                dist_clip = float(torch.mean(1.0 - sim_clip.max(dim=1).values))

            sim_dino = F.cosine_similarity(
                q_dino[mask_idx].unsqueeze(0),
                ref_comp['dino_image'],
                dim=1,
            )
            dist_dino = float(1.0 - sim_dino.max())
            sim_geo = F.cosine_similarity(
                q_geo[mask_idx].unsqueeze(0),
                ref_comp['geo'],
                dim=1,
            )
            dist_geo = float(1.0 - sim_geo.max())
            debug['component_dist_clip'].append(float(dist_clip))
            debug['component_dist_dino'].append(dist_dino)
            debug['component_dist_geo'].append(dist_geo)
            dist, resolved_weights = self._combine_multi_gecm_distance(
                cls_name,
                dist_clip=dist_clip,
                dist_dino=dist_dino,
                dist_geo=dist_geo,
            )
            debug['clip_weight'] = resolved_weights['clip_weight']
            debug['dino_weight'] = resolved_weights['dino_weight']
            debug['geo_weight'] = resolved_weights['geo_weight']
            anomaly_map_dist[component_mask > 0] += dist

        score = float(anomaly_map_dist.max() / 2.0)
        debug['anomaly_map_dist_max'] = float(anomaly_map_dist.max())
        debug['anomaly_map_dist_mean'] = float(anomaly_map_dist.mean())
        map_tensor = anomaly_map_dist.unsqueeze(0).unsqueeze(0) / 2.0
        return (score, map_tensor, debug) if return_debug else (score, map_tensor)

    # ------------------------------------------------------------------
    # train() override
    # ------------------------------------------------------------------

    def train(self, mode=True):
        """Keep both backbones in eval mode."""
        super().train(mode)
        self.clip.eval()
        self.dinov2.eval()
        return self
