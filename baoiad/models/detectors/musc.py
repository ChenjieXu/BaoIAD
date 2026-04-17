"""MuSc: Mutual Scoring for Zero-Shot Anomaly Detection (ECCV 2024).

Reference: https://github.com/xrli-U/MuSc

Zero-shot method using CLIP/DINOv2 patch tokens with LNAMD aggregation,
MSM mutual scoring, and RsCIN refinement.
"""
import copy
import logging
import math
from contextlib import nullcontext
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from baoiad.structures import ADDataSample
from baoiad.registry import MODELS
from baoiad.models.base_ad_model import VisionLanguageADModel

logger = logging.getLogger(__name__)


# ============================================================================
# LNAMD Module (Local Neighborhood Aggregation with Multi-scale Dilation)
# Copied from ref/MuSc/models/modules/_LNAMD.py
# ============================================================================

class PatchMaker:
    """Patch maker for local neighborhood aggregation."""

    def __init__(self, patchsize: int, stride: int = None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        padding = int((self.patchsize - 1) / 2)
        unfolder = nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features


class MeanMapper(nn.Module):
    """Mean mapper for feature preprocessing."""

    def __init__(self, preprocessing_dim: int):
        super().__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)


class Preprocessing(nn.Module):
    """Preprocessing module for multi-layer features."""

    def __init__(self, input_layers: int, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self.preprocessing_modules = nn.ModuleList()
        for _ in range(input_layers):
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

    def forward(self, features):
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        return torch.stack(_features, dim=1)


class LNAMD(nn.Module):
    """Local Neighborhood Aggregation with Multi-scale Dilation.

    Args:
        device: Device to run on.
        feature_dim: Feature dimension.
        feature_layer: List of layer indices.
        r: Aggregation radius.
        patchstride: Stride for patch extraction.
    """

    def __init__(
        self,
        device,
        feature_dim: int = 1024,
        feature_layer: List[int] = None,
        r: int = 3,
        patchstride: int = 1
    ):
        super().__init__()
        if feature_layer is None:
            feature_layer = [1, 2, 3, 4]

        self.device = device
        self.r = r
        self.patch_maker = PatchMaker(r, stride=patchstride)
        self.LNA = Preprocessing(len(feature_layer), feature_dim)

    def _embed(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Embed features with local neighborhood aggregation.

        Args:
            features: List of (B, L, C) tensors (with CLS token at index 0).

        Returns:
            Aggregated features (B, L, layer, C).
        """
        B = features[0].shape[0]

        features_layers = []
        for feature in features:
            # Remove CLS token and reshape to spatial
            feature = feature[:, 1:, :]  # (B, L-1, C)
            H = int(math.sqrt(feature.shape[1]))
            W = H
            feature = feature.reshape(feature.shape[0], H, W, feature.shape[2])
            feature = feature.permute(0, 3, 1, 2)  # (B, C, H, W)

            # Apply LayerNorm
            feature = nn.LayerNorm([feature.shape[1], feature.shape[2], feature.shape[3]]).to(self.device)(feature)
            features_layers.append(feature)

        if self.r != 1:
            # Divide into patches
            features_layers = [self.patch_maker.patchify(x, return_spatial_info=True) for x in features_layers]
            patch_shapes = [x[1] for x in features_layers]
            features_layers = [x[0] for x in features_layers]
        else:
            patch_shapes = [f.shape[-2:] for f in features_layers]
            features_layers = [f.reshape(f.shape[0], f.shape[1], -1, 1, 1).permute(0, 2, 1, 3, 4) for f in features_layers]

        # Align patches across layers
        ref_num_patches = patch_shapes[0]
        for i in range(1, len(features_layers)):
            patch_dims = patch_shapes[i]
            if patch_dims[0] == ref_num_patches[0] and patch_dims[1] == ref_num_patches[1]:
                continue
            _features = features_layers[i]
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features_layers[i] = _features

        features_layers = [x.reshape(-1, *x.shape[-3:]) for x in features_layers]

        # Aggregation
        features_layers = self.LNA(features_layers)
        features_layers = features_layers.reshape(B, -1, *features_layers.shape[-2:])  # (B, L, layer, C)

        return features_layers.detach().cpu()


# ============================================================================
# MSM Module (Mutual Scoring Mechanism)
# Copied from ref/MuSc/models/modules/_MSM.py
# ============================================================================

def compute_scores_fast(Z: torch.Tensor, i: int, device, topmin_min: float = 0, topmin_max: float = 0.3):
    """Compute anomaly scores for image i against all other images (fast version).

    WARNING: This version uses more GPU memory. Use compute_scores_slow for large datasets.

    Args:
        Z: All features (N, L, C).
        i: Index of query image.
        device: Device to run on.
        topmin_min: Minimum percentile for topmin (0-1).
        topmin_max: Maximum percentile for topmin (0-1).

    Returns:
        Anomaly scores for each patch (L,).
    """
    image_num, patch_num, c = Z.shape

    # Reference: all images except the query
    Z_ref = torch.cat((Z[:i], Z[i + 1:]), dim=0)

    # Compute distances: (patch_num, image_num-1, patch_num)
    patch2image = torch.cdist(Z[i:i + 1], Z_ref.reshape(-1, c)).reshape(patch_num, image_num - 1, patch_num)
    patch2image = torch.min(patch2image, -1)[0]  # (patch_num, image_num-1)

    # Interval average (topmin scoring)
    k_max = topmin_max
    k_min = topmin_min
    if k_max < 1:
        k_max = int(patch2image.shape[1] * k_max)
    if k_min < 1:
        k_min = int(patch2image.shape[1] * k_min)
    if k_max < k_min:
        k_max, k_min = k_min, k_max

    # Get values in [k_min, k_max] range
    vals, _ = torch.topk(patch2image.float(), k_max, largest=False, sorted=True)
    vals, _ = torch.topk(vals.float(), k_max - k_min, largest=True, sorted=True)

    return torch.mean(vals, dim=1)


def compute_scores_batched(Z: torch.Tensor, i: int, device, topmin_min: float = 0,
                           topmin_max: float = 0.3, batch_size: int = 64):
    """Compute anomaly scores for image i against all other images (batched version).

    This version processes reference images in batches for better GPU utilization
    while keeping memory usage bounded.

    Args:
        Z: All features (N, L, C).
        i: Index of query image.
        device: Device to run on.
        topmin_min: Minimum percentile for topmin (0-1).
        topmin_max: Maximum percentile for topmin (0-1).
        batch_size: Number of reference images to process at once.

    Returns:
        Anomaly scores for each patch (L,).
    """
    N, L, C = Z.shape
    query = Z[i]  # (L, C)

    # Pre-allocate result tensor
    all_min_dists = torch.zeros(L, N - 1, device=device, dtype=torch.float32)

    # Process reference images in batches
    ref_indices = [j for j in range(N) if j != i]
    for batch_start in range(0, len(ref_indices), batch_size):
        batch_end = min(batch_start + batch_size, len(ref_indices))
        batch_indices = ref_indices[batch_start:batch_end]
        batch_refs = torch.stack([Z[j] for j in batch_indices], dim=0)  # (B, L, C)

        # Compute distances: query (L, C) vs batch_refs (B, L, C)
        # Result: (L, B) - min distance from each query patch to each ref image
        for b_idx, ref in enumerate(batch_refs):
            dist = torch.cdist(query.unsqueeze(0), ref.unsqueeze(0))[0]  # (L, L)
            min_dist = torch.min(dist, dim=1)[0]  # (L,)
            all_min_dists[:, batch_start + b_idx] = min_dist

    # Interval average (topmin scoring)
    k_max = topmin_max
    k_min = topmin_min
    if k_max < 1:
        k_max = int(all_min_dists.shape[1] * k_max)
    if k_min < 1:
        k_min = int(all_min_dists.shape[1] * k_min)
    if k_max < k_min:
        k_max, k_min = k_min, k_max

    # Get values in [k_min, k_max] range
    vals, _ = torch.topk(all_min_dists.float(), k_max, largest=False, sorted=True)
    vals, _ = torch.topk(vals.float(), k_max - k_min, largest=True, sorted=True)

    return torch.mean(vals, dim=1)


def compute_scores_slow(Z: torch.Tensor, i: int, device, topmin_min: float = 0, topmin_max: float = 0.3):
    """Compute anomaly scores for image i against all other images (memory-efficient version).

    This version processes one reference image at a time to avoid OOM.

    Args:
        Z: All features (N, L, C).
        i: Index of query image.
        device: Device to run on.
        topmin_min: Minimum percentile for topmin (0-1).
        topmin_max: Maximum percentile for topmin (0-1).

    Returns:
        Anomaly scores for each patch (L,).
    """
    patch2image = torch.tensor([]).to(device)
    for j in range(Z.shape[0]):
        if j != i:
            # Compute min distance from each patch in Z[i] to all patches in Z[j]
            dist = torch.cdist(Z[i], Z[j])
            min_dist = torch.min(dist, dim=1)[0]  # (L,)
            patch2image = torch.cat((patch2image, min_dist.unsqueeze(1)), dim=1)

    # Interval average (topmin scoring)
    k_max = topmin_max
    k_min = topmin_min
    if k_max < 1:
        k_max = int(patch2image.shape[1] * k_max)
    if k_min < 1:
        k_min = int(patch2image.shape[1] * k_min)
    if k_max < k_min:
        k_max, k_min = k_min, k_max

    # Get values in [k_min, k_max] range
    vals, _ = torch.topk(patch2image.float(), k_max, largest=False, sorted=True)
    vals, _ = torch.topk(vals.float(), k_max - k_min, largest=True, sorted=True)
    patch2image = vals.clone()

    return torch.mean(patch2image, dim=1)


def MSM(Z: torch.Tensor, device, topmin_min: float = 0, topmin_max: float = 0.3) -> torch.Tensor:
    """Mutual Scoring Mechanism.

    Compute anomaly scores for all images by mutual scoring.
    Uses chunked processing to balance memory and speed.

    Args:
        Z: Aggregated features (N, L, C).
        device: Device to run on.
        topmin_min: Minimum percentile for topmin (0-1).
        topmin_max: Maximum percentile for topmin (0-1).

    Returns:
        Anomaly map (N, L).
    """
    N, L, C = Z.shape
    Z = Z.to(device)

    # Pre-compute k values for topmin
    k_max = int(N * topmin_max) if topmin_max < 1 else int(topmin_max)
    k_min = int(N * topmin_min) if topmin_min < 1 else int(topmin_min)
    if k_max < k_min:
        k_max, k_min = k_min, k_max
    if k_max == 0:
        k_max = 1

    # Chunk size for reference processing (balance memory vs speed)
    chunk_size = min(32, N - 1)

    anomaly_scores_matrix = torch.zeros(N, L, device=device, dtype=torch.float64)

    for i in tqdm(range(N), desc="MSM scoring"):
        # Collect all patch-to-image distances for query i
        # For each query patch, we need min distance to each reference image
        patch2image = []

        # Process references in chunks
        for chunk_start in range(0, N, chunk_size):
            chunk_end = min(chunk_start + chunk_size, N)

            # Skip if query is in this chunk
            if chunk_start <= i < chunk_end:
                # Handle refs before i
                if chunk_start < i:
                    Z_chunk = Z[chunk_start:i]
                    # cdist: (L, C) vs (chunk_size-1, L, C) -> need to reshape
                    # Z[i]: (L, C), Z_chunk: (chunk_size-1, L, C)
                    # We want: for each query patch, min distance to each ref image
                    # Reshape Z_chunk to (ref_images * L, C) for batched cdist
                    Z_chunk_flat = Z_chunk.reshape(-1, C)  # (chunk_refs * L, C)
                    dist = torch.cdist(Z[i].unsqueeze(0), Z_chunk_flat.unsqueeze(0))[0]  # (L, chunk_refs * L)
                    dist = dist.reshape(L, i - chunk_start, L)  # (L, num_refs, L)
                    min_dist = dist.min(dim=-1)[0]  # (L, num_refs)
                    patch2image.append(min_dist)
                # Handle refs after i
                if i + 1 < chunk_end:
                    Z_chunk = Z[i + 1:chunk_end]
                    Z_chunk_flat = Z_chunk.reshape(-1, C)
                    dist = torch.cdist(Z[i].unsqueeze(0), Z_chunk_flat.unsqueeze(0))[0]
                    dist = dist.reshape(L, chunk_end - i - 1, L)
                    min_dist = dist.min(dim=-1)[0]
                    patch2image.append(min_dist)
            else:
                # Process whole chunk
                Z_chunk = Z[chunk_start:chunk_end]
                Z_chunk_flat = Z_chunk.reshape(-1, C)
                dist = torch.cdist(Z[i].unsqueeze(0), Z_chunk_flat.unsqueeze(0))[0]  # (L, chunk_size * L)
                dist = dist.reshape(L, chunk_end - chunk_start, L)  # (L, num_refs, L)
                min_dist = dist.min(dim=-1)[0]  # (L, num_refs)
                patch2image.append(min_dist)

        # Concatenate all distances: (L, N-1)
        patch2image = torch.cat(patch2image, dim=1)

        # Topmin scoring
        vals, _ = torch.topk(patch2image.float(), k_max, largest=False, sorted=True)
        vals, _ = torch.topk(vals.float(), k_max - k_min, largest=True, sorted=True)
        anomaly_scores_matrix[i] = vals.mean(dim=1)

    return anomaly_scores_matrix


# ============================================================================
# RsCIN Module (Refined Scoring with Class-token Induced Neighborhood)
# Copied from ref/MuSc/models/modules/_RsCIN.py
# ============================================================================

def MMO(W: torch.Tensor, score: torch.Tensor, k_list: List[int] = None):
    """Multi-scale Manifold Optimization.

    Args:
        W: Similarity matrix (N, N).
        score: Initial scores (N,).
        k_list: List of k values for neighborhood.

    Returns:
        Refined scores (N,).
    """
    if k_list is None:
        k_list = [1, 2, 3]

    S_list = []
    for k in k_list:
        # Top-k smallest similarities (most different images)
        _, topk_matrix = torch.topk(W.float(), W.shape[0] - k, largest=False, sorted=True)
        W_mask = W.clone()
        for i in range(W.shape[0]):
            W_mask[i, topk_matrix[i]] = 0

        n = W.shape[-1]
        D_ = torch.zeros_like(W).float()
        for i in range(n):
            D_[i, i] = 1 / (W_mask[i, :].sum() + 1e-8)

        P = D_ @ W_mask
        S = score.clone().unsqueeze(-1)
        S = P @ S
        S_list.append(S)

    S = torch.concat(S_list, -1).mean(-1)
    return S


def RsCIN(scores_old: np.ndarray, cls_tokens: List[np.ndarray] = None, k_list: List[int] = None) -> np.ndarray:
    """Refined Scoring with Class-token Induced Neighborhood.

    Args:
        scores_old: Initial image-level scores.
        cls_tokens: List of CLS token features.
        k_list: List of k values for MMO.

    Returns:
        Refined image-level scores.
    """
    if k_list is None:
        k_list = [1, 2, 3]

    if cls_tokens is None or 0 in k_list:
        return scores_old

    cls_tokens = np.array(cls_tokens)

    # Normalize scores
    scores = (scores_old - scores_old.min()) / (scores_old.max() - scores_old.min() + 1e-8)

    # Compute similarity matrix from CLS tokens
    similarity_matrix = cls_tokens @ cls_tokens.T
    similarity_matrix = torch.tensor(similarity_matrix)

    # Apply MMO
    scores_new = MMO(similarity_matrix.clone().float(), score=torch.tensor(scores).clone().float(), k_list=k_list)
    scores_new = scores_new.numpy()

    return scores_new


# ============================================================================
# MuScDetector
# ============================================================================

@MODELS.register_module()
class MuScDetector(VisionLanguageADModel):
    """MuSc: Mutual Scoring anomaly detector.

    Zero-shot method using CLIP/DINOv2 backbone with:
    - LNAMD: Local Neighborhood Aggregation with Multi-scale Dilation
    - MSM: Mutual Scoring Mechanism with topmin
    - RsCIN: Refined Scoring with Class-token Induced Neighborhood

    Args:
        backbone: Backbone config dict. Default uses MuScCLIPBackbone with ViT-L-14-336.
        feature_layers: List of 1-indexed layer numbers to extract features from.
        r_list: List of aggregation radii for LNAMD.
        image_size: Input image size.
        topmin_min: Minimum percentile for MSM topmin (0-1).
        topmin_max: Maximum percentile for MSM topmin (0-1).
        k_list: List of k values for RsCIN MMO.
        data_preprocessor: Data preprocessor.
        init_cfg: Initialization config.
    """

    def __init__(
        self,
        backbone=None,
        feature_layers: List[int] = None,
        r_list: List[int] = None,
        image_size: int = 518,
        topmin_min: float = 0.0,
        topmin_max: float = 0.3,
        k_list: List[int] = None,
        data_preprocessor=None,
        init_cfg=None,
        **kwargs
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        # Default parameters
        if feature_layers is None:
            feature_layers = [5, 11, 17, 23]
        if r_list is None:
            r_list = [1, 3, 5]
        if k_list is None:
            k_list = [1, 2, 3]

        self.feature_layers = feature_layers
        self.r_list = r_list
        self.image_size = image_size
        self.topmin_min = topmin_min
        self.topmin_max = topmin_max
        self.k_list = k_list

        # Build backbone
        if backbone is None:
            backbone = dict(
                type='MuScCLIPBackbone',
                model_name='ViT-L-14-336',
                pretrained='openai',
                feature_layers=feature_layers,
                image_size=image_size,
                frozen=True,
            )
        elif isinstance(backbone, dict):
            backbone = copy.deepcopy(backbone)
        else:
            raise ValueError(f"backbone must be None or dict, got {type(backbone)}")

        self.backbone = MODELS.build(backbone)

        # Get feature dimension
        self.feature_dim = getattr(self.backbone, 'width', 1024)
        self.resolved_feature_layers = list(
            getattr(self.backbone, 'resolved_feature_layers', self.feature_layers)
        )

        # MuSc requires dataset-level post-processing
        self.requires_full_test_postprocess = True

        # Storage for deferred scoring
        self._pending_patch_tokens: List[List[torch.Tensor]] = []  # [(B, L, C) per batch]
        self._pending_cls_tokens: List[torch.Tensor] = []  # [(B, D) per batch]
        self._pending_samples: List[ADDataSample] = []

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Extract multi-layer features from backbone.

        Args:
            x: Input images (B, 3, H, W).

        Returns:
            image_features: (B, D) CLS token features.
            patch_tokens: List of (B, L+1, C) tensors for each layer.
        """
        return self.backbone.encode_image(x, self.feature_layers)

    def forward(self, inputs, data_samples=None, mode='tensor'):
        """Forward pass.

        Args:
            inputs: Input images.
            data_samples: Data samples.
            mode: Forward mode ('loss', 'predict', 'tensor').

        Returns:
            For 'predict': list of ADDataSample with placeholder scores.
            For 'tensor': extracted features.
        """
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        if mode == 'loss':
            # Zero-shot: no training needed
            return {'loss': torch.tensor(0.0, device=inputs.device, requires_grad=True)}

        elif mode == 'predict':
            # Extract features
            image_features, patch_tokens = self.extract_features(inputs)

            # Store for deferred scoring
            self._pending_patch_tokens.append([pt.cpu() for pt in patch_tokens])
            self._pending_cls_tokens.append(image_features.cpu())

            results = []
            for i in range(inputs.shape[0]):
                r = data_samples[i] if data_samples else ADDataSample()
                # Placeholder outputs: score_all() will update these
                r.pred_score = 0.0
                r.pred_anomaly_map = torch.zeros(1, self.image_size, self.image_size)
                results.append(r)
                self._pending_samples.append(r)

            return results

        # mode == 'tensor'
        return self.extract_features(inputs)

    def score_all(self, image_size=None):
        """Compute mutual scores using all collected test images.

        This method is called after all test images have been processed
        to compute the mutual anomaly scores.

        Args:
            image_size: Override image size for interpolation.

        Returns:
            List of ADDataSample with updated scores.
        """
        if not self._pending_patch_tokens:
            return []

        device = next(self.backbone.parameters()).device
        num_images = len(self._pending_samples)
        target_size = image_size if image_size else self.image_size

        # Collect all patch tokens by layer
        # patch_tokens_all[layer] = (N, L, C)
        num_layers = len(self.feature_layers)
        patch_tokens_all = [[] for _ in range(num_layers)]
        cls_tokens_all = []

        for batch_idx, patch_tokens_batch in enumerate(self._pending_patch_tokens):
            for layer_idx, pt in enumerate(patch_tokens_batch):
                patch_tokens_all[layer_idx].append(pt)
            cls_tokens_all.append(self._pending_cls_tokens[batch_idx])

        # Concatenate: (N, L+1, C) per layer
        patch_tokens_all = [torch.cat(pt, dim=0) for pt in patch_tokens_all]
        cls_tokens_all = torch.cat(cls_tokens_all, dim=0).numpy()  # (N, D)

        # Process each aggregation radius
        anomaly_maps_all_r = []

        for r in self.r_list:
            logger.info(f'LNAMD with r={r}...')

            # LNAMD: Aggregate features
            lnamd = LNAMD(
                device=device,
                r=r,
                feature_dim=self.feature_dim,
                feature_layer=self.resolved_feature_layers,
            )

            Z_layers = {}
            for im in range(num_images):
                patch_tokens = [pt[im:im + 1].to(device) for pt in patch_tokens_all]
                autocast_ctx = torch.amp.autocast('cuda') if device.type == 'cuda' else nullcontext()
                with autocast_ctx:
                    features = lnamd._embed(patch_tokens)
                    features = features / features.norm(dim=-1, keepdim=True)

                for layer_index in range(num_layers):
                    if str(layer_index) not in Z_layers:
                        Z_layers[str(layer_index)] = []
                    # Store on CPU to save GPU memory
                    Z_layers[str(layer_index)].append(features[:, :, layer_index, :].cpu())

                # Clear GPU memory immediately
                del features
                del patch_tokens

            del lnamd
            torch.cuda.empty_cache()

            # MSM: Mutual scoring for each layer
            anomaly_maps_layers = []
            for layer_key in tqdm(sorted(Z_layers.keys()), desc=f"MSM r={r}"):
                # Process one layer at a time on GPU
                Z = torch.cat(Z_layers[layer_key], dim=0).to(device)  # (N, L, C)
                anomaly_map = MSM(Z, device, self.topmin_min, self.topmin_max)
                anomaly_maps_layers.append(anomaly_map.unsqueeze(0).cpu())
                # Clear GPU memory
                del Z
                del anomaly_map
                torch.cuda.empty_cache()

            # Clear Z_layers to free memory
            del Z_layers
            torch.cuda.empty_cache()

            # Average across layers (keep on CPU)
            anomaly_maps_layers = torch.mean(torch.cat(anomaly_maps_layers, dim=0), dim=0)
            anomaly_maps_all_r.append(anomaly_maps_layers.unsqueeze(0))

        # Average across r values (on CPU first, then move final result to GPU)
        anomaly_maps = torch.mean(torch.cat(anomaly_maps_all_r, dim=0), dim=0)  # (N, L) on CPU

        # Interpolate to image size
        B, L = anomaly_maps.shape
        H = int(np.sqrt(L))
        anomaly_maps = F.interpolate(
            anomaly_maps.view(B, 1, H, H),
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=True
        ).squeeze(1)  # (N, H, W)

        anomaly_maps_np = anomaly_maps.cpu().numpy()

        # Image-level scores: max over spatial
        image_scores = anomaly_maps_np.reshape(B, -1).max(-1)

        # RsCIN: Refine image scores using CLS token similarity
        refined_scores = RsCIN(image_scores, cls_tokens_all, self.k_list)

        # Update data samples
        for i, sample in enumerate(self._pending_samples):
            sample.pred_score = float(refined_scores[i])
            sample.pred_anomaly_map = torch.from_numpy(anomaly_maps_np[i:i + 1])

        results = list(self._pending_samples)

        # Clear pending data
        self._pending_patch_tokens = []
        self._pending_cls_tokens = []
        self._pending_samples = []

        return results

    def train(self, mode=True):
        """Override train to keep backbone frozen."""
        super().train(mode)
        self.backbone.eval()
        return self
