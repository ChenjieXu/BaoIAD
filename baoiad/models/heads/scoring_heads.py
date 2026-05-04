"""Composable scoring heads for anomaly detection.

Each ScoringHead encapsulates a complete anomaly scoring pipeline:
  - feature processing (dim reduction, selection, aggregation)
  - normal representation modeling (memory bank, Gaussian, PCA)
  - anomaly scoring (kNN, Mahalanobis, reconstruction error)

All heads share the same lifecycle:
  1. loss() — called during training to collect features or compute training loss
  2. fit()  — called after training to build the scoring model
  3. predict() — called during inference to compute anomaly scores
"""

import math
import random
from abc import abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from torch import Tensor

from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS
from baoiad.structures import ADDataSample

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


# ============================================================
# Base
# ============================================================

@MODELS.register_module(force=True)
class BaseScoringHead(BaseModule):
    """Abstract base class for all composable scoring heads.

    Lifecycle:
      1. loss(feats, data_samples) — training: collect features or compute loss
      2. fit()                     — post-training: build scoring model
      3. predict(feats, data_samples) — inference: compute anomaly scores

    Args:
        input_size: Target spatial size (H, W) for anomaly map upsampling.
        blur_sigma: Gaussian blur sigma for post-processing. 0 to disable.
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (256, 256),
        blur_sigma: float = 4.0,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.input_size = input_size
        self.blur_sigma = blur_sigma
        if blur_sigma > 0:
            kernel_size = 2 * int(4.0 * blur_sigma + 0.5) + 1
            self._blur = _GaussianBlur2d(sigma=blur_sigma, kernel_size=kernel_size)
        else:
            self._blur = None

    @abstractmethod
    def loss(
        self,
        feats: Tuple[Tensor, ...],
        data_samples: Optional[List] = None,
    ) -> Dict[str, Tensor]:
        """Training step: collect features or compute training loss."""

    def fit(self) -> None:
        """Post-training: build the scoring model (coreset, Gaussian, PCA, etc.)."""

    @abstractmethod
    def predict(
        self,
        feats: Tuple[Tensor, ...],
        data_samples: Optional[List] = None,
    ) -> List:
        """Inference: compute anomaly scores and return predictions."""

    def _postprocess(self, score_map: Tensor) -> Tensor:
        """Upsample and optionally blur anomaly map."""
        if score_map.shape[-2:] != self.input_size:
            score_map = F.interpolate(
                score_map, size=self.input_size,
                mode='bilinear', align_corners=False,
            )
        if self._blur is not None:
            if self._blur.kernel.device != score_map.device:
                self._blur.to(score_map.device)
            score_map = self._blur(score_map)
        return score_map


# ============================================================
# Gaussian Blur helper
# ============================================================

class _GaussianBlur2d(nn.Module):
    """Depthwise 2D Gaussian blur."""

    def __init__(self, sigma: float, kernel_size: int):
        super().__init__()
        x = torch.arange(kernel_size).float() - kernel_size // 2
        gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        kernel_2d = gauss.unsqueeze(1) * gauss.unsqueeze(0)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
        self.register_buffer('kernel', kernel_2d)
        self.padding = kernel_size // 2

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        kernel = self.kernel.expand(C, -1, -1, -1)
        return F.conv2d(x, kernel, padding=self.padding, groups=C)


# ============================================================
# KNNScoringHead (PatchCore tricks)
# ============================================================

@MODELS.register_module(force=True)
class KNNScoringHead(BaseScoringHead):
    """kNN-based anomaly scoring with coreset subsampling.

    Reimplements PatchCore's scoring pipeline as a composable head.

    Args:
        coreset_ratio: Fraction of features to keep.
        num_neighbors: k for kNN search.
        patchsize: Neighborhood aggregation size.
        feature_selection: 'coreset' (PatchCore) or 'full' (SPADE).
        dim_reduction: 'none', 'random', or 'pca'.
        n_random_dims: Dimensions to keep when dim_reduction='random'.
        n_pca_dims: Dimensions to keep when dim_reduction='pca'.
        coreset_method: 'approx_greedy' or 'random'.
        coreset_projection_dim: Projection dim for approx greedy.
    """

    def __init__(
        self,
        coreset_ratio: float = 0.1,
        num_neighbors: int = 9,
        patchsize: int = 3,
        feature_selection: str = 'coreset',
        dim_reduction: str = 'none',
        n_random_dims: int = 128,
        n_pca_dims: int = 128,
        coreset_method: str = 'approx_greedy',
        coreset_projection_dim: int = 128,
        input_size: Tuple[int, int] = (256, 256),
        blur_sigma: float = 4.0,
        init_cfg=None,
    ):
        super().__init__(input_size=input_size, blur_sigma=blur_sigma, init_cfg=init_cfg)
        self.coreset_ratio = coreset_ratio
        self.num_neighbors = num_neighbors
        self.patchsize = patchsize
        self.feature_selection = feature_selection
        self.dim_reduction = dim_reduction
        self.n_random_dims = n_random_dims
        self.n_pca_dims = n_pca_dims
        self.coreset_method = coreset_method
        self.coreset_projection_dim = coreset_projection_dim

        self._train_features: List[np.ndarray] = []
        self.memory_bank: Optional[np.ndarray] = None
        self._nn_index = None
        # Dim reduction state
        self._pca_components: Optional[Tensor] = None
        self._pca_mean: Optional[Tensor] = None
        self._random_idx: Optional[Tensor] = None

    # ------ feature processing ------

    def _patchify_and_aggregate(self, concat: Tensor) -> Tensor:
        """Patch neighborhood aggregation (PatchCore trick)."""
        B, C, H, W = concat.shape
        if self.patchsize <= 1:
            return concat.permute(0, 2, 3, 1).reshape(-1, C)
        padding = (self.patchsize - 1) // 2
        unfolded = F.unfold(
            concat, kernel_size=self.patchsize,
            stride=1, padding=padding,
        )
        unfolded = unfolded.reshape(B, C, self.patchsize ** 2, -1)
        aggregated = unfolded.mean(dim=2)  # (B, C, H*W)
        return aggregated.permute(0, 2, 1).reshape(-1, C)

    def _apply_dim_reduction(self, patches: np.ndarray) -> np.ndarray:
        if self.dim_reduction == 'none' or patches.shape[1] <= 1:
            return patches
        patches_t = torch.from_numpy(patches.astype(np.float32))
        if self.dim_reduction == 'random':
            if self._random_idx is None:
                rng = torch.Generator()
                rng.manual_seed(42)
                D = patches.shape[1]
                n = min(self.n_random_dims, D)
                self._random_idx = torch.randperm(D, generator=rng)[:n]
            patches_t = patches_t[:, self._random_idx]
        elif self.dim_reduction == 'pca':
            if self._pca_components is None:
                # Fit PCA on first batch of collected features
                self._fit_pca(torch.from_numpy(
                    self._train_features[0].astype(np.float32)
                ) if self._train_features else patches_t)
            patches_t = (patches_t - self._pca_mean) @ self._pca_components
        return patches_t.numpy().astype(np.float32)

    def _fit_pca(self, data: Tensor) -> None:
        mean = data.mean(dim=0)
        centered = data - mean
        _, _, Vh = torch.linalg.svd(centered.double(), full_matrices=False)
        n = min(self.n_pca_dims, Vh.shape[0])
        self._pca_components = Vh[:n].T.float()
        self._pca_mean = mean.float()

    # ------ lifecycle ------

    def loss(self, feats: Tuple[Tensor, ...], data_samples=None) -> Dict[str, Tensor]:
        concat = torch.cat(feats, dim=1)
        patches = self._patchify_and_aggregate(concat)
        self._train_features.append(patches.cpu().numpy())
        return {'loss': torch.tensor(0.0, requires_grad=True)}

    def fit(self) -> None:
        if not self._train_features:
            return
        all_features = np.concatenate(self._train_features, axis=0).astype(np.float32)
        self._train_features.clear()

        # Dim reduction before selection
        all_features = self._apply_dim_reduction(all_features)

        # Feature selection
        n_total = all_features.shape[0]
        n_select = max(1, int(n_total * self.coreset_ratio))

        if self.feature_selection == 'coreset' and n_total > n_select:
            if self.coreset_method == 'approx_greedy':
                self.memory_bank = self._approx_coreset(all_features, n_select)
            else:
                rng = np.random.default_rng(42)
                idx = rng.choice(n_total, n_select, replace=False)
                self.memory_bank = all_features[idx]
        else:
            self.memory_bank = all_features

        # Build kNN index
        data = np.ascontiguousarray(self.memory_bank.astype(np.float32))
        if _HAS_FAISS:
            self._nn_index = faiss.IndexFlatL2(data.shape[1])
            self._nn_index.add(data)
        else:
            from sklearn.neighbors import NearestNeighbors
            self._nn_index = NearestNeighbors(
                n_neighbors=self.num_neighbors, metric='euclidean')
            self._nn_index.fit(data)

    def _approx_coreset(self, features: np.ndarray, n_select: int) -> np.ndarray:
        """Approximate greedy coreset subsampling."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ft = torch.from_numpy(features).to(device)
        N, D = ft.shape
        gen = torch.Generator(device=device)
        gen.manual_seed(42)
        if D > self.coreset_projection_dim:
            proj = torch.randn(D, self.coreset_projection_dim,
                               generator=gen, device=device, dtype=ft.dtype)
            reduced = ft @ proj
        else:
            reduced = ft
        start_count = min(10, N)
        start_idx = torch.randperm(N, generator=gen, device=device)[:start_count]
        start_feats = reduced[start_idx]
        norms = (reduced ** 2).sum(dim=1, keepdim=True)
        start_norms = (start_feats ** 2).sum(dim=1).unsqueeze(0)
        dists = norms + start_norms - 2.0 * (reduced @ start_feats.T)
        dists.clamp_(min=0)
        min_dists = dists.mean(dim=1)
        selected = []
        for _ in range(n_select):
            idx = int(torch.argmax(min_dists).item())
            selected.append(idx)
            anchor = reduced[idx:idx + 1]
            anchor_norm = (anchor ** 2).sum(dim=1)
            d = norms.squeeze(1) + anchor_norm - 2.0 * torch.mv(reduced, anchor.squeeze(0))
            d.clamp_(min=0)
            min_dists = torch.minimum(min_dists, d)
        return features[np.array(selected, dtype=np.int64)]

    def predict(self, feats: Tuple[Tensor, ...], data_samples=None) -> List:
        assert self.memory_bank is not None, 'Call fit() before predict()'
        concat = torch.cat(feats, dim=1)
        B, C, H, W = concat.shape
        patches = self._patchify_and_aggregate(concat).cpu().numpy().astype(np.float32)
        patches = self._apply_dim_reduction(patches)

        # kNN search
        k = self.num_neighbors
        if _HAS_FAISS and self._nn_index is not None:
            distances, _ = self._nn_index.search(patches, k)
            distances = np.sqrt(np.maximum(distances, 0.0)).astype(np.float32)
        else:
            distances, _ = self._nn_index.kneighbors(patches, n_neighbors=k)

        patch_scores = distances[:, 0].reshape(B, H, W)
        device = feats[0].device
        results = []
        for i in range(B):
            raw_map = torch.from_numpy(patch_scores[i]).float().unsqueeze(0).unsqueeze(0).to(device)
            score_map = self._postprocess(raw_map)
            img_score = float(score_map.max().item())
            results.append(self._build_result(data_samples, i, img_score, score_map))
        return results

    @staticmethod
    def _build_result(data_samples, idx, img_score, score_map):
        if data_samples is not None and idx < len(data_samples):
            sample = data_samples[idx]
        else:
            sample = ADDataSample()
        sample.pred_score = img_score
        sample.pred_anomaly_map = score_map
        return sample


# ============================================================
# GaussianScoringHead (PaDiM tricks)
# ============================================================

@MODELS.register_module(force=True)
class GaussianScoringHead(BaseScoringHead):
    """Per-position Gaussian modeling with Mahalanobis distance scoring.

    Reimplements PaDiM's scoring pipeline as a composable head.

    Args:
        d_reduced: Number of dimensions after random selection.
        eps: Regularization for covariance matrix.
        feature_selection: 'full' or 'coreset'.
        dim_reduction: 'random' (PaDiM default), 'pca', or 'none'.
        coreset_ratio: Ratio when feature_selection='coreset'.
    """

    def __init__(
        self,
        d_reduced: int = 550,
        eps: float = 0.01,
        feature_selection: str = 'full',
        dim_reduction: str = 'random',
        coreset_ratio: float = 0.1,
        input_size: Tuple[int, int] = (256, 256),
        blur_sigma: float = 4.0,
        init_cfg=None,
    ):
        super().__init__(input_size=input_size, blur_sigma=blur_sigma, init_cfg=init_cfg)
        self.d_reduced = d_reduced
        self.eps = eps
        self.feature_selection = feature_selection
        self.dim_reduction = dim_reduction
        self.coreset_ratio = coreset_ratio

        self._features: List[Tensor] = []
        self.register_buffer('mean', None)
        self.register_buffer('cov_inv', None)
        self.register_buffer('idx', None)  # random dim selection indices
        self._total_dim: Optional[int] = None

    def _process_features(self, feats: Tuple[Tensor, ...]) -> Tensor:
        """Resize to layer1 size, concatenate, apply dim reduction."""
        # Resize all to first layer spatial size
        H, W = feats[0].shape[2], feats[0].shape[3]
        resized = [feats[0]]
        for f in feats[1:]:
            resized.append(F.interpolate(f, size=(H, W), mode='nearest'))
        feat = torch.cat(resized, dim=1)  # B, C_total, H, W

        # Dim reduction
        total_dim = feat.shape[1]
        if self.dim_reduction == 'random':
            if self.idx is None or self._total_dim != total_dim:
                rng = random.Random(42)
                d = min(self.d_reduced, total_dim)
                self.register_buffer(
                    'idx',
                    torch.tensor(rng.sample(range(total_dim), d), dtype=torch.long),
                )
                self._total_dim = total_dim
            feat = feat[:, self.idx]
        elif self.dim_reduction == 'pca':
            # PCA is applied in fit() after collection
            pass
        return feat

    def loss(self, feats: Tuple[Tensor, ...], data_samples=None) -> Dict[str, Tensor]:
        feat = self._process_features(feats)
        for i in range(feat.shape[0]):
            self._features.append(feat[i].cpu())
        return {'loss': torch.tensor(0.0, requires_grad=True)}

    def fit(self) -> None:
        if not self._features:
            return
        all_feats = torch.stack(self._features, dim=0)  # N, d, H, W

        # Optional coreset selection
        if self.feature_selection == 'coreset':
            N, d, H, W = all_feats.shape
            n_select = max(1, int(N * self.coreset_ratio))
            if N > n_select:
                # Simple random coreset for Gaussian (greedy on mean is overkill)
                rng = torch.Generator()
                rng.manual_seed(42)
                idx = torch.randperm(N, generator=rng)[:n_select]
                all_feats = all_feats[idx]

        N, d, H, W = all_feats.shape
        # Per-position Gaussian: (H*W, N, d)
        all_feats = all_feats.permute(2, 3, 0, 1).reshape(H * W, N, d)
        mean = all_feats.mean(dim=1)  # (H*W, d)
        diff = all_feats - mean.unsqueeze(1)
        cov = torch.bmm(diff.transpose(1, 2), diff) / max(N - 1, 1)
        cov += self.eps * torch.eye(d, device=cov.device).unsqueeze(0)
        cov_inv = torch.linalg.inv(cov)

        device = next(self.parameters()).device if len(list(self.parameters())) > 0 else cov_inv.device
        self.register_buffer('mean', mean.to(device))
        self.register_buffer('cov_inv', cov_inv.to(device))
        self._features.clear()

    def predict(self, feats: Tuple[Tensor, ...], data_samples=None) -> List:
        assert self.mean is not None, 'Call fit() before predict()'
        feat = self._process_features(feats)
        B, d, H, W = feat.shape
        device = feat.device
        feat_r = feat.permute(0, 2, 3, 1).reshape(B, H * W, d)
        mean = self.mean.to(device)
        cov_inv = self.cov_inv.to(device)
        diff = feat_r - mean.unsqueeze(0)
        maha = torch.einsum('bhd,hde,bhe->bh', diff, cov_inv, diff)
        maha = torch.sqrt(torch.clamp(maha, min=0))
        score_map = maha.reshape(B, 1, H, W)
        score_map = self._postprocess(score_map)
        img_scores = score_map.view(B, -1).max(dim=1).values
        return build_predict_results(data_samples, img_scores, score_map)


# ============================================================
# PCAScoringHead (DFM tricks)
# ============================================================

@MODELS.register_module(force=True)
class PCAScoringHead(BaseScoringHead):
    """PCA subspace modeling with reconstruction error scoring.

    Reimplements DFM's scoring pipeline as a composable head.

    Args:
        pca_level: Variance ratio to keep (0.97) or fixed component count (>1).
        scoring: 'fre' (feature reconstruction error) or 'nll' (Gaussian NLL).
        feature_selection: 'full' or 'coreset'.
        dim_reduction: 'none' or 'random'.
        coreset_ratio: Ratio when feature_selection='coreset'.
        pooling_kernel_size: Spatial avg pooling before flattening.
    """

    def __init__(
        self,
        pca_level: float = 0.97,
        scoring: str = 'fre',
        feature_selection: str = 'full',
        dim_reduction: str = 'none',
        coreset_ratio: float = 0.1,
        n_random_dims: int = 128,
        pooling_kernel_size: int = 4,
        input_size: Tuple[int, int] = (256, 256),
        blur_sigma: float = 0.0,
        init_cfg=None,
    ):
        super().__init__(input_size=input_size, blur_sigma=blur_sigma, init_cfg=init_cfg)
        self.pca_level = pca_level
        self.scoring = scoring
        self.feature_selection = feature_selection
        self.dim_reduction = dim_reduction
        self.coreset_ratio = coreset_ratio
        self.n_random_dims = n_random_dims
        self.pooling_kernel_size = pooling_kernel_size

        self._memory_bank: List[Tensor] = []
        self.register_buffer('singular_vectors', torch.empty(0))
        self.register_buffer('singular_values', torch.empty(0))
        self.register_buffer('pca_mean', torch.empty(0))
        self._feat_shape: Optional[Tuple[int, ...]] = None
        # NLL scoring state
        self.register_buffer('gauss_mean', torch.empty(0))
        self.register_buffer('gauss_u', torch.empty(0))
        self.register_buffer('gauss_sigma', torch.empty(0))
        # Random dim state
        self._random_idx: Optional[Tensor] = None

    def _extract_flat(self, feats: Tuple[Tensor, ...]) -> Tuple[Tensor, Tuple]:
        """Flatten features for PCA (use first layer)."""
        f = feats[0] if len(feats) == 1 else feats[-1]
        B = f.shape[0]
        if self.pooling_kernel_size > 1:
            f = F.avg_pool2d(f, kernel_size=self.pooling_kernel_size)
        shape = f.shape
        flat = f.view(B, -1).detach()

        if self.dim_reduction == 'random':
            D = flat.shape[1]
            if self._random_idx is None:
                rng = torch.Generator()
                rng.manual_seed(42)
                n = min(self.n_random_dims, D)
                self._random_idx = torch.randperm(D, generator=rng)[:n]
            flat = flat[:, self._random_idx]
        return flat, shape

    def loss(self, feats: Tuple[Tensor, ...], data_samples=None) -> Dict[str, Tensor]:
        flat, shape = self._extract_flat(feats)
        self._feat_shape = shape
        self._memory_bank.append(flat.cpu())
        return {'loss': torch.tensor(0.0, requires_grad=True)}

    def fit(self) -> None:
        if not self._memory_bank:
            return
        all_feats = torch.vstack(self._memory_bank)
        self._memory_bank.clear()

        # Coreset selection
        if self.feature_selection == 'coreset':
            N = all_feats.shape[0]
            n_select = max(1, int(N * self.coreset_ratio))
            if N > n_select:
                rng = torch.Generator()
                rng.manual_seed(42)
                idx = torch.randperm(N, generator=rng)[:n_select]
                all_feats = all_feats[idx]

        # PCA fitting
        mean = all_feats.mean(dim=0)
        centered = all_feats - mean
        _, sig, Vh = torch.linalg.svd(centered.double(), full_matrices=False)

        if self.pca_level <= 1.0:
            total_var = (sig ** 2).sum()
            ratios = torch.cumsum(sig ** 2, dim=0) / total_var
            qualified = torch.nonzero(ratios >= self.pca_level, as_tuple=False)
            n_components = int(qualified[0].item()) + 1 if len(qualified) > 0 else 1
        else:
            n_components = int(self.pca_level)

        self.register_buffer('singular_vectors', Vh.T[:, :n_components].float())
        self.register_buffer('singular_values', sig[:n_components].float())
        self.register_buffer('pca_mean', mean.float())

        # NLL Gaussian fitting
        if self.scoring == 'nll':
            projected = (all_feats.float() - self.pca_mean) @ self.singular_vectors
            self._fit_gaussian(projected.T)  # (d, N)

    def _fit_gaussian(self, dataset: Tensor) -> None:
        """Fit single-class Gaussian via SVD for NLL scoring."""
        N = dataset.shape[1]
        mean = dataset.mean(dim=1)
        centered = (dataset - mean.reshape(-1, 1)) / math.sqrt(N)
        u, sigma, _ = torch.linalg.svd(centered, full_matrices=False)
        self.register_buffer('gauss_mean', mean)
        self.register_buffer('gauss_u', u)
        self.register_buffer('gauss_sigma', sigma)

    def predict(self, feats: Tuple[Tensor, ...], data_samples=None) -> List:
        assert self.singular_vectors.numel() > 0, 'Call fit() before predict()'
        features, feat_shape = self._extract_flat(feats)
        B = features.shape[0]
        device = features.device

        sv = self.singular_vectors.to(device)
        pm = self.pca_mean.to(device)
        projected = (features.float() - pm) @ sv

        if self.scoring == 'nll':
            transformed = projected - self.gauss_mean.to(device)
            transformed = transformed @ (self.gauss_u.to(device) / self.gauss_sigma.to(device))
            scores = transformed.pow(2).sum(dim=1) + 2 * self.gauss_sigma.to(device).log().sum()
            # Image-level only for NLL
            score_map = torch.zeros(B, 1, self.input_size[0], self.input_size[1], device=device)
        else:
            reconstructed = projected @ sv.T + pm
            error = (features.float() - reconstructed).pow(2)
            # Reshape to spatial map
            C, H, W = feat_shape[1], feat_shape[2], feat_shape[3]
            if self.dim_reduction == 'random' and self._random_idx is not None:
                # Map back to full spatial with zeros for non-selected dims
                full_error = torch.zeros(B, C * H * W, device=device)
                full_error[:, self._random_idx] = error
                error = full_error
            score_map = error.reshape(B, C, H, W).mean(dim=1, keepdim=True)
            score_map = F.interpolate(score_map, size=self.input_size, mode='bilinear', align_corners=False)
            if self._blur is not None:
                score_map = self._blur(score_map)
            scores = error.reshape(B, C, H, W).mean(dim=[1, 2, 3])

        return build_predict_results(data_samples, scores, score_map)
