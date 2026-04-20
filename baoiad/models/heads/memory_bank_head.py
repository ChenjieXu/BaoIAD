"""Memory bank head for PatchCore-style anomaly detection."""

from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.model import BaseModule
from scipy.ndimage import gaussian_filter
from torch import Tensor

from baoiad.registry import MODELS
from baoiad.structures import ADDataSample

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    from sklearn.neighbors import NearestNeighbors
    HAS_FAISS = False

HAS_FAISS_GPU = (
    HAS_FAISS
    and hasattr(faiss, 'StandardGpuResources')
    and hasattr(faiss, 'index_cpu_to_gpu')
)


@MODELS.register_module()
class MemoryBankHead(BaseModule):
    """Memory bank head using coreset subsampling and kNN scoring.

    Args:
        coreset_ratio: Fraction of training features to keep in memory bank.
        num_neighbors: Number of nearest neighbors for anomaly scoring.
        distance: Distance metric ('euclidean').
    """

    def __init__(
        self,
        coreset_ratio: float = 0.1,
        num_neighbors: int = 1,
        distance: str = 'euclidean',
        patchsize: int = 3,
        patchstride: int = 1,
        reweight_scores: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
        blur_sigma: float = 4.0,
        image_score_source: str = 'postprocessed',
        patch_score_neighbors: int = 1,
        patch_score_reduction: str = 'first',
        coreset_sampling_method: str = 'approx_greedy',
        coreset_projection_dim: int = 128,
        coreset_starting_points: int = 10,
        coreset_device: str = 'auto',
        prefilter_max_size: int = 30000,
        prefilter_multiplier: int = 3,
        faiss_on_gpu: bool = False,
        faiss_gpu_id: int = 0,
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.coreset_ratio = coreset_ratio
        self.num_neighbors = num_neighbors
        self.distance = distance
        self.patchsize = patchsize
        self.patchstride = patchstride
        self.reweight_scores = reweight_scores
        self.input_size = input_size  # (H, W) for upsampling anomaly map
        self.blur_sigma = blur_sigma
        self.image_score_source = image_score_source
        self.patch_score_neighbors = patch_score_neighbors
        self.patch_score_reduction = patch_score_reduction
        self.coreset_sampling_method = coreset_sampling_method
        self.coreset_projection_dim = coreset_projection_dim
        self.coreset_starting_points = coreset_starting_points
        self.coreset_device = coreset_device
        self.prefilter_max_size = prefilter_max_size
        self.prefilter_multiplier = prefilter_multiplier
        self.faiss_on_gpu = faiss_on_gpu
        self.faiss_gpu_id = faiss_gpu_id

        # Will be populated during training phase
        self.memory_bank: Optional[np.ndarray] = None
        self._nn_index = None  # faiss.IndexFlatL2 or sklearn NearestNeighbors
        self._faiss_resources = None

        # Buffer to collect training features
        self._train_features: List[np.ndarray] = []

        # Persist the memory bank inside state_dict so that checkpoints round-trip
        # correctly. The FAISS / sklearn index is rebuilt on load from the bank.
        self._register_state_dict_hook(self._save_memory_bank_to_state_dict)
        self._register_load_state_dict_pre_hook(self._load_memory_bank_from_state_dict)

    @staticmethod
    def _save_memory_bank_to_state_dict(module, state_dict, prefix, local_metadata):
        if module.memory_bank is not None:
            state_dict[prefix + 'memory_bank'] = torch.from_numpy(
                np.ascontiguousarray(module.memory_bank.astype(np.float32))
            )
        return state_dict

    def _load_memory_bank_from_state_dict(
        self, state_dict, prefix, local_metadata,
        strict, missing_keys, unexpected_keys, error_msgs,
    ):
        key = prefix + 'memory_bank'
        if key in state_dict:
            tensor = state_dict.pop(key)
            self.memory_bank = tensor.detach().cpu().numpy().astype(np.float32)
            self._build_nn_index_from_bank()

    def _build_nn_index_from_bank(self) -> None:
        if self.memory_bank is None:
            return
        data = np.ascontiguousarray(self.memory_bank.astype(np.float32))
        if HAS_FAISS:
            dim = data.shape[1]
            self._nn_index = faiss.IndexFlatL2(dim)
            if self.faiss_on_gpu and HAS_FAISS_GPU:
                try:
                    self._faiss_resources = faiss.StandardGpuResources()
                    self._nn_index = faiss.index_cpu_to_gpu(
                        self._faiss_resources, self.faiss_gpu_id, self._nn_index)
                except Exception as exc:
                    warnings.warn(
                        f'Failed to initialize GPU FAISS index on load: {exc}. '
                        'Falling back to CPU IndexFlatL2.',
                        RuntimeWarning,
                    )
                    self._nn_index = faiss.IndexFlatL2(dim)
                    self._faiss_resources = None
            self._nn_index.add(data)
        else:
            self._nn_index = NearestNeighbors(
                n_neighbors=self.num_neighbors,
                metric='euclidean',
                algorithm='auto',
            )
            self._nn_index.fit(data)

    def _patchify_and_aggregate(self, concat: Tensor) -> Tensor:
        """Apply patchsize-neighborhood aggregation (PatchCore paper, patchsize=3).

        For each spatial position, unfold a patchsize x patchsize neighborhood,
        then average-pool over the neighborhood to get the patch embedding.

        Args:
            concat: (B, C, H, W) concatenated multi-scale features.

        Returns:
            (B*H*W, C) patch embeddings with neighborhood context.
        """
        B, C, H, W = concat.shape
        if self.patchsize <= 1:
            return concat.permute(0, 2, 3, 1).reshape(-1, C)

        padding = (self.patchsize - 1) // 2
        # Unfold: (B, C * patchsize * patchsize, H * W)
        unfolded = F.unfold(
            concat,
            kernel_size=self.patchsize,
            stride=self.patchstride,
            padding=padding,
        )
        # Reshape to (B, C, patchsize*patchsize, H*W)
        unfolded = unfolded.reshape(B, C, self.patchsize * self.patchsize, -1)
        # Average over neighborhood: (B, C, H*W)
        aggregated = unfolded.mean(dim=2)
        # (B, H*W, C) -> (B*H*W, C)
        aggregated = aggregated.permute(0, 2, 1).reshape(-1, C)
        return aggregated

    def collect_features(self, feats: Tuple[Tensor, ...]) -> None:
        """Collect patch features during training (feature extraction phase).

        Args:
            feats: Tuple of pooled feature tensors, each (B, C, H, W).
        """
        concat = torch.cat(feats, dim=1)  # (B, C_total, H, W)
        patches = self._patchify_and_aggregate(concat)
        self._train_features.append(patches.cpu().numpy())

    def build_memory_bank(self) -> None:
        """Build coreset memory bank from collected training features."""
        if not self._train_features:
            raise RuntimeError('No training features collected for memory bank.')

        all_features = np.concatenate(self._train_features, axis=0)
        self._train_features.clear()

        n_total = all_features.shape[0]
        n_coreset = max(1, int(n_total * self.coreset_ratio))

        # For large patch banks, sample a manageable candidate pool first.
        rng = np.random.default_rng(42)
        pre_n = min(
            n_total,
            max(n_coreset * self.prefilter_multiplier, self.prefilter_max_size),
        )
        if n_total > pre_n:
            indices = rng.choice(n_total, pre_n, replace=False)
            coreset_source = all_features[indices]
        else:
            coreset_source = all_features

        if coreset_source.shape[0] > n_coreset:
            if self.coreset_sampling_method == 'approx_greedy':
                self.memory_bank = self._approximate_coreset_sampling(
                    coreset_source,
                    n_coreset,
                )
            elif self.coreset_sampling_method == 'greedy':
                self.memory_bank = self._coreset_sampling(coreset_source, n_coreset)
            elif self.coreset_sampling_method == 'random':
                indices = rng.choice(coreset_source.shape[0], n_coreset, replace=False)
                self.memory_bank = coreset_source[indices]
            else:
                raise ValueError(
                    f'Unsupported coreset_sampling_method: {self.coreset_sampling_method}')
        else:
            self.memory_bank = coreset_source

        if self.faiss_on_gpu and HAS_FAISS and not HAS_FAISS_GPU:
            warnings.warn(
                'faiss_on_gpu=True but GPU FAISS symbols are unavailable; '
                'falling back to CPU IndexFlatL2.',
                RuntimeWarning,
            )
        self._build_nn_index_from_bank()

    @staticmethod
    def _coreset_sampling(features: np.ndarray, n_samples: int) -> np.ndarray:
        """Greedy coreset subsampling via iterative farthest-point sampling.

        Uses chunked distance computation to avoid massive temporary arrays.

        Args:
            features: (N, D) feature array.
            n_samples: Number of samples to select.

        Returns:
            (n_samples, D) selected features.
        """
        if n_samples >= features.shape[0]:
            return features

        N, D = features.shape
        features = features.astype(np.float32)

        # Precompute squared norms for efficient distance computation
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        norms_sq = np.sum(features ** 2, axis=1)  # (N,)

        rng = np.random.default_rng(42)
        selected = [rng.integers(N)]
        min_distances = np.full(N, np.inf, dtype=np.float32)

        for _ in range(n_samples - 1):
            last_idx = selected[-1]
            last = features[last_idx]  # (D,)
            # ||features - last||^2 = norms_sq + ||last||^2 - 2 * features @ last
            dists = norms_sq + norms_sq[last_idx] - 2.0 * (features @ last)
            np.maximum(dists, 0, out=dists)  # clamp numerical noise
            np.minimum(min_distances, dists, out=min_distances)
            selected.append(int(np.argmax(min_distances)))

        return features[np.array(selected)]

    def _resolve_coreset_device(self) -> torch.device:
        """Pick the execution device for approximate coreset sampling."""
        if self.coreset_device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.coreset_device)

    def _approximate_coreset_sampling(
        self,
        features: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """Approximate PatchCore coreset with projection + greedy updates.

        This follows the reference idea more closely than the full greedy CPU path
        while remaining practical on large memory banks.
        """
        if n_samples >= features.shape[0]:
            return features

        device = self._resolve_coreset_device()
        features_t = torch.from_numpy(
            np.ascontiguousarray(features.astype(np.float32))).to(device)
        N, D = features_t.shape

        gen = torch.Generator(device=device)
        gen.manual_seed(42)
        if D > self.coreset_projection_dim:
            projection = torch.randn(
                (D, self.coreset_projection_dim),
                device=device,
                generator=gen,
                dtype=features_t.dtype,
            )
            reduced = features_t @ projection
        else:
            reduced = features_t

        start_count = min(self.coreset_starting_points, N)
        start_points = torch.randperm(N, generator=gen, device=device)[:start_count]

        start_features = reduced[start_points]
        reduced_norms = torch.sum(reduced ** 2, dim=1, keepdim=True)
        start_norms = torch.sum(start_features ** 2, dim=1).unsqueeze(0)
        start_dists = reduced_norms + start_norms - 2.0 * (reduced @ start_features.T)
        start_dists.clamp_(min=0.0)
        min_distances = torch.mean(start_dists, dim=1)

        selected = []
        for _ in range(n_samples):
            select_idx = int(torch.argmax(min_distances).item())
            selected.append(select_idx)
            anchor = reduced[select_idx:select_idx + 1]
            anchor_norm = torch.sum(anchor ** 2, dim=1)
            dists = reduced_norms.squeeze(1) + anchor_norm - 2.0 * torch.mv(reduced, anchor.squeeze(0))
            dists.clamp_(min=0.0)
            min_distances = torch.minimum(min_distances, dists)

        return features[np.array(selected, dtype=np.int64)]

    def loss(
        self,
        feats: Tuple[Tensor, ...],
        data_samples: Optional[List] = None,
    ) -> Dict[str, Tensor]:
        """PatchCore has no training loss; collect features instead.

        Returns:
            Empty dict (no gradient-based training).
        """
        self.collect_features(feats)
        # Return dummy zero loss (requires_grad for mmengine backward compatibility)
        dummy = sum(p.sum() * 0 for p in self.parameters()) if len(list(self.parameters())) > 0 else torch.tensor(0.0, requires_grad=True)
        return {'loss': dummy}

    def _compute_image_score(self, score_map: Tensor) -> float:
        """Aggregate raw patch scores into an image-level anomaly score."""
        flat = score_map.flatten()
        if self.reweight_scores:
            weights = torch.softmax(flat, dim=0)
            return float((flat * weights).sum())
        return float(flat.max())

    @staticmethod
    def _euclidean_dist(x: Tensor, y: Tensor) -> Tensor:
        """Compute Euclidean distance with broadcasted batch dimensions."""
        x_norm = x.pow(2).sum(dim=-1, keepdim=True)
        y_norm = y.pow(2).sum(dim=-1, keepdim=True)
        dist = -2.0 * torch.matmul(x, y.transpose(-2, -1))
        dist = dist + x_norm + y_norm.transpose(-2, -1)
        return dist.clamp_min_(0).sqrt_()

    def _search_nn(self, queries: np.ndarray, n_neighbors: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search nearest neighbors and return Euclidean distances."""
        k = max(1, int(n_neighbors))
        if HAS_FAISS and self._nn_index is not None:
            distances, locations = self._nn_index.search(queries, k)
            distances = np.sqrt(np.maximum(distances, 0.0)).astype(np.float32, copy=False)
        elif self._nn_index is not None:
            distances, locations = self._nn_index.kneighbors(queries, n_neighbors=k)
        else:
            bank = np.ascontiguousarray(self.memory_bank.astype(np.float32))
            diff = queries[:, None, :] - bank[None, :, :]
            distances = np.linalg.norm(diff, axis=-1)
            locations = np.argsort(distances, axis=1)[:, :k]
            distances = np.take_along_axis(distances, locations, axis=1)
        return distances.astype(np.float32, copy=False), locations

    def _compute_weighted_patchcore_score(
        self,
        patch_scores: Tensor,
        locations: np.ndarray,
        patch_embeddings: np.ndarray,
    ) -> Tensor:
        """Compute anomalib-style weighted image score from patch scores."""
        batch_size, num_patches = patch_scores.shape
        if self.num_neighbors <= 1:
            return patch_scores.max(dim=1).values

        max_patches = torch.argmax(patch_scores, dim=1)
        batch_indices = torch.arange(batch_size, device=patch_scores.device)
        score = patch_scores[batch_indices, max_patches]

        patch_embeddings_t = torch.from_numpy(patch_embeddings).to(
            device=patch_scores.device, dtype=patch_scores.dtype)
        max_patch_features = patch_embeddings_t[batch_indices, max_patches]

        nn_index = locations[batch_indices.cpu().numpy(), max_patches.cpu().numpy()]
        nn_samples = np.ascontiguousarray(self.memory_bank[nn_index].astype(np.float32))
        support_k = min(self.num_neighbors, self.memory_bank.shape[0])
        _, support_locations = self._search_nn(nn_samples, support_k)
        support_bank = torch.from_numpy(
            np.ascontiguousarray(self.memory_bank[support_locations].astype(np.float32))
        ).to(device=patch_scores.device, dtype=patch_scores.dtype)

        distances = self._euclidean_dist(
            max_patch_features.unsqueeze(1),
            support_bank,
        ).squeeze(1)
        weights = (1 - torch.softmax(distances, dim=1))[:, 0]
        return weights * score

    def _reduce_patch_scores(self, distances: np.ndarray) -> np.ndarray:
        """Reduce kNN distances into a single patch score."""
        if self.patch_score_reduction == 'first':
            return distances[:, 0]
        if self.patch_score_reduction == 'mean':
            return distances.mean(axis=1)
        raise ValueError(f'Unsupported patch_score_reduction: {self.patch_score_reduction}')

    def _postprocess_score_map(self, score_map: Tensor) -> Tensor:
        """Upsample and smooth the patch-level score map for pixel metrics."""
        if self.input_size is not None:
            score_map = F.interpolate(
                score_map,
                size=self.input_size,
                mode='bilinear',
                align_corners=False,
            )

        score_map = score_map.squeeze(0)  # (1, H, W)
        if self.blur_sigma > 0:
            smoothed = gaussian_filter(
                score_map.squeeze(0).cpu().numpy(),
                sigma=self.blur_sigma,
            ).astype(np.float32, copy=False)
            score_map = torch.from_numpy(np.ascontiguousarray(smoothed)).unsqueeze(0)

        return score_map

    def _upsample_without_blur(self, score_map: Tensor) -> Tensor:
        """Upsample the patch-level score map without smoothing."""
        if self.input_size is not None:
            score_map = F.interpolate(
                score_map,
                size=self.input_size,
                mode='bilinear',
                align_corners=False,
            )
        return score_map.squeeze(0)

    def predict(
        self,
        feats: Tuple[Tensor, ...],
        data_samples: Optional[List] = None,
    ) -> List:
        """Predict anomaly scores and maps.

        Args:
            feats: Tuple of pooled feature tensors, each (B, C, H, W).
            data_samples: List of data samples to attach predictions to.

        Returns:
            Updated list of data samples with anomaly predictions.
        """
        assert self._nn_index is not None, 'Memory bank not built yet.'

        concat = torch.cat(feats, dim=1)  # (B, C_total, H, W)
        B, C, H, W = concat.shape
        patches_np = self._patchify_and_aggregate(concat).cpu().numpy().astype(np.float32)

        # kNN search for patch map generation. The image-level score can still
        # use a separate weighted scheme when enabled.
        distances, locations = self._search_nn(patches_np, self.patch_score_neighbors)
        patch_scores = self._reduce_patch_scores(distances).reshape(B, H, W)
        patch_locations = locations.reshape(B, H * W, -1)
        patch_embeddings = patches_np.reshape(B, H * W, -1)

        results = []
        for i in range(B):
            raw_score_map = torch.from_numpy(patch_scores[i]).float()
            score_map = self._postprocess_score_map(
                raw_score_map.unsqueeze(0).unsqueeze(0))
            if self.reweight_scores:
                image_score = float(self._compute_weighted_patchcore_score(
                    raw_score_map.view(1, -1),
                    patch_locations[i:i + 1, :, 0],
                    patch_embeddings[i:i + 1],
                )[0].item())
            else:
                if self.image_score_source == 'postprocessed':
                    image_score = float(score_map.max().item())
                elif self.image_score_source == 'upsampled':
                    image_score = float(self._upsample_without_blur(
                        raw_score_map.unsqueeze(0).unsqueeze(0)
                    ).max().item())
                elif self.image_score_source == 'raw':
                    image_score = float(raw_score_map.max().item())
                else:
                    raise ValueError(
                        f'Unsupported image_score_source: {self.image_score_source}')

            result = data_samples[i] if data_samples else ADDataSample()
            result.pred_score = image_score
            result.pred_anomaly_map = score_map
            results.append(result)

        return results

    def forward(self, feats: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        """Tensor mode: return features as-is."""
        return feats
