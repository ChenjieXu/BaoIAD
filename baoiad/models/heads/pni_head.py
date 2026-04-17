"""PNI head aligned to the official Position and Neighborhood Information flow."""

from typing import Dict, List, Optional, Tuple

import copy
import gc
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from scipy.ndimage import gaussian_filter
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, random_split

try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from sklearn.random_projection import SparseRandomProjection

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from baoiad.registry import MODELS
from baoiad.structures import ADDataSample


@MODELS.register_module()
class PNIHead(BaseModule):
    """PNI head with official patch-embedding, coreset, and nb+coor scoring."""

    def __init__(
        self,
        coreset_ratio: float = 0.01,
        distribution_size: int = 2048,
        neighborhood_size: int = 9,
        mlp_layers: int = 10,
        mlp_channels: int = 2048,
        temperature: float = 2.0,
        lambda_param: float = 1.0,
        num_neighbors: int = 3,
        input_size: Optional[Tuple[int, int]] = None,
        mlp_epochs: int = 15,
        mlp_lr: float = 1e-3,
        prob_gamma: float = 0.99,
        softmax_nb_gamma: float = 0.5,
        softmax_coor_gamma: float = 0.5,
        blur_sigma: float = 8.0,
        mlp_batch_size: int = 2048,
        max_train_samples: int = 0,
        candidate_neighbors: int = 100,
        patchsize: int = 5,
        patchstride: int = 1,
        pretrain_embed_dimension: int = 1024,
        target_embed_dimension: int = 1024,
        approximate_coreset: bool = False,
        mlp_val_ratio: float = 0.1,
        coreset_prefilter_size: int = 12000,
        coreset_projection_dim: int = 128,
        search_chunk_size: int = 1024,
        histogram_chunk_size: int = 16,
        log_predict_stats: bool = False,
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.coreset_ratio = coreset_ratio
        self.distribution_size = distribution_size
        self.neighborhood_size = neighborhood_size
        self.mlp_layers = mlp_layers
        self.mlp_channels = mlp_channels
        self.temperature = temperature
        self.lambda_param = lambda_param
        self.num_neighbors = num_neighbors
        self.input_size = input_size
        self.mlp_epochs = mlp_epochs
        self.mlp_lr = mlp_lr
        self.prob_gamma = prob_gamma
        self.softmax_nb_gamma = softmax_nb_gamma
        self.softmax_coor_gamma = softmax_coor_gamma
        self.blur_sigma = blur_sigma
        self.mlp_batch_size = mlp_batch_size
        self.max_train_samples = max_train_samples
        self.candidate_neighbors = candidate_neighbors
        self.patchsize = patchsize
        self.patchstride = patchstride
        self.pretrain_embed_dimension = pretrain_embed_dimension
        self.target_embed_dimension = target_embed_dimension
        self.approximate_coreset = approximate_coreset
        self.mlp_val_ratio = mlp_val_ratio
        self.coreset_prefilter_size = coreset_prefilter_size
        self.coreset_projection_dim = coreset_projection_dim
        self.search_chunk_size = search_chunk_size
        self.histogram_chunk_size = histogram_chunk_size
        self.log_predict_stats = log_predict_stats

        self.patch_padding = (self.patchsize - 1) // 2
        self.dist_padding = (self.neighborhood_size - 1) // 2

        self.embedding_coreset: Optional[np.ndarray] = None
        self.dist_coreset: Optional[np.ndarray] = None
        self.embedding_coreset_with_edge: Optional[np.ndarray] = None

        self._embedding_index = None
        self._embedding_index_with_edge = None
        self._dist_index = None
        self._faiss_gpu_resources = None

        self.coor_model: Optional[np.ndarray] = None
        self.coor_model_with_edge: Optional[np.ndarray] = None

        self.emb_to_dist: Optional[np.ndarray] = None

        self.mlp: Optional[nn.Module] = None

        self._train_features: List[np.ndarray] = []
        self._train_features_with_edge: List[np.ndarray] = []
        self._spatial_shape: Optional[Tuple[int, int]] = None
        self._valid_spatial_shape: Optional[Tuple[int, int]] = None
        self._feature_dim: Optional[int] = None
        self.last_build_info: Dict[str, object] = {}
        self.last_predict_summary: Optional[Dict[str, float]] = None
        self.last_scoring_summary: Optional[Dict[str, float]] = None
        self.last_mlp_fit_info: Optional[Dict[str, float]] = None

    def _patchify(self, features: Tensor) -> Tuple[Tensor, List[int]]:
        """Extract patch neighborhoods around each spatial location."""
        padding = self.patch_padding
        unfolded = F.unfold(
            features,
            kernel_size=self.patchsize,
            stride=self.patchstride,
            padding=padding,
        )
        patch_shape = []
        for size in features.shape[-2:]:
            n_patches = (
                size + 2 * padding - (self.patchsize - 1) - 1
            ) / self.patchstride + 1
            patch_shape.append(int(n_patches))
        unfolded = unfolded.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded = unfolded.permute(0, 4, 1, 2, 3).contiguous()
        return unfolded, patch_shape

    def _generate_embeddings(self, feats: Tuple[Tensor, ...]) -> Tensor:
        """Generate official PNI patch embeddings for a multi-scale feature tuple."""
        features = [self._patchify(feat) for feat in feats]
        patch_shapes = [item[1] for item in features]
        patch_tensors = [item[0] for item in features]

        ref_num_patches = patch_shapes[0]
        for idx in range(1, len(patch_tensors)):
            layer_features = patch_tensors[idx]
            layer_shape = patch_shapes[idx]
            layer_features = layer_features.reshape(
                layer_features.shape[0],
                layer_shape[0],
                layer_shape[1],
                *layer_features.shape[2:],
            )
            layer_features = layer_features.permute(0, -3, -2, -1, 1, 2)
            base_shape = layer_features.shape
            layer_features = layer_features.reshape(-1, *layer_features.shape[-2:])
            layer_features = F.interpolate(
                layer_features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)
            layer_features = layer_features.reshape(
                *base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            layer_features = layer_features.permute(0, -2, -1, 1, 2, 3)
            layer_features = layer_features.reshape(
                len(layer_features), -1, *layer_features.shape[-3:]
            )
            patch_tensors[idx] = layer_features

        flat_layers = [item.reshape(-1, *item.shape[-3:]) for item in patch_tensors]
        pooled_layers = []
        for item in flat_layers:
            item = item.reshape(len(item), 1, -1)
            item = F.adaptive_avg_pool1d(item, self.pretrain_embed_dimension).squeeze(1)
            pooled_layers.append(item)

        embeddings = torch.stack(pooled_layers, dim=1)
        embeddings = embeddings.reshape(len(embeddings), 1, -1)
        embeddings = F.adaptive_avg_pool1d(
            embeddings, self.target_embed_dimension
        ).reshape(len(embeddings), -1)

        batch_size = feats[0].shape[0]
        return embeddings.reshape(
            batch_size,
            ref_num_patches[0],
            ref_num_patches[1],
            self.target_embed_dimension,
        )

    def _crop_valid_embeddings(self, embeddings: Tensor) -> Tensor:
        if self.patch_padding == 0:
            return embeddings
        return embeddings[
            :,
            self.patch_padding:-self.patch_padding,
            self.patch_padding:-self.patch_padding,
            :,
        ]

    def collect_features(self, feats: Tuple[Tensor, ...]) -> None:
        """Collect patch embeddings for memory-bank fitting."""
        embeddings = self._generate_embeddings(feats)
        valid_embeddings = self._crop_valid_embeddings(embeddings)

        if self._spatial_shape is None:
            _, full_h, full_w, embed_dim = embeddings.shape
            _, valid_h, valid_w, _ = valid_embeddings.shape
            self._spatial_shape = (full_h, full_w)
            self._valid_spatial_shape = (valid_h, valid_w)
            self._feature_dim = embed_dim

        self._train_features_with_edge.append(
            embeddings.detach().cpu().numpy().astype(np.float32)
        )
        self._train_features.append(
            valid_embeddings.detach().cpu().numpy().astype(np.float32)
        )

    def build_memory_bank(self) -> None:
        """Build dual coreset, coordinate models, and neighborhood MLP."""
        if self._dist_index is not None:
            print('  Memory bank already built, skipping...', flush=True)
            return

        if not self._train_features:
            print('  No features collected, skipping build...', flush=True)
            return

        print('  [1/7] Concatenating embeddings...', flush=True)
        valid_features = np.concatenate(self._train_features, axis=0)
        full_features = np.concatenate(self._train_features_with_edge, axis=0)

        n_valid, valid_h, valid_w, embed_dim = valid_features.shape
        n_full, full_h, full_w, _ = full_features.shape
        print(
            f'  Valid embeddings: N={n_valid}, H={valid_h}, W={valid_w}, C={embed_dim}',
            flush=True,
        )
        print(
            f'  Full embeddings: N={n_full}, H={full_h}, W={full_w}, C={embed_dim}',
            flush=True,
        )

        self._spatial_shape = (full_h, full_w)
        self._valid_spatial_shape = (valid_h, valid_w)

        stage_times = {}

        print('  [2/7] Building dual coreset...', flush=True)
        stage_start = time.perf_counter()
        self._build_dual_coreset(valid_features, full_features)
        stage_times['dual_coreset_s'] = time.perf_counter() - stage_start
        gc.collect()

        print('  [3/7] Building coordinate model (without edge)...', flush=True)
        stage_start = time.perf_counter()
        self._build_position_histograms(full_features, with_edge=False)
        stage_times['coor_without_edge_s'] = time.perf_counter() - stage_start
        gc.collect()

        print('  [4/7] Building coordinate model (with edge)...', flush=True)
        stage_start = time.perf_counter()
        self._build_position_histograms(full_features, with_edge=True)
        stage_times['coor_with_edge_s'] = time.perf_counter() - stage_start
        gc.collect()

        print('  [5/7] Training neighborhood MLP...', flush=True)
        stage_start = time.perf_counter()
        self._build_neighborhood_mlp(full_features)
        stage_times['mlp_train_s'] = time.perf_counter() - stage_start
        gc.collect()

        print('  [6/7] Building search indices...', flush=True)
        stage_start = time.perf_counter()
        self._build_faiss_indices()
        stage_times['index_build_s'] = time.perf_counter() - stage_start

        print('  [7/7] Building embedding-to-distribution mapping...', flush=True)
        stage_start = time.perf_counter()
        self._build_emb_to_dist_mapping()
        stage_times['emb_to_dist_s'] = time.perf_counter() - stage_start

        self._train_features.clear()
        self._train_features_with_edge.clear()
        gc.collect()
        self.last_build_info = {
            'valid_shape': (n_valid, valid_h, valid_w, embed_dim),
            'full_shape': (n_full, full_h, full_w, embed_dim),
            'embedding_coreset_shape': tuple(self.embedding_coreset.shape),
            'dist_coreset_shape': tuple(self.dist_coreset.shape),
            'embedding_coreset_with_edge_shape': tuple(self.embedding_coreset_with_edge.shape),
            'coor_model_shape': tuple(self.coor_model.shape) if self.coor_model is not None else None,
            'coor_model_with_edge_shape': tuple(self.coor_model_with_edge.shape)
            if self.coor_model_with_edge is not None else None,
            'emb_to_dist_shape': tuple(self.emb_to_dist.shape) if self.emb_to_dist is not None else None,
            'approximate_coreset': self.approximate_coreset,
            'mlp_fit_info': self.last_mlp_fit_info,
            'stage_times': stage_times,
        }
        print(
            '  stage timing: '
            + ', '.join(f'{k}={v:.2f}s' for k, v in stage_times.items()),
            flush=True,
        )
        print('  build_memory_bank() completed!', flush=True)

    def _select_coreset_indices(
        self, features: np.ndarray, sample_size: int
    ) -> np.ndarray:
        """Select coreset indices with official or bounded sampling."""
        if self.approximate_coreset:
            return self._select_coreset_indices_bounded(features, sample_size)
        return self._select_coreset_indices_official(features, sample_size)

    def _select_coreset_indices_official(
        self, features: np.ndarray, sample_size: int
    ) -> np.ndarray:
        """Official k-center-greedy selection with sparse random projection."""
        n_total = features.shape[0]
        if sample_size >= n_total:
            return np.arange(n_total)

        flat_features = features.reshape(n_total, -1).astype(np.float32)
        devices = [torch.device('cuda')] if torch.cuda.is_available() else []
        devices.append(torch.device('cpu'))
        last_exc = None

        for device in devices:
            try:
                if HAS_SKLEARN and flat_features.shape[1] > 1:
                    projector = SparseRandomProjection(n_components='auto', eps=0.9)
                    projected = np.asarray(
                        projector.fit_transform(flat_features), dtype=np.float32
                    )
                    if projected.ndim == 1:
                        projected = projected[:, None]
                    feature_tensor = self._move_numpy_to_device(projected, device)
                else:
                    feature_tensor = self._move_numpy_to_device(flat_features, device)

                norms = torch.sum(feature_tensor * feature_tensor, dim=1)
                min_distances = torch.full((n_total,), float('inf'), device=device)
                selected = []
                idx = int(torch.randint(high=n_total, size=(1,), device=device).item())

                for step in range(sample_size):
                    center = feature_tensor[idx]
                    distances = norms + norms[idx] - 2.0 * (feature_tensor @ center)
                    min_distances = torch.minimum(min_distances, distances)
                    selected.append(idx)
                    min_distances[idx] = 0.0
                    idx = int(torch.argmax(min_distances).item())

                    if (step + 1) % 512 == 0 or step + 1 == sample_size:
                        print(
                            f'      coreset selected {step + 1}/{sample_size}',
                            flush=True,
                        )

                return np.asarray(selected, dtype=np.int64)
            except (RuntimeError, torch.AcceleratorError) as exc:
                if 'out of memory' not in str(exc).lower() or device.type != 'cuda':
                    raise
                last_exc = exc
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

        if last_exc is not None:
            raise last_exc
        raise RuntimeError('Failed to select official coreset indices.')

    def _select_coreset_indices_bounded(
        self, features: np.ndarray, sample_size: int
    ) -> np.ndarray:
        """Approximate farthest-point coreset on a deterministic prefiltered subset."""
        n_total = features.shape[0]
        if sample_size >= n_total:
            return np.arange(n_total)

        rng = np.random.default_rng(42)
        prefilter_size = min(n_total, max(self.coreset_prefilter_size, sample_size * 4))
        if prefilter_size < n_total:
            pre_idx = np.sort(rng.choice(n_total, size=prefilter_size, replace=False))
            subset = features[pre_idx]
        else:
            pre_idx = np.arange(n_total)
            subset = features

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        subset_t = torch.from_numpy(subset.astype(np.float32)).to(device)

        proj_dim = min(self.coreset_projection_dim, subset_t.shape[1])
        if proj_dim > 0 and subset_t.shape[1] > proj_dim:
            gen = torch.Generator(device=device)
            gen.manual_seed(42)
            projection = torch.randn(
                subset_t.shape[1],
                proj_dim,
                device=device,
                generator=gen,
                dtype=subset_t.dtype,
            )
            subset_t = subset_t @ projection
            subset_t = subset_t / float(proj_dim) ** 0.5

        norms = torch.sum(subset_t * subset_t, dim=1)
        min_distances = torch.full((subset_t.shape[0],), float('inf'), device=device)
        selected = []
        idx = int(rng.integers(subset_t.shape[0]))

        for step in range(sample_size):
            center = subset_t[idx]
            dists = norms + norms[idx] - 2.0 * (subset_t @ center)
            min_distances = torch.minimum(min_distances, dists)
            selected.append(idx)
            min_distances[idx] = 0
            idx = int(torch.argmax(min_distances).item())

            if (step + 1) % 512 == 0 or step + 1 == sample_size:
                print(
                    f'      coreset selected {step + 1}/{sample_size}',
                    flush=True,
                )

        selected = np.asarray(selected, dtype=np.int64)
        return pre_idx[selected]

    def _build_dual_coreset(
        self, valid_features: np.ndarray, full_features: np.ndarray
    ) -> None:
        """Build embedding/dist coresets from official valid/full embeddings."""
        flat_valid = valid_features.reshape(-1, valid_features.shape[-1]).astype(np.float32)
        flat_full = full_features.reshape(-1, full_features.shape[-1]).astype(np.float32)

        n_total = flat_valid.shape[0]
        n_emb = max(1, int(n_total * self.coreset_ratio))
        n_dist = min(self.distribution_size, n_total)
        max_size = max(n_emb, n_dist)

        print(
            f'    Without edge: n_total={n_total}, n_emb={n_emb}, n_dist={n_dist}',
            flush=True,
        )
        if max_size >= n_total:
            selected_idx = np.arange(n_total)
        else:
            selected_idx = self._select_coreset_indices(flat_valid, max_size)

        self.embedding_coreset = flat_valid[selected_idx[:n_emb]].astype(np.float32)
        self.dist_coreset = flat_valid[selected_idx[:n_dist]].astype(np.float32)

        n_total_with_edge = flat_full.shape[0]
        n_emb_with_edge = max(1, int(n_total_with_edge * self.coreset_ratio))
        print(
            f'    With edge: n_total={n_total_with_edge}, n_emb={n_emb_with_edge}',
            flush=True,
        )
        if n_emb_with_edge >= n_total_with_edge:
            selected_idx_with_edge = np.arange(n_total_with_edge)
        else:
            selected_idx_with_edge = self._select_coreset_indices(
                flat_full, n_emb_with_edge
            )

        self.embedding_coreset_with_edge = flat_full[selected_idx_with_edge].astype(
            np.float32
        )

    def _search_index(
        self, index, queries: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search a FAISS index or a raw array-backed fallback."""
        if hasattr(index, 'search'):
            distances, indices = index.search(queries.astype(np.float32), k)
            distances = np.sqrt(np.maximum(distances, 0.0))
            return distances.astype(np.float32), indices.astype(np.int32)

        if hasattr(index, 'kneighbors'):
            distances, indices = index.kneighbors(
                queries.astype(np.float32), n_neighbors=k
            )
            return distances.astype(np.float32), indices.astype(np.int32)

        queries_t = torch.from_numpy(queries.astype(np.float32))
        if torch.is_tensor(index):
            index_t = index
        else:
            index_t = torch.from_numpy(index.astype(np.float32))
        if index_t.device.type != queries_t.device.type:
            queries_t = queries_t.to(index_t.device, non_blocking=True)
        elif torch.cuda.is_available() and queries_t.device.type != 'cuda':
            queries_t = queries_t.cuda(non_blocking=True)
            index_t = index_t.cuda(non_blocking=True)

        distances_all = []
        indices_all = []
        chunk_size = self.search_chunk_size
        for start in range(0, len(queries_t), chunk_size):
            chunk = queries_t[start:start + chunk_size]
            distances = torch.cdist(chunk, index_t)
            values, indices = torch.topk(distances, k=k, largest=False, dim=1)
            distances_all.append(values.cpu())
            indices_all.append(indices.cpu())

        distances = torch.cat(distances_all, dim=0).numpy().astype(np.float32)
        indices = torch.cat(indices_all, dim=0).numpy().astype(np.int32)
        return distances, indices

    def _build_position_histograms(
        self, full_features: np.ndarray, with_edge: bool = False
    ) -> None:
        """Build coordinate probability tables on the full patch grid."""
        if not self.approximate_coreset:
            self._build_position_histograms_exact(full_features, with_edge=with_edge)
            return

        n_images, full_h, full_w, embed_dim = full_features.shape
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if with_edge:
            emb_coreset = self.embedding_coreset_with_edge
            valid_mask = None
        else:
            emb_coreset = self.embedding_coreset
            valid_mask = torch.zeros((1, 1, full_h, full_w), device=device)
            valid_mask[
                :,
                :,
                self.patch_padding:full_h - self.patch_padding,
                self.patch_padding:full_w - self.patch_padding,
            ] = 1.0

        num_emb = len(emb_coreset)
        if HAS_FAISS:
            emb_index = self._build_search_index(emb_coreset, device=device)
        else:
            emb_index = torch.from_numpy(emb_coreset.astype(np.float32)).to(device)

        kernel = torch.ones(
            (num_emb, 1, self.neighborhood_size, self.neighborhood_size),
            device=device,
            dtype=torch.float32,
        )
        coor_counts = torch.zeros((num_emb, full_h, full_w), device=device)

        for start in range(0, n_images, self.histogram_chunk_size):
            end = min(start + self.histogram_chunk_size, n_images)
            chunk = full_features[start:end].astype(np.float32)
            chunk_n = end - start
            flat_chunk = chunk.reshape(-1, embed_dim)
            _, indices = self._search_index(emb_index, flat_chunk, 1)
            indices_t = torch.from_numpy(indices.reshape(chunk_n, full_h, full_w)).to(
                device=device, dtype=torch.long
            )
            one_hot = F.one_hot(indices_t, num_classes=num_emb).permute(0, 3, 1, 2)
            one_hot = one_hot.to(dtype=torch.float32)
            if valid_mask is not None:
                one_hot = one_hot * valid_mask
            counts = F.conv2d(
                one_hot,
                kernel,
                padding=self.dist_padding,
                groups=num_emb,
            )
            coor_counts += counts.sum(dim=0)
            print(f'      Processed {end}/{n_images} images', flush=True)

        coor_model = coor_counts.permute(1, 2, 0).cpu().numpy().astype(np.float32)
        sums = coor_model.sum(axis=-1, keepdims=True)
        coor_model = (coor_model / (sums + 1e-8)).reshape(-1, num_emb).astype(np.float32)

        if with_edge:
            self.coor_model_with_edge = coor_model
        else:
            self.coor_model = coor_model

    def _build_position_histograms_exact(
        self, full_features: np.ndarray, with_edge: bool = False
    ) -> None:
        """Build exact coordinate probability tables without dense one-hot tensors."""
        devices = [torch.device('cuda')] if torch.cuda.is_available() else []
        devices.append(torch.device('cpu'))
        last_exc = None

        for device in devices:
            try:
                self._build_position_histograms_exact_on_device(
                    full_features,
                    with_edge=with_edge,
                    device=device,
                )
                return
            except (RuntimeError, torch.AcceleratorError) as exc:
                if 'out of memory' not in str(exc).lower() or device.type != 'cuda':
                    raise
                last_exc = exc
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

        if last_exc is not None:
            raise last_exc
        raise RuntimeError('Failed to build exact coordinate histograms.')

    def _build_position_histograms_exact_on_device(
        self,
        full_features: np.ndarray,
        *,
        with_edge: bool,
        device: torch.device,
    ) -> None:
        """Build exact coordinate probability tables on one specific device."""
        n_images, full_h, full_w, embed_dim = full_features.shape
        if with_edge:
            emb_coreset = self.embedding_coreset_with_edge
        else:
            emb_coreset = self.embedding_coreset

        num_emb = len(emb_coreset)
        if HAS_FAISS:
            emb_index = self._build_search_index(emb_coreset, device=device)
        else:
            emb_index = self._move_numpy_to_device(emb_coreset.astype(np.float32), device)

        _, indices = self._search_index(
            emb_index,
            full_features.reshape(-1, embed_dim).astype(np.float32),
            1,
        )
        index_maps = indices.reshape(n_images, full_h, full_w).astype(np.int32)
        coor_counts = np.zeros((full_h * full_w, num_emb), dtype=np.float32)

        if with_edge:
            valid_center_mask = np.ones((full_h, full_w), dtype=bool)
        else:
            valid_center_mask = np.zeros((full_h, full_w), dtype=bool)
            valid_center_mask[
                self.patch_padding:full_h - self.patch_padding,
                self.patch_padding:full_w - self.patch_padding,
            ] = True

        row_indices = np.repeat(
            np.arange(full_h * full_w, dtype=np.int64),
            self.neighborhood_size * self.neighborhood_size,
        )
        row_indices_t = self._move_numpy_to_device(row_indices, device, dtype=torch.long)

        chunk_size = max(1, self.histogram_chunk_size)
        for start in range(0, n_images, chunk_size):
            end = min(start + chunk_size, n_images)
            chunk_maps = index_maps[start:end].copy()
            if not with_edge:
                chunk_maps[:, ~valid_center_mask] = -1

            chunk_t = self._move_numpy_to_device(chunk_maps, device, dtype=torch.long)
            padded = torch.full(
                (
                    end - start,
                    full_h + 2 * self.dist_padding,
                    full_w + 2 * self.dist_padding,
                ),
                fill_value=-1,
                device=device,
                dtype=torch.long,
            )
            padded[
                :,
                self.dist_padding:self.dist_padding + full_h,
                self.dist_padding:self.dist_padding + full_w,
            ] = chunk_t

            windows = padded.unfold(1, self.neighborhood_size, 1).unfold(
                2, self.neighborhood_size, 1
            )
            window_values = windows.contiguous().view(end - start, -1)
            valid = window_values >= 0
            flat_rows = row_indices_t.repeat(end - start)[valid.reshape(-1)]
            flat_cols = window_values.reshape(-1)[valid.reshape(-1)]
            flat_indices = flat_rows * num_emb + flat_cols
            counts = torch.bincount(
                flat_indices,
                minlength=full_h * full_w * num_emb,
            ).to(dtype=torch.float32)
            coor_counts += counts.view(full_h * full_w, num_emb).cpu().numpy()

            print(f'      Processed {end}/{n_images} images', flush=True)

        sums = coor_counts.sum(axis=-1, keepdims=True)
        coor_model = (coor_counts / (sums + 1e-8)).astype(np.float32)
        if with_edge:
            self.coor_model_with_edge = coor_model
        else:
            self.coor_model = coor_model

    @staticmethod
    def _move_numpy_to_device(
        array: np.ndarray,
        device: torch.device,
        *,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Move a numpy array to the preferred device, falling back to CPU on OOM."""
        tensor = torch.from_numpy(array)
        target_dtype = dtype if dtype is not None else tensor.dtype
        if device.type != 'cuda':
            return tensor.to(dtype=target_dtype)
        try:
            return tensor.to(device=device, dtype=target_dtype)
        except (RuntimeError, torch.AcceleratorError) as exc:
            if 'out of memory' not in str(exc).lower():
                raise
            return tensor.to(dtype=target_dtype)

    def _build_faiss_l2_index(self, vectors: np.ndarray):
        """Build an L2 FAISS index, preferring GPU when available."""
        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors.astype(np.float32))

        if not torch.cuda.is_available():
            return index
        if not hasattr(faiss, 'StandardGpuResources'):
            return index
        if not hasattr(faiss, 'index_cpu_to_gpu'):
            return index

        try:
            if self._faiss_gpu_resources is None:
                self._faiss_gpu_resources = faiss.StandardGpuResources()
            return faiss.index_cpu_to_gpu(self._faiss_gpu_resources, 0, index)
        except Exception:
            return index

    def _build_search_index(
        self,
        vectors: np.ndarray,
        *,
        device: Optional[torch.device] = None,
    ):
        """Build a search index with a GPU tensor fallback for CPU-only FAISS."""
        array = vectors.astype(np.float32)
        if HAS_FAISS:
            has_gpu_faiss = hasattr(faiss, 'StandardGpuResources') and hasattr(
                faiss, 'index_cpu_to_gpu'
            )
            if device is not None and device.type == 'cuda' and not has_gpu_faiss:
                return self._move_numpy_to_device(array, device)
            return self._build_faiss_l2_index(array)

        if device is not None:
            return self._move_numpy_to_device(array, device)
        return array

    def _build_faiss_indices(self) -> None:
        """Build nearest-neighbor indices for embedding and distribution search."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if HAS_FAISS:
            self._embedding_index = self._build_search_index(
                self.embedding_coreset,
                device=device,
            )
            self._embedding_index_with_edge = self._build_search_index(
                self.embedding_coreset_with_edge,
                device=device,
            )
            self._dist_index = self._build_search_index(
                self.dist_coreset,
                device=device,
            )
        else:
            self._embedding_index = self.embedding_coreset.astype(np.float32)
            self._embedding_index_with_edge = self.embedding_coreset_with_edge.astype(
                np.float32
            )
            self._dist_index = self.dist_coreset.astype(np.float32)

    def _build_emb_to_dist_mapping(self) -> None:
        """Map each embedding-center index to its nearest distribution center."""
        _, indices = self._search_index(
            self._dist_index, self.embedding_coreset.astype(np.float32), 1
        )
        self.emb_to_dist = indices.reshape(-1).astype(np.int32)

    def _build_neighborhood_mlp(self, full_features: np.ndarray) -> None:
        """Train the neighborhood MLP on full embeddings with valid centers only."""
        if not self.approximate_coreset:
            self._build_neighborhood_mlp_exact(full_features)
            return

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        n_images, full_h, full_w, embed_dim = full_features.shape
        valid_h = full_h - 2 * self.patch_padding
        valid_w = full_w - 2 * self.patch_padding
        num_valid_positions = valid_h * valid_w
        kernel_elems = self.neighborhood_size * self.neighborhood_size
        center_idx = kernel_elems // 2
        neighborhood_dim = embed_dim * (kernel_elems - 1)

        if self.max_train_samples > 0:
            samples_per_image = max(
                1, min(num_valid_positions, self.max_train_samples // max(n_images, 1))
            )
        else:
            samples_per_image = num_valid_positions
        print(
            f'    samples_per_image={samples_per_image}, valid_positions={num_valid_positions}',
            flush=True,
        )

        if HAS_FAISS:
            dist_index = self._build_search_index(self.dist_coreset, device=device)
        else:
            dist_index = torch.from_numpy(self.dist_coreset.astype(np.float32)).to(device)

        rng = np.random.default_rng(42)
        all_inputs = []
        all_labels = []
        mask = torch.ones(kernel_elems, dtype=torch.bool, device=device)
        mask[center_idx] = False

        chunk_size = 1
        for chunk_start in range(0, n_images, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_images)
            chunk_np = full_features[chunk_start:chunk_end].astype(np.float32)
            chunk_count = chunk_end - chunk_start

            if samples_per_image >= num_valid_positions:
                sampled_valid = np.arange(num_valid_positions)
            else:
                sampled_valid = np.sort(
                    rng.choice(num_valid_positions, size=samples_per_image, replace=False)
                )

            rows = sampled_valid // valid_w + self.patch_padding
            cols = sampled_valid % valid_w + self.patch_padding
            sampled_full = rows * full_w + cols

            with torch.no_grad():
                chunk_tensor = torch.from_numpy(chunk_np).permute(0, 3, 1, 2).to(device)
                unfolded = F.unfold(
                    chunk_tensor,
                    kernel_size=self.neighborhood_size,
                    padding=self.dist_padding,
                    stride=1,
                )
                unfolded = unfolded.transpose(1, 2).reshape(
                    chunk_count,
                    full_h * full_w,
                    embed_dim,
                    kernel_elems,
                )
                unfolded = unfolded[:, :, :, mask].reshape(
                    chunk_count, full_h * full_w, neighborhood_dim
                )
                sampled_inputs = unfolded[:, sampled_full, :].reshape(-1, neighborhood_dim)
                all_inputs.append(sampled_inputs.cpu().numpy())

            centers = chunk_np[
                :,
                self.patch_padding:-self.patch_padding,
                self.patch_padding:-self.patch_padding,
                :,
            ].reshape(chunk_count, num_valid_positions, embed_dim)
            centers = centers[:, sampled_valid, :].reshape(-1, embed_dim).astype(np.float32)
            _, labels = self._search_index(dist_index, centers, 1)
            all_labels.append(labels.reshape(-1))

            if chunk_end % 50 == 0 or chunk_end == n_images:
                print(f'      Processed {chunk_end}/{n_images} images', flush=True)

        inputs = np.concatenate(all_inputs, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        del all_inputs, all_labels
        gc.collect()

        if self.max_train_samples > 0 and len(inputs) > self.max_train_samples:
            keep = rng.choice(len(inputs), size=self.max_train_samples, replace=False)
            inputs = inputs[keep]
            labels = labels[keep]
            print(f'    Subsampled to {len(inputs)} MLP samples', flush=True)

        self.mlp = self._build_mlp(neighborhood_dim, len(self.dist_coreset))
        self._train_mlp(inputs.astype(np.float32), labels.astype(np.int64))

    def _build_neighborhood_mlp_exact(self, full_features: np.ndarray) -> None:
        """Train the exact official neighborhood MLP without materializing all samples."""
        n_images, full_h, full_w, embed_dim = full_features.shape
        valid_h = full_h - 2 * self.patch_padding
        valid_w = full_w - 2 * self.patch_padding
        num_valid_positions = valid_h * valid_w
        kernel_elems = self.neighborhood_size * self.neighborhood_size
        neighborhood_dim = embed_dim * (kernel_elems - 1)

        if self.max_train_samples > 0:
            samples_per_image = max(
                1, min(num_valid_positions, self.max_train_samples // max(n_images, 1))
            )
        else:
            samples_per_image = num_valid_positions
        print(
            f'    samples_per_image={samples_per_image}, valid_positions={num_valid_positions}',
            flush=True,
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if HAS_FAISS:
            dist_index = self._build_search_index(self.dist_coreset, device=device)
        else:
            dist_index = torch.from_numpy(self.dist_coreset.astype(np.float32)).to(device)

        _, labels = self._search_index(
            dist_index,
            full_features.reshape(-1, embed_dim).astype(np.float32),
            1,
        )
        dist_index_maps = labels.reshape(n_images, full_h, full_w).astype(np.int64)

        sample_coords = self._build_mlp_sample_coords(
            n_images=n_images,
            full_h=full_h,
            full_w=full_w,
            samples_per_image=samples_per_image,
        )
        if self.max_train_samples > 0 and len(sample_coords) > self.max_train_samples:
            keep = torch.randperm(len(sample_coords))[:self.max_train_samples].numpy()
            sample_coords = sample_coords[keep]
            print(f'    Subsampled to {len(sample_coords)} MLP samples', flush=True)

        self.mlp = self._build_mlp(neighborhood_dim, len(self.dist_coreset))
        self._train_mlp_exact(
            full_features=full_features,
            dist_index_maps=dist_index_maps,
            sample_coords=sample_coords,
        )

    def _build_mlp_sample_coords(
        self,
        *,
        n_images: int,
        full_h: int,
        full_w: int,
        samples_per_image: int,
    ) -> np.ndarray:
        """Build deterministic valid-center coordinates for neighborhood MLP samples."""
        valid_h = full_h - 2 * self.patch_padding
        valid_w = full_w - 2 * self.patch_padding
        num_valid_positions = valid_h * valid_w
        coords = []

        for image_idx in range(n_images):
            if samples_per_image >= num_valid_positions:
                sampled_valid = np.arange(num_valid_positions, dtype=np.int64)
            else:
                sampled_valid = np.sort(
                    torch.randperm(num_valid_positions)[:samples_per_image].cpu().numpy()
                )

            rows = sampled_valid // valid_w + self.patch_padding
            cols = sampled_valid % valid_w + self.patch_padding
            image_ids = np.full_like(rows, image_idx)
            coords.append(np.stack([image_ids, rows, cols], axis=1))

            processed = image_idx + 1
            if processed % 50 == 0 or processed == n_images:
                print(f'      Processed {processed}/{n_images} images', flush=True)

        return np.concatenate(coords, axis=0).astype(np.int64)

    def _split_mlp_sample_coords(self, sample_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split coordinates into train/val subsets with official 90/10 semantics."""
        total = len(sample_coords)
        val_size = int(total * self.mlp_val_ratio)
        if val_size <= 0 or val_size >= total:
            return sample_coords, sample_coords[:0]

        permutation = torch.randperm(total).cpu().numpy()
        val_indices = permutation[:val_size]
        train_indices = permutation[val_size:]
        return sample_coords[train_indices], sample_coords[val_indices]

    def _neighbor_offsets(self) -> np.ndarray:
        """Return flattened neighborhood offsets excluding the center."""
        offsets = []
        for row in range(-self.dist_padding, self.dist_padding + 1):
            for col in range(-self.dist_padding, self.dist_padding + 1):
                if row == 0 and col == 0:
                    continue
                offsets.append((row, col))
        return np.asarray(offsets, dtype=np.int64)

    def _extract_neighbor_batch(
        self,
        full_features: np.ndarray,
        dist_index_maps: np.ndarray,
        sample_coords: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract one neighborhood batch by vectorized coordinate gathering."""
        offsets = self._neighbor_offsets()
        batch_inputs = []
        batch_labels = []

        image_ids = sample_coords[:, 0]
        rows = sample_coords[:, 1]
        cols = sample_coords[:, 2]

        for image_idx in np.unique(image_ids):
            mask = image_ids == image_idx
            local_rows = rows[mask]
            local_cols = cols[mask]
            padded = np.pad(
                full_features[image_idx],
                ((self.dist_padding, self.dist_padding),
                 (self.dist_padding, self.dist_padding),
                 (0, 0)),
                mode='constant',
            )
            padded_rows = local_rows + self.dist_padding
            padded_cols = local_cols + self.dist_padding
            neighbors = padded[
                padded_rows[:, None] + offsets[:, 0],
                padded_cols[:, None] + offsets[:, 1],
            ]
            batch_inputs.append(
                neighbors.reshape(len(local_rows), -1).astype(np.float32, copy=False)
            )
            batch_labels.append(dist_index_maps[image_idx, local_rows, local_cols])

        return np.concatenate(batch_inputs, axis=0), np.concatenate(batch_labels, axis=0)

    def _iterate_mlp_batches(
        self,
        full_features: np.ndarray,
        dist_index_maps: np.ndarray,
        sample_coords: np.ndarray,
        *,
        shuffle: bool,
    ):
        """Yield vectorized neighborhood MLP batches."""
        if len(sample_coords) == 0:
            return

        if shuffle:
            order = torch.randperm(len(sample_coords)).cpu().numpy()
            coords = sample_coords[order]
        else:
            coords = sample_coords

        batch_size = min(self.mlp_batch_size, len(coords))
        for start in range(0, len(coords), batch_size):
            batch_coords = coords[start:start + batch_size]
            yield self._extract_neighbor_batch(full_features, dist_index_maps, batch_coords)

    def _train_mlp_exact(
        self,
        *,
        full_features: np.ndarray,
        dist_index_maps: np.ndarray,
        sample_coords: np.ndarray,
    ) -> None:
        """Train the neighborhood MLP with exact lazy neighborhood extraction."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mlp = self.mlp.to(device)
        optimizer = torch.optim.Adam(self.mlp.parameters(), lr=self.mlp_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        criterion = nn.CrossEntropyLoss()

        train_coords, val_coords = self._split_mlp_sample_coords(sample_coords)
        best_state = copy.deepcopy(self.mlp.state_dict())
        best_val_loss = float('inf')
        last_train_loss = float('nan')

        for epoch in range(self.mlp_epochs):
            self.mlp.train()
            total_loss = 0.0
            total_count = 0
            for batch_inputs, batch_labels in self._iterate_mlp_batches(
                full_features,
                dist_index_maps,
                train_coords,
                shuffle=True,
            ):
                batch_inputs_t = torch.from_numpy(batch_inputs).to(device)
                batch_labels_t = torch.from_numpy(batch_labels.astype(np.int64)).to(device)

                optimizer.zero_grad()
                logits = self.mlp(batch_inputs_t)
                loss = criterion(logits, batch_labels_t)
                loss.backward()
                optimizer.step()

                batch_count = batch_inputs_t.shape[0]
                total_loss += float(loss.item()) * batch_count
                total_count += batch_count

            last_train_loss = total_loss / max(total_count, 1)

            val_loss = None
            if len(val_coords) > 0:
                self.mlp.eval()
                total_val_loss = 0.0
                total_val_count = 0
                with torch.no_grad():
                    for batch_inputs, batch_labels in self._iterate_mlp_batches(
                        full_features,
                        dist_index_maps,
                        val_coords,
                        shuffle=False,
                    ):
                        batch_inputs_t = torch.from_numpy(batch_inputs).to(device)
                        batch_labels_t = torch.from_numpy(batch_labels.astype(np.int64)).to(device)
                        logits = self.mlp(batch_inputs_t)
                        loss = criterion(logits, batch_labels_t)
                        batch_count = batch_inputs_t.shape[0]
                        total_val_loss += float(loss.item()) * batch_count
                        total_val_count += batch_count

                val_loss = total_val_loss / max(total_val_count, 1)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(self.mlp.state_dict())
            else:
                best_val_loss = last_train_loss
                best_state = copy.deepcopy(self.mlp.state_dict())

            scheduler.step()
            if (epoch + 1) % 5 == 0 or epoch + 1 == self.mlp_epochs:
                if val_loss is None:
                    message = (
                        f'    MLP epoch {epoch + 1}/{self.mlp_epochs}, '
                        f'train_loss={last_train_loss:.4f}'
                    )
                else:
                    message = (
                        f'    MLP epoch {epoch + 1}/{self.mlp_epochs}, '
                        f'train_loss={last_train_loss:.4f}, '
                        f'val_loss={val_loss:.4f}'
                    )
                print(message, flush=True)

        self.mlp.load_state_dict(best_state)
        self.mlp.eval()
        self.last_mlp_fit_info = {
            'train_size': int(len(train_coords)),
            'val_size': int(len(val_coords)),
            'best_val_loss': float(best_val_loss),
            'last_train_loss': float(last_train_loss),
        }

    def _build_mlp(self, input_dim: int, output_dim: int) -> nn.Module:
        """Build the official distribution MLP."""
        if self.mlp_layers <= 1:
            return nn.Linear(input_dim, output_dim)

        layers = [
            nn.Linear(input_dim, self.mlp_channels),
            nn.BatchNorm1d(self.mlp_channels),
            nn.ReLU(),
            nn.Dropout(),
        ]
        for _ in range(self.mlp_layers - 2):
            layers.extend(
                [
                    nn.Linear(self.mlp_channels, self.mlp_channels),
                    nn.BatchNorm1d(self.mlp_channels),
                    nn.ReLU(),
                    nn.Dropout(),
                ]
            )
        layers.append(nn.Linear(self.mlp_channels, output_dim))
        return nn.Sequential(*layers)

    def _train_mlp(self, inputs: np.ndarray, labels: np.ndarray) -> None:
        """Train the neighborhood MLP with the official train/val split."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mlp = self.mlp.to(device)
        self.mlp.train()

        dataset = TensorDataset(
            torch.from_numpy(inputs.astype(np.float32)),
            torch.from_numpy(labels.astype(np.int64)),
        )
        val_size = int(len(dataset) * self.mlp_val_ratio)
        if val_size <= 0 or val_size >= len(dataset):
            train_dataset = dataset
            val_dataset = None
        else:
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = random_split(
                dataset, [train_size, val_size]
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=min(self.mlp_batch_size, len(train_dataset)),
            shuffle=True,
            num_workers=0,
        )
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=min(self.mlp_batch_size, len(val_dataset)),
                shuffle=False,
                num_workers=0,
            )

        optimizer = torch.optim.Adam(self.mlp.parameters(), lr=self.mlp_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        best_state = copy.deepcopy(self.mlp.state_dict())
        best_val_loss = float('inf')
        last_train_loss = float('nan')

        for epoch in range(self.mlp_epochs):
            total_loss = 0.0
            train_count = 0
            for batch_inputs, batch_labels in train_loader:
                batch_inputs = batch_inputs.to(device)
                batch_labels = batch_labels.to(device)

                optimizer.zero_grad()
                logits = self.mlp(batch_inputs)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()
                batch_size = batch_inputs.shape[0]
                total_loss += float(loss.item()) * batch_size
                train_count += batch_size

            last_train_loss = total_loss / max(train_count, 1)

            val_loss = None
            if val_loader is not None:
                self.mlp.eval()
                total_val_loss = 0.0
                val_count = 0
                with torch.no_grad():
                    for batch_inputs, batch_labels in val_loader:
                        batch_inputs = batch_inputs.to(device)
                        batch_labels = batch_labels.to(device)
                        logits = self.mlp(batch_inputs)
                        loss = criterion(logits, batch_labels)
                        batch_size = batch_inputs.shape[0]
                        total_val_loss += float(loss.item()) * batch_size
                        val_count += batch_size
                val_loss = total_val_loss / max(val_count, 1)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(self.mlp.state_dict())
                self.mlp.train()
            else:
                best_val_loss = last_train_loss
                best_state = copy.deepcopy(self.mlp.state_dict())

            scheduler.step()
            if (epoch + 1) % 5 == 0 or epoch + 1 == self.mlp_epochs:
                if val_loss is None:
                    message = (
                        f'    MLP epoch {epoch + 1}/{self.mlp_epochs}, '
                        f'train_loss={last_train_loss:.4f}'
                    )
                else:
                    message = (
                        f'    MLP epoch {epoch + 1}/{self.mlp_epochs}, '
                        f'train_loss={last_train_loss:.4f}, '
                        f'val_loss={val_loss:.4f}'
                    )
                print(message, flush=True)

        self.mlp.load_state_dict(best_state)
        self.mlp.eval()
        self.last_mlp_fit_info = {
            'train_size': int(len(train_dataset)),
            'val_size': int(len(val_dataset)) if val_dataset is not None else 0,
            'best_val_loss': float(best_val_loss),
            'last_train_loss': float(last_train_loss),
        }
        del dataset, train_loader, val_loader
        gc.collect()

    def _extract_neighborhood_features(self, embeddings: Tensor) -> np.ndarray:
        """Extract 9x9 neighborhoods from full embeddings for all full-grid positions."""
        batch_size, full_h, full_w, embed_dim = embeddings.shape
        kernel_elems = self.neighborhood_size * self.neighborhood_size
        center_idx = kernel_elems // 2

        tensor = embeddings.permute(0, 3, 1, 2).contiguous()
        unfolded = F.unfold(
            tensor,
            kernel_size=self.neighborhood_size,
            padding=self.dist_padding,
            stride=1,
        )
        unfolded = unfolded.transpose(1, 2).reshape(
            batch_size, full_h * full_w, embed_dim, kernel_elems
        )
        mask = torch.ones(kernel_elems, dtype=torch.bool, device=embeddings.device)
        mask[center_idx] = False
        unfolded = unfolded[:, :, :, mask].reshape(batch_size * full_h * full_w, -1)
        return unfolded.detach().cpu().numpy().astype(np.float32)

    @staticmethod
    def _gather_rank_aligned_mask(mask_by_id: np.ndarray, ranked_indices: np.ndarray) -> np.ndarray:
        """Gather a bool/probability table from id order into nearest-rank order."""
        row_indices = np.arange(mask_by_id.shape[0], dtype=np.int64)[:, None]
        return mask_by_id[row_indices, ranked_indices]

    def _compute_predict_artifacts(self, feats: Tuple[Tensor, ...]) -> Dict[str, object]:
        """Compute intermediate predict artifacts for scoring/debugging."""
        assert self._embedding_index is not None, 'Memory bank not built yet.'
        assert self.coor_model is not None, 'Coordinate model not built.'
        assert self.mlp is not None, 'Neighborhood MLP not built.'
        assert self.emb_to_dist is not None, 'Embedding-to-distribution mapping missing.'

        embeddings = self._generate_embeddings(feats)
        batch_size, full_h, full_w, embed_dim = embeddings.shape
        flat_embeddings = embeddings.reshape(-1, embed_dim).detach().cpu().numpy().astype(
            np.float32
        )

        num_emb = len(self.embedding_coreset)
        num_dist = len(self.dist_coreset)
        embed_distances, embed_indices = self._search_index(
            self._embedding_index, flat_embeddings, num_emb
        )
        embed_probs = self.prob_gamma * np.exp(-self.prob_gamma * embed_distances)

        pos_indices = np.tile(np.arange(full_h * full_w, dtype=np.int64), batch_size)
        coor_probs = self.coor_model[pos_indices]
        coor_threshold = self.softmax_coor_gamma / num_emb
        coor_mask_by_id = coor_probs > coor_threshold
        coor_mask = self._gather_rank_aligned_mask(coor_mask_by_id, embed_indices)
        coor_mask[:, -1] = True

        neighborhood_inputs = self._extract_neighborhood_features(embeddings)
        mlp_device = next(self.mlp.parameters()).device
        with torch.no_grad():
            logits = self.mlp(torch.from_numpy(neighborhood_inputs).to(mlp_device))
            logits = logits / self.temperature
            nb_probs = torch.softmax(logits, dim=-1).cpu().numpy()

        nb_threshold = self.softmax_nb_gamma / num_emb
        dist_active = nb_probs > nb_threshold
        # Official rank-based nb_mask: for each embedding member e,
        # emb_to_dist[e] gives dist member d; check MLP activation for
        # the actual dist member at rank d for each test position.
        _, dist_indices_full = self._search_index(
            self._dist_index, flat_embeddings, num_dist
        )
        rank_positions = self.emb_to_dist  # (emb_size,) dist member IDs
        actual_dist_at_rank = dist_indices_full[:, rank_positions]
        row_idx = np.arange(flat_embeddings.shape[0])[:, None]
        nb_mask = dist_active[row_idx, actual_dist_at_rank]
        nb_mask[:, -1] = True

        anomaly_nb = -np.log(np.max(embed_probs * nb_mask, axis=1) + 1e-10)
        anomaly_coor = -np.log(np.max(embed_probs * coor_mask, axis=1) + 1e-10)

        anomaly_nb = anomaly_nb.reshape(batch_size, full_h, full_w)
        anomaly_coor = anomaly_coor.reshape(batch_size, full_h, full_w)

        if self.patch_padding > 0:
            anomaly_map_raw = 0.5 * (
                anomaly_nb[
                    :,
                    self.patch_padding:-self.patch_padding,
                    self.patch_padding:-self.patch_padding,
                ]
                + anomaly_coor[
                    :,
                    self.patch_padding:-self.patch_padding,
                    self.patch_padding:-self.patch_padding,
                ]
            )
            anomaly_map = np.pad(
                anomaly_map_raw,
                (
                    (0, 0),
                    (self.patch_padding, self.patch_padding),
                    (self.patch_padding, self.patch_padding),
                ),
                mode='edge',
            )
        else:
            anomaly_map_raw = 0.5 * (anomaly_nb + anomaly_coor)
            anomaly_map = anomaly_map_raw

        return {
            'embeddings': embeddings,
            'batch_size': batch_size,
            'full_h': full_h,
            'full_w': full_w,
            'num_emb': num_emb,
            'num_dist': num_dist,
            'embed_distances': embed_distances,
            'embed_indices': embed_indices,
            'embed_probs': embed_probs,
            'coor_probs': coor_probs,
            'coor_threshold': coor_threshold,
            'coor_mask_by_id': coor_mask_by_id,
            'coor_mask': coor_mask,
            'nb_probs': nb_probs,
            'nb_threshold': nb_threshold,
            'ranked_dist_indices': actual_dist_at_rank,
            'nb_mask': nb_mask,
            'anomaly_nb': anomaly_nb,
            'anomaly_coor': anomaly_coor,
            'anomaly_map_raw': anomaly_map_raw,
            'anomaly_map': anomaly_map,
        }

    def debug_predict_summary(self, feats: Tuple[Tensor, ...]) -> Dict[str, float]:
        """Summarize predict-time scoring internals."""
        artifacts = self._compute_predict_artifacts(feats)
        summary = {
            'batch_size': int(artifacts['batch_size']),
            'full_h': int(artifacts['full_h']),
            'full_w': int(artifacts['full_w']),
            'num_emb': int(artifacts['num_emb']),
            'num_dist': int(artifacts['num_dist']),
            'embed_distance_mean': float(np.mean(artifacts['embed_distances'])),
            'embed_distance_std': float(np.std(artifacts['embed_distances'])),
            'embed_prob_mean': float(np.mean(artifacts['embed_probs'])),
            'embed_prob_std': float(np.std(artifacts['embed_probs'])),
            'coor_prob_mean': float(np.mean(artifacts['coor_probs'])),
            'coor_prob_max_mean': float(np.mean(np.max(artifacts['coor_probs'], axis=1))),
            'coor_threshold': float(artifacts['coor_threshold']),
            'coor_mask_rate': float(np.mean(artifacts['coor_mask'].astype(np.float32))),
            'nb_prob_mean': float(np.mean(artifacts['nb_probs'])),
            'nb_prob_max_mean': float(np.mean(np.max(artifacts['nb_probs'], axis=1))),
            'nb_threshold': float(artifacts['nb_threshold']),
            'nb_mask_rate': float(np.mean(artifacts['nb_mask'].astype(np.float32))),
            'anomaly_nb_mean': float(np.mean(artifacts['anomaly_nb'])),
            'anomaly_nb_std': float(np.std(artifacts['anomaly_nb'])),
            'anomaly_coor_mean': float(np.mean(artifacts['anomaly_coor'])),
            'anomaly_coor_std': float(np.std(artifacts['anomaly_coor'])),
            'anomaly_map_raw_mean': float(np.mean(artifacts['anomaly_map_raw'])),
            'anomaly_map_raw_std': float(np.std(artifacts['anomaly_map_raw'])),
            'anomaly_map_raw_max': float(np.max(artifacts['anomaly_map_raw'])),
            'anomaly_map_mean': float(np.mean(artifacts['anomaly_map'])),
            'anomaly_map_std': float(np.std(artifacts['anomaly_map'])),
        }
        self.last_scoring_summary = summary
        return summary

    def loss(
        self,
        feats: Tuple[Tensor, ...],
        data_samples: Optional[List] = None,
    ) -> Dict[str, Tensor]:
        """PNI collects embeddings during the fitting phase."""
        if self._embedding_index is None:
            self.collect_features(feats)
        dummy = torch.tensor(0.0, device=feats[0].device, requires_grad=True)
        return {'loss': dummy}

    def predict(
        self,
        feats: Tuple[Tensor, ...],
        data_samples: Optional[List] = None,
    ) -> List:
        """Predict anomaly scores using official nb+coor PNI scoring."""
        assert self._embedding_index is not None, 'Memory bank not built yet.'
        assert self.coor_model is not None, 'Coordinate model not built.'
        assert self.mlp is not None, 'Neighborhood MLP not built.'
        assert self.emb_to_dist is not None, 'Embedding-to-distribution mapping missing.'

        artifacts = self._compute_predict_artifacts(feats)
        batch_size = artifacts['batch_size']
        anomaly_map = artifacts['anomaly_map']
        anomaly_map_raw = artifacts['anomaly_map_raw']

        results = []
        for batch_idx in range(batch_size):
            score_map = gaussian_filter(anomaly_map[batch_idx], sigma=self.blur_sigma)
            score_map = torch.from_numpy(score_map).unsqueeze(0).unsqueeze(0)

            if self.input_size is not None:
                score_map = F.interpolate(
                    score_map,
                    size=self.input_size,
                    mode='bilinear',
                    align_corners=False,
                )

            score_map = score_map.squeeze(0)
            image_score = float(np.max(anomaly_map_raw[batch_idx]))

            result = data_samples[batch_idx] if data_samples else ADDataSample()
            result.pred_score = image_score
            result.pred_anomaly_map = score_map
            results.append(result)

        if self.log_predict_stats and results:
            scores = np.asarray([float(result.pred_score) for result in results], dtype=np.float32)
            maps = torch.stack([result.pred_anomaly_map.detach().float().cpu() for result in results])
            self.last_predict_summary = {
                'score_min': float(scores.min()),
                'score_max': float(scores.max()),
                'score_mean': float(scores.mean()),
                'map_min': float(maps.min().item()),
                'map_max': float(maps.max().item()),
                'map_std': float(maps.std(unbiased=False).item()),
            }
            print(
                '  predict stats: '
                f'score_min={self.last_predict_summary["score_min"]:.4f}, '
                f'score_max={self.last_predict_summary["score_max"]:.4f}, '
                f'score_mean={self.last_predict_summary["score_mean"]:.4f}, '
                f'map_min={self.last_predict_summary["map_min"]:.4f}, '
                f'map_max={self.last_predict_summary["map_max"]:.4f}, '
                f'map_std={self.last_predict_summary["map_std"]:.4f}',
                flush=True,
            )
        else:
            self.last_predict_summary = None

        return results

    def forward(self, feats: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        """Tensor mode keeps the incoming feature tuple unchanged."""
        return feats
