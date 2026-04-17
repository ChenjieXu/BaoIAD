"""AnomalyDINO: training-free anomaly detection with DINOv2 patch tokens.

Reference: https://arxiv.org/abs/2405.14529
Code: https://github.com/dammsi/AnomalyDINO
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Optional, Sequence, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS
from baoiad.structures import ADDataSample
from baoiad.models.base_ad_model import VisionLanguageADModel


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply Gaussian blur to a tensor."""
    if sigma <= 0:
        return x

    kernel_size = int(2 * round(3 * sigma) + 1)
    coords = torch.arange(kernel_size, dtype=torch.float32, device=x.device) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = g / g.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d.unsqueeze(0).unsqueeze(0).repeat(x.shape[1], 1, 1, 1)

    pad = kernel_size // 2
    x = F.pad(x, [pad] * 4, mode='reflect')
    return F.conv2d(x, kernel, groups=x.shape[1])


def _rotate_tensor(x: torch.Tensor, angle: float) -> torch.Tensor:
    """Rotate an image tensor in-place size with bilinear interpolation."""
    normalized_angle = angle % 360.0
    if math.isclose(normalized_angle, 0.0, abs_tol=1e-6):
        return x

    radians = math.radians(normalized_angle)
    theta = x.new_tensor([
        [math.cos(radians), -math.sin(radians), 0.0],
        [math.sin(radians), math.cos(radians), 0.0],
    ]).unsqueeze(0).repeat(x.shape[0], 1, 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(
        x,
        grid,
        mode='bilinear',
        padding_mode='reflection',
        align_corners=False,
    )


@MODELS.register_module()
class AnomalyDINODetector(VisionLanguageADModel):
    """Training-free anomaly detection using DINOv2 patch tokens.

    The strict official path follows `dammsi/AnomalyDINO`:
    - short-edge resize to 448 with aspect ratio preserved
    - crop to patch-size multiples before DINOv2 token extraction
    - deterministic few-shot support selection by sorted file names
    - optional 8-angle support rotation
    - official PCA foreground masking on test images, with support masking
      controlled separately via ``mask_ref_images``.

    Legacy configs without ``preprocess`` remain supported for backwards
    compatibility.
    """

    PCA_MASKING_CATEGORIES = frozenset({
        'capsule', 'hazelnut', 'pill', 'screw', 'toothbrush',
    })
    INFORMED_ROTATION_CATEGORIES = frozenset({'hazelnut', 'screw'})
    LEGACY_ROTATION_ANGLES = (0.0, 90.0, 180.0, 270.0)
    OFFICIAL_ROTATION_ANGLES = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)
    SUPPORTED_PREPROCESS = frozenset({
        'agnostic',
        'informed',
        'masking_only',
        'agnostic_no_mask',
        'informed_no_mask',
        'force_no_mask_no_rotation',
        'force_mask_no_rotation',
        'force_no_mask_rotation',
        'force_mask_rotation',
    })
    build_memory_bank_from_dataloader_only = True

    def __init__(
        self,
        backbone: Union[str, dict],
        k: int = 1,
        pca_foreground: Union[bool, str] = False,
        top_ratio: float = 0.01,
        max_memory_bank_size: int = 0,
        gaussian_sigma: float = 4.0,
        few_shot: Optional[int] = None,
        rotation_aug: bool = True,
        preprocess: Optional[str] = None,
        mask_ref_images: bool = False,
        few_shot_seed: int = 0,
        rotation_angles: Optional[Sequence[float]] = None,
        mask_threshold: float = 10.0,
        mask_kernel_size: int = 3,
        mask_border: float = 0.2,
        data_preprocessor=None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        if isinstance(backbone, dict):
            self.backbone = MODELS.build(backbone)
        else:
            raise ValueError(f'backbone must be dict, got {type(backbone)}')

        if preprocess is not None and preprocess not in self.SUPPORTED_PREPROCESS:
            raise ValueError(
                f'Unsupported preprocess={preprocess!r}. '
                f'Expected one of {sorted(self.SUPPORTED_PREPROCESS)}.'
            )

        self.k = k
        self.pca_foreground = pca_foreground
        self.top_ratio = top_ratio
        self.max_memory_bank_size = max_memory_bank_size
        self.gaussian_sigma = gaussian_sigma
        self.few_shot = few_shot
        self.rotation_aug = rotation_aug
        self.preprocess = preprocess
        self.mask_ref_images = mask_ref_images
        self.few_shot_seed = few_shot_seed
        self.rotation_angles = tuple(rotation_angles) if rotation_angles is not None else None
        self.mask_threshold = mask_threshold
        self.mask_kernel_size = mask_kernel_size
        self.mask_border = mask_border

        self.register_buffer('memory_bank', torch.empty(0))

        self._collected_refs: list[dict[str, Any]] = []
        self._current_img_idx = 0

    def _sample_meta(self, sample: Optional[ADDataSample], key: str, default=None):
        """Read metadata from a data sample with BaseDataElement compatibility."""
        if sample is None:
            return default

        if hasattr(sample, 'get'):
            try:
                value = sample.get(key, None)
            except Exception:
                value = None
            if value is not None:
                return value

        if hasattr(sample, key):
            value = getattr(sample, key)
            if value is not None:
                return value

        metainfo = getattr(sample, 'metainfo', None)
        if isinstance(metainfo, dict):
            return metainfo.get(key, default)

        return default

    def _get_patch_tokens(self, x: torch.Tensor):
        """Extract DINOv2 patch tokens from input images."""
        patch_size = 14
        _, _, h, w = x.shape
        crop_h = h - h % patch_size
        crop_w = w - w % patch_size
        if crop_h <= 0 or crop_w <= 0:
            raise RuntimeError(
                f'Input spatial shape {(h, w)} is too small for patch_size={patch_size}.'
            )
        if crop_h != h or crop_w != w:
            x = x[:, :, :crop_h, :crop_w]

        out = self.backbone.encoder.forward_features(x)
        tokens = out['x_norm_patchtokens']
        return tokens, crop_h // patch_size, crop_w // patch_size

    def _should_apply_masking(self, cls_name: Optional[str]) -> bool:
        """Return whether official PCA masking should run for the given class."""
        if self.preprocess is None:
            if self.pca_foreground == 'auto':
                return cls_name in self.PCA_MASKING_CATEGORIES
            return bool(self.pca_foreground)

        if self.preprocess in {'agnostic', 'informed', 'masking_only'}:
            return cls_name in self.PCA_MASKING_CATEGORIES
        if self.preprocess in {'force_mask_no_rotation', 'force_mask_rotation'}:
            return True
        return False

    def _should_apply_rotation(self, cls_name: Optional[str]) -> bool:
        """Return whether support rotations should run for the given class."""
        if self.preprocess is None:
            return self.rotation_aug

        if self.preprocess in {'agnostic', 'agnostic_no_mask', 'force_no_mask_rotation', 'force_mask_rotation'}:
            return True
        if self.preprocess in {'informed', 'informed_no_mask'}:
            return cls_name in self.INFORMED_ROTATION_CATEGORIES
        return False

    def _rotation_angles_for_category(self, cls_name: Optional[str]) -> tuple[float, ...]:
        """Return non-zero support rotation angles for the active mode."""
        if not self._should_apply_rotation(cls_name):
            return ()

        if self.rotation_angles is not None:
            angles = tuple(float(angle) for angle in self.rotation_angles)
        elif self.preprocess is None:
            angles = self.LEGACY_ROTATION_ANGLES
        else:
            angles = self.OFFICIAL_ROTATION_ANGLES

        extra_angles = []
        for angle in angles:
            normalized = angle % 360.0
            if not math.isclose(normalized, 0.0, abs_tol=1e-6):
                extra_angles.append(float(normalized))
        return tuple(extra_angles)

    def _build_background_mask(
        self,
        tokens: torch.Tensor,
        spatial_h: int,
        spatial_w: int,
    ) -> torch.Tensor:
        """Approximate the official PCA foreground mask used by AnomalyDINO."""
        if tokens.ndim == 3:
            if tokens.shape[0] != 1:
                raise ValueError('Expected a single sample when building a background mask.')
            tokens = tokens[0]

        if tokens.numel() == 0 or tokens.shape[0] <= 1:
            return torch.ones(tokens.shape[0], dtype=torch.bool)

        feats = tokens.detach().float().cpu()
        centered = feats - feats.mean(dim=0, keepdim=True)
        try:
            _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        except RuntimeError:
            return torch.ones(tokens.shape[0], dtype=torch.bool)

        first_pc = centered @ vh[0]
        mask = first_pc > self.mask_threshold
        grid = mask.reshape(spatial_h, spatial_w).numpy()

        start_h = int(spatial_h * self.mask_border)
        end_h = max(start_h + 1, int(spatial_h * (1 - self.mask_border)))
        start_w = int(spatial_w * self.mask_border)
        end_w = max(start_w + 1, int(spatial_w * (1 - self.mask_border)))
        center = grid[start_h:end_h, start_w:end_w]
        if center.size > 0 and center.mean() <= 0.35:
            mask = (-first_pc) > self.mask_threshold
            grid = mask.reshape(spatial_h, spatial_w).numpy()

        kernel = np.ones((self.mask_kernel_size, self.mask_kernel_size), dtype=np.uint8)
        grid_u8 = grid.astype(np.uint8)
        grid_u8 = cv2.dilate(grid_u8, kernel).astype(np.uint8)
        grid_u8 = cv2.morphologyEx(grid_u8, cv2.MORPH_CLOSE, kernel).astype(np.uint8)
        mask = torch.from_numpy(grid_u8.reshape(-1).astype(bool))

        if not mask.any():
            return torch.ones(spatial_h * spatial_w, dtype=torch.bool)
        return mask

    def _knn_distances(self, queries: torch.Tensor, memory_bank: torch.Tensor) -> torch.Tensor:
        """Compute top-k cosine distances from queries to the memory bank."""
        if queries.numel() == 0:
            return torch.empty((0, self.k), device=queries.device, dtype=queries.dtype)

        topk_dists = torch.full(
            (queries.shape[0], self.k),
            float('inf'),
            device=queries.device,
            dtype=queries.dtype,
        )
        mem_chunk_size = 8192
        for start in range(0, memory_bank.shape[0], mem_chunk_size):
            end = min(start + mem_chunk_size, memory_bank.shape[0])
            sims = queries @ memory_bank[start:end].T
            dists_chunk = 1 - sims
            combined = torch.cat([topk_dists, dists_chunk], dim=1)
            topk_dists, _ = torch.topk(combined, k=self.k, dim=1, largest=False)
        return topk_dists

    def _collect_reference_features(
        self,
        inputs: torch.Tensor,
        data_samples: Optional[Sequence[ADDataSample]] = None,
    ) -> None:
        """Collect support features for later memory-bank construction."""
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)
        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(0)

        tokens, spatial_h, spatial_w = self._get_patch_tokens(inputs)
        batch_size = tokens.shape[0]
        data_samples = list(data_samples or [])

        for index in range(batch_size):
            sample = data_samples[index] if index < len(data_samples) else None
            cls_name = self._sample_meta(sample, 'cls_name', None)
            img_path = self._sample_meta(sample, 'img_path', None)
            if not img_path:
                img_path = f'__image_{len(self._collected_refs):08d}'

            self._collected_refs.append(dict(
                img_path=img_path,
                cls_name=cls_name,
                tokens=tokens[index].detach().cpu(),
                spatial_h=spatial_h,
                spatial_w=spatial_w,
            ))

            for angle in self._rotation_angles_for_category(cls_name):
                rotated = _rotate_tensor(inputs[index:index + 1], angle)
                rot_tokens, rot_h, rot_w = self._get_patch_tokens(rotated)
                self._collected_refs.append(dict(
                    img_path=img_path,
                    cls_name=cls_name,
                    tokens=rot_tokens[0].detach().cpu(),
                    spatial_h=rot_h,
                    spatial_w=rot_w,
                ))

    def _selected_reference_paths_from_dataset(self, dataset) -> list[str]:
        """Select support image paths directly from dataset metadata."""
        if hasattr(dataset, 'full_init'):
            dataset.full_init()
        data_infos = [dataset.get_data_info(index) for index in range(len(dataset))]
        grouped_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for info in data_infos:
            grouped_records[info['img_path']].append(info)
        return self._selected_reference_paths(grouped_records)

    def forward(
        self,
        inputs: torch.Tensor,
        data_samples: Optional[Sequence[ADDataSample]] = None,
        mode: str = 'tensor',
    ):
        """Forward pass."""
        if mode == 'loss':
            if isinstance(inputs, (list, tuple)):
                inputs = torch.stack(inputs)
            elif torch.is_tensor(inputs) and inputs.ndim == 3:
                inputs = inputs.unsqueeze(0)

            batch_size = inputs.shape[0]
            selected_positions = self._selected_batch_positions(batch_size)
            if selected_positions:
                selected_inputs = inputs[selected_positions]
                selected_samples = None
                if data_samples is not None:
                    selected_samples = [data_samples[index] for index in selected_positions]
                self._collect_reference_features(selected_inputs, selected_samples)

            self._current_img_idx += batch_size
            return {'loss': torch.zeros(1, requires_grad=True, device=inputs.device)}

        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        tokens, spatial_h, spatial_w = self._get_patch_tokens(inputs)

        if mode == 'predict':
            if self.memory_bank.numel() == 0:
                raise RuntimeError('Memory bank is empty. Call build_memory_bank() first.')

            masks = []
            batch_size = tokens.shape[0]
            for index in range(batch_size):
                sample = data_samples[index] if data_samples and index < len(data_samples) else None
                cls_name = self._sample_meta(sample, 'cls_name', None)
                if self._should_apply_masking(cls_name):
                    mask = self._build_background_mask(tokens[index], spatial_h, spatial_w)
                else:
                    mask = torch.ones(tokens.shape[1], dtype=torch.bool)
                masks.append(mask.to(tokens.device))

            anomaly_map = self._compute_anomaly_map(
                tokens,
                spatial_h=spatial_h,
                spatial_w=spatial_w,
                masks=torch.stack(masks, dim=0),
            )
            anomaly_map = F.interpolate(
                anomaly_map,
                size=inputs.shape[2:],
                mode='bilinear',
                align_corners=False,
            )
            anomaly_map = _gaussian_blur(anomaly_map, self.gaussian_sigma)

            flat = anomaly_map.flatten(1)
            k_top = max(1, int(flat.shape[1] * self.top_ratio))
            top_values, _ = torch.topk(flat, k=k_top, dim=1)
            img_scores = top_values.mean(dim=1)
            return build_predict_results(data_samples, img_scores, anomaly_map)

        return tokens

    def _compute_anomaly_map(
        self,
        tokens: torch.Tensor,
        spatial_h: int,
        spatial_w: int,
        masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute anomaly map via k-NN distance against memory bank."""
        if self.memory_bank.numel() == 0:
            raise RuntimeError('Memory bank is empty. Call build_memory_bank() first.')

        batch_size, num_tokens, channels = tokens.shape
        memory_bank = F.normalize(self.memory_bank.to(tokens.device), dim=-1)

        dist_maps = []
        for index in range(batch_size):
            sample_tokens = tokens[index]
            sample_mask = None if masks is None else masks[index].bool()
            if sample_mask is None:
                query_tokens = sample_tokens
            else:
                query_tokens = sample_tokens[sample_mask]
                if query_tokens.numel() == 0:
                    sample_mask = None
                    query_tokens = sample_tokens

            queries = F.normalize(query_tokens.reshape(-1, channels), dim=-1)
            query_dists = self._knn_distances(queries, memory_bank).mean(dim=1)

            dist_map = torch.zeros(num_tokens, device=tokens.device, dtype=tokens.dtype)
            if sample_mask is None:
                dist_map = query_dists
            else:
                dist_map[sample_mask] = query_dists
            dist_maps.append(dist_map)

        return torch.stack(dist_maps, dim=0).view(batch_size, 1, spatial_h, spatial_w)

    def _selected_reference_paths(self, grouped_records: dict[str, list[dict[str, Any]]]) -> list[str]:
        """Select support images using the official sorted-slice few-shot rule."""
        all_paths = sorted(grouped_records)
        if self.few_shot is None or self.few_shot <= 0:
            return all_paths

        start = self.few_shot_seed * self.few_shot
        selected = all_paths[start:start + self.few_shot]
        if not selected:
            raise RuntimeError(
                'No support images selected for '
                f'few_shot={self.few_shot}, few_shot_seed={self.few_shot_seed}.'
            )
        return selected

    def _selected_batch_positions(self, batch_size: int) -> list[int]:
        """Return which positions of the current batch should collect features."""
        if self.preprocess is None or self.few_shot is None or self.few_shot <= 0:
            return list(range(batch_size))

        start = self.few_shot_seed * self.few_shot
        end = start + self.few_shot
        selected = []
        for batch_index in range(batch_size):
            global_index = self._current_img_idx + batch_index
            if start <= global_index < end:
                selected.append(batch_index)
        return selected

    def build_memory_bank(self, dataloader=None):
        """Build the memory bank from collected reference features."""
        selected_paths_override: Optional[list[str]] = None
        if dataloader is not None and self.few_shot is not None and self.few_shot > 0:
            dataset = dataloader.dataset
            selected_paths = self._selected_reference_paths_from_dataset(dataset)
            selected_paths_override = list(selected_paths)
            path_to_indices: dict[str, list[int]] = defaultdict(list)
            for index in range(len(dataset)):
                info = dataset.get_data_info(index)
                path_to_indices[info['img_path']].append(index)

            self._collected_refs.clear()
            device = next(self.backbone.parameters()).device
            for img_path in selected_paths:
                for index in path_to_indices[img_path]:
                    sample = dataset[index]
                    inputs = sample['inputs']
                    if torch.is_tensor(inputs):
                        inputs = inputs.unsqueeze(0).to(device)
                    else:
                        inputs = torch.stack(inputs).to(device)
                    data_samples = sample['data_samples']
                    if not isinstance(data_samples, (list, tuple)):
                        data_samples = [data_samples]
                    self._collect_reference_features(inputs, data_samples)

        if not self._collected_refs:
            return

        grouped_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in self._collected_refs:
            grouped_records[record['img_path']].append(record)

        feats_list = []
        selected_paths = (
            selected_paths_override
            if selected_paths_override is not None
            else self._selected_reference_paths(grouped_records)
        )
        for img_path in selected_paths:
            for record in grouped_records[img_path]:
                tokens = record['tokens']
                if self.mask_ref_images and self._should_apply_masking(record['cls_name']):
                    mask = self._build_background_mask(
                        tokens,
                        spatial_h=record['spatial_h'],
                        spatial_w=record['spatial_w'],
                    )
                    tokens = tokens[mask]

                if tokens.numel() == 0:
                    continue
                feats_list.append(tokens.reshape(-1, tokens.shape[-1]))

        self._collected_refs.clear()
        self._current_img_idx = 0

        if not feats_list:
            raise RuntimeError('No valid reference features were collected for the memory bank.')

        feats = torch.cat(feats_list, dim=0)
        if self.max_memory_bank_size > 0 and feats.shape[0] > self.max_memory_bank_size:
            indices = torch.randperm(feats.shape[0], device=feats.device)[:self.max_memory_bank_size]
            feats = feats[indices]

        self.memory_bank = feats

    def fit(self):
        """Alias for build_memory_bank for compatibility."""
        self.build_memory_bank()

    def train(self, mode: bool = True):
        """Keep the frozen backbone in eval mode."""
        super().train(mode)
        self.backbone.eval()
        return self
