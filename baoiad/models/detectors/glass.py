"""GLASS anomaly detector with legacy and strict official-alignment paths."""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.optim import OptimWrapperDict

from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS
from baoiad.optional import require_optional_module
from baoiad.runtime import OfflineModeError
from baoiad.utils.glass_utils import (
    distribution_judge,
    resolve_dtd_texture_paths,
    tensor_to_bgr_image,
)
from baoiad.models.base_ad_model import BaseADModel

logger = logging.getLogger(__name__)


class PatchMaker:
    """Patchify helpers shared by the GLASS/SimpleNet-style path."""

    def __init__(self, patchsize: int, stride: int = 1):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features: torch.Tensor, return_spatial_info: bool = False):
        padding = int((self.patchsize - 1) / 2)
        unfolder = nn.Unfold(
            kernel_size=self.patchsize,
            stride=self.stride,
            padding=padding,
            dilation=1,
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for spatial_size in features.shape[-2:]:
            n_patches = (spatial_size + 2 * padding - (self.patchsize - 1) - 1) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2],
            self.patchsize,
            self.patchsize,
            -1,
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)
        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    @staticmethod
    def unpatch_scores(scores: torch.Tensor, batchsize: int) -> torch.Tensor:
        return scores.reshape(batchsize, -1, *scores.shape[1:])

    @staticmethod
    def score(scores: torch.Tensor) -> torch.Tensor:
        scores = scores[:, :, 0]
        return torch.max(scores, dim=1).values


class MeanMapper(nn.Module):
    """Adaptive average pooling on flattened patch features."""

    def __init__(self, preprocessing_dim: int):
        super().__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)


class Preprocessing(nn.Module):
    """Per-layer adaptive pooling to a common dimension, then stack."""

    def __init__(self, input_dims, output_dim: int):
        super().__init__()
        self.preprocessing_modules = nn.ModuleList([MeanMapper(output_dim) for _ in input_dims])

    def forward(self, features):
        outputs = []
        for module, feature in zip(self.preprocessing_modules, features):
            outputs.append(module(feature))
        return torch.stack(outputs, dim=1)


class Aggregator(nn.Module):
    """Aggregate stacked layer features into ``target_dim`` via average pooling."""

    def __init__(self, target_dim: int):
        super().__init__()
        self.target_dim = target_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)


class Projection(nn.Module):
    """Single or multi-layer linear projection."""

    def __init__(self, in_planes: int, out_planes: Optional[int] = None, n_layers: int = 1, layer_type: int = 0):
        super().__init__()
        if out_planes is None:
            out_planes = in_planes
        self.layers = nn.Sequential()
        current_in = in_planes
        for i in range(n_layers):
            self.layers.add_module(f'{i}fc', nn.Linear(current_in, out_planes))
            if i < n_layers - 1 and layer_type > 1:
                self.layers.add_module(f'{i}relu', nn.LeakyReLU(0.2))
            current_in = out_planes
        self._init_weight()

    def _init_weight(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Discriminator(nn.Module):
    """Discriminator used by both GLASS paths."""

    def __init__(self, in_planes: int = 1536, n_layers: int = 2, hidden: int = 1024):
        super().__init__()
        current_hidden = in_planes if hidden is None else hidden
        self.body = nn.Sequential()
        for i in range(n_layers - 1):
            current_in = in_planes if i == 0 else current_hidden
            current_hidden = int(current_hidden // 1.5) if hidden is None else hidden
            self.body.add_module(
                f'block{i + 1}',
                nn.Sequential(
                    nn.Linear(current_in, current_hidden),
                    nn.BatchNorm1d(current_hidden),
                    nn.LeakyReLU(0.2),
                ),
            )
        self.tail = nn.Linear(current_hidden, 1, bias=False)
        self._init_weight()

    def _init_weight(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x)
        return self.tail(x)


class LegacyFocalLossGLASS(nn.Module):
    """Binary focal loss used by the existing lightweight GLASS path."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = prob.clamp(min=1e-7, max=1 - 1e-7)
        pt = target * prob + (1 - target) * (1 - prob)
        focal_weight = (1 - pt) ** self.gamma
        bce = -target * torch.log(prob) - (1 - target) * torch.log(1 - prob)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        return (alpha_weight * focal_weight * bce).mean()


class ReferenceFocalLossGLASS(nn.Module):
    """Official GLASS focal loss on 2-class probabilities."""

    def __init__(self, alpha=None, gamma: float = 2.0, smooth: float = 1e-5, size_average: bool = True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

    def forward(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_class = logit.shape[1]
        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))

        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)

        alpha = self.alpha
        if alpha is None:
            alpha = torch.ones(num_class, 1, device=logit.device, dtype=logit.dtype)
        elif isinstance(alpha, (list, np.ndarray)):
            alpha = torch.as_tensor(alpha, dtype=logit.dtype, device=logit.device).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1, device=logit.device, dtype=logit.dtype)
            alpha = alpha * (1 - self.alpha)
            alpha[0] = self.alpha
        else:
            raise TypeError(f'Unsupported alpha type: {type(alpha)!r}')

        idx = target.long()
        one_hot = torch.zeros(target.size(0), num_class, device=logit.device, dtype=logit.dtype)
        one_hot = one_hot.scatter_(1, idx, 1)
        if self.smooth:
            one_hot = torch.clamp(one_hot, self.smooth / (num_class - 1), 1.0 - self.smooth)

        pt = (one_hot * logit).sum(1) + self.smooth
        logpt = pt.log()
        alpha = alpha[idx].squeeze()
        loss = -alpha * torch.pow(1 - pt, self.gamma) * logpt
        return loss.mean() if self.size_average else loss.sum()


class RescaleSegmentor:
    """Official GLASS score-map upsampling with Gaussian smoothing."""

    def __init__(self, target_size: int = 288, smoothing: float = 4.0):
        self.target_size = target_size
        self.smoothing = smoothing

    def convert_to_segmentation(self, patch_scores: torch.Tensor | np.ndarray):
        with torch.no_grad():
            if isinstance(patch_scores, np.ndarray):
                patch_scores = torch.from_numpy(patch_scores)
            scores = patch_scores.float().unsqueeze(1)
            scores = F.interpolate(
                scores,
                size=self.target_size,
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)
            maps = scores.cpu().numpy()
        return [ndimage.gaussian_filter(score_map, sigma=self.smoothing) for score_map in maps]


@MODELS.register_module(force=True)
class GLASSDetector(BaseADModel):
    """GLASS detector with a fast legacy path and a strict official path."""

    def __init__(
        self,
        backbone='wide_resnet50_2',
        target_dim: int = 1536,
        pretrain_embed_dim: int = 1536,
        patchsize: int = 3,
        patchstride: int = 1,
        pre_proj: int = 1,
        proj_layer_type: int = 0,
        dsc_layers: int = 2,
        dsc_hidden: int = 1024,
        strict: bool = False,
        # Legacy path params
        gas_steps: int = 5,
        gas_lr: float = 0.5,
        gas_truncate: bool = True,
        dtd_path: Optional[str] = 'auto',
        anomaly_ratio: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        bce_weight: float = 1.0,
        las_weight: float = 1.0,
        # Strict official path params
        mining: int = 1,
        noise: float = 0.015,
        radius: float = 0.75,
        p: float = 0.5,
        svd: int = 0,
        distribution: int = 0,
        step: Optional[int] = None,
        limit: int = 392,
        smoothing: float = 4.0,
        image_size: int = 288,
        distribution_meta_path: Optional[str] = None,
        data_preprocessor=None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        if isinstance(backbone, str):
            backbone = dict(
                type='FeatureExtractor',
                backbone_name=backbone,
                out_indices=(2, 3),
                frozen=True,
            )
        self.feature_extractor = MODELS.build(backbone)
        channels = self.feature_extractor.channels

        self.patch_maker = PatchMaker(patchsize, stride=patchstride)
        self.preprocessing = Preprocessing(input_dims=[channels[0], channels[1]], output_dim=pretrain_embed_dim)
        self.aggregator = Aggregator(target_dim=target_dim)

        self.pre_proj = pre_proj
        if self.pre_proj > 0:
            self.projection = Projection(target_dim, target_dim, n_layers=pre_proj, layer_type=proj_layer_type)

        self.discriminator = Discriminator(in_planes=target_dim, n_layers=dsc_layers, hidden=dsc_hidden)

        self.strict = bool(strict)
        self.gas_steps = gas_steps if step is None else int(step)
        self.gas_lr = gas_lr
        self.gas_truncate = gas_truncate
        self.anomaly_ratio = anomaly_ratio
        self.dtd_path = dtd_path
        self._dtd_texture_paths = self._load_dtd_textures(dtd_path) if not self.strict else []
        self._texture_cache = None

        self.legacy_focal_loss = LegacyFocalLossGLASS(alpha=focal_alpha, gamma=focal_gamma)
        self.strict_focal_loss = ReferenceFocalLossGLASS(gamma=focal_gamma)
        self.bce_weight = bce_weight
        self.las_weight = las_weight

        self.mining = int(mining)
        self.noise_std = float(noise)
        self.radius = float(radius)
        self.p = float(p)
        self.svd = int(svd)
        self.distribution = int(distribution)
        self.limit = int(limit)
        self.distribution_meta_path = distribution_meta_path
        self.segmentor = RescaleSegmentor(target_size=image_size, smoothing=smoothing)
        self.strict_center: Optional[torch.Tensor] = None

    def _load_dtd_textures(self, dtd_path: Optional[str]) -> list[str]:
        if not dtd_path:
            return []
        try:
            paths = resolve_dtd_texture_paths(dtd_path)
            logger.info('GLASS: loaded %d DTD textures for legacy LAS path', len(paths))
            return paths
        except OfflineModeError:
            raise
        except Exception as exc:  # pragma: no cover - depends on external assets/network
            logger.warning('GLASS: failed to resolve DTD textures (%s); legacy LAS will fall back to random noise.', exc)
            return []

    def _prepare_inputs(self, inputs) -> torch.Tensor:
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)
        return inputs

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor):
        """Extract patch-level features following the official GLASS embedding path."""
        raw_features = self.feature_extractor(x)
        batch_size = raw_features[0].shape[0]
        features = [self.patch_maker.patchify(feat, return_spatial_info=True) for feat in raw_features]
        patch_shapes = [item[1] for item in features]
        features = [item[0] for item in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            feat = features[i]
            patch_dims = patch_shapes[i]
            feat = feat.reshape(feat.shape[0], patch_dims[0], patch_dims[1], *feat.shape[2:])
            feat = feat.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = feat.shape
            feat = feat.reshape(-1, *feat.shape[-2:])
            feat = F.interpolate(
                feat.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)
            feat = feat.reshape(*perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1])
            feat = feat.permute(0, -2, -1, 1, 2, 3)
            feat = feat.reshape(len(feat), -1, *feat.shape[-3:])
            features[i] = feat

        features = [feat.reshape(-1, *feat.shape[-3:]) for feat in features]
        features = self.preprocessing(features)
        features = self.aggregator(features)

        h, w = ref_num_patches
        return features, batch_size, h, w

    def _sample_dtd_texture(self, h: int, w: int, device) -> torch.Tensor:
        if self._texture_cache is None and self._dtd_texture_paths and torch.cuda.is_available():
            textures = []
            for path in self._dtd_texture_paths[:500]:
                texture = cv2.imread(path, cv2.IMREAD_COLOR)
                if texture is None:
                    continue
                texture = cv2.resize(texture, (max(h, w), max(h, w)))
                texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
                texture = texture.astype(np.float32) / 255.0
                textures.append(torch.from_numpy(texture).permute(2, 0, 1))
            if textures:
                self._texture_cache = torch.stack(textures).to(device)

        if self._texture_cache is not None:
            idx = int(torch.randint(0, len(self._texture_cache), (1,), device=device).item())
            texture = self._texture_cache[idx]
            if texture.shape[-2:] != (h, w):
                texture = F.interpolate(
                    texture.unsqueeze(0),
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(0)
            return texture
        return torch.rand(3, h, w, device=device)

    def _generate_las_batch(self, inputs: torch.Tensor):
        batch_size, _, h, w = inputs.shape
        device = inputs.device
        las_imgs = inputs.clone()
        las_masks = torch.zeros(batch_size, h, w, device=device)
        apply_las = torch.rand(batch_size, device=device) < self.anomaly_ratio
        las_indices = apply_las.nonzero(as_tuple=True)[0]
        for idx in las_indices:
            mask = torch.rand(h, w, device=device)
            mask = (mask > 0.5).float()
            if float(mask.sum().item()) == 0:
                continue
            texture = self._sample_dtd_texture(h, w, device)
            las_imgs[idx] = inputs[idx] * (1 - mask) + texture * mask
            las_masks[idx] = mask
        return las_imgs, las_masks

    def _gas_anomaly_synthesis(self, feats: torch.Tensor) -> torch.Tensor:
        fake = feats.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([fake], lr=self.gas_lr)
        for _ in range(self.gas_steps):
            optimizer.zero_grad(set_to_none=True)
            score = self.discriminator(fake)
            loss = -score.mean()
            loss.backward()
            optimizer.step()

        if self.gas_truncate:
            center = feats.mean(0, keepdim=True)
            delta = fake.detach() - center
            max_norm = (feats - center).norm(dim=-1).max() * 1.5
            delta_norm = delta.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            scale = torch.clamp(max_norm / delta_norm, max=1.0)
            fake = center + delta * scale
        return fake.detach()

    def _stack_strict_training_fields(self, data_samples, device) -> tuple[torch.Tensor, torch.Tensor]:
        if not data_samples:
            raise ValueError('Strict GLASS training requires data_samples with aug and mask_s metadata.')
        aug_imgs = []
        masks_s = []
        for sample in data_samples:
            if not hasattr(sample, 'aug'):
                raise ValueError('Strict GLASS training requires aug in data_samples metainfo.')
            if not hasattr(sample, 'mask_s'):
                raise ValueError('Strict GLASS training requires mask_s in data_samples metainfo.')
            aug_imgs.append(sample.aug.to(device))
            masks_s.append(sample.mask_s.to(device))
        return torch.stack(aug_imgs), torch.stack(masks_s)

    def _distribution_from_file(self, meta_path: str, class_key: str) -> int:
        if not meta_path:
            raise FileNotFoundError('GLASS strict training requires a distribution metadata path.')
        pandas = require_optional_module(
            'pandas', extra='glass', feature='GLASS distribution metadata')
        frame = pandas.read_excel(meta_path)
        row = frame.loc[frame['Class'] == class_key]
        if row.empty:
            raise KeyError(f'GLASS distribution metadata has no row for {class_key!r} in {meta_path}.')
        return int(row.iloc[0]['Distribution'])

    def _resolve_strict_svd(self, train_dataloader) -> int:
        dataset = getattr(train_dataloader, 'dataset', None)
        cls_names = getattr(dataset, 'cls_names', []) or []
        cls_name = cls_names[0] if cls_names else 'unknown'
        class_key = f"{getattr(dataset, 'dataset_name', 'mvtec')}_{cls_name}"
        distribution_mode = int(getattr(dataset, 'distribution', self.distribution))
        meta_path = getattr(dataset, 'distribution_meta_path', None) or self.distribution_meta_path

        if distribution_mode == 2:
            return 0
        if distribution_mode == 3:
            return 1
        if distribution_mode == 4:
            return 1 - self._distribution_from_file(meta_path, class_key)
        if distribution_mode == 1:
            avg_img = None
            count = 0
            with torch.no_grad():
                for data_batch in train_dataloader:
                    batch = self.data_preprocessor(data_batch, False)
                    inputs = self._prepare_inputs(batch['inputs'])
                    batch_mean = inputs.mean(dim=0)
                    avg_img = batch_mean if avg_img is None else avg_img + batch_mean
                    count += 1
            if avg_img is None or count == 0:
                raise RuntimeError('GLASS strict distribution auto-judge received an empty dataloader.')
            avg_img = avg_img / count
            return distribution_judge(tensor_to_bgr_image(avg_img))
        return self._distribution_from_file(meta_path, class_key)

    @torch.no_grad()
    def prepare_strict_epoch(self, train_dataloader) -> None:
        """Recompute the official GLASS center at the start of every epoch."""
        self.svd = self._resolve_strict_svd(train_dataloader)
        center = None
        batch_count = 0
        self.eval()
        for data_batch in train_dataloader:
            batch = self.data_preprocessor(data_batch, False)
            inputs = self._prepare_inputs(batch['inputs'])
            feats, batch_size, _, _ = self.extract_features(inputs)
            if self.pre_proj > 0:
                feats = self.projection(feats)
            feats = feats.reshape(batch_size, -1, feats.shape[-1])
            batch_mean = feats.mean(dim=0)
            center = batch_mean if center is None else center + batch_mean
            batch_count += 1
        if center is None or batch_count == 0:
            raise RuntimeError('GLASS strict center computation received an empty dataloader.')
        self.strict_center = center / batch_count

    def _legacy_loss(self, inputs: torch.Tensor):
        feats, batch_size, _, _ = self.extract_features(inputs)
        if self.pre_proj > 0:
            feats = self.projection(feats)

        normal_scores = torch.sigmoid(self.discriminator(feats))
        normal_loss = F.binary_cross_entropy(normal_scores, torch.zeros_like(normal_scores))

        gas_feats = self._gas_anomaly_synthesis(feats)
        gas_scores = torch.sigmoid(self.discriminator(gas_feats))
        gas_loss = F.binary_cross_entropy(gas_scores, torch.ones_like(gas_scores))

        las_imgs, las_masks = self._generate_las_batch(inputs)
        with torch.no_grad():
            las_feats, _, h_las, w_las = self.extract_features(las_imgs)
        if self.pre_proj > 0:
            las_feats = self.projection(las_feats)
        las_scores = torch.sigmoid(self.discriminator(las_feats)).reshape(batch_size, h_las, w_las)
        las_scores_up = F.interpolate(
            las_scores.unsqueeze(1),
            size=inputs.shape[-2:],
            mode='bilinear',
            align_corners=False,
        ).squeeze(1)
        las_loss = self.legacy_focal_loss(las_scores_up, las_masks)
        total_loss = self.bce_weight * (normal_loss + gas_loss) + self.las_weight * las_loss
        return {
            'loss': total_loss,
            'loss_normal': normal_loss,
            'loss_gas': gas_loss,
            'loss_las': las_loss,
        }

    def _strict_loss(self, inputs: torch.Tensor, data_samples):
        if self.strict_center is None:
            raise RuntimeError(
                'GLASS strict path requires prepare_strict_epoch() to run before training. '
                'Use GLASSTrainLoop or call prepare_strict_epoch() manually.'
            )

        aug_inputs, mask_s = self._stack_strict_training_fields(data_samples, inputs.device)
        if self.pre_proj > 0:
            fake_feats = self.projection(self.extract_features(aug_inputs)[0])
            true_feats = self.projection(self.extract_features(inputs)[0])
        else:
            fake_feats = self.extract_features(aug_inputs)[0]
            true_feats = self.extract_features(inputs)[0]

        mask_s_gt = mask_s.reshape(-1, 1).float()
        if int(mask_s_gt.sum().item()) == 0:
            raise RuntimeError('GLASS strict path received an empty mask_s after packing.')

        noise = torch.normal(0, self.noise_std, true_feats.shape, device=inputs.device)
        gaus_feats = true_feats + noise

        center = self.strict_center.to(inputs.device).repeat(inputs.shape[0], 1, 1).reshape(-1, true_feats.shape[-1])
        normal_mask = mask_s_gt[:, 0] == 0
        anomaly_mask = mask_s_gt[:, 0] == 1

        true_points = torch.cat([fake_feats[normal_mask], true_feats], dim=0)
        center_points = torch.cat([center[normal_mask], center], dim=0)
        dist_t = torch.norm(true_points - center_points, dim=1)
        r_t = torch.quantile(dist_t, q=self.radius)

        bce_loss = None
        for step_idx in range(self.gas_steps + 1):
            scores = torch.sigmoid(self.discriminator(torch.cat([true_feats, gaus_feats], dim=0)))
            true_scores = scores[:len(true_feats)]
            gaus_scores = scores[len(true_feats):]
            true_loss = F.binary_cross_entropy(true_scores, torch.zeros_like(true_scores))
            gaus_loss = F.binary_cross_entropy(gaus_scores, torch.ones_like(gaus_scores))
            bce_loss = true_loss + gaus_loss

            if step_idx == self.gas_steps:
                break
            if self.mining == 0:
                break

            grad = torch.autograd.grad(gaus_loss, [gaus_feats], retain_graph=True)[0]
            grad_norm = torch.norm(grad, dim=1, keepdim=True)
            grad_normalized = grad / (grad_norm + 1e-10)
            with torch.no_grad():
                gaus_feats.add_(0.001 * grad_normalized)

            if (step_idx + 1) % 5 == 0:
                dist_g = torch.norm(gaus_feats - center, dim=1)
                proj_feats = center if self.svd == 1 else true_feats
                radius = r_t if self.svd == 1 else torch.tensor(0.5, device=inputs.device, dtype=true_feats.dtype)
                h = gaus_feats - proj_feats
                h_norm = dist_g if self.svd == 1 else torch.norm(h, dim=1)
                alpha = torch.clamp(h_norm, min=float(radius.item()), max=float((2 * radius).item()))
                proj = (alpha / (h_norm + 1e-10)).view(-1, 1)
                gaus_feats = proj_feats + proj * h

        assert bce_loss is not None

        fake_points = fake_feats[anomaly_mask]
        true_points = true_feats[anomaly_mask]
        center_fake = center[anomaly_mask]
        if fake_points.numel() == 0:
            raise RuntimeError('GLASS strict path received no anomalous feature positions from mask_s.')

        dist_f = torch.norm(fake_points - center_fake, dim=1)
        proj_feats = center_fake if self.svd == 1 else true_points
        projection_radius = r_t if self.svd == 1 else torch.tensor(1.0, device=inputs.device, dtype=true_feats.dtype)

        if self.svd == 1:
            h = fake_points - proj_feats
            h_norm = dist_f
            alpha = torch.clamp(h_norm, min=float((2 * projection_radius).item()), max=float((4 * projection_radius).item()))
            proj = (alpha / (h_norm + 1e-10)).view(-1, 1)
            fake_points = proj_feats + proj * h
            fake_feats = fake_feats.clone()
            fake_feats[anomaly_mask] = fake_points

        fake_scores = torch.sigmoid(self.discriminator(fake_feats))
        if self.p > 0:
            fake_dist = (fake_scores - mask_s_gt) ** 2
            hard_threshold = torch.quantile(fake_dist, q=self.p)
            hard_mask = fake_dist >= hard_threshold
            fake_scores_selected = fake_scores[hard_mask].view(-1, 1)
            mask_selected = mask_s_gt[hard_mask].view(-1, 1)
        else:
            fake_scores_selected = fake_scores
            mask_selected = mask_s_gt

        output = torch.cat([1 - fake_scores_selected, fake_scores_selected], dim=1)
        focal_loss = self.strict_focal_loss(output, mask_selected)
        total_loss = bce_loss + focal_loss
        return {
            'loss': total_loss,
            'loss_bce': bce_loss,
            'loss_focal': focal_loss,
            'svd': torch.tensor(float(self.svd), device=inputs.device),
        }

    def _legacy_predict(self, inputs: torch.Tensor, data_samples=None):
        feats, batch_size, h_p, w_p = self.extract_features(inputs)
        if self.pre_proj > 0:
            feats = self.projection(feats)
        patch_scores = torch.sigmoid(self.discriminator(feats)).squeeze(-1)
        score_map = patch_scores.reshape(batch_size, h_p, w_p)
        score_map = F.interpolate(
            score_map.unsqueeze(1),
            size=inputs.shape[-2:],
            mode='bilinear',
            align_corners=False,
        ).squeeze(1)
        img_scores = score_map.view(batch_size, -1).max(dim=1).values
        return build_predict_results(data_samples, img_scores, score_map)

    def _strict_predict(self, inputs: torch.Tensor, data_samples=None):
        feats, batch_size, h_p, w_p = self.extract_features(inputs)
        if self.pre_proj > 0:
            feats = self.projection(feats)

        patch_scores = torch.sigmoid(self.discriminator(feats))
        score_map = self.patch_maker.unpatch_scores(patch_scores, batchsize=batch_size)
        score_map = score_map.reshape(batch_size, h_p, w_p)
        maps = self.segmentor.convert_to_segmentation(score_map)

        img_scores = self.patch_maker.score(self.patch_maker.unpatch_scores(patch_scores, batchsize=batch_size))
        if isinstance(img_scores, torch.Tensor):
            img_scores = img_scores.detach().cpu()
        return build_predict_results(data_samples, img_scores, maps)

    def train_step(self, data, optim_wrapper):
        if not self.strict:
            data = self.data_preprocessor(data, True)
            inputs = self._prepare_inputs(data['inputs'])
            losses = self._legacy_loss(inputs)
            loss = losses['loss']
            if isinstance(optim_wrapper, OptimWrapperDict):
                for ow in optim_wrapper.values():
                    ow.zero_grad()
                loss.backward()
                for ow in optim_wrapper.values():
                    ow.step()
            else:
                optim_wrapper.zero_grad()
                loss.backward()
                optim_wrapper.step()
            return losses

        if not isinstance(optim_wrapper, OptimWrapperDict) or 'discriminator' not in optim_wrapper:
            raise TypeError(
                'Strict GLASS training requires an OptimWrapperDict with at least a discriminator optimizer.'
            )

        data = self.data_preprocessor(data, True)
        inputs = self._prepare_inputs(data['inputs'])
        data_samples = data.get('data_samples', None)

        projection_optim = optim_wrapper['projection'] if 'projection' in optim_wrapper else None
        discriminator_optim = optim_wrapper['discriminator']

        if projection_optim is not None:
            projection_optim.zero_grad()
        discriminator_optim.zero_grad()

        losses = self(inputs, data_samples, mode='loss')
        scaled_loss = discriminator_optim.scale_loss(losses['loss'])
        discriminator_optim.backward(scaled_loss)

        if projection_optim is not None:
            projection_optim._inner_count += 1
            if projection_optim.should_update():
                projection_optim.step()
                projection_optim.zero_grad()

        if discriminator_optim.should_update():
            discriminator_optim.step()
            discriminator_optim.zero_grad()

        detached = {}
        for key, value in losses.items():
            detached[key] = value.detach() if torch.is_tensor(value) else value
        return detached

    def forward(self, inputs, data_samples=None, mode: str = 'tensor'):
        inputs = self._prepare_inputs(inputs)

        if mode == 'loss':
            if self.strict:
                return self._strict_loss(inputs, data_samples)
            return self._legacy_loss(inputs)

        if mode == 'predict':
            if self.strict:
                return self._strict_predict(inputs, data_samples)
            return self._legacy_predict(inputs, data_samples)

        feats, _, _, _ = self.extract_features(inputs)
        if self.pre_proj > 0:
            feats = self.projection(feats)
        return feats

    def train(self, mode: bool = True):
        super().train(mode)
        self.feature_extractor.eval()
        return self
