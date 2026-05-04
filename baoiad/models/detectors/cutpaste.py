"""CutPaste: Self-Supervised Learning for Anomaly Detection (CVPR 2021).

Reference: Li et al., "CutPaste: Self-Supervised Learning for Anomaly
Detection and Localization", CVPR 2021.

Self-supervised augmentation-based anomaly detection.
Training: apply CutPaste/CutPaste-Scar augmentation, train 3-way classifier.
Testing: GDE (Gaussian Density Estimation) Mahalanobis distance scoring.

Architecture (from paper Sec 4.1):
- Pretrained backbone (ResNet-18 or EfficientNet-B4)
- Projection head: Linear(enc_dim, 256) → ReLU → Linear(256, 128)
- 3-way classifier: Linear(128, 3) for normal/CutPaste/Scar
- GDE: Mahalanobis distance on 512-dim backbone embeddings (not 128-dim projection head!)
  Reference: pytorch-cutpaste/eval.py:73-74 returns embeds from self.resnet18(x)
"""
import math
import random
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.covariance import LedoitWolf
from torch.utils.data import DataLoader

from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS
from baoiad.models.base_ad_model import BaseADModel


class CutPasteAugmentation:
    """CutPaste and CutPaste-Scar augmentations.

    Operates on tensors in ``[0, 1]`` and mirrors the sampling logic from
    Runinho/pytorch-cutpaste as closely as possible while staying inside the
    detector.
    """

    def __init__(self, area_ratio=(0.02, 0.15), aspect_ratio=0.3,
                 scar_width=(2, 16), scar_length=(10, 25), scar_rotation=(-45, 45),
                 scar_min_changed_ratio=0.0, scar_max_attempts=1,
                 color_jitter=True):
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio  # Single value for log-uniform sampling
        self.scar_width = scar_width
        self.scar_length = scar_length
        self.scar_rotation = scar_rotation
        self.scar_min_changed_ratio = scar_min_changed_ratio
        self.scar_max_attempts = max(1, int(scar_max_attempts))
        self.color_jitter = color_jitter

        # ColorJitter params matching reference: brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
        if color_jitter:
            self.jitter = torchvision.transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
        else:
            self.jitter = None

    @staticmethod
    def _sample_offset(limit: int) -> int:
        if limit <= 0:
            return 0
        return int(random.uniform(0, limit))

    def cutpaste(self, img):
        """Apply CutPaste augmentation to image tensor (C, H, W)."""
        _, H, W = img.shape
        ratio_area = random.uniform(self.area_ratio[0], self.area_ratio[1]) * H * W

        log_ratio = torch.log(torch.tensor((self.aspect_ratio, 1.0 / self.aspect_ratio)))
        aspect = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

        cut_w = max(1, min(int(round(math.sqrt(ratio_area * aspect))), W))
        cut_h = max(1, min(int(round(math.sqrt(ratio_area / aspect))), H))

        from_location_h = self._sample_offset(H - cut_h)
        from_location_w = self._sample_offset(W - cut_w)
        patch = img[:, from_location_h:from_location_h + cut_h,
                    from_location_w:from_location_w + cut_w].clone()

        if self.jitter is not None:
            patch_pil = torchvision.transforms.ToPILImage()(patch.detach().cpu())
            patch_pil = self.jitter(patch_pil)
            patch = torchvision.transforms.ToTensor()(patch_pil).to(
                device=img.device,
                dtype=img.dtype,
            )

        to_location_h = self._sample_offset(H - cut_h)
        to_location_w = self._sample_offset(W - cut_w)

        result = img.clone()
        result[:, to_location_h:to_location_h + cut_h,
               to_location_w:to_location_w + cut_w] = patch
        return result

    def cutpaste_scar(self, img):
        """Apply CutPaste-Scar augmentation (thin strip).

        Reference params: width=[2,16], height=[10,25], rotation=[-45,45].
        Uses RGBA rotation + alpha mask blending to match the reference code.
        """
        best_result = None
        best_ratio = -1.0
        for _ in range(self.scar_max_attempts):
            result, changed_ratio = self._cutpaste_scar_once(img)
            if changed_ratio > best_ratio:
                best_result = result
                best_ratio = changed_ratio
            if changed_ratio >= self.scar_min_changed_ratio:
                return result
        return best_result if best_result is not None else img.clone()

    def _cutpaste_scar_once(self, img):
        _, H, W = img.shape
        cut_w = max(1, min(int(random.uniform(*self.scar_width)), W))
        cut_h = max(1, min(int(random.uniform(*self.scar_length)), H))

        from_location_h = self._sample_offset(H - cut_h)
        from_location_w = self._sample_offset(W - cut_w)
        patch = img[:, from_location_h:from_location_h + cut_h,
                    from_location_w:from_location_w + cut_w].clone()

        if self.jitter is not None:
            patch_pil = torchvision.transforms.ToPILImage()(patch.detach().cpu())
            patch_pil = self.jitter(patch_pil)
            patch = torchvision.transforms.ToTensor()(patch_pil).to(
                device=img.device,
                dtype=img.dtype,
            )

        from PIL import Image

        patch_np = (patch.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        patch_pil = Image.fromarray(patch_np).convert('RGBA')
        patch_rotated = patch_pil.rotate(
            random.uniform(*self.scar_rotation),
            expand=True,
            resample=Image.BILINEAR,
        )

        alpha_np = np.asarray(patch_rotated.split()[-1], dtype=np.float32) / 255.0
        patch_rgb_np = np.asarray(patch_rotated.convert('RGB'), dtype=np.float32)
        patch_rgb_np = patch_rgb_np.transpose(2, 0, 1) / 255.0

        rot_h, rot_w = alpha_np.shape
        rot_h = min(rot_h, H)
        rot_w = min(rot_w, W)

        to_location_h = self._sample_offset(H - rot_h)
        to_location_w = self._sample_offset(W - rot_w)

        result = img.clone()
        patch_tensor = torch.from_numpy(patch_rgb_np[:, :rot_h, :rot_w]).to(
            device=img.device,
            dtype=img.dtype,
        )
        alpha_tensor = torch.from_numpy(alpha_np[:rot_h, :rot_w]).to(
            device=img.device,
            dtype=img.dtype,
        ).unsqueeze(0)
        region = result[:, to_location_h:to_location_h + rot_h,
                        to_location_w:to_location_w + rot_w]
        result[:, to_location_h:to_location_h + rot_h,
               to_location_w:to_location_w + rot_w] = (
            alpha_tensor * patch_tensor + (1 - alpha_tensor) * region
        )
        changed_ratio = self._changed_ratio(img, result)
        return result, changed_ratio

    @staticmethod
    def _changed_ratio(original, augmented, threshold=0.05):
        diff = (augmented - original).abs().mean(dim=0)
        return float((diff > threshold).float().mean().item())

    def __call__(self, img):
        """Randomly apply one of the two augmentations."""
        if random.random() > 0.5:
            return self.cutpaste(img)
        return self.cutpaste_scar(img)


@MODELS.register_module(force=True)
class CutPasteDetector(BaseADModel):
    """CutPaste anomaly detector with self-supervised augmentation.

    Args:
        backbone: Backbone config or name.
        proj_dim: Projection head output dimension (default 128, paper spec).
        num_classes: 2 (normal vs CutPaste) or 3 (normal, CutPaste, Scar).
        freeze_epochs: Legacy warmup length for epoch-based training, or the
            reference warmup steps when ``freeze_iters`` is unset.
        freeze_iters: Explicit warmup steps before unfreezing the backbone in
            iteration-based training. Use this for CutPaste reference configs.
        reference_total_iters: Reference training budget used to interpret the
            legacy ``freeze_epochs`` value in iteration-based training.
        head_dims: Projection head layer dimensions. Paper uses (256, 128).
        use_bn: Whether to use BatchNorm in projection head. Paper doesn't use BN.
        pre_cutpaste_jitter: Whether to apply ColorJitter to whole image before CutPaste.
    """

    def __init__(self, backbone: Union[str, dict] = 'resnet18', proj_dim=128,
                 num_classes=3, freeze_epochs=10,
                 freeze_iters: Optional[int] = None,
                 reference_total_iters=256,
                 head_dims=(256, 128),
                 use_bn=False,
                 force_backbone_eval_while_frozen=True,
                 keep_backbone_bn_eval=False,
                 stop_grad_backbone_while_frozen=False,
                 embedding_source: Optional[str] = None,
                 train_embedding_source='features_only',
                 density_embedding_source='features_only',
                 normalize_embeddings=True,
                 pre_cutpaste_jitter=True,
                 cutpaste_area_ratio=(0.02, 0.15),
                 cutpaste_aspect_ratio=0.3,
                 scar_width=(2, 16),
                 scar_length=(10, 25),
                 scar_rotation=(-45, 45),
                 scar_min_changed_ratio=0.0,
                 scar_max_attempts=1,
                 loss=dict(type='CrossEntropyLoss'),
                 data_preprocessor=None, init_cfg=None, **kwargs):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        from baoiad.models.backbone_utils import build_feature_extractor
        self.backbone = build_feature_extractor(
            backbone, default_out_indices=(4,), default_frozen=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if embedding_source is not None:
            train_embedding_source = embedding_source
            density_embedding_source = embedding_source
        self.train_embedding_source = self._validate_embedding_source(train_embedding_source)
        self.density_embedding_source = self._validate_embedding_source(density_embedding_source)

        feat_dim = self._backbone_embedding_dim(self.train_embedding_source)

        # Projection head aligned with pytorch-cutpaste reference:
        # Linear → BatchNorm → ReLU for ALL layers including final
        # Reference: pytorch-cutpaste/model.py:16-21
        head_dims = list(head_dims)
        proj_layers = []

        # All layers: Linear → BatchNorm → ReLU (matches original)
        for i in range(len(head_dims)):
            in_dim = feat_dim if i == 0 else head_dims[i - 1]
            out_dim = head_dims[i]
            proj_layers.append(nn.Linear(in_dim, out_dim, bias=True))
            if use_bn:
                proj_layers.append(nn.BatchNorm1d(out_dim))
            proj_layers.append(nn.ReLU(inplace=True))

        self.head = nn.Sequential(*proj_layers)
        self.classifier = nn.Linear(head_dims[-1], num_classes)

        self.num_classes = num_classes
        self.normalize_embeddings = normalize_embeddings
        self.force_backbone_eval_while_frozen = force_backbone_eval_while_frozen
        self.keep_backbone_bn_eval = keep_backbone_bn_eval
        self.stop_grad_backbone_while_frozen = stop_grad_backbone_while_frozen
        # ImageNet normalization stats (used by default pipeline)
        self.register_buffer('imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('imagenet_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # ColorJitter for whole image BEFORE CutPaste (matches reference run_training.py:57)
        # Reference: train_transform.transforms.append(ColorJitter(0.1, 0.1, 0.1, 0.1))
        self.pre_cutpaste_jitter = pre_cutpaste_jitter
        if pre_cutpaste_jitter:
            self.color_jitter = torchvision.transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
        else:
            self.color_jitter = None

        # Enable ColorJitter since we denormalize before augmentation
        self.augmentation = CutPasteAugmentation(
            area_ratio=cutpaste_area_ratio,
            aspect_ratio=cutpaste_aspect_ratio,
            scar_width=scar_width,
            scar_length=scar_length,
            scar_rotation=scar_rotation,
            scar_min_changed_ratio=scar_min_changed_ratio,
            scar_max_attempts=scar_max_attempts,
            color_jitter=True,
        )
        self.loss_fn = MODELS.build(loss)
        self.freeze_epochs = freeze_epochs
        self.freeze_iters = freeze_iters
        self.reference_total_iters = reference_total_iters
        self._backbone_unfrozen = False

        # Re-fit GDE at every validation (MemoryBankHook checks this flag)
        self.always_refit = True

        # GDE (Gaussian Density Estimator)
        self._gde_mean: Optional[torch.Tensor] = None
        self._gde_cov_inv: Optional[torch.Tensor] = None
        self._last_fit_num_samples = 0

    def _denormalize(self, x):
        """Convert ImageNet-normalized tensor to [0,1] range."""
        return x * self.imagenet_std.to(x.device) + self.imagenet_mean.to(x.device)

    def _normalize(self, x):
        """Convert [0,1] tensor to ImageNet-normalized."""
        return (x - self.imagenet_mean.to(x.device)) / self.imagenet_std.to(x.device)

    @staticmethod
    def _validate_embedding_source(source: str) -> str:
        if source not in {'features_only', 'pre_logits'}:
            raise ValueError(f'Unsupported embedding_source: {source}')
        return source

    def _backbone_embedding_dim(self, source):
        if source == 'pre_logits':
            if getattr(self.backbone, 'num_features', None) is not None:
                return int(self.backbone.num_features)
            if hasattr(self.backbone, 'net') and getattr(self.backbone.net, 'num_features', None) is not None:
                return int(self.backbone.net.num_features)
        ch = getattr(self.backbone, 'out_channels', None)
        if ch:
            return int(ch[-1])
        raise ValueError(f'Unable to infer CutPaste embedding dim for source={self.embedding_source!r}')

    def _extract_pre_logits_embedding(self, x):
        if hasattr(self.backbone, 'forward_pre_logits'):
            embeddings = self.backbone.forward_pre_logits(x)
        elif hasattr(self.backbone, 'net') and hasattr(self.backbone.net, 'forward_features'):
            features = self.backbone.net.forward_features(x)
            if hasattr(self.backbone.net, 'forward_head'):
                embeddings = self.backbone.net.forward_head(features, pre_logits=True)
            else:
                embeddings = features
        else:  # pragma: no cover - guarded by config/tests
            raise AttributeError('Backbone does not support pre_logits extraction.')
        if embeddings.ndim > 2:
            embeddings = self.avgpool(embeddings).flatten(1)
        return embeddings

    def _extract_features_only_embedding(self, x):
        if hasattr(self.backbone, 'forward_intermediates'):
            feats = self.backbone.forward_intermediates(
                x,
                indices=getattr(self.backbone, 'out_indices', None),
            )
        else:
            feats = self.backbone(x)
        feat = feats[-1] if isinstance(feats, (list, tuple)) else feats
        if feat.ndim > 2:
            return self.avgpool(feat).flatten(1)
        return feat.flatten(1)

    def extract_backbone_embedding(self, x, source=None):
        """Extract pooled backbone embeddings.

        Defaults to the density estimator source for backward compatibility.
        """
        source = self.density_embedding_source if source is None else self._validate_embedding_source(source)
        if source == 'pre_logits':
            return self._extract_pre_logits_embedding(x)
        return self._extract_features_only_embedding(x)

    def extract_train_backbone_embedding(self, x):
        if not self._backbone_unfrozen and self.stop_grad_backbone_while_frozen:
            with torch.no_grad():
                return self.extract_backbone_embedding(x, source=self.train_embedding_source)
        return self.extract_backbone_embedding(x, source=self.train_embedding_source)

    def extract_density_backbone_embedding(self, x):
        return self.extract_backbone_embedding(x, source=self.density_embedding_source)

    def extract_head_embedding(self, x):
        return self.head(self.extract_train_backbone_embedding(x))

    def extract_embedding(self, x):
        return self.extract_head_embedding(x)

    def set_epoch_info(self, epoch, max_epochs):
        """Called by MemoryBankHook. Used for two-phase training (epoch-based)."""
        if not self._backbone_unfrozen and epoch >= self.freeze_epochs:
            # Unfreeze all backbone parameters (including BN) - match original
            for param in self.backbone.parameters():
                param.requires_grad = True
            self._backbone_unfrozen = True
            self._apply_backbone_bn_policy()

    def set_iter_info(self, iter, max_iters):
        """Called by MemoryBankHook. Used for two-phase training (iteration-based).

        The reference implementation unfreezes at step 20 out of 256 total steps.
        Prefer ``freeze_iters`` to encode that directly. ``freeze_epochs`` is kept
        as a backward-compatible fallback for older configs.
        """
        if not self._backbone_unfrozen:
            if self.freeze_iters is not None:
                freeze_iters = self.freeze_iters
            else:
                freeze_iters = int(self.freeze_epochs / self.reference_total_iters * max_iters)
            if iter >= freeze_iters:
                for param in self.backbone.parameters():
                    param.requires_grad = True
                self._backbone_unfrozen = True
                self._apply_backbone_bn_policy()

    def _apply_backbone_bn_policy(self):
        if not self.keep_backbone_bn_eval:
            return
        for module in self.backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    def _build_fit_dataloader(self, dataloader):
        """Build a deterministic loader over the original train split.

        The reference ``eval.py`` fits density on the plain MVTec train split
        with ``shuffle=False``. Training loaders in BaoIAD may wrap that split
        in ``RepeatDataset`` with shuffling for optimization, so we unwrap the
        repeated dataset here before fitting the Gaussian density estimator.
        """
        dataset = getattr(dataloader, 'dataset', None)
        if dataset is None:
            return dataloader

        if hasattr(dataset, 'times') and hasattr(dataset, 'dataset'):
            dataset = dataset.dataset

        return DataLoader(
            dataset,
            batch_size=getattr(dataloader, 'batch_size', 1) or 1,
            shuffle=False,
            num_workers=0,
            collate_fn=getattr(dataloader, 'collate_fn', None),
            drop_last=False,
        )

    @torch.no_grad()
    def fit_gaussian_density(self, dataloader, embedding_type='backbone'):
        """Fit Gaussian density on selected embeddings.

        Args:
            dataloader: Train dataloader or wrapper.
            embedding_type: ``'backbone'`` or ``'head'``.
        """
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        all_embeds = []
        fit_dataloader = self._build_fit_dataloader(dataloader)
        for data in fit_dataloader:
            imgs = data['inputs']
            if isinstance(imgs, (list, tuple)):
                imgs = torch.stack(imgs)
            imgs = imgs.to(device)
            if embedding_type in {'backbone', 'density_backbone'}:
                embeds = self.extract_density_backbone_embedding(imgs)
            elif embedding_type == 'train_backbone':
                embeds = self.extract_train_backbone_embedding(imgs)
            elif embedding_type == 'head':
                embeds = self.extract_head_embedding(imgs)
            else:  # pragma: no cover - guarded by caller/tests
                raise ValueError(f'Unsupported embedding_type: {embedding_type}')
            all_embeds.append(embeds.cpu())
        if not all_embeds:
            if was_training:
                self.train()
            return {
                'mean': None,
                'cov_inv': None,
                'num_samples': 0,
                'embedding_type': embedding_type,
            }
        all_embeds = torch.cat(all_embeds, dim=0)
        if self.normalize_embeddings:
            all_embeds = torch.nn.functional.normalize(all_embeds, p=2, dim=1)
        mean = all_embeds.mean(dim=0)
        lw = LedoitWolf()
        lw.fit(all_embeds.numpy())
        cov_inv = torch.from_numpy(lw.precision_).to(all_embeds.device)
        if was_training:
            self.train()
        return {
            'mean': mean,
            'cov_inv': cov_inv,
            'num_samples': int(all_embeds.shape[0]),
            'embedding_type': embedding_type,
        }

    @torch.no_grad()
    def fit(self, dataloader):
        """Fit GDE on training embeddings. Called by MemoryBankHook.

        Uses LedoitWolf shrinkage estimator for covariance (matches original
        pytorch-cutpaste/density.py:21). This is critical for stability with
        high-dimensional embeddings and small sample sizes.

        IMPORTANT: Uses 512-dim backbone embeddings (not 128-dim projection head)
        to match original pytorch-cutpaste behavior.
        Reference: pytorch-cutpaste/eval.py:73-74 + model.py:33-34

        NOTE: Density fitting must iterate the original train split once with
        ``shuffle=False``. Using the shuffled RepeatDataset batches from the
        training loop skews the train embedding distribution.
        """
        stats = self.fit_gaussian_density(dataloader, embedding_type='backbone')
        self._last_fit_num_samples = stats['num_samples']
        self._gde_mean = stats['mean']
        self._gde_cov_inv = stats['cov_inv']

    def _mahalanobis_score_from_embeddings(self, embeddings, mean, cov_inv):
        if self.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        delta = embeddings - mean.to(embeddings.device)
        cov_inv = cov_inv.to(embeddings.device)
        dist = (delta @ cov_inv * delta).sum(dim=1)
        dist = dist.clamp(min=1e-10)
        return dist.sqrt()

    def score_with_backbone_mahalanobis(self, inputs, mean=None, cov_inv=None):
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)
        if mean is None or cov_inv is None:
            mean = self._gde_mean
            cov_inv = self._gde_cov_inv
        embeddings = self.extract_density_backbone_embedding(inputs)
        return self._mahalanobis_score_from_embeddings(embeddings, mean, cov_inv)

    def score_with_head_mahalanobis(self, inputs, mean, cov_inv):
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)
        embeddings = self.extract_head_embedding(inputs)
        return self._mahalanobis_score_from_embeddings(embeddings, mean, cov_inv)

    def score_with_classifier_prob(self, inputs):
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)
        embeddings = self.extract_head_embedding(inputs)
        logits = self.classifier(embeddings)
        probs = torch.softmax(logits, dim=1)
        return 1.0 - probs[:, 0]

    def _mahalanobis_score(self, inputs):
        """Compute Mahalanobis distance from the fitted backbone GDE."""
        return self.score_with_backbone_mahalanobis(inputs)

    def build_augmented_views(self, inputs):
        """Build the 3-way CutPaste training batch.

        Returns both normalized tensors used by the classifier and their
        corresponding ``[0, 1]`` images for diagnostics.
        """
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        batch_size = inputs.shape[0]
        inputs_01 = self._denormalize(inputs).clamp(0.0, 1.0)

        if self.color_jitter is not None:
            jittered_01 = torch.stack([self.color_jitter(img) for img in inputs_01])
        else:
            jittered_01 = inputs_01
        jittered_01 = jittered_01.clamp(0.0, 1.0)
        normal = self._normalize(jittered_01)

        cutpaste_01 = torch.stack([self.augmentation.cutpaste(jittered_01[i]) for i in range(batch_size)])
        cutpaste_01 = cutpaste_01.clamp(0.0, 1.0)
        cutpaste = self._normalize(cutpaste_01)

        outputs = {
            'inputs_01': inputs_01,
            'normal_01': jittered_01,
            'normal': normal,
            'cutpaste_01': cutpaste_01,
            'cutpaste': cutpaste,
        }

        all_imgs = [normal, cutpaste]
        labels = [
            torch.zeros(batch_size, dtype=torch.long, device=inputs.device),
            torch.ones(batch_size, dtype=torch.long, device=inputs.device),
        ]

        if self.num_classes == 3:
            scar_01 = torch.stack([self.augmentation.cutpaste_scar(jittered_01[i]) for i in range(batch_size)])
            scar_01 = scar_01.clamp(0.0, 1.0)
            scar = self._normalize(scar_01)
            outputs['scar_01'] = scar_01
            outputs['scar'] = scar
            all_imgs.append(scar)
            labels.append(torch.full((batch_size,), 2, dtype=torch.long, device=inputs.device))

        outputs['all_imgs'] = torch.cat(all_imgs, dim=0)
        outputs['labels'] = torch.cat(labels, dim=0)
        return outputs

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        if mode == 'loss':
            augmented = self.build_augmented_views(inputs)
            embeddings = self.extract_embedding(augmented['all_imgs'])
            logits = self.classifier(embeddings)
            loss = self.loss_fn(logits, augmented['labels'])

            return {'loss': loss}

        elif mode == 'predict':
            # Use GDE scoring if fitted, otherwise fall back to softmax
            if self._gde_mean is not None and self._gde_cov_inv is not None:
                # _mahalanobis_score now takes inputs and extracts backbone embedding
                img_scores = self._mahalanobis_score(inputs)
            else:
                embeddings = self.extract_embedding(inputs)
                logits = self.classifier(embeddings)
                probs = torch.softmax(logits, dim=1)
                img_scores = 1.0 - probs[:, 0]

            B, _, H, W = inputs.shape
            score_map = img_scores.view(B, 1, 1, 1).expand(B, 1, H, W)
            return build_predict_results(data_samples, img_scores, score_map)

        return self.extract_embedding(inputs)

    def train(self, mode=True):
        super().train(mode)
        if not self._backbone_unfrozen:
            if self.force_backbone_eval_while_frozen:
                self.backbone.eval()
            elif mode:
                # Match official CutPaste semantics: frozen backbone weights can
                # still keep BatchNorm layers in training mode so running stats
                # keep tracking the current data stream.
                nn.Module.train(self.backbone, True)
        elif mode:
            self._apply_backbone_bn_policy()
        return self
