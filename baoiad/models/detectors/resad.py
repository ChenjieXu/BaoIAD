"""ResAD: Residual-based Anomaly Detection with Conditional Normalizing Flows.

NeurIPS 2024 Spotlight (arXiv:2410.20047).
Reference: https://github.com/xcyao00/ResAD

Key components:
1. Frozen backbone (WRN-50-2) for multi-scale feature extraction
2. Multi-scale Vector Quantization (VQ) for distribution alignment
3. Feature Constraintor with log-barrier one-class loss
4. Conditional Normalizing Flows for density estimation

Training uses residual features between test images and a few-shot reference bank.
"""
import math
import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.optim import OptimWrapperDict

import FrEIA.framework as Ff
import FrEIA.modules as Fm

from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS
from baoiad.utils.freia import patch_freia_soft_permutation_rvs
from baoiad.models.base_ad_model import BaseADModel


_GCONST_ = -0.9189385332046727  # ln(sqrt(2*pi))


class BoundaryAverager:
    """Moving average tracker for normal boundaries per feature level.

    Used in official ResAD to track the decision boundary between normal
    and abnormal log-likelihoods during training.
    """

    def __init__(self, num_levels: int = 3, momentum: float = 0.9):
        self.boundaries = [0.0 for _ in range(num_levels)]
        self.momentum = momentum

    def update_boundary(self, boundary: float, level: int):
        """Update boundary with exponential moving average."""
        self.boundaries[level] = self.boundaries[level] * self.momentum + (1 - self.momentum) * boundary

    def get_boundary(self, level: int) -> float:
        return self.boundaries[level]


class FocalLoss(nn.Module):
    """Focal Loss for binary classification with class imbalance.

    Copy from official ResAD implementation.
    """

    def __init__(self, alpha=None, gamma: float = 2, smooth: float = 1e-5, size_average: bool = True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

    def forward(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            logit: Softmax probabilities (N, 2)
            target: Labels (N, 1), 0 for normal, 1 for abnormal

        Returns:
            Focal loss value
        """
        num_class = logit.shape[1]

        # Flatten if needed
        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))

        target = target.view(-1, 1)

        # Build alpha
        if self.alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            alpha = torch.FloatTensor(self.alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[0] = self.alpha
        else:
            alpha = torch.ones(num_class, 1)

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        # Build one-hot encoding
        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)

        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        alpha = alpha[idx]
        alpha = alpha.squeeze(-1)
        loss = -1 * alpha * torch.pow((1 - pt), self.gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        return loss


def positionalencoding2d(D, H, W):
    """2D sinusoidal positional encoding. D must be divisible by 4."""
    if D % 4 != 0:
        raise ValueError(f"Cannot use sin/cos positional encoding with dim={D}")
    P = torch.zeros(D, H, W)
    D2 = D // 2
    div_term = torch.exp(torch.arange(0.0, D2, 2) * -(math.log(1e4) / D2))
    pos_w = torch.arange(0.0, W).unsqueeze(1)
    pos_h = torch.arange(0.0, H).unsqueeze(1)
    P[0:D2:2, :, :] = torch.sin(pos_w * div_term).T.unsqueeze(1).repeat(1, H, 1)
    P[1:D2:2, :, :] = torch.cos(pos_w * div_term).T.unsqueeze(1).repeat(1, H, 1)
    P[D2::2, :, :] = torch.sin(pos_h * div_term).T.unsqueeze(2).repeat(1, 1, W)
    P[D2 + 1::2, :, :] = torch.cos(pos_h * div_term).T.unsqueeze(2).repeat(1, 1, W)
    return P


def subnet_fc(dims_in, dims_out):
    """Fully connected subnet for FrEIA coupling blocks."""
    return nn.Sequential(
        nn.Linear(dims_in, 2 * dims_in),
        nn.ReLU(),
        nn.Linear(2 * dims_in, dims_out),
    )


def build_flow_head(n_feat, condition_dim, coupling_layers, clamp_alpha):
    """Build a conditional normalizing flow using FrEIA SequenceINN."""
    patch_freia_soft_permutation_rvs()
    coder = Ff.SequenceINN(n_feat)
    for _ in range(coupling_layers):
        coder.append(
            Fm.AllInOneBlock,
            cond=0,
            cond_shape=(condition_dim,),
            subnet_constructor=subnet_fc,
            affine_clamping=clamp_alpha,
            global_affine_type='SOFTPLUS',
            permute_soft=True,
        )
    return coder


class VectorQuantizer(nn.Module):
    """Vector Quantization layer with straight-through estimator.

    Aligned to official ResAD implementation: only quantizes normal samples
    (mask == 0) during training, and uses both commitment and codebook losses.

    Args:
        num_embeddings: Number of codebook vectors
        embedding_dim: Dimension of each embedding vector
        beta: Codebook loss weight (default: 0.25)
    """

    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z, mask=None):
        """Quantize input tensor.

        Args:
            z: Input tensor of shape (B, C, H, W)
            mask: Optional mask tensor of shape (B, H, W), 0=normal, 1=anomaly.
                  If provided, only normal samples (mask==0) are quantized.

        Returns:
            z_q: Quantized tensor (B, C, H, W)
            loss: VQ loss (commitment + codebook for training, commitment only for inference)
        """
        B, C, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, C)

        if mask is not None:
            # Training mode: only quantize normal samples (official behavior)
            mask_flat = mask.view(-1)
            z_normal = z_flat[mask_flat == 0]

            if z_normal.shape[0] == 0:
                # No normal samples, return zero loss
                return z, z.new_tensor(0.0)

            # Compute distances using torch.cdist (matches official implementation)
            min_encoding_indices = torch.argmin(
                torch.cdist(z_normal, self.embedding.weight), dim=1
            )
            z_q_normal = self.embedding(min_encoding_indices)

            # Official loss: commitment + codebook
            # loss = mean((z_q.detach() - z)^2) + beta * mean((z_q - z.detach())^2)
            loss = torch.mean((z_q_normal.detach() - z_normal) ** 2) + \
                   self.beta * torch.mean((z_q_normal - z_normal.detach()) ** 2)

            # Reconstruct full z_q: quantized for normal, original for anomaly
            z_q_flat = z_flat.clone()
            # Straight-through estimator for normal samples
            z_q_flat[mask_flat == 0] = z_normal + (z_q_normal - z_normal).detach()
            z_q = z_q_flat.view(B, H, W, C).permute(0, 3, 1, 2)
        else:
            # Inference mode: quantize all positions
            min_encoding_indices = torch.argmin(
                torch.cdist(z_flat, self.embedding.weight), dim=1
            )
            z_q_flat = self.embedding(min_encoding_indices)
            z_q = z_q_flat.view(B, H, W, C).permute(0, 3, 1, 2)

            # Straight-through estimator
            z_q = z + (z_q - z).detach()

            # Commitment loss only for inference
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2)

        return z_q, loss


class MultiScaleVQ(nn.Module):
    """Multi-scale Vector Quantization module.

    In training mode, only computes VQ loss for codebook learning.
    In inference mode, returns quantized features for EFDM.

    Args:
        channels: List of channel dimensions for each scale
        num_embeddings: Number of codebook vectors (default: 1536)
    """

    def __init__(self, channels: List[int], num_embeddings: int = 1536):
        super().__init__()
        self.vq_layers = nn.ModuleList([
            VectorQuantizer(num_embeddings, ch) for ch in channels
        ])

    def forward(self, features: List[torch.Tensor], masks: List[torch.Tensor] = None):
        """Forward pass for VQ.

        Args:
            features: List of tensors [(B,C1,H1,W1), (B,C2,H2,W2), ...]
            masks: Optional list of mask tensors [(B,H1,W1), (B,H2,W2), ...],
                   where 0=normal, 1=anomaly. Required for training.

        Returns:
            If masks is not None (training mode):
                z_q_list: Empty list (not used in training)
                total_loss: Sum of VQ losses
            If masks is None (inference mode):
                z_q_list: List of quantized tensors
                total_loss: Sum of VQ losses
        """
        if masks is not None:
            # Training mode: only compute loss for codebook learning
            # Official implementation does NOT use quantized features during training
            total_loss = 0.0
            for feat, vq, mask in zip(features, self.vq_layers, masks):
                _, loss = vq(feat, mask)
                total_loss = total_loss + loss
            return [], total_loss
        else:
            # Inference mode: return quantized features for EFDM
            z_q_list = []
            total_loss = 0.0
            for feat, vq in zip(features, self.vq_layers):
                z_q, loss = vq(feat, mask=None)
                z_q_list.append(z_q)
                total_loss = total_loss + loss
            return z_q_list, total_loss


class ConvBnAct(nn.Module):
    """Conv2d + BatchNorm + ReLU block."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MultiScaleConv(nn.Module):
    """Multi-scale feature constraintor (one ConvBnAct per level).

    Matches official ResAD signature: forward(layer1_x, layer2_x, layer3_x)
    """

    def __init__(self, channels: List[int]):
        super().__init__()
        self.convs = nn.ModuleList([
            ConvBnAct(ch, ch) for ch in channels
        ])

    def forward(self, *args):
        """Apply constraintor to each feature level.

        Args:
            *args: Either unpacked tensors (layer1, layer2, layer3) or a list of tensors.
        """
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            features = args[0]
        else:
            features = args
        return [conv(f) for f, conv in zip(features, self.convs)]


@MODELS.register_module(force=True)
class ResADDetector(BaseADModel):
    """ResAD: Residual-based Anomaly Detection with Conditional Normalizing Flows.

    Uses residual features between test images and a reference bank,
    combined with VQ-based distribution alignment and conditional flows.

    Args:
        backbone: Backbone config (TIMMBackbone with WRN-50-2)
        n_shot: Number of reference images per forward pass (default: 4)
        num_embeddings: VQ codebook size (default: 1536)
        pos_embed_dim: Positional encoding dimension (default: 256)
        coupling_layers: Number of flow coupling layers (default: 10)
        clamp_alpha: Affine clamping for flow (default: 1.9)
        fdm_alpha: EFDM interpolation weight (default: 0.4)
        r_max: Maximum radius for OCC loss (default: 0.4)
        occ_lambda: OCC loss weight (default: 1.0)
        flow_lambda: Flow loss weight (default: 1.0)
        first_stage_epochs: Epochs for stage 1 training (default: 10)
        smooth_sigma: Gaussian blur sigma for anomaly map (default: 4.0)
        data_root: Dataset root used to sample per-class normal references.
        input_size: Reference image preprocessing size.
        strict_ref_features: If True, require official pre-extracted few-shot
            features to exist before building the evaluation memory bank.
    """

    def __init__(
        self,
        backbone: Union[str, dict] = 'wide_resnet50_2',
        n_shot: int = 4,
        num_embeddings: int = 1536,
        pos_embed_dim: int = 256,
        coupling_layers: int = 10,
        clamp_alpha: float = 1.9,
        fdm_alpha: float = 0.4,
        r_max: float = 0.4,
        occ_lambda: float = 1.0,
        flow_lambda: float = 1.0,
        first_stage_epochs: int = 10,
        smooth_sigma: float = 4.0,
        margin_tau: float = 0.1,
        bgspp_lambda: float = 1.0,
        pos_beta: float = 0.05,
        ref_feature_dir: str = '',
        num_ref_shot: int = 4,
        total_ref_shot: int = 4,
        data_root: str = '',
        input_size: int = 224,
        strict_ref_features: bool = False,
        data_preprocessor=None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        # Build backbone
        if isinstance(backbone, dict) and backbone.get('type') == 'TIMMBackbone':
            backbone_cfg = backbone.copy()
            backbone_cfg.setdefault('out_indices', (1, 2, 3))
            backbone_cfg.setdefault('pretrained', True)
            backbone_cfg.setdefault('frozen', True)
            self.backbone = MODELS.build(backbone_cfg)
        else:
            from baoiad.models.backbone_utils import build_feature_extractor
            self.backbone = build_feature_extractor(
                backbone, default_out_indices=(2, 3, 4), default_frozen=True)

        # Get output channels
        channels = self.backbone.out_channels

        # Build modules
        self.vq = MultiScaleVQ(channels, num_embeddings)
        self.constraintor = MultiScaleConv(channels)
        self.flows = nn.ModuleList([
            build_flow_head(ch, pos_embed_dim, coupling_layers, clamp_alpha)
            for ch in channels
        ])

        # Store hyperparameters
        self.n_shot = n_shot
        self.pos_embed_dim = pos_embed_dim
        self.fdm_alpha = fdm_alpha
        self.r_max = r_max
        self.r_min = r_max * 0.99
        self.occ_lambda = occ_lambda
        self.flow_lambda = flow_lambda
        self.first_stage_epochs = first_stage_epochs
        self.smooth_sigma = smooth_sigma
        self.margin_tau = margin_tau
        self.bgspp_lambda = bgspp_lambda
        self.pos_beta = pos_beta
        self.ref_feature_dir = ref_feature_dir
        self.num_ref_shot = num_ref_shot
        self.total_ref_shot = total_ref_shot
        self.data_root = data_root
        self.input_size = int(input_size)
        self.strict_ref_features = strict_ref_features

        # Reference bank (populated by build_memory_bank)
        self.ref_bank: Optional[Union[List[torch.Tensor], Dict[str, List[torch.Tensor]]]] = None
        self.current_epoch = 0

        # Positional encoding cache
        self._pe_cache = {}
        self._train_good_paths: Dict[str, List[str]] = {}
        self._ref_image_transform = None

        # ResAD scoring uses dataset-level max normalization on the full
        # evaluation set. We compute provisional per-batch outputs during
        # predict() and let ADValLoop/ADTestLoop call score_all() to finalize.
        self.requires_full_test_postprocess = True
        self._pending_logp_batches: List[List[torch.Tensor]] = []
        self._pending_logp_a_batches: List[List[torch.Tensor]] = []
        self._pending_samples = []
        self._pending_output_size: Optional[tuple[int, int]] = None

    @torch.no_grad()
    def extract_features(self, x):
        """Extract multi-scale features from frozen backbone."""
        feats = self.backbone(x)
        return list(feats)

    def _get_positional_encoding(self, H, W, device, dtype):
        """Get or create cached positional encoding."""
        key = (H, W, str(device), str(dtype))
        pe = self._pe_cache.get(key)
        if pe is None:
            pe = positionalencoding2d(self.pos_embed_dim, H, W).to(device=device, dtype=dtype)
            self._pe_cache[key] = pe
        return pe

    def _resolve_cls_names(self, data_samples) -> List[str]:
        """Extract per-sample class names from ``ADDataSample`` objects."""
        if not data_samples:
            return []
        return [getattr(sample, 'cls_name', '') for sample in data_samples]

    def _get_ref_image_transform(self):
        """Build the torchvision-style preprocessing used by upstream ResAD."""
        if self._ref_image_transform is None:
            from torchvision import transforms as T

            self._ref_image_transform = T.Compose([
                T.Resize(self.input_size, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(self.input_size),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        return self._ref_image_transform

    def _get_train_good_paths(self, class_name: str) -> List[str]:
        """Cache ``train/good`` image paths for reference sampling."""
        if class_name in self._train_good_paths:
            return self._train_good_paths[class_name]

        good_dir = os.path.join(self.data_root, class_name, 'train', 'good')
        if not os.path.isdir(good_dir):
            raise RuntimeError(f'ResAD training references not found: {good_dir}')

        image_paths = sorted(
            os.path.join(good_dir, name)
            for name in os.listdir(good_dir)
            if name.lower().endswith(('.png', '.jpg', '.bmp'))
        )
        if not image_paths:
            raise RuntimeError(f'ResAD found no normal training images in {good_dir}')

        self._train_good_paths[class_name] = image_paths
        return image_paths

    @torch.no_grad()
    def _sample_reference_from_dataset(self, data_samples, device: torch.device):
        """Sample per-class few-shot normal references from ``train/good``."""
        class_names = sorted(set(self._resolve_cls_names(data_samples)))
        if not class_names:
            raise RuntimeError('ResAD requires class metadata to sample train-time references.')

        from PIL import Image

        transform = self._get_ref_image_transform()
        ref_bank = {}
        for class_name in class_names:
            image_paths = self._get_train_good_paths(class_name)
            sample_count = max(1, self.n_shot)
            indices = np.random.randint(len(image_paths), size=sample_count)
            images = []
            for idx in indices:
                with Image.open(image_paths[idx]) as img:
                    images.append(transform(img.convert('RGB')))
            ref_inputs = torch.stack(images, dim=0).to(device)
            ref_feats = self.extract_features(ref_inputs)
            ref_bank[class_name] = [
                feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1]).contiguous()
                for feat in ref_feats
            ]
        return ref_bank

    def _match_reference_single(self, features: List[torch.Tensor], ref_bank: List[torch.Tensor]):
        """Match features to reference bank using cosine similarity NN.

        Args:
            features: List of feature tensors [(B,C,H,W), ...]
            ref_bank: List of reference tensors [(N,C,1,1) or (N,C,H,W), ...]

        Returns:
            matched: List of matched reference tensors
        """
        matched = []
        for feat, ref in zip(features, ref_bank):
            B, C, H, W = feat.shape
            feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, C).contiguous()  # (B*H*W, C)
            feat_norm = F.normalize(feat_flat, p=2, dim=1)

            if ref.ndim == 4:
                ref_flat = ref.permute(0, 2, 3, 1).reshape(-1, C).contiguous()
            else:
                ref_flat = ref.reshape(-1, C).contiguous()

            ref_norm = F.normalize(ref_flat, p=2, dim=1)
            sim = feat_norm @ ref_norm.T
            idx = torch.argmax(sim, dim=1)
            matched_feat = ref_flat[idx].reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            matched.append(matched_feat)

        return matched

    def _match_reference_multi(
        self,
        features: List[torch.Tensor],
        ref_bank: Dict[str, List[torch.Tensor]],
        class_names: List[str],
    ):
        """Match each sample against the reference bank of its own class."""
        if len(class_names) != features[0].shape[0]:
            raise RuntimeError('ResAD class metadata does not match the input batch size.')

        matched = [[] for _ in range(len(features))]
        for sample_idx, class_name in enumerate(class_names):
            if class_name not in ref_bank:
                raise RuntimeError(f'Reference bank for class "{class_name}" not found.')
            class_refs = ref_bank[class_name]

            for level, feat in enumerate(features):
                sample_feat = feat[sample_idx:sample_idx + 1]
                _, C, H, W = sample_feat.shape
                feat_flat = sample_feat.permute(0, 2, 3, 1).reshape(-1, C).contiguous()
                feat_norm = F.normalize(feat_flat, p=2, dim=1)

                ref = class_refs[level]
                if ref.ndim == 4:
                    ref_flat = ref.permute(0, 2, 3, 1).reshape(-1, C).contiguous()
                else:
                    ref_flat = ref.reshape(-1, C).contiguous()

                ref_norm = F.normalize(ref_flat, p=2, dim=1)
                sim = feat_norm @ ref_norm.T
                idx = torch.argmax(sim, dim=1)
                matched_feat = ref_flat[idx].reshape(1, H, W, C).permute(0, 3, 1, 2).contiguous()
                matched[level].append(matched_feat)

        return [torch.cat(items, dim=0) for items in matched]

    def _match_reference(
        self,
        features: List[torch.Tensor],
        ref_bank: Union[List[torch.Tensor], Dict[str, List[torch.Tensor]]],
        data_samples=None,
    ):
        """Dispatch reference matching for global or per-class reference banks."""
        if isinstance(ref_bank, dict):
            return self._match_reference_multi(features, ref_bank, self._resolve_cls_names(data_samples))
        return self._match_reference_single(features, ref_bank)

    def _compute_residuals(self, features: List[torch.Tensor], matched: List[torch.Tensor]):
        """Compute element-wise MSE residuals."""
        residuals = []
        for f, m in zip(features, matched):
            r = (f - m) ** 2  # Element-wise squared difference
            residuals.append(r)
        return residuals

    def _apply_efdm(self, residuals: List[torch.Tensor], vq_entries: List[torch.Tensor]):
        """Apply Exact Feature Distribution Matching via sorted interpolation.

        Args:
            residuals: List of residual tensors
            vq_entries: List of VQ quantized tensors

        Returns:
            aligned: Distribution-aligned residuals
        """
        aligned = []
        alpha = 1.0 - self.fdm_alpha

        for r, vq in zip(residuals, vq_entries):
            B, C, H, W = r.shape
            r_flat = r.reshape(B, C, -1)
            vq_flat = vq.reshape(B, C, -1)

            r_sorted, sort_idx = torch.sort(r_flat, dim=-1)
            vq_sorted, _ = torch.sort(vq_flat, dim=-1)
            aligned_sorted = r_sorted + (vq_sorted - r_sorted) * alpha
            inv_idx = sort_idx.argsort(dim=-1)
            aligned_flat = aligned_sorted.gather(-1, inv_idx)
            aligned.append(aligned_flat.view(B, C, H, W))

        return aligned

    def _occ_loss(self, features: List[torch.Tensor], masks: List[torch.Tensor] = None,
                  targets: List[torch.Tensor] = None):
        """Compute the upstream log-barrier bi-OCC loss.

        Args:
            features: List of constraintor output tensors
            masks: Optional list of level-wise masks (0=normal, 1=anomaly)
            targets: Optional list of target tensors (residuals before constraintor)

        Returns:
            total_loss: Sum of OCC losses
        """
        total_loss = 0.0

        for i, f in enumerate(features):
            B, C, H, W = f.shape
            e = f.permute(0, 2, 3, 1).reshape(-1, C)

            # Compute radius: sqrt(||e||^2 + 1) - 1
            A = torch.sqrt(e.norm(dim=1) + 1.0) - 1.0

            # Get mask for this level
            if masks is not None:
                m = masks[i].reshape(-1)
            else:
                m = torch.zeros(e.shape[0], dtype=torch.long, device=e.device)

            # Determine r_max and r_min
            Aa = A[m == 1]
            if Aa.numel() > 0:  # Has anomaly samples
                r_max = min(0.9 * Aa.min().item(), 0.4)
                r_min = r_max * 0.99
            else:  # First stage or no anomalies
                r_max = self.r_max
                r_min = self.r_min

            # Normal samples: log-barrier bi-OCC loss
            An = A[m == 0]
            if An.numel() > 0:
                An_larger = An[An > r_max]
                if An_larger.numel() > 0:
                    weights = torch.exp(An_larger - r_max).detach()
                    loss_larger = torch.mean(-F.logsigmoid(-(An_larger - r_max)) * weights)
                else:
                    loss_larger = 0.0

                An_lower = An[An < r_min]
                if An_lower.numel() > 0:
                    weights = torch.exp(r_min - An_lower).detach()
                    loss_lower = torch.mean(-F.logsigmoid(-(r_min - An_lower)) * weights)
                else:
                    loss_lower = 0.0

                total_loss = total_loss + loss_larger + loss_lower

            # Anomaly samples: invariant loss (if targets provided)
            if targets is not None and Aa.numel() > 0:
                t = targets[i].permute(0, 2, 3, 1).reshape(-1, C)
                ano_features = e[m == 1]
                target_features = t[m == 1]

                loss_mse = F.mse_loss(ano_features, target_features)
                loss_cos = torch.mean(1 - F.cosine_similarity(ano_features, target_features))
                loss_inv = loss_mse + loss_cos

                # Push anomaly features out of boundary
                boundary = r_max + 0.1
                Aa_lower = Aa[Aa < boundary]
                if Aa_lower.numel() > 0:
                    weights = torch.exp(boundary - Aa_lower).detach()
                    loss_lower = torch.mean(-F.logsigmoid(-(boundary - Aa_lower)) * weights)
                else:
                    loss_lower = 0.0

                total_loss = total_loss + loss_inv + loss_lower

        return total_loss

    def _flow_forward(self, features: List[torch.Tensor]):
        """Forward pass through conditional flows.

        Returns:
            logp_list: List of log-likelihood tensors per level
            logp_a_list: List of shifted log-likelihood tensors per level
        """
        logp_list = []
        logp_a_list = []

        for feat, flow in zip(features, self.flows):
            B, C, H, W = feat.shape
            S = H * W
            E = B * S

            # Flatten features: (B*H*W, C)
            e = feat.permute(0, 2, 3, 1).reshape(E, C)

            # Get positional encoding
            pe = self._get_positional_encoding(H, W, feat.device, feat.dtype)
            pe = pe.unsqueeze(0).expand(B, -1, -1, -1)  # (B, D, H, W)
            pe = pe.permute(0, 2, 3, 1).reshape(E, self.pos_embed_dim)  # (B*H*W, D)

            # Flow forward
            z, log_jac_det = flow(e, [pe])

            # The upstream implementation normalizes both score branches by the
            # feature dimension before any downstream scoring.
            logp = (C * _GCONST_ - 0.5 * z.pow(2).sum(1) + log_jac_det) / C
            logp = logp.reshape(B, H, W)

            logp_a = (C * _GCONST_ - 0.5 * (z - 1).pow(2).sum(1) + log_jac_det) / C
            logp_a = logp_a.reshape(B, H, W)

            logp_list.append(logp)
            logp_a_list.append(logp_a)

        return logp_list, logp_a_list

    def _sample_reference_from_batch(self, features: List[torch.Tensor]):
        """Sample n_shot reference features from current batch."""
        B = features[0].shape[0]
        n_shot = min(self.n_shot, B)

        if n_shot >= B:
            # Use all samples as reference
            return [f.detach() for f in features]

        # Random sample
        idx = torch.randperm(B, device=features[0].device)[:n_shot]
        return [f[idx].detach() for f in features]

    def _load_reference_features_from_dir(
        self,
        class_name: str,
        device: torch.device,
    ) -> List[torch.Tensor]:
        """Load official pre-extracted few-shot reference features.

        The upstream ResAD repo stores flattened patch-level features as
        `layer1.npy/layer2.npy/layer3.npy` under `<root>/<class_name>/`.
        """
        class_dir = os.path.join(self.ref_feature_dir, class_name)
        refs = []
        for layer_name in ('layer1.npy', 'layer2.npy', 'layer3.npy'):
            layer = np.load(os.path.join(class_dir, layer_name))
            layer = torch.from_numpy(layer).to(device)
            if self.total_ref_shot > 0 and self.num_ref_shot > 0 and layer.shape[0] >= self.total_ref_shot:
                per_shot = layer.shape[0] // self.total_ref_shot
                keep = per_shot * min(self.num_ref_shot, self.total_ref_shot)
                if keep > 0:
                    layer = layer[:keep]
            refs.append(layer.float())
        return refs

    def _has_reference_feature_dir(self) -> bool:
        return bool(self.ref_feature_dir and os.path.isdir(self.ref_feature_dir))

    def _clear_pending_postprocess(self) -> None:
        """Clear deferred-scoring buffers."""
        self._pending_logp_batches = []
        self._pending_logp_a_batches = []
        self._pending_samples = []
        self._pending_output_size = None

    def _smooth_anomaly_map(self, score_map: torch.Tensor) -> torch.Tensor:
        """Apply the upstream scipy Gaussian smoothing on each sample map."""
        if self.smooth_sigma <= 0:
            return score_map

        from scipy.ndimage import gaussian_filter

        score_np = score_map.detach().cpu().numpy()
        for index in range(score_np.shape[0]):
            score_np[index] = gaussian_filter(score_np[index], sigma=self.smooth_sigma)
        return torch.from_numpy(score_np).to(device=score_map.device, dtype=score_map.dtype)

    def _compute_score_maps(
        self,
        logp_list: List[torch.Tensor],
        logp_a_list: List[torch.Tensor],
        output_size,
        level_maxes: Optional[List[torch.Tensor]] = None,
    ):
        """Reproduce the upstream dual-branch score aggregation."""
        normal_maps = []
        abnormal_maps = []
        for level, (logp, logp_a) in enumerate(zip(logp_list, logp_a_list)):
            level_max = level_maxes[level] if level_maxes is not None else logp.max()
            if not torch.is_tensor(level_max):
                level_max = logp.new_tensor(level_max)
            level_max = level_max.to(device=logp.device, dtype=logp.dtype)

            probs = torch.exp(logp - level_max)
            normal_map = F.interpolate(
                probs.unsqueeze(1),
                size=output_size,
                mode='bilinear',
                align_corners=True,
            ).squeeze(1)
            normal_maps.append(normal_map)

            logits = torch.stack([logp, logp_a], dim=-1)
            abnormal_map = F.softmax(logits, dim=-1)[..., 1]
            abnormal_map = F.interpolate(
                abnormal_map.unsqueeze(1),
                size=output_size,
                mode='bilinear',
                align_corners=True,
            ).squeeze(1)
            abnormal_maps.append(abnormal_map)

        score1 = sum(normal_maps)
        score1 = score1.max() - score1
        score1 = self._smooth_anomaly_map(score1)

        score2 = sum(abnormal_maps) / len(abnormal_maps)
        score2 = self._smooth_anomaly_map(score2)

        return score1, score2, (score1 + score2) / 2

    def score_all(self):
        """Finalize dataset-level scores using global level-wise max statistics."""
        if not self._pending_samples:
            return []

        if self._pending_output_size is None:
            raise RuntimeError('ResAD deferred scoring requires a known output size.')

        num_levels = len(self._pending_logp_batches[0])
        logp_list = []
        logp_a_list = []
        for level in range(num_levels):
            logp_level = torch.cat([batch[level] for batch in self._pending_logp_batches], dim=0)
            logp_a_level = torch.cat([batch[level] for batch in self._pending_logp_a_batches], dim=0)
            logp_list.append(logp_level)
            logp_a_list.append(logp_a_level)

        level_maxes = [logp.max() for logp in logp_list]
        _, _, score_map = self._compute_score_maps(
            logp_list,
            logp_a_list,
            self._pending_output_size,
            level_maxes=level_maxes,
        )

        batch_size = score_map.shape[0]
        img_scores = torch.topk(score_map.view(batch_size, -1), k=1, dim=1).values.mean(dim=1)

        for index, sample in enumerate(self._pending_samples):
            sample.pred_score = float(img_scores[index].item())
            sample.pred_anomaly_map = score_map[index:index + 1].detach().cpu()

        results = list(self._pending_samples)
        self._clear_pending_postprocess()
        return results

    def forward(self, inputs, data_samples=None, mode='tensor'):
        """Forward pass.

        Args:
            inputs: Input images (B, 3, H, W)
            data_samples: List of ADDataSample
            mode: 'tensor', 'loss', or 'predict'

        Returns:
            mode='tensor': List of features
            mode='loss': Dict with 'loss' key
            mode='predict': List of ADDataSample with predictions
        """
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        feats = self.extract_features(inputs)
        B = inputs.shape[0]

        if mode == 'loss':
            if self.data_root and data_samples:
                ref = self._sample_reference_from_dataset(data_samples, inputs.device)
            else:
                ref = self._sample_reference_from_batch(feats)
            matched = self._match_reference(feats, ref, data_samples=data_samples)

            # Compute residuals
            residuals = self._compute_residuals(feats, matched)

            # Generate level-wise masks from gt_label (0=normal, 1=anomaly)
            # Official ResAD only quantizes normal samples during training
            gt_labels = torch.tensor([s.gt_label for s in data_samples], device=inputs.device)
            lvl_masks = []
            for feat in feats:
                _, _, h, w = feat.shape
                m = F.interpolate(gt_labels.float().view(-1, 1, 1, 1), size=(h, w), mode='nearest').squeeze(1)
                lvl_masks.append(m)

            # VQ forward with masks (only quantize normal samples)
            z_q, vq_loss = self.vq(residuals, lvl_masks)

            # EFDM (identity during training - VQ still learning)
            # During training, we use residuals directly for constraintor
            z_efdm = residuals

            # Save residuals before constraintor for OCC loss target
            residuals_before_constraintor = [r.detach().clone() for r in residuals]

            # Constraintor forward (official uses *rfeatures unpacking)
            z_c = self.constraintor(*z_efdm)

            # OCC loss with target (official passes target for anomaly invariant loss)
            occ_loss = self._occ_loss(z_c, lvl_masks, residuals_before_constraintor)

            # Probe / generic loss-mode should mirror the staged official flow
            # objective as closely as possible, even though strict training
            # uses the custom train_step with separate optimizers.
            z_flow = [z.detach().clone() for z in z_c]
            logp_list, logp_a_list = self._flow_forward(z_flow)

            current_epoch = getattr(self, 'current_epoch', 0)
            is_first_stage = current_epoch < self.first_stage_epochs
            flow_loss = inputs.new_tensor(0.0)
            if not hasattr(self, '_boundary_averager'):
                self._boundary_averager = BoundaryAverager(num_levels=len(self.flows))

            for level, (logp, logp_a, lvl_mask) in enumerate(zip(logp_list, logp_a_list, lvl_masks)):
                mask = lvl_mask.reshape(-1)
                logp_flat = logp.reshape(-1)
                logp_a_flat = logp_a.reshape(-1)

                batch_boundary = self._get_normal_boundary(logp_flat.detach(), mask)
                self._boundary_averager.update_boundary(batch_boundary, level)

                if is_first_stage:
                    flow_loss = flow_loss + self._compute_flow_loss_stage1(logp_flat, mask)
                    continue

                b_n = self._boundary_averager.get_boundary(level)
                b_a = b_n - self.margin_tau
                flow_loss = flow_loss + self._compute_flow_loss_stage2(
                    logp_flat,
                    logp_a_flat,
                    mask,
                    (b_n, b_a),
                    bgspp_lambda=self.bgspp_lambda,
                )

            # Total loss
            total_loss = vq_loss + self.occ_lambda * occ_loss + self.flow_lambda * flow_loss

            return {'loss': total_loss}

        elif mode == 'predict':
            # Inference: use full reference bank
            if self.ref_bank is None:
                raise RuntimeError("Reference bank not built. Call build_memory_bank() first.")

            matched = self._match_reference(feats, self.ref_bank, data_samples=data_samples)
            residuals = self._compute_residuals(feats, matched)

            # VQ forward (no grad)
            with torch.no_grad():
                z_q, _ = self.vq(residuals)

            # EFDM at inference
            z_efdm = self._apply_efdm(residuals, z_q)

            # Constraintor forward (official uses *rfeatures unpacking)
            with torch.no_grad():
                z_c = self.constraintor(*z_efdm)

            # Flow forward
            logp_list, logp_a_list = self._flow_forward(z_c)
            _, _, score_map = self._compute_score_maps(logp_list, logp_a_list, inputs.shape[-2:])

            # Upstream image score uses topk(1).mean(), which is equivalent to max.
            img_scores = torch.topk(score_map.view(B, -1), k=1, dim=1).values.mean(dim=1)
            results = build_predict_results(data_samples, img_scores, score_map)

            if self.requires_full_test_postprocess:
                if self._pending_output_size is None:
                    self._pending_output_size = tuple(inputs.shape[-2:])
                elif self._pending_output_size != tuple(inputs.shape[-2:]):
                    raise RuntimeError(
                        'ResAD deferred scoring requires a consistent evaluation input size. '
                        f'Got {self._pending_output_size} and {tuple(inputs.shape[-2:])}.'
                    )

                self._pending_logp_batches.append([logp.detach().cpu() for logp in logp_list])
                self._pending_logp_a_batches.append([logp.detach().cpu() for logp in logp_a_list])
                self._pending_samples.extend(results)

            return results

        return feats

    def build_memory_bank(self, data_loader):
        """Build reference bank from training data.

        Called by MemoryBankHook after training completes.

        Args:
            data_loader: Training data loader
        """
        self.eval()
        device = next(self.parameters()).device

        if self.strict_ref_features and not self._has_reference_feature_dir():
            raise RuntimeError(
                'ResAD strict reference mode requires pre-extracted few-shot features at '
                f'"{self.ref_feature_dir}".'
            )

        if self._has_reference_feature_dir():
            dataset = getattr(data_loader, 'dataset', None)
            cls_names = getattr(dataset, 'cls_names', None) if dataset is not None else None
            if not cls_names:
                cls_names = sorted(
                    d for d in os.listdir(self.ref_feature_dir)
                    if os.path.isdir(os.path.join(self.ref_feature_dir, d))
                )
            if len(cls_names) == 1:
                self.ref_bank = self._load_reference_features_from_dir(cls_names[0], device)
            else:
                self.ref_bank = {
                    cls_name: self._load_reference_features_from_dir(cls_name, device)
                    for cls_name in cls_names
                    if os.path.isdir(os.path.join(self.ref_feature_dir, cls_name))
                }
            return

        class_features: Dict[str, List[List[torch.Tensor]]] = {}
        all_features = [[] for _ in range(len(self.flows))]

        with torch.no_grad():
            for data in data_loader:
                if isinstance(data, dict):
                    inputs = data['inputs']
                    data_samples = data.get('data_samples')
                else:
                    inputs = data[0]
                    data_samples = None

                if isinstance(inputs, (list, tuple)):
                    inputs = torch.stack(inputs)

                inputs = inputs.to(device)
                feats = self.extract_features(inputs)
                class_names = self._resolve_cls_names(data_samples)

                if class_names and len(class_names) == inputs.shape[0]:
                    for sample_idx, class_name in enumerate(class_names):
                        if class_name not in class_features:
                            class_features[class_name] = [[] for _ in range(len(self.flows))]
                        for level, feat in enumerate(feats):
                            _, C, _, _ = feat[sample_idx:sample_idx + 1].shape
                            flattened = feat[sample_idx:sample_idx + 1].permute(0, 2, 3, 1).reshape(-1, C).cpu()
                            class_features[class_name][level].append(flattened)
                    continue

                for i, f in enumerate(feats):
                    B, C, H, W = f.shape
                    flattened = f.permute(0, 2, 3, 1).reshape(-1, C).cpu()
                    all_features[i].append(flattened)

        max_patches = 2048  # Limit memory usage
        if class_features:
            ref_bank = {}
            for class_name, feat_lists in class_features.items():
                ref_bank[class_name] = []
                for feat_list in feat_lists:
                    feats = torch.cat(feat_list, dim=0)
                    if feats.shape[0] > max_patches:
                        idx = torch.randperm(feats.shape[0])[:max_patches]
                        feats = feats[idx]
                    ref_bank[class_name].append(feats.to(device))
            self.ref_bank = ref_bank if len(ref_bank) > 1 else next(iter(ref_bank.values()))
            return

        self.ref_bank = []
        for feat_list in all_features:
            feats = torch.cat(feat_list, dim=0)
            if feats.shape[0] > max_patches:
                idx = torch.randperm(feats.shape[0])[:max_patches]
                feats = feats[idx]
            self.ref_bank.append(feats.to(device))

    def set_epoch_info(self, epoch, max_epochs):
        """Set current epoch for stage switching."""
        self.current_epoch = epoch
        # Initialize boundary averager if not exists
        if not hasattr(self, '_boundary_averager'):
            self._boundary_averager = BoundaryAverager(num_levels=len(self.flows))

    def _get_normal_boundary(self, logps: torch.Tensor, mask: torch.Tensor) -> float:
        """Find the decision boundary from normal log-likelihood distribution.

        Args:
            logps: Log-likelihoods (N,)
            mask: 0 for normal, 1 for abnormal (N,)

        Returns:
            Boundary value
        """
        normal_logps = logps[mask == 0]
        if normal_logps.numel() == 0:
            return 0.0

        n_idx = int((normal_logps.numel() * self.pos_beta))
        n_idx = max(0, min(n_idx, normal_logps.numel() - 1))
        sorted_indices = torch.sort(normal_logps)[1]
        n_idx = sorted_indices[n_idx]
        return normal_logps[n_idx].item()

    def _compute_flow_loss_stage1(self, logps: torch.Tensor, mask: torch.Tensor):
        """First stage flow loss: only ML loss on normal samples.

        Args:
            logps: Log-likelihoods (N,)
            mask: 0 for normal, 1 for abnormal (N,)

        Returns:
            loss: ML loss
        """
        normal_logps = logps[mask == 0]
        if normal_logps.numel() == 0:
            return torch.tensor(0.0, device=logps.device)
        return -F.logsigmoid(normal_logps).mean()

    def _compute_flow_loss_stage2(
        self,
        logps: torch.Tensor,
        logps_a: torch.Tensor,
        mask: torch.Tensor,
        boundaries: tuple,
        bgspp_lambda: float = 1.0,
    ):
        """Second stage flow loss: ML + Focal + BG-SPP loss.

        Args:
            logps: Normal log-likelihoods (N,)
            logps_a: Abnormal log-likelihoods (N,)
            mask: 0 for normal, 1 for abnormal (N,)
            boundaries: (b_n, b_a) boundary values
            bgspp_lambda: Weight for bg-spp loss

        Returns:
            loss: Combined loss
        """
        if mask.sum() == 0:
            return -F.logsigmoid(logps).mean()

        # Official ResAD applies LogSigmoid to both normal and abnormal ML terms.
        loss_ml_n = -F.logsigmoid(logps[mask == 0]).mean() if (mask == 0).sum() > 0 else 0.0
        loss_ml_a = -F.logsigmoid(logps_a[mask == 1]).mean() if mask.sum() > 0 else 0.0
        loss_ml = loss_ml_n + loss_ml_a

        # Focal loss
        if mask.sum() > 0:
            logits = torch.stack([logps, logps_a], dim=-1)  # (N, 2)
            s = torch.softmax(logits, dim=-1)
            focal_loss_fn = FocalLoss()
            loss_focal = focal_loss_fn(s, mask.unsqueeze(-1))
        else:
            loss_focal = 0.0

        # BG-SPP loss
        b_n, b_a = boundaries
        normal_logps = logps[mask == 0]
        anomaly_logps = logps[mask == 1]

        # Normal: push into b_n
        if normal_logps.numel() > 0:
            normal_inter = normal_logps[normal_logps <= b_n]
            if normal_inter.numel() > 0:
                loss_bg_n = -F.logsigmoid(-(b_n - normal_inter)).mean()
            else:
                loss_bg_n = 0.0
        else:
            loss_bg_n = 0.0

        # Anomaly: push away from b_a
        if anomaly_logps.numel() > 0:
            anomaly_inter = anomaly_logps[anomaly_logps >= b_a]
            if anomaly_inter.numel() > 0:
                loss_bg_a = -F.logsigmoid(-(anomaly_inter - b_a)).mean()
            else:
                loss_bg_a = 0.0
        else:
            loss_bg_a = 0.0

        loss_bgspp = loss_bg_n + loss_bg_a

        return loss_ml + loss_focal + bgspp_lambda * loss_bgspp

    def train_step(self, data, optim_wrapper, epoch=None, first_stage_epochs=10, N_batch=4096):
        """Train step with staged optimization (strictly aligned to official).

        Official ResAD training loop:
        1. VQ loss → backward → VQ optimizer step
        2. Constraintor forward + OCC loss → backward → constraintor optimizer step
        3. Flow loss (with N_batch sub-sampling and two-stage training)

        Args:
            data: Input data batch
            optim_wrapper: OptimWrapperDict with 'vq', 'constraintor', 'flow' keys
            epoch: Current epoch (used for stage switching)
            first_stage_epochs: Number of epochs for first stage (default: 10)
            N_batch: Batch size for flow training sub-sampling (default: 4096)
        """
        if not isinstance(optim_wrapper, OptimWrapperDict):
            return super().train_step(data, optim_wrapper)

        if not all(k in optim_wrapper for k in ['vq', 'constraintor', 'flow']):
            raise TypeError(
                'ResAD train_step requires OptimWrapperDict with vq, constraintor, and flow optimizers.'
            )

        data = self.data_preprocessor(data, True)
        inputs = data['inputs']
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)
        data_samples = data.get('data_samples', None)

        vq_optim = optim_wrapper['vq']
        constraintor_optim = optim_wrapper['constraintor']
        flow_optim = optim_wrapper['flow']

        # Extract features
        feats = self.extract_features(inputs)

        # Get references
        if self.data_root and data_samples:
            ref = self._sample_reference_from_dataset(data_samples, inputs.device)
        else:
            ref = self._sample_reference_from_batch(feats)
        matched = self._match_reference(feats, ref, data_samples=data_samples)

        # Compute residuals
        residuals = self._compute_residuals(feats, matched)

        # Generate level-wise masks from gt_label (0=normal, 1=anomaly)
        gt_labels = torch.tensor([s.gt_label for s in data_samples], device=inputs.device)
        lvl_masks = []
        for feat in feats:
            _, _, h, w = feat.shape
            m = F.interpolate(gt_labels.float().view(-1, 1, 1, 1), size=(h, w), mode='nearest').squeeze(1)
            lvl_masks.append(m)

        # ========== Stage 1: VQ training ==========
        vq_optim.zero_grad()
        _, vq_loss = self.vq(residuals, lvl_masks)
        vq_optim.backward(vq_loss)
        vq_optim.step()

        # ========== Stage 2: Constraintor training ==========
        # Save residuals before constraintor for OCC loss target
        residuals_before_constraintor = [r.detach().clone() for r in residuals]

        constraintor_optim.zero_grad()
        z_c = self.constraintor(*residuals)
        occ_loss = self._occ_loss(z_c, lvl_masks, residuals_before_constraintor)
        constraintor_optim.backward(occ_loss)
        constraintor_optim.step()

        # ========== Stage 3: Flow training (official with N_batch sub-sampling) ==========
        # Detach constraintor output for flow training
        z_c_detached = [z.detach().clone() for z in z_c]

        # Determine current epoch (default to 0 if not provided)
        current_epoch = epoch if epoch is not None else self.current_epoch
        is_first_stage = current_epoch < first_stage_epochs

        flow_loss_total = 0.0
        flow_num_batches = 0

        # Process each feature level
        for level, (feat, flow) in enumerate(zip(z_c_detached, self.flows)):
            bs, dim, h, w = feat.shape
            total_pixels = bs * h * w

            # Flatten features and mask for this level
            e = feat.permute(0, 2, 3, 1).reshape(-1, dim)
            m = lvl_masks[level].reshape(-1)

            # Get positional encoding
            pe = self._get_positional_encoding(h, w, feat.device, feat.dtype)
            pe = pe.unsqueeze(0).expand(bs, -1, -1, -1)
            pe = pe.permute(0, 2, 3, 1).reshape(-1, self.pos_embed_dim)

            # N_batch sub-sampling (official behavior)
            perm = torch.randperm(total_pixels, device=feat.device)
            num_batches = total_pixels // N_batch

            for i in range(num_batches):
                idx = torch.arange(i * N_batch, (i + 1) * N_batch)
                e_b = e[perm[idx]]
                pe_b = pe[perm[idx]]
                m_b = m[perm[idx]]

                # Flow forward
                z, log_jac_det = flow(e_b, [pe_b])

                # Compute log-likelihoods
                logps = (dim * _GCONST_ - 0.5 * z.pow(2).sum(1) + log_jac_det) / dim
                logps_a = (dim * _GCONST_ - 0.5 * (z - 1).pow(2).sum(1) + log_jac_det) / dim

                if is_first_stage:
                    # First stage: only ML loss on normal samples
                    flow_loss = self._compute_flow_loss_stage1(logps, m_b)

                    # Update boundary averager
                    if hasattr(self, '_boundary_averager'):
                        b_n = self._get_normal_boundary(logps.detach(), m_b)
                        self._boundary_averager.update_boundary(b_n, level)
                else:
                    # Second stage: ML + Focal + BG-SPP loss
                    if hasattr(self, '_boundary_averager'):
                        b_n = self._boundary_averager.get_boundary(level)
                        b_a = b_n - self.margin_tau
                        boundaries = (b_n, b_a)
                    else:
                        boundaries = (0.0, -0.1)

                    flow_loss = self._compute_flow_loss_stage2(
                        logps, logps_a, m_b, boundaries, bgspp_lambda=self.bgspp_lambda
                    )

                    # Update boundary
                    if hasattr(self, '_boundary_averager'):
                        b_n = self._get_normal_boundary(logps.detach(), m_b)
                        self._boundary_averager.update_boundary(b_n, level)

                flow_optim.zero_grad()
                flow_optim.backward(flow_loss)
                flow_optim.step()

                flow_loss_total += flow_loss.item() if torch.is_tensor(flow_loss) else flow_loss
                flow_num_batches += 1

        avg_flow_loss = flow_loss_total / max(flow_num_batches, 1)

        total_loss = vq_loss.detach() + occ_loss.detach() + torch.tensor(avg_flow_loss, device=inputs.device)
        return {
            'loss': total_loss,
            'vq_loss': vq_loss.detach(),
            'occ_loss': occ_loss.detach(),
            'flow_loss': avg_flow_loss,
        }

    def train(self, mode=True):
        """Override to keep backbone in eval mode."""
        super().train(mode)
        self.backbone.eval()
        return self
