"""DRAEM anomaly detector (ICCV 2021).

Faithful reimplementation aligned with anomalib/ADer:
- Reconstructive network: base_width=128, NO skip connections (information bottleneck)
- Discriminative network: 6 encoder + 5 decoder stages with full skip connections
- Loss: MSE + SSIM (x2 weight) + FocalLoss (matching anomalib)
- Images in [0, 1] range (no ImageNet normalization)
- Augmentation happens in DRAEMDataset, not in the model
"""
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS
from baoiad.models.base_ad_model import ReconstructionADModel

logger = logging.getLogger(__name__)


# ==================== SSIM Loss (no kornia dependency) ====================

def _gaussian_window(window_size, sigma=1.5):
    """Create 1D Gaussian window."""
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g


def _create_window(window_size, channel):
    """Create 2D Gaussian window for SSIM."""
    _1D = _gaussian_window(window_size, 1.5).unsqueeze(1)
    _2D = _1D.mm(_1D.t()).unsqueeze(0).unsqueeze(0)
    return _2D.expand(channel, 1, window_size, window_size).contiguous()


def _ssim(img1, img2, window_size=11):
    """Compute SSIM between two images. Returns mean SSIM score."""
    channel = img1.size(1)
    window = _create_window(window_size, channel).to(img1.device, img1.dtype)
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2

    # For [0,1] range images
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    return ssim_map.mean()


class SSIMLoss(nn.Module):
    """SSIM-based loss: 1 - SSIM."""

    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size

    def forward(self, img1, img2):
        return 1.0 - _ssim(img1, img2, self.window_size)


# ==================== Focal Loss (matching ADer implementation) ====================

class FocalLoss(nn.Module):
    """Focal Loss matching ADer's implementation.

    ADer's FocalLoss expects softmax probabilities (not raw logits) and uses
    label smoothing. This is fundamentally different from kornia's FocalLoss
    which expects raw logits and uses cross_entropy.

    Key differences from kornia:
    - Input: softmax probabilities (not logits)
    - Uses label smoothing (smooth=1e-5)
    - Manual one-hot encoding of targets
    """

    def __init__(self, alpha=None, gamma=2, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, prob, target):
        """
        Args:
            prob: (B, C, H, W) softmax probabilities
            target: (B, H, W) long class indices
        """
        num_class = prob.shape[1]
        B, _, H, W = prob.shape

        # Reshape: (B, C, H, W) -> (B, H, W, C) -> (B*H*W, C)
        prob = prob.view(B, num_class, -1).permute(0, 2, 1).contiguous()
        prob = prob.view(-1, num_class)  # (B*H*W, C)

        target = target.view(-1, 1)  # (B*H*W, 1)

        # One-hot encoding with label smoothing
        one_hot = torch.zeros(target.size(0), num_class, device=prob.device)
        one_hot = one_hot.scatter_(1, target, 1)
        # Apply label smoothing: clamp away from 0 and 1
        one_hot = torch.clamp(one_hot, self.smooth / (num_class - 1), 1.0 - self.smooth)

        # Compute pt = probability of correct class
        pt = (one_hot * prob).sum(1) + self.smooth
        logpt = pt.log()

        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha = self.alpha
            else:
                alpha = self.alpha[target.squeeze()]
            loss = -alpha * torch.pow(1 - pt, self.gamma) * logpt
        else:
            loss = -torch.pow(1 - pt, self.gamma) * logpt

        return loss.mean()


# ==================== Perlin Noise ====================

def _lerp_np(x, y, w):
    return (y - x) * w + x


def rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    """Generate 2D Perlin noise."""
    h, w = shape
    delta = (res[0] / h, res[1] / w)
    d = (h // res[0], w // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    tile_grads = lambda s1, s2: np.repeat(
        np.repeat(gradients[s1[0]:s1[1], s2[0]:s2[1]], d[0], axis=0), d[1], axis=1)
    dot = lambda grad, shift: (
        np.stack((grid[:h, :w, 0] + shift[0], grid[:h, :w, 1] + shift[1]), axis=-1) * grad[:h, :w]
    ).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:h, :w])
    return math.sqrt(2) * _lerp_np(_lerp_np(n00, n10, t[..., 0]), _lerp_np(n01, n11, t[..., 0]), t[..., 1])


def generate_perlin_mask(H, W, perlin_scale=6, min_perlin_scale=0, threshold=0.5):
    """Generate a binary Perlin noise mask with continuous rotation (anomalib-style)."""
    scalex = 2 ** np.random.randint(min_perlin_scale, perlin_scale)
    scaley = 2 ** np.random.randint(min_perlin_scale, perlin_scale)
    noise = rand_perlin_2d_np((H, W), (scalex, scaley))

    # Anomalib-style: rescale noise when no values exceed threshold
    if not (noise > threshold).any():
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        noise = noise * 2 - 1  # remap to [-1, 1]

    # Continuous rotation (-90, 90) instead of discrete rot90
    import cv2
    angle = np.random.uniform(-90, 90)
    center = (W / 2, H / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    noise = cv2.warpAffine(noise.astype(np.float32), M, (W, H),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                           borderValue=0)

    mask = (noise > threshold).astype(np.float32)
    return mask


# ==================== Reconstructive SubNetwork (NO skip connections) ====================

def _conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(True),
    )


class EncoderReconstructive(nn.Module):
    """5-stage encoder: 3->b->2b->4b->8b->8b with pooling."""

    def __init__(self, in_channels, base_width):
        super().__init__()
        b = base_width
        self.block1 = _conv_block(in_channels, b)
        self.mp1 = nn.MaxPool2d(2)
        self.block2 = _conv_block(b, b * 2)
        self.mp2 = nn.MaxPool2d(2)
        self.block3 = _conv_block(b * 2, b * 4)
        self.mp3 = nn.MaxPool2d(2)
        self.block4 = _conv_block(b * 4, b * 8)
        self.mp4 = nn.MaxPool2d(2)
        self.block5 = _conv_block(b * 8, b * 8)

    def forward(self, x):
        b1 = self.block1(x)
        b2 = self.block2(self.mp1(b1))
        b3 = self.block3(self.mp2(b2))
        b4 = self.block4(self.mp3(b3))
        b5 = self.block5(self.mp4(b4))
        return b5  # No skip outputs


class DecoderReconstructive(nn.Module):
    """4-stage decoder WITHOUT skip connections (information bottleneck)."""

    def __init__(self, base_width, out_channels=3):
        super().__init__()
        b = base_width
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(b * 8, b * 8, 3, padding=1), nn.BatchNorm2d(b * 8), nn.ReLU(True))
        self.db1 = nn.Sequential(
            nn.Conv2d(b * 8, b * 8, 3, padding=1), nn.BatchNorm2d(b * 8), nn.ReLU(True),
            nn.Conv2d(b * 8, b * 4, 3, padding=1), nn.BatchNorm2d(b * 4), nn.ReLU(True))

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(b * 4, b * 4, 3, padding=1), nn.BatchNorm2d(b * 4), nn.ReLU(True))
        self.db2 = nn.Sequential(
            nn.Conv2d(b * 4, b * 4, 3, padding=1), nn.BatchNorm2d(b * 4), nn.ReLU(True),
            nn.Conv2d(b * 4, b * 2, 3, padding=1), nn.BatchNorm2d(b * 2), nn.ReLU(True))

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(b * 2, b * 2, 3, padding=1), nn.BatchNorm2d(b * 2), nn.ReLU(True))
        self.db3 = nn.Sequential(
            nn.Conv2d(b * 2, b * 2, 3, padding=1), nn.BatchNorm2d(b * 2), nn.ReLU(True),
            nn.Conv2d(b * 2, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(True))

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(True))
        self.db4 = nn.Sequential(
            nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(True),
            nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(True))

        self.fin_out = nn.Conv2d(b, out_channels, 3, padding=1)

    def forward(self, b5):
        d1 = self.db1(self.up1(b5))
        d2 = self.db2(self.up2(d1))
        d3 = self.db3(self.up3(d2))
        d4 = self.db4(self.up4(d3))
        return self.fin_out(d4)


# ==================== Discriminative SubNetwork (6 enc + 5 dec with skips) ====================

class EncoderDiscriminative(nn.Module):
    """6-stage encoder with outputs for skip connections."""

    def __init__(self, in_channels, base_width):
        super().__init__()
        b = base_width
        self.block1 = _conv_block(in_channels, b)
        self.mp1 = nn.MaxPool2d(2)
        self.block2 = _conv_block(b, b * 2)
        self.mp2 = nn.MaxPool2d(2)
        self.block3 = _conv_block(b * 2, b * 4)
        self.mp3 = nn.MaxPool2d(2)
        self.block4 = _conv_block(b * 4, b * 8)
        self.mp4 = nn.MaxPool2d(2)
        self.block5 = _conv_block(b * 8, b * 8)
        self.mp5 = nn.MaxPool2d(2)
        self.block6 = _conv_block(b * 8, b * 8)

    def forward(self, x):
        b1 = self.block1(x)
        b2 = self.block2(self.mp1(b1))
        b3 = self.block3(self.mp2(b2))
        b4 = self.block4(self.mp3(b3))
        b5 = self.block5(self.mp4(b4))
        b6 = self.block6(self.mp5(b5))
        return b1, b2, b3, b4, b5, b6


class DecoderDiscriminative(nn.Module):
    """5-stage decoder with skip connections from encoder."""

    def __init__(self, base_width, out_channels=2):
        super().__init__()
        b = base_width

        # b6 -> up -> cat(b5) -> db_b
        self.up_b = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(b * 8, b * 8, 3, padding=1), nn.BatchNorm2d(b * 8), nn.ReLU(True))
        self.db_b = nn.Sequential(
            nn.Conv2d(b * (8 + 8), b * 8, 3, padding=1), nn.BatchNorm2d(b * 8), nn.ReLU(True),
            nn.Conv2d(b * 8, b * 8, 3, padding=1), nn.BatchNorm2d(b * 8), nn.ReLU(True))

        # -> up -> cat(b4) -> db1
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(b * 8, b * 4, 3, padding=1), nn.BatchNorm2d(b * 4), nn.ReLU(True))
        self.db1 = nn.Sequential(
            nn.Conv2d(b * (4 + 8), b * 4, 3, padding=1), nn.BatchNorm2d(b * 4), nn.ReLU(True),
            nn.Conv2d(b * 4, b * 4, 3, padding=1), nn.BatchNorm2d(b * 4), nn.ReLU(True))

        # -> up -> cat(b3) -> db2
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(b * 4, b * 2, 3, padding=1), nn.BatchNorm2d(b * 2), nn.ReLU(True))
        self.db2 = nn.Sequential(
            nn.Conv2d(b * (2 + 4), b * 2, 3, padding=1), nn.BatchNorm2d(b * 2), nn.ReLU(True),
            nn.Conv2d(b * 2, b * 2, 3, padding=1), nn.BatchNorm2d(b * 2), nn.ReLU(True))

        # -> up -> cat(b2) -> db3
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(b * 2, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(True))
        self.db3 = nn.Sequential(
            nn.Conv2d(b * (2 + 1), b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(True),
            nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(True))

        # -> up -> cat(b1) -> db4
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(True))
        self.db4 = nn.Sequential(
            nn.Conv2d(b * 2, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(True),
            nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(True))

        self.fin_out = nn.Conv2d(b, out_channels, 3, padding=1)

    def forward(self, b1, b2, b3, b4, b5, b6):
        up_b = self.up_b(b6)
        db_b = self.db_b(torch.cat([up_b, b5], dim=1))

        up1 = self.up1(db_b)
        db1 = self.db1(torch.cat([up1, b4], dim=1))

        up2 = self.up2(db1)
        db2 = self.db2(torch.cat([up2, b3], dim=1))

        up3 = self.up3(db2)
        db3 = self.db3(torch.cat([up3, b2], dim=1))

        up4 = self.up4(db3)
        db4 = self.db4(torch.cat([up4, b1], dim=1))

        return self.fin_out(db4)


# ==================== Weight Init ====================

def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# ==================== Detector ====================

@MODELS.register_module(force=True)
class DRAEMDetector(ReconstructionADModel):
    """DRAEM: reconstruction + discrimination for anomaly detection.

    Aligned with anomalib implementation:
    - Loss: MSE + SSIM (x2) + FocalLoss (not CrossEntropy)
    - Images expected in [0, 1] range (no ImageNet normalization)
    - Augmentation is done in DRAEMDataset, not here

    Args:
        base_width: Base channel width for reconstructive network (paper: 128).
        disc_base_width: Base channel width for discriminative network (paper: 64).
        ssim_weight: Weight multiplier for SSIM loss (anomalib uses 2.0).
    """

    def __init__(
        self,
        base_width=128,
        disc_base_width=64,
        ssim_weight=1.0,
        data_preprocessor=None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        # Reconstructive network (no skip connections)
        self.encoder_recon = EncoderReconstructive(3, base_width)
        self.decoder_recon = DecoderReconstructive(base_width, out_channels=3)

        # Discriminative network (full skip connections)
        self.encoder_disc = EncoderDiscriminative(6, disc_base_width)
        self.decoder_disc = DecoderDiscriminative(disc_base_width, out_channels=2)

        # Init weights
        self.encoder_recon.apply(_weights_init)
        self.decoder_recon.apply(_weights_init)
        self.encoder_disc.apply(_weights_init)
        self.decoder_disc.apply(_weights_init)

        self.ssim_weight = ssim_weight

        # Loss functions (matching anomalib: MSE + SSIM*2 + Focal)
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIMLoss(window_size=11)
        self.focal_loss = FocalLoss(alpha=1.0, gamma=2.0)

    def reconstruct(self, x):
        b5 = self.encoder_recon(x)
        return torch.sigmoid(self.decoder_recon(b5))

    def discriminate(self, x):
        b1, b2, b3, b4, b5, b6 = self.encoder_disc(x)
        return self.decoder_disc(b1, b2, b3, b4, b5, b6)

    def forward(self, inputs, data_samples=None, mode='tensor'):
        """Forward pass.

        Args:
            inputs: Original images (B, C, H, W) in [0, 1] range
            data_samples: ADDataSample with augmented_img and anomaly_mask in metainfo
            mode: 'loss', 'predict', or 'tensor'
        """
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        if mode == 'loss':
            # Get pre-augmented data from data_samples (done by DRAEMDataset)
            # Move to same device as inputs since preprocessor only moves 'inputs'
            augmented = torch.stack([ds.augmented_img for ds in data_samples]).to(inputs.device)
            masks = torch.stack([ds.anomaly_mask for ds in data_samples]).to(inputs.device)

            recon = self.reconstruct(augmented)
            joined = torch.cat([recon, augmented], dim=1)  # ADer order: [recon, augmented]
            pred = self.discriminate(joined)

            # Loss: MSE + SSIM*2 + Focal (matching ADer DraemLoss)
            loss_mse = self.mse_loss(recon, inputs)
            loss_ssim = self.ssim_loss(recon, inputs) * self.ssim_weight
            # ADer passes softmax probabilities to FocalLoss (not raw logits)
            pred_softmax = torch.softmax(pred, dim=1)
            loss_focal = self.focal_loss(pred_softmax, masks.long())

            return {'loss': loss_mse + loss_ssim + loss_focal}

        elif mode == 'predict':
            recon = self.reconstruct(inputs)
            joined = torch.cat([recon, inputs], dim=1)  # ADer order: [recon, input]
            pred = self.discriminate(joined)
            prob = torch.softmax(pred, dim=1)[:, 1]  # (B, H, W) anomaly probability

            # Image score: avg_pool2d(21x21) -> max (matching ADer: pooling_ks=[21, 21])
            B, H, W = prob.shape
            prob_pooled = F.avg_pool2d(prob.unsqueeze(1), kernel_size=21, stride=1, padding=10).squeeze(1)
            img_scores = prob_pooled.view(B, -1).max(dim=1).values

            return build_predict_results(data_samples, img_scores, prob)

        return self.reconstruct(inputs)
