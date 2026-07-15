"""DSR: Dual Subspace Re-projection Network for Surface Anomaly Detection (ECCV 2022).

Faithful reimplementation with:
- DiscreteLatentModel: dual VQ-VAE (top/bot) with general appearance decoder
- SubspaceRestrictionModules (hi/lo): InstanceNorm-based autoencoders
- ImageReconstructionNetwork: object-specific decoder
- AnomalyDetectionModule: UNet comparing object-specific vs general decoded images
- UpsamplingModule: refines anomaly map to full resolution
- Three training phases:
  Phase 1: Pre-train discrete latent model (loaded from checkpoint)
  Phase 2: Train subspace restriction + image recon + anomaly detection
  Phase 3: Train upsampling module with Perlin noise anomalies
"""
import logging
import math
import os
import warnings
import zipfile
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import v2
from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS
from baoiad.runtime import OfflineModeError
from baoiad.models.base_ad_model import ReconstructionADModel

logger = logging.getLogger(__name__)


# ==================== MultiRandomChoice Transform ====================

class MultiRandomChoice(v2.Transform):
    """Randomly applies num_transforms transforms (with replacement).

    Unlike RandomChoice which picks ONE transform, this applies multiple
    transforms sequentially, mimicking anomalib's MultiRandomChoice behavior.

    Args:
        transforms: List of transforms to choose from
        num_transforms: Number of transforms to apply per sample (default: 3)
    """

    def __init__(self, transforms: list, num_transforms: int = 3):
        super().__init__()
        self.transforms = transforms
        self.num_transforms = num_transforms

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        for _ in range(self.num_transforms):
            idx = torch.randint(len(self.transforms), (1,)).item()
            img = self.transforms[idx](img)
        return img

# Pretrained VQ-VAE weights from anomalib GitHub release (via gh-proxy mirror)
_VQVAE_WEIGHTS_URL = (
    "https://gh-proxy.com/https://github.com/openvinotoolkit/anomalib/releases/download/"
    "dsr_pretrained_weights/dsr_vq_model_pretrained.zip"
)
_VQVAE_WEIGHTS_DIR = "pre_trained"
_VQVAE_WEIGHTS_FILE = "vq_model_pretrained_128_4096.pckl"  # anomalib uses .pckl format


def _download_vqvae_weights(target_dir=_VQVAE_WEIGHTS_DIR, force_redownload: bool = False):
    """Download pretrained VQ-VAE weights from anomalib release if not present."""
    target_path = os.path.join(target_dir, _VQVAE_WEIGHTS_FILE)
    if force_redownload and os.path.isfile(target_path):
        os.remove(target_path)
    if os.path.isfile(target_path):
        return target_path

    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, "dsr_vq_model_pretrained.zip")
    if force_redownload and os.path.isfile(zip_path):
        os.remove(zip_path)

    if not os.path.isfile(zip_path):
        from baoiad.runtime import download_url

        logger.info("Downloading DSR pretrained VQ-VAE weights from anomalib...")
        try:
            download_url(
                _VQVAE_WEIGHTS_URL,
                zip_path,
                action='download DSR VQ-VAE weights',
                timeout=300,
            )
        except OfflineModeError:
            raise
        except Exception as e:
            if os.path.isfile(zip_path):
                os.remove(zip_path)
            raise RuntimeError(
                f"Failed to download DSR VQ-VAE weights: {e}. "
                f"Please manually download from {_VQVAE_WEIGHTS_URL} and extract to {target_dir}/"
            )
    logger.info("Extracting DSR VQ-VAE weights...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)
    except zipfile.BadZipFile as error:
        if os.path.isfile(zip_path):
            os.remove(zip_path)
        if not force_redownload:
            return _download_vqvae_weights(target_dir=target_dir, force_redownload=True)
        raise RuntimeError(
            f"Downloaded archive at {zip_path} is not a valid zip file."
        ) from error

    # Find and rename extracted file
    for root, dirs, files in os.walk(target_dir):
        for f in files:
            if f.endswith('.pckl') or f.endswith('.pth'):
                src = os.path.join(root, f)
                if src != target_path:
                    import shutil
                    shutil.move(src, target_path)
                break

    # Clean up zip file
    if os.path.isfile(zip_path):
        os.remove(zip_path)

    if not os.path.isfile(target_path):
        raise FileNotFoundError(
            f"Failed to obtain DSR VQ-VAE weights under {target_path}. "
            f"Please manually download from {_VQVAE_WEIGHTS_URL} and extract to {target_dir}/"
        )

    return target_path


# ==================== Perlin Noise ====================

def _lerp_np(x, y, w):
    return (y - x) * w + x


def rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3):
    h, w = shape
    delta = (res[0] / h, res[1] / w)
    d = (h // res[0], w // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    def tile_grads(s1, s2):
        return np.repeat(
            np.repeat(gradients[s1[0]:s1[1], s2[0]:s2[1]], d[0], axis=0),
            d[1],
            axis=1,
        )

    def dot(grad, shift):
        return (
            np.stack((grid[:h, :w, 0] + shift[0], grid[:h, :w, 1] + shift[1]), axis=-1) * grad[:h, :w]
        ).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:h, :w])
    return math.sqrt(2) * _lerp_np(_lerp_np(n00, n10, t[..., 0]), _lerp_np(n01, n11, t[..., 0]), t[..., 1])


def _generate_perlin_anomaly_mask(H, W, p_anomalous=0.5):
    """Generate a binary Perlin noise anomaly mask (B=1)."""
    if np.random.rand() > p_anomalous:
        return torch.zeros(1, 1, H, W)
    scalex = 2 ** np.random.randint(0, 6)
    scaley = 2 ** np.random.randint(0, 6)
    noise = rand_perlin_2d_np((H, W), (scalex, scaley))
    k = np.random.randint(0, 4)
    noise = np.rot90(noise, k)
    threshold = 0.5
    if not (noise > threshold).any():
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        noise = noise * 2 - 1
    mask = (noise > threshold).astype(np.float32)
    return torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)


def _generate_perlin_anomaly_batch(batch_size, H, W, p_anomalous=0.5):
    masks = [_generate_perlin_anomaly_mask(H, W, p_anomalous) for _ in range(batch_size)]
    return torch.cat(masks, dim=0)


# ==================== Residual Blocks ====================

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, num_residual_hiddens):
        super().__init__()
        self._block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_residual_hiddens, 3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_residual_hiddens, out_channels, 1, stride=1, bias=False),
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super().__init__()
        self._layers = nn.ModuleList([
            Residual(in_channels, num_hiddens, num_residual_hiddens)
            for _ in range(num_residual_layers)
        ])

    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return F.relu(x)


# ==================== Vector Quantizer ====================

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(num_embeddings, embedding_dim)
        self._embedding.weight.data.normal_()
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.zeros(num_embeddings, embedding_dim))
        self._ema_w.data.normal_()

    @property
    def embedding(self):
        return self._embedding

    def forward(self, inputs):
        inputs_perm = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs_perm.shape
        flat_input = inputs_perm.view(-1, self._embedding_dim)
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        quantized = inputs_perm + (quantized - inputs_perm).detach()
        return quantized.permute(0, 3, 1, 2).contiguous()


# ==================== Discrete Latent Model ====================

class EncoderBot(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super().__init__()
        self._conv_1 = nn.Conv2d(in_channels, num_hiddens // 2, 4, stride=2, padding=1)
        self._conv_2 = nn.Conv2d(num_hiddens // 2, num_hiddens, 4, stride=2, padding=1)
        self._conv_3 = nn.Conv2d(num_hiddens, num_hiddens, 3, stride=1, padding=1)
        self._residual_stack = ResidualStack(num_hiddens, num_hiddens, num_residual_layers, num_residual_hiddens)

    def forward(self, x):
        x = F.relu(self._conv_1(x))
        x = F.relu(self._conv_2(x))
        x = self._conv_3(x)
        return self._residual_stack(x)


class EncoderTop(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super().__init__()
        self._conv_1 = nn.Conv2d(in_channels, num_hiddens, 4, stride=2, padding=1)
        self._conv_2 = nn.Conv2d(num_hiddens, num_hiddens, 3, stride=1, padding=1)
        self._residual_stack = ResidualStack(num_hiddens, num_hiddens, num_residual_layers, num_residual_hiddens)

    def forward(self, x):
        x = F.relu(self._conv_1(x))
        x = F.relu(self._conv_2(x))
        return self._residual_stack(x)


class DecoderBot(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super().__init__()
        self._conv_1 = nn.Conv2d(in_channels, num_hiddens, 3, stride=1, padding=1)
        self._residual_stack = ResidualStack(num_hiddens, num_hiddens, num_residual_layers, num_residual_hiddens)
        self._conv_trans_1 = nn.ConvTranspose2d(num_hiddens, num_hiddens // 2, 4, stride=2, padding=1)
        self._conv_trans_2 = nn.ConvTranspose2d(num_hiddens // 2, 3, 4, stride=2, padding=1)

    def forward(self, x):
        x = self._conv_1(x)
        x = self._residual_stack(x)
        x = F.relu(self._conv_trans_1(x))
        return self._conv_trans_2(x)


class DiscreteLatentModel(nn.Module):
    def __init__(self, num_hiddens=128, num_residual_layers=2, num_residual_hiddens=64,
                 num_embeddings=4096, embedding_dim=128):
        super().__init__()
        self._encoder_b = EncoderBot(3, num_hiddens, num_residual_layers, num_residual_hiddens)
        self._encoder_t = EncoderTop(num_hiddens, num_hiddens, num_residual_layers, num_residual_hiddens)
        self._pre_vq_conv_top = nn.Conv2d(num_hiddens, embedding_dim, 1, stride=1)
        self._pre_vq_conv_bot = nn.Conv2d(num_hiddens + embedding_dim, embedding_dim, 1, stride=1)
        self._vq_vae_top = VectorQuantizer(num_embeddings, embedding_dim)
        self._vq_vae_bot = VectorQuantizer(num_embeddings, embedding_dim)
        self._decoder_b = DecoderBot(embedding_dim * 2, num_hiddens, num_residual_layers, num_residual_hiddens)
        self.upsample_t = nn.ConvTranspose2d(embedding_dim, embedding_dim, 4, stride=2, padding=1)

    @property
    def vq_vae_top(self):
        return self._vq_vae_top

    @property
    def vq_vae_bot(self):
        return self._vq_vae_bot

    @staticmethod
    def generate_fake_anomalies_joined(features, embeddings, memory_torch_original, mask, strength):
        device = embeddings.device
        random_embeddings = torch.zeros(
            (embeddings.shape[0], embeddings.shape[2] * embeddings.shape[3], memory_torch_original.shape[1]),
        )
        inputs = features.permute(0, 2, 3, 1).contiguous()

        for k in range(embeddings.shape[0]):
            memory_torch = memory_torch_original
            flat_input = inputs[k].view(-1, memory_torch.shape[1])
            distances_b = (
                torch.sum(flat_input ** 2, dim=1, keepdim=True)
                + torch.sum(memory_torch ** 2, dim=1)
                - 2 * torch.matmul(flat_input, memory_torch.t())
            )
            percentage_vectors = strength[k]
            topk = max(1, min(int(percentage_vectors * memory_torch.shape[0]) + 1, memory_torch.shape[0] - 1))
            _, topk_indices = torch.topk(distances_b, topk, dim=1, largest=False)
            topk_indices = topk_indices[:, int(memory_torch.shape[0] * 0.05):]
            topk = topk_indices.shape[1]
            if topk < 1:
                topk = 1
                topk_indices = topk_indices[:, :1]
            random_indices_hik = torch.randint(topk, size=(topk_indices.shape[0],))
            random_indices_t = topk_indices[torch.arange(random_indices_hik.shape[0]), random_indices_hik]
            random_embeddings[k] = memory_torch[random_indices_t, :]

        random_embeddings = random_embeddings.reshape(
            random_embeddings.shape[0], embeddings.shape[2], embeddings.shape[3], random_embeddings.shape[2],
        )
        random_embeddings_tensor = random_embeddings.permute(0, 3, 1, 2).to(device)

        down_ratio_y = int(mask.shape[2] / embeddings.shape[2])
        down_ratio_x = int(mask.shape[3] / embeddings.shape[3])
        anomaly_mask = F.max_pool2d(mask, (down_ratio_y, down_ratio_x)).float()

        return anomaly_mask * random_embeddings_tensor + (1.0 - anomaly_mask) * embeddings

    def forward(self, batch, anomaly_mask=None, anom_str_lo=None, anom_str_hi=None):
        device = batch.device
        enc_b = self._encoder_b(batch)
        enc_t = self._encoder_t(enc_b)
        zt = self._pre_vq_conv_top(enc_t)
        quantized_t = self._vq_vae_top(zt)
        up_quantized_t = self.upsample_t(quantized_t)
        feat = torch.cat((enc_b, up_quantized_t), dim=1)
        zb = self._pre_vq_conv_bot(feat)
        quantized_b = self._vq_vae_bot(zb)

        outputs = {'quantized_b': quantized_b, 'quantized_t': quantized_t}

        if anomaly_mask is not None:
            anomaly_embedding_lo = self.generate_fake_anomalies_joined(
                zt, quantized_t, self._vq_vae_top.embedding.weight, anomaly_mask, anom_str_lo,
            )
            up_quantized_t_defect = self.upsample_t(anomaly_embedding_lo)
            feat_defect = torch.cat((enc_b, up_quantized_t_defect), dim=1)
            zb_defect = self._pre_vq_conv_bot(feat_defect)
            quantized_b_defect = self._vq_vae_bot(zb_defect)

            anomaly_embedding_hi = self.generate_fake_anomalies_joined(
                zb_defect, quantized_b_defect, self._vq_vae_bot.embedding.weight, anomaly_mask, anom_str_hi,
            )

            use_both = torch.randint(0, 2, (batch.shape[0], 1, 1, 1)).to(device).float()
            use_lo = torch.randint(0, 2, (batch.shape[0], 1, 1, 1)).to(device).float()
            use_hi = 1 - use_lo

            anomaly_embedding_hi_usebot = self.generate_fake_anomalies_joined(
                zb, quantized_b, self._vq_vae_bot.embedding.weight, anomaly_mask, anom_str_hi,
            )
            anomaly_embedding_lo_usebot = quantized_t
            anomaly_embedding_hi_usetop = quantized_b
            anomaly_embedding_lo_usetop = anomaly_embedding_lo

            anomaly_embedding_hi_not_both = use_hi * anomaly_embedding_hi_usebot + use_lo * anomaly_embedding_hi_usetop
            anomaly_embedding_lo_not_both = use_hi * anomaly_embedding_lo_usebot + use_lo * anomaly_embedding_lo_usetop

            anomaly_embedding_hi = (
                anomaly_embedding_hi * use_both + anomaly_embedding_hi_not_both * (1.0 - use_both)
            ).detach().clone()
            anomaly_embedding_lo = (
                anomaly_embedding_lo * use_both + anomaly_embedding_lo_not_both * (1.0 - use_both)
            ).detach().clone()

            up_quantized_anomaly_t = self.upsample_t(anomaly_embedding_lo.clone())
            quant_join_anomaly = torch.cat((up_quantized_anomaly_t, anomaly_embedding_hi.clone()), dim=1)
            recon_image = self._decoder_b(quant_join_anomaly)

            down_ratio_x_hi = int(anomaly_mask.shape[3] / quantized_b_defect.shape[3])
            anomaly_mask_hi = F.max_pool2d(anomaly_mask, (down_ratio_x_hi, down_ratio_x_hi)).float()
            anomaly_mask_hi = F.interpolate(anomaly_mask_hi, scale_factor=down_ratio_x_hi)
            down_ratio_x_lo = int(anomaly_mask.shape[3] / quantized_t.shape[3])
            anomaly_mask_lo = F.max_pool2d(anomaly_mask, (down_ratio_x_lo, down_ratio_x_lo)).float()
            anomaly_mask_lo = F.interpolate(anomaly_mask_lo, scale_factor=down_ratio_x_lo)
            final_mask = anomaly_mask_lo * use_both + (anomaly_mask_lo * use_lo + anomaly_mask_hi * use_hi) * (
                1.0 - use_both
            )

            outputs['recon_image'] = recon_image
            outputs['anomaly_mask'] = final_mask
            outputs['anomaly_embedding_lo'] = anomaly_embedding_lo
            outputs['anomaly_embedding_hi'] = anomaly_embedding_hi
        else:
            up_quantized_t2 = self.upsample_t(quantized_t)
            quant_join = torch.cat((up_quantized_t2, quantized_b), dim=1)
            recon_image = self._decoder_b(quant_join)
            outputs['recon_image'] = recon_image

        return outputs


# ==================== Subspace Restriction Network (InstanceNorm) ====================

def _inst_conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.InstanceNorm2d(out_ch),
        nn.ReLU(True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.InstanceNorm2d(out_ch),
        nn.ReLU(True),
    )


class FeatureEncoder(nn.Module):
    def __init__(self, in_channels, base_width):
        super().__init__()
        self.block1 = _inst_conv_block(in_channels, base_width)
        self.mp1 = nn.MaxPool2d(2)
        self.block2 = _inst_conv_block(base_width, base_width * 2)
        self.mp2 = nn.MaxPool2d(2)
        self.block3 = _inst_conv_block(base_width * 2, base_width * 4)

    def forward(self, x):
        b1 = self.block1(x)
        b2 = self.block2(self.mp1(b1))
        b3 = self.block3(self.mp2(b2))
        return b1, b2, b3


class FeatureDecoder(nn.Module):
    """Decoder: only uses b3, ignores b1/b2 (as per anomalib source)."""
    def __init__(self, base_width, out_channels):
        super().__init__()
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width * 4, base_width * 2, 3, padding=1),
            nn.InstanceNorm2d(base_width * 2), nn.ReLU(True),
        )
        self.db2 = _inst_conv_block(base_width * 2, base_width * 2)
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width * 2, base_width, 3, padding=1),
            nn.InstanceNorm2d(base_width), nn.ReLU(True),
        )
        self.db3 = _inst_conv_block(base_width, base_width)
        self.fin_out = nn.Conv2d(base_width, out_channels, 3, padding=1)

    def forward(self, _b1, _b2, b3):
        up2 = self.up2(b3)
        db2 = self.db2(up2)
        up3 = self.up3(db2)
        db3 = self.db3(up3)
        return self.fin_out(db3)


class SubspaceRestrictionModule(nn.Module):
    def __init__(self, base_width):
        super().__init__()
        self.encoder = FeatureEncoder(base_width, base_width)
        self.decoder = FeatureDecoder(base_width, base_width)

    def forward(self, x, quantization):
        b1, b2, b3 = self.encoder(x)
        recon = self.decoder(b1, b2, b3)
        quantized = quantization(recon)
        return recon, quantized


# ==================== Image Reconstruction Network ====================

class ImageReconstructionNetwork(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super().__init__()
        norm_layer = nn.InstanceNorm2d
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1), norm_layer(in_channels), nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels * 2, 3, padding=1), norm_layer(in_channels * 2), nn.ReLU(True),
        )
        self.mp1 = nn.MaxPool2d(2)
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 2, 3, padding=1), norm_layer(in_channels * 2), nn.ReLU(True),
            nn.Conv2d(in_channels * 2, in_channels * 4, 3, padding=1), norm_layer(in_channels * 4), nn.ReLU(True),
        )
        self.mp2 = nn.MaxPool2d(2)
        self.pre_vq_conv = nn.Conv2d(in_channels * 4, 64, 1, stride=1)
        self.upblock1 = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)
        self.upblock2 = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)
        self._conv_1 = nn.Conv2d(64, num_hiddens, 3, stride=1, padding=1)
        self._residual_stack = ResidualStack(num_hiddens, num_hiddens, num_residual_layers, num_residual_hiddens)
        self._conv_trans_1 = nn.ConvTranspose2d(num_hiddens, num_hiddens // 2, 4, stride=2, padding=1)
        self._conv_trans_2 = nn.ConvTranspose2d(num_hiddens // 2, 3, 4, stride=2, padding=1)

    def forward(self, x):
        x = self.block1(x)
        x = self.mp1(x)
        x = self.block2(x)
        x = self.mp2(x)
        x = self.pre_vq_conv(x)
        x = F.relu(self.upblock1(x))
        x = F.relu(self.upblock2(x))
        x = self._conv_1(x)
        x = self._residual_stack(x)
        x = F.relu(self._conv_trans_1(x))
        return self._conv_trans_2(x)


# ==================== UNet for Anomaly Detection / Upsampling ====================

class UnetEncoder(nn.Module):
    def __init__(self, in_channels, base_width):
        super().__init__()
        norm = nn.InstanceNorm2d
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, 3, padding=1), norm(base_width), nn.ReLU(True),
            nn.Conv2d(base_width, base_width, 3, padding=1), norm(base_width), nn.ReLU(True),
        )
        self.mp1 = nn.MaxPool2d(2)
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width, base_width * 2, 3, padding=1), norm(base_width * 2), nn.ReLU(True),
            nn.Conv2d(base_width * 2, base_width * 2, 3, padding=1), norm(base_width * 2), nn.ReLU(True),
        )
        self.mp2 = nn.MaxPool2d(2)
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 4, 3, padding=1), norm(base_width * 4), nn.ReLU(True),
            nn.Conv2d(base_width * 4, base_width * 4, 3, padding=1), norm(base_width * 4), nn.ReLU(True),
        )
        self.mp3 = nn.MaxPool2d(2)
        self.block4 = nn.Sequential(
            nn.Conv2d(base_width * 4, base_width * 4, 3, padding=1), norm(base_width * 4), nn.ReLU(True),
            nn.Conv2d(base_width * 4, base_width * 4, 3, padding=1), norm(base_width * 4), nn.ReLU(True),
        )

    def forward(self, x):
        b1 = self.block1(x)
        b2 = self.block2(self.mp1(b1))
        b3 = self.block3(self.mp2(b2))
        b4 = self.block4(self.mp3(b3))
        return b1, b2, b3, b4


class UnetDecoder(nn.Module):
    def __init__(self, base_width, out_channels):
        super().__init__()
        norm = nn.InstanceNorm2d
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width * 4, base_width * 4, 3, padding=1), norm(base_width * 4), nn.ReLU(True),
        )
        self.db1 = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 4, 3, padding=1), norm(base_width * 4), nn.ReLU(True),
            nn.Conv2d(base_width * 4, base_width * 4, 3, padding=1), norm(base_width * 4), nn.ReLU(True),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width * 4, base_width * 2, 3, padding=1), norm(base_width * 2), nn.ReLU(True),
        )
        self.db2 = nn.Sequential(
            nn.Conv2d(base_width * 4, base_width * 2, 3, padding=1), norm(base_width * 2), nn.ReLU(True),
            nn.Conv2d(base_width * 2, base_width * 2, 3, padding=1), norm(base_width * 2), nn.ReLU(True),
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width * 2, base_width, 3, padding=1), norm(base_width), nn.ReLU(True),
        )
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width, 3, padding=1), norm(base_width), nn.ReLU(True),
            nn.Conv2d(base_width, base_width, 3, padding=1), norm(base_width), nn.ReLU(True),
        )
        self.fin_out = nn.Conv2d(base_width, out_channels, 3, padding=1)

    def forward(self, b1, b2, b3, b4):
        up1 = self.up1(b4)
        db1 = self.db1(torch.cat([up1, b3], dim=1))
        up2 = self.up2(db1)
        db2 = self.db2(torch.cat([up2, b2], dim=1))
        up3 = self.up3(db2)
        db3 = self.db3(torch.cat([up3, b1], dim=1))
        return self.fin_out(db3)


class UnetModel(nn.Module):
    def __init__(self, in_channels, out_channels, base_width):
        super().__init__()
        self.encoder = UnetEncoder(in_channels, base_width)
        self.decoder = UnetDecoder(base_width, out_channels)

    def forward(self, x):
        b1, b2, b3, b4 = self.encoder(x)
        return self.decoder(b1, b2, b3, b4)


class AnomalyDetectionModule(nn.Module):
    """Compares object-specific decoded image vs general decoded image."""
    def __init__(self, in_channels, out_channels, base_width):
        super().__init__()
        self.unet = UnetModel(in_channels, out_channels, base_width)

    def forward(self, batch_real, batch_anomaly):
        return self.unet(torch.cat((batch_real, batch_anomaly), dim=1))


class UpsamplingModule(nn.Module):
    """Refines anomaly map to full resolution."""
    def __init__(self, in_channels, out_channels, base_width):
        super().__init__()
        self.unet = UnetModel(in_channels, out_channels, base_width)

    def forward(self, batch_real, batch_anomaly, batch_segmentation_map):
        return self.unet(torch.cat((batch_real, batch_anomaly, batch_segmentation_map), dim=1))


# ==================== DSR Detector ====================

@MODELS.register_module(force=True)
class DSRDetector(ReconstructionADModel):
    """DSR: Dual Subspace Re-projection Network for Surface Anomaly Detection.

    Three training phases:
    - Phase 1: Pre-train discrete latent model (VQ-VAE) — loaded from checkpoint
    - Phase 2: Train subspace restriction + image reconstruction + anomaly detection
    - Phase 3: Train upsampling module with Perlin noise anomalies

    Args:
        embedding_dim: Dimension of VQ-VAE embeddings. Default 128.
        num_embeddings: Number of VQ-VAE codebook entries. Default 4096.
        latent_anomaly_strength: Strength of generated latent anomalies. Default 0.2.
        num_hiddens: Hidden channels in residual blocks. Default 128.
        num_residual_layers: Number of residual layers. Default 2.
        num_residual_hiddens: Intermediate channels in residual blocks. Default 64.
        phase: Training phase (1, 2, or 3). Default 2.
        pretrained_vqvae_path: Path to pretrained VQ-VAE checkpoint (phase 2/3).
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_embeddings: int = 4096,
        latent_anomaly_strength: float = 0.2,
        num_hiddens: int = 128,
        num_residual_layers: int = 2,
        num_residual_hiddens: int = 64,
        phase: int = 2,
        upsampling_train_ratio: float = 0.7,
        pretrained_vqvae_path: Optional[str] = None,
        recon_loss=dict(type='MSELoss'),
        feat_loss=dict(type='MSELoss'),
        seg_loss=dict(type='FocalLoss', alpha=1.0, gamma=2.0),
        data_preprocessor=None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.phase = phase
        self.upsampling_train_ratio = upsampling_train_ratio
        self._phase_switch_epoch = None  # set when max_epochs known
        self.latent_anomaly_strength = latent_anomaly_strength
        self.image_dim = 3
        self.anomaly_map_dim = 2

        self.discrete_latent_model = DiscreteLatentModel(
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )

        self.image_reconstruction_network = ImageReconstructionNetwork(
            in_channels=embedding_dim * 2,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )

        self.subspace_restriction_module_lo = SubspaceRestrictionModule(base_width=embedding_dim)
        self.subspace_restriction_module_hi = SubspaceRestrictionModule(base_width=embedding_dim)

        self.anomaly_detection_module = AnomalyDetectionModule(
            in_channels=2 * self.image_dim,
            out_channels=self.anomaly_map_dim,
            base_width=64,
        )

        self.upsampling_module = UpsamplingModule(
            in_channels=(2 * self.image_dim) + self.anomaly_map_dim,
            out_channels=self.anomaly_map_dim,
            base_width=64,
        )

        resolved_vqvae_path = self._resolve_vqvae_path(pretrained_vqvae_path)
        checkpoint_loaded = False
        if resolved_vqvae_path is not None:
            checkpoint_loaded = self._try_load_discrete_latent_model_weights(
                resolved_vqvae_path,
                allow_auto_redownload=pretrained_vqvae_path == 'auto',
            )
        if not checkpoint_loaded and phase in (2, 3):
            warnings.warn(
                'DSR phase 2/3 is configured without a valid pretrained VQ-VAE checkpoint. '
                'Using random latent model may severely degrade performance.',
                UserWarning,
                stacklevel=2,
            )

        # Freeze discrete latent model for phase 2/3
        if phase in (2, 3):
            for p in self.discrete_latent_model.parameters():
                p.requires_grad = False

        # Loss functions
        self.recon_loss = MODELS.build(recon_loss)
        self.feat_loss = MODELS.build(feat_loss)
        self.seg_loss = MODELS.build(seg_loss)

        # Phase 3 augmentation: aligned with anomalib PerlinAnomalyGenerator
        # anomalib uses MultiRandomChoice with num_transforms=3, fixed_num_transforms=True
        # This is CRITICAL for pixel-level performance - each sample gets 3 random augmentations
        self.phase3_augmenters = MultiRandomChoice(
            transforms=[
                v2.ColorJitter(contrast=(0.5, 2.0)),
                v2.RandomPhotometricDistort(
                    brightness=(0.8, 1.2),
                    contrast=(1.0, 1.0),
                    saturation=(1.0, 1.0),
                    hue=(0.0, 0.0),
                    p=1.0,
                ),
                v2.RandomAdjustSharpness(sharpness_factor=2.0, p=1.0),
                v2.ColorJitter(hue=[-50/360, 50/360], saturation=[0.5, 1.5]),
                v2.RandomSolarize(threshold=128/255, p=1.0),  # threshold will be dynamically set
                v2.RandomPosterize(bits=4, p=1.0),
                v2.RandomInvert(p=1.0),
                v2.AutoAugment(),
                v2.RandomEqualize(p=1.0),
                v2.RandomAffine(degrees=(-45, 45), interpolation=v2.InterpolationMode.BILINEAR, fill=0),
            ],
            num_transforms=3,  # Key: apply 3 transforms per sample (anomalib behavior)
        )

    @staticmethod
    def _resolve_vqvae_path(pretrained_vqvae_path: Optional[str]) -> Optional[str]:
        if not pretrained_vqvae_path:
            return None
        if pretrained_vqvae_path == 'auto':
            # Prefer the .pckl file (anomalib format, matching key names)
            candidates = [
                'pre_trained/vq_model_pretrained_128_4096.pckl',
                'pretrained/vq_model_pretrained_128_4096.pckl',
                'pre_trained/vq_model_pretrained_128_4096.pth',
                'pretrained/vq_model_pretrained_128_4096.pth',
            ]
            for c in candidates:
                if os.path.exists(c):
                    return c
            try:
                return _download_vqvae_weights()
            except OfflineModeError:
                raise
            except Exception as e:
                warnings.warn(
                    f"Failed to auto-download DSR VQ-VAE weights: {e}. "
                    "Using random weights may severely degrade performance.",
                    UserWarning,
                    stacklevel=2,
                )
                return None
        candidate_queue = [pretrained_vqvae_path]
        seen = set()
        while candidate_queue:
            candidate = candidate_queue.pop(0)
            if candidate in seen:
                continue
            seen.add(candidate)
            if os.path.exists(candidate):
                return candidate
            alias_candidates = [
                candidate.replace('pretrained/', 'pre_trained/'),
                candidate.replace('pre_trained/', 'pretrained/'),
            ]
            if candidate.endswith('.pth'):
                alias_candidates.append(candidate[:-4] + '.pckl')
            elif candidate.endswith('.pckl'):
                alias_candidates.append(candidate[:-5] + '.pth')
            for alias in alias_candidates:
                if alias not in seen:
                    candidate_queue.append(alias)
        return None

    @staticmethod
    def _load_vqvae_checkpoint(path: str):
        weights_only = not path.endswith('.pckl')
        checkpoint = torch.load(path, map_location='cpu', weights_only=weights_only)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        return checkpoint

    @staticmethod
    def _is_compatible_vqvae_checkpoint(checkpoint) -> bool:
        if not isinstance(checkpoint, dict):
            return False
        required_keys = {
            '_encoder_b._conv_1.weight',
            '_encoder_t._conv_1.weight',
            '_vq_vae_top._embedding.weight',
            '_vq_vae_bot._embedding.weight',
            '_decoder_b._conv_1.weight',
            'upsample_t.weight',
        }
        return required_keys.issubset(set(checkpoint.keys()))

    def _try_load_discrete_latent_model_weights(
        self,
        checkpoint_path: str,
        allow_auto_redownload: bool = False,
    ) -> bool:
        def _load_from(path: str) -> bool:
            checkpoint = self._load_vqvae_checkpoint(path)
            if not self._is_compatible_vqvae_checkpoint(checkpoint):
                raise RuntimeError(
                    f'Checkpoint at {path} is not compatible with anomalib DSR VQ-VAE weights.'
                )
            self.discrete_latent_model.load_state_dict(checkpoint)
            return True

        try:
            return _load_from(checkpoint_path)
        except OfflineModeError:
            raise
        except Exception as error:
            if not allow_auto_redownload:
                raise
            logger.warning(
                'DSR: failed to load local auto VQ-VAE checkpoint %s (%s). '
                'Retrying anomalib download.',
                checkpoint_path,
                error,
            )
            try:
                redownloaded_path = _download_vqvae_weights(force_redownload=True)
                return _load_from(redownloaded_path)
            except OfflineModeError:
                raise
            except Exception as retry_error:
                warnings.warn(
                    f'Failed to auto-load compatible DSR VQ-VAE weights: {retry_error}. '
                    'Using random weights may severely degrade performance.',
                    UserWarning,
                    stacklevel=2,
                )
                return False

    def _forward_phase1_loss(self, inputs):
        """Phase 1: Train VQ-VAE (discrete latent model)."""
        outputs = self.discrete_latent_model(inputs)
        recon = outputs['recon_image']
        loss_recon = self.recon_loss(recon, inputs)
        return {'loss': loss_recon}

    def _forward_phase2_loss(self, inputs):
        """Phase 2: Train subspace restriction + image recon + anomaly detection."""
        B, C, H, W = inputs.shape
        device = inputs.device

        # Generate Perlin noise anomaly masks
        anomaly_mask = _generate_perlin_anomaly_batch(B, H, W, p_anomalous=0.5).to(device)

        # Generate anomaly strength
        anom_str_lo = (
            torch.rand(B) * (1.0 - self.latent_anomaly_strength) + self.latent_anomaly_strength
        ).to(device)
        anom_str_hi = (
            torch.rand(B) * (1.0 - self.latent_anomaly_strength) + self.latent_anomaly_strength
        ).to(device)

        with torch.no_grad():
            latent_outputs = self.discrete_latent_model(inputs, anomaly_mask, anom_str_lo, anom_str_hi)

        gen_image_def = latent_outputs['recon_image']
        true_anomaly_map = latent_outputs['anomaly_mask']
        embd_top = latent_outputs['quantized_t']
        embd_bot = latent_outputs['quantized_b']
        embd_top_def = latent_outputs['anomaly_embedding_lo']
        embd_bot_def = latent_outputs['anomaly_embedding_hi']

        # Subspace restriction
        recon_feat_hi, recon_embd_hi = self.subspace_restriction_module_hi(
            embd_bot_def, self.discrete_latent_model.vq_vae_bot
        )
        recon_feat_lo, recon_embd_lo = self.subspace_restriction_module_lo(
            embd_top_def, self.discrete_latent_model.vq_vae_top
        )

        # Object-specific image reconstruction
        up_quantized_recon_t = self.discrete_latent_model.upsample_t(recon_embd_lo)
        quant_join = torch.cat((up_quantized_recon_t, recon_embd_hi), dim=1)
        obj_spec_image = self.image_reconstruction_network(quant_join)

        # Anomaly detection
        out_mask = self.anomaly_detection_module(obj_spec_image.detach(), gen_image_def.detach())

        # Losses (aligned with anomalib DsrSecondStageLoss)
        loss_feat_hi = self.feat_loss(recon_feat_hi, embd_bot)
        loss_feat_lo = self.feat_loss(recon_feat_lo, embd_top)
        loss_img = self.recon_loss(obj_spec_image, inputs) * 10  # anomalib uses *10 weight

        # Segmentation loss: anomalib uses FocalLoss(alpha=1, reduction='mean')
        # Approximate with cross_entropy (focal with gamma=0 is CE; anomalib default gamma=2)
        target = F.interpolate(true_anomaly_map, size=out_mask.shape[-2:], mode='nearest')
        target = target.squeeze(1).long().clamp(0, 1)
        loss_seg = self.seg_loss(out_mask, target)

        loss = loss_feat_hi + loss_feat_lo + loss_img + loss_seg
        return {'loss': loss}

    def _forward_phase3_loss(self, inputs):
        """Phase 3: Train upsampling module with Perlin-corrupted images.

        Anomalib approach: generate Perlin anomaly on the image, feed corrupted
        image to the full model pipeline, train upsampling module to detect
        the corrupted regions. The anomaly detection module output is used
        as input to the upsampling module (with no_grad).

        Key fixes (v4) aligned with anomalib PerlinAnomalyGenerator:
        - Use RandomAffine for Perlin noise rotation (any angle, not just 90°)
        - Dynamically set RandomSolarize threshold (32/255 to 128/255)
        - Apply 3 augmentations via MultiRandomChoice (not just 1)
        """
        B, C, H, W = inputs.shape
        device = inputs.device

        # Dynamically set RandomSolarize threshold (anomalib: uniform 32/255 to 128/255)
        random_solarize_threshold = torch.empty(1).uniform_(32/255, 128/255).item()
        for t in self.phase3_augmenters.transforms:
            if isinstance(t, v2.RandomSolarize):
                t.threshold = random_solarize_threshold

        # Generate Perlin anomaly masks and corrupt the images
        anomaly_masks = []
        corrupted_images = []
        for i in range(B):
            # 1. Generate Perlin noise with random scale
            scalex = 2 ** np.random.randint(0, 6)
            scaley = 2 ** np.random.randint(0, 6)
            perlin_noise = rand_perlin_2d_np((H, W), (scalex, scaley))

            # 2. Rescaling logic (aligned with anomalib)
            threshold = 0.5
            if not (perlin_noise > threshold).any():
                perlin_noise = (perlin_noise - perlin_noise.min()) / (perlin_noise.max() - perlin_noise.min() + 1e-8)
                perlin_noise = perlin_noise * 2 - 1

            # 3. Random rotation using RandomAffine (anomalib: degrees=(-90, 90))
            perlin_tensor = torch.from_numpy(perlin_noise.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            rotation_transform = v2.RandomAffine(degrees=(-90, 90), fill=0)
            perlin_rotated = rotation_transform(perlin_tensor).squeeze().cpu().numpy()

            # 4. Generate mask from threshold
            mask = (perlin_rotated > threshold).astype(np.float32)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)

            # 5. Use Perlin noise as anomaly texture (anomalib: noise * 0.5 + 0.25)
            anomaly_texture = torch.from_numpy(perlin_rotated.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            anomaly_texture = anomaly_texture.repeat(1, 3, 1, 1)  # (1, 3, H, W)
            anomaly_texture = anomaly_texture * 0.5 + 0.25  # Scale to [0.25, 0.75] range

            # 6. Apply image augmentation (3 transforms via MultiRandomChoice)
            anomaly_texture_chw = anomaly_texture.squeeze(0)  # [3, H, W]
            anomaly_augmented = self.phase3_augmenters(anomaly_texture_chw)
            anomaly_augmented = anomaly_augmented.unsqueeze(0)  # [1, 3, H, W]

            # 7. Blend with original image (beta ∈ [0.2, 1.0], same as anomalib)
            beta = np.random.uniform(0.2, 1.0)
            corrupted = inputs[i:i+1] * (1 - mask_tensor) + beta * anomaly_augmented * mask_tensor + (1 - beta) * inputs[i:i+1] * mask_tensor

            anomaly_masks.append(mask_tensor.squeeze(0))  # (1, H, W)
            corrupted_images.append(corrupted.squeeze(0))

        anomaly_mask_batch = torch.stack(anomaly_masks)  # (B, 1, H, W)
        corrupted_batch = torch.stack(corrupted_images)  # (B, C, H, W)

        with torch.no_grad():
            latent_outputs = self.discrete_latent_model(corrupted_batch)
            gen_image = latent_outputs['recon_image']
            embd_top = latent_outputs['quantized_t']
            embd_bot = latent_outputs['quantized_b']

            _, recon_embd_bot = self.subspace_restriction_module_hi(
                embd_bot, self.discrete_latent_model.vq_vae_bot
            )
            _, recon_embd_top = self.subspace_restriction_module_lo(
                embd_top, self.discrete_latent_model.vq_vae_top
            )

            up_quantized_recon_t = self.discrete_latent_model.upsample_t(recon_embd_top)
            quant_join = torch.cat((up_quantized_recon_t, recon_embd_bot), dim=1)
            obj_spec_image = self.image_reconstruction_network(quant_join)

            out_mask = self.anomaly_detection_module(obj_spec_image, gen_image)
            out_mask_sm = torch.softmax(out_mask, dim=1)

        upsampled_mask = self.upsampling_module(obj_spec_image, gen_image, out_mask_sm)

        # Target: the Perlin anomaly mask
        target = F.interpolate(anomaly_mask_batch, size=upsampled_mask.shape[-2:], mode='nearest')
        target = target.squeeze(1).long().clamp(0, 1)
        loss = self.seg_loss(upsampled_mask, target)
        return {'loss': loss}

    def _forward_predict(self, inputs):
        """Inference: generate anomaly map and score."""
        with torch.no_grad():
            latent_outputs = self.discrete_latent_model(inputs)
            gen_image = latent_outputs['recon_image']
            embd_top = latent_outputs['quantized_t']
            embd_bot = latent_outputs['quantized_b']

            _, recon_embd_bot = self.subspace_restriction_module_hi(
                embd_bot, self.discrete_latent_model.vq_vae_bot
            )
            _, recon_embd_top = self.subspace_restriction_module_lo(
                embd_top, self.discrete_latent_model.vq_vae_top
            )

            up_quantized_recon_t = self.discrete_latent_model.upsample_t(recon_embd_top)
            quant_join = torch.cat((up_quantized_recon_t, recon_embd_bot), dim=1)
            obj_spec_image = self.image_reconstruction_network(quant_join)

            out_mask = self.anomaly_detection_module(obj_spec_image, gen_image)
            out_mask_sm = torch.softmax(out_mask, dim=1)

        upsampled_mask = self.upsampling_module(obj_spec_image, gen_image, out_mask_sm)
        out_mask_sm_up = torch.softmax(upsampled_mask, dim=1)

        # Anomaly map: channel 1 (anomalous class probability)
        anomaly_map = out_mask_sm_up[:, 1:, :, :]  # (B, 1, H, W)

        # Image score: max of smoothed anomaly map
        smoothed = F.avg_pool2d(out_mask_sm[:, 1:, :, :], 21, stride=1, padding=10)
        img_scores = torch.amax(smoothed, dim=(2, 3)).squeeze(1)

        return anomaly_map, img_scores

    def set_epoch_info(self, epoch, max_epochs):
        """Called by MemoryBankHook before each training epoch."""
        self._current_epoch = epoch
        if self._phase_switch_epoch is None and max_epochs > 0:
            self._phase_switch_epoch = int(max_epochs * self.upsampling_train_ratio)
            logger.info(f"DSR: will switch to phase 3 at epoch {self._phase_switch_epoch}/{max_epochs}")
        self._maybe_switch_phase()

    def _maybe_switch_phase(self):
        """Auto-switch from phase 2 to phase 3 based on epoch.

        Mimics anomalib: phase 2 for first upsampling_train_ratio * max_epochs,
        then phase 3 for the rest.
        """
        if self.phase != 2 or self.upsampling_train_ratio <= 0:
            return

        if self._phase_switch_epoch is None:
            return

        current_epoch = getattr(self, '_current_epoch', 0)
        if current_epoch >= self._phase_switch_epoch:
            logger.info(f"DSR: switching to phase 3 (upsampling) at epoch {current_epoch}")
            self.phase = 3
            # Freeze everything except upsampling module
            for name, param in self.named_parameters():
                if 'upsampling_module' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        if mode == 'loss':
            self._maybe_switch_phase()
            if self.phase == 1:
                return self._forward_phase1_loss(inputs)
            elif self.phase == 2:
                return self._forward_phase2_loss(inputs)
            else:
                return self._forward_phase3_loss(inputs)

        elif mode == 'predict':
            anomaly_map, img_scores = self._forward_predict(inputs)
            return build_predict_results(data_samples, img_scores, anomaly_map)

        # tensor mode
        latent_outputs = self.discrete_latent_model(inputs)
        return latent_outputs
