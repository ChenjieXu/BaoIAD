"""RegAD: Registration-Based Few-Shot Anomaly Detection (ECCV 2022).

Key components:
1. STN (Spatial Transformer Network) for feature registration
2. Encoder-Predictor for metric learning (BYOL-style)
3. Mahalanobis distance scoring on multi-scale embeddings
4. Support set augmentation (24 variants)

Reference: https://github.com/MediaBrain-SJTU/RegAD
"""
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from baoiad.models.base_ad_model import MemoryBankADModel
from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS

logger = logging.getLogger(__name__)


# ============================================================================
# Helper functions
# ============================================================================

def embedding_concat(x, y):
    """Concatenate embeddings from different scales.

    Args:
        x: (B, C1, H1, W1) - larger feature map
        y: (B, C2, H2, W2) - smaller feature map

    Returns:
        (B, C1+C2, H1, W1) - concatenated at larger resolution
    """
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = H1 // H2  # scale factor

    # Unfold x to match y's spatial size
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)  # (B, C1*s*s, H2*W2)
    x = x.view(B, C1, s*s, H2, W2)

    # Repeat y to match unfolded x
    y = y.unsqueeze(2).expand(-1, -1, s*s, -1, -1)  # (B, C2, s*s, H2, W2)

    # Concatenate and fold back
    z = torch.cat([x, y], dim=1)  # (B, C1+C2, s*s, H2, W2)
    z = z.view(B, -1, H2*W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
    return z


def mahalanobis_distance(u, v, cov_inv):
    """Compute Mahalanobis distance.

    Args:
        u: (C,) - sample vector
        v: (C,) - mean vector
        cov_inv: (C, C) - inverse covariance matrix

    Returns:
        scalar distance
    """
    delta = u - v
    m = torch.dot(delta, torch.matmul(cov_inv, delta))
    return torch.sqrt(m.clamp(min=1e-10))


# ============================================================================
# Encoder and Predictor networks (BYOL-style)
# ============================================================================

class Encoder(nn.Module):
    """3-layer conv encoder for feature transformation."""

    def __init__(self, in_channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.conv3 = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return x


class Predictor(nn.Module):
    """2-layer conv predictor for BYOL-style prediction."""

    def __init__(self, in_channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 1, bias=False)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return x


# ============================================================================
# STN (Spatial Transformer Network)
# ============================================================================

class STNModule(nn.Module):
    """Spatial Transformer Network for feature registration.

    Uses rotation_scale mode (3 params: angle, scale_x, scale_y).
    """

    def __init__(self, in_channels, feat_size, stn_mode='rotation_scale'):
        super().__init__()
        self.feat_size = feat_size  # Feature map size after pooling
        self.stn_mode = stn_mode
        self.n_params = 3  # rotation_scale mode

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        self.fc = nn.Sequential(
            nn.Linear(16 * self.feat_size * self.feat_size, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.n_params),
        )

        # Initialize to identity transform
        self.fc[2].weight.data.zero_()
        self.fc[2].bias.data.copy_(torch.tensor([0.0, 1.0, 1.0]))  # angle=0, scale_x=1, scale_y=1

    def forward(self, x):
        """
        Returns:
            transformed: transformed feature map
            theta: (B, 2, 3) transformation matrix
        """
        batch_size = x.size(0)
        conv_x = self.conv(x)
        theta_params = self.fc(conv_x.view(batch_size, -1))

        # Build rotation_scale transformation matrix
        angle = theta_params[:, 0]
        scale_x = theta_params[:, 1]
        scale_y = theta_params[:, 2]

        theta = x.new_zeros(batch_size, 2, 3)
        theta[:, 0, 0] = torch.cos(angle) * scale_x
        theta[:, 0, 1] = -torch.sin(angle)
        theta[:, 1, 0] = torch.sin(angle)
        theta[:, 1, 1] = torch.cos(angle) * scale_y

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        transformed = F.grid_sample(x, grid, padding_mode='reflection', align_corners=False)

        return transformed, theta


def inverse_transform(theta):
    """Compute inverse transformation matrix.

    Args:
        theta: (B, 2, 3) affine transformation matrix

    Returns:
        theta_inv: (B, 2, 3) inverse transformation matrix
    """
    batch_size = theta.size(0)
    # Add row [0, 0, 1] to make 3x3 matrix
    theta_3x3 = theta.new_zeros(batch_size, 3, 3)
    theta_3x3[:, :2, :] = theta
    theta_3x3[:, 2, 2] = 1.0

    # Compute inverse
    theta_inv = torch.linalg.inv(theta_3x3)
    return theta_inv[:, :2, :]


# ============================================================================
# ResNet-18 Backbone with STN
# ============================================================================

class ResNet18STN(nn.Module):
    """ResNet-18 backbone with STN modules at each layer.

    Outputs registered features at multiple scales.
    """

    def __init__(self, pretrained=True, stn_mode='rotation_scale', img_size=256):
        super().__init__()
        from torchvision.models import resnet18
        import torchvision
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = resnet18(weights=weights)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels

        # Compute feature sizes for STN
        # After conv1+bn1+relu+maxpool: img_size/4
        # After layer1: img_size/4
        # After layer2: img_size/8
        # After layer3: img_size/16
        feat_size_1 = img_size // 4  # layer1 output size
        feat_size_2 = img_size // 8  # layer2 output size
        feat_size_3 = img_size // 16  # layer3 output size

        # STN conv has two MaxPool2d(3, stride=2, padding=1)
        # Output size formula: floor((input_size + 2*padding - kernel_size) / stride) + 1
        # For kernel=3, stride=2, padding=1: floor((input_size + 2 - 3) / 2) + 1 = floor((input_size - 1) / 2) + 1
        def compute_stn_output_size(input_size):
            size = input_size
            # First MaxPool2d(3, stride=2, padding=1)
            size = (size + 2 * 1 - 3) // 2 + 1
            # Second MaxPool2d(3, stride=2, padding=1)
            size = (size + 2 * 1 - 3) // 2 + 1
            return max(1, size)

        # STN modules with computed feature sizes (after STN conv's 2 pooling layers)
        self.stn1 = STNModule(64, compute_stn_output_size(feat_size_1), stn_mode)
        self.stn2 = STNModule(128, compute_stn_output_size(feat_size_2), stn_mode)
        self.stn3 = STNModule(256, compute_stn_output_size(feat_size_3), stn_mode)

        # Output channels
        self.channel_dims = {1: 64, 2: 128, 3: 256}

    def forward(self, x):
        """Extract features with STN registration.

        Returns:
            dict with keys 'layer1', 'layer2', 'layer3' containing
            registered features at each scale.
        """
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Layer 1 + STN1
        x = self.layer1(x)
        x, theta1 = self.stn1(x)
        theta1_inv = inverse_transform(theta1)
        stn1_output = self._apply_transform(x.detach(), theta1_inv)

        # Layer 2 + STN2
        x = self.layer2(x)
        x, theta2 = self.stn2(x)
        theta2_inv = inverse_transform(theta2)
        # Apply inverse transforms in sequence (theta1_inv, then theta2_inv)
        stn2_output = self._apply_transform(
            self._apply_transform(x.detach(), theta2_inv), theta1_inv)

        # Layer 3 + STN3
        x = self.layer3(x)
        x, theta3 = self.stn3(x)
        theta3_inv = inverse_transform(theta3)
        # Apply inverse transforms in sequence
        stn3_output = self._apply_transform(
            self._apply_transform(
                self._apply_transform(x.detach(), theta3_inv), theta2_inv), theta1_inv)

        return {
            'layer1': stn1_output,
            'layer2': stn2_output,
            'layer3': stn3_output,
            'final_feat': x  # For encoder input
        }

    def _apply_transform(self, x, theta):
        """Apply transformation to feature map."""
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, padding_mode='reflection', align_corners=False)


# ============================================================================
# Support Set Augmentation
# ============================================================================

def augment_support_set(support_img):
    """Apply 24 augmentations to support images.

    Args:
        support_img: (K, C, H, W) support images

    Returns:
        (24*K, C, H, W) augmented support images
    """
    augmented = [support_img]

    # 8 small rotations
    for angle in [-math.pi/4, -3*math.pi/16, -math.pi/8, -math.pi/16,
                  math.pi/16, math.pi/8, 3*math.pi/16, math.pi/4]:
        rotated = rot_img(support_img, angle)
        augmented.append(rotated)

    # 8 translations
    for a, b in [(0.2, 0.2), (-0.2, 0.2), (-0.2, -0.2), (0.2, -0.2),
                 (0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1), (0.1, -0.1)]:
        translated = translation_img(support_img, a, b)
        augmented.append(translated)

    # Horizontal flip
    flipped = hflip_img(support_img)
    augmented.append(flipped)

    # Grayscale
    greyed = grey_img(support_img)
    augmented.append(greyed)

    # Rotations 90, 180, 270
    for k in [1, 2, 3]:
        rotated90 = rot90_img(support_img, k)
        augmented.append(rotated90)

    # Concatenate all
    result = torch.cat(augmented, dim=0)
    # Shuffle
    result = result[torch.randperm(result.size(0))]
    return result


def rot_img(x, theta):
    """Rotate image by angle theta."""
    batch_size = x.size(0)
    rot_mat = x.new_zeros(batch_size, 2, 3)
    rot_mat[:, 0, 0] = math.cos(theta)
    rot_mat[:, 0, 1] = -math.sin(theta)
    rot_mat[:, 1, 0] = math.sin(theta)
    rot_mat[:, 1, 1] = math.cos(theta)
    grid = F.affine_grid(rot_mat, x.size(), align_corners=False)
    return F.grid_sample(x, grid, padding_mode='reflection', align_corners=False)


def translation_img(x, a, b):
    """Translate image by (a, b)."""
    batch_size = x.size(0)
    trans_mat = x.new_zeros(batch_size, 2, 3)
    trans_mat[:, 0, 0] = 1.0
    trans_mat[:, 0, 2] = a
    trans_mat[:, 1, 1] = 1.0
    trans_mat[:, 1, 2] = b
    grid = F.affine_grid(trans_mat, x.size(), align_corners=False)
    return F.grid_sample(x, grid, padding_mode='reflection', align_corners=False)


def hflip_img(x):
    """Horizontal flip."""
    return torch.flip(x, dims=[-1])


def grey_img(x):
    """Convert to grayscale (but keep 3 channels)."""
    # RGB to grayscale: 0.299*R + 0.587*G + 0.114*B
    grey = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    return grey.expand(-1, 3, -1, -1)


def rot90_img(x, k):
    """Rotate by k*90 degrees."""
    return torch.rot90(x, k, dims=[-2, -1])


# ============================================================================
# Cosine Loss (BYOL-style)
# ============================================================================

def cos_loss(data1, data2, mean=True):
    """Cosine similarity loss with stop-gradient.

    Args:
        data1: (B, C, H, W)
        data2: (B, C, H, W)
        mean: if True, return mean; else return per-position values

    Returns:
        negative cosine similarity (to minimize)
    """
    data2 = data2.detach()  # Stop gradient
    cos_sim = F.cosine_similarity(data1, data2, dim=1)  # (B, H, W)
    if mean:
        return -cos_sim.mean()
    else:
        return -cos_sim


# ============================================================================
# Gaussian Blur
# ============================================================================

class GaussianBlur2d(nn.Module):
    """2D Gaussian blur using depthwise convolution."""

    def __init__(self, sigma=4.0, kernel_size=None):
        super().__init__()
        if kernel_size is None:
            kernel_size = 2 * int(4.0 * sigma + 0.5) + 1

        x = torch.arange(kernel_size).float() - kernel_size // 2
        gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        kernel_2d = gauss.unsqueeze(1) * gauss.unsqueeze(0)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
        self.register_buffer('kernel', kernel_2d)
        self.padding = kernel_size // 2

    def forward(self, x):
        B, C, H, W = x.shape
        kernel = self.kernel.expand(C, -1, -1, -1)
        return F.conv2d(x, kernel, padding=self.padding, groups=C)


# ============================================================================
# RegAD Detector
# ============================================================================

@MODELS.register_module()
class RegADDetector(MemoryBankADModel):
    """RegAD: Registration-based Few-Shot Anomaly Detection.

    Key components:
    1. ResNet-18 backbone with STN modules
    2. Encoder-Predictor for metric learning
    3. Mahalanobis distance scoring on multi-scale embeddings
    4. Support set augmentation
    """

    def __init__(self,
                 backbone='resnet18',
                 layers=(1, 2, 3),
                 sigma=4.0,
                 stn_mode='rotation_scale',
                 encoder_channels=256,
                 img_size=256,
                 few_shot=None,
                 pretrained_backbone=True,
                 freeze_backbone=False,
                 data_preprocessor=None,
                 init_cfg=None,
                 target_cls=None,
                 data_root=None,
                 **kwargs):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.few_shot = few_shot  # None = use all data; K = use K random samples
        self.img_size = img_size
        self.target_cls = target_cls  # For cross-category training
        self.data_root = data_root  # For loading support images from target category
        self.pretrained_backbone = pretrained_backbone
        self.freeze_backbone = freeze_backbone

        # Build backbone with STN
        if isinstance(backbone, dict):
            self.backbone = MODELS.build(backbone)
        else:
            self.backbone = ResNet18STN(
                pretrained=pretrained_backbone,
                stn_mode=stn_mode,
                img_size=img_size,
            )

        self.layers = layers

        # Encoder and Predictor
        self.encoder = Encoder(encoder_channels)
        self.predictor = Predictor(encoder_channels)

        # Support set memory
        self.register_buffer('support_feat', None)  # Mean support feature
        self.register_buffer('embedding_mean', None)  # For Mahalanobis
        self.register_buffer('embedding_cov_inv', None)  # For Mahalanobis
        self._cached_support_images = None

        # Gaussian blur for score map
        self.blur = GaussianBlur2d(sigma=sigma) if sigma > 0 else None

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                               missing_keys, unexpected_keys, error_msgs):
        """Override to properly load buffers that may be None initially.

        The support_feat, embedding_mean, and embedding_cov_inv buffers are
        saved in checkpoints but may not be loaded by default because they
        are None when the model is created.
        """
        # Handle buffer keys that exist in checkpoint
        buffer_keys = ['support_feat', 'embedding_mean', 'embedding_cov_inv']
        for key in buffer_keys:
            full_key = prefix + key
            if full_key in state_dict:
                # Get the tensor and keep it on CPU for now
                # (will be moved to correct device when model is moved)
                tensor = state_dict[full_key]
                self.register_buffer(key, tensor, persistent=True)
                # Remove from unexpected_keys if it was added
                if full_key in unexpected_keys:
                    unexpected_keys.remove(full_key)

        # Call parent implementation
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                       missing_keys, unexpected_keys, error_msgs)

    def extract_features(self, x):
        """Extract multi-scale features with STN registration."""
        if self.freeze_backbone:
            with torch.no_grad():
                return self.backbone(x)
        return self.backbone(x)

    @staticmethod
    def _prepare_inputs(inputs):
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)
        if torch.is_tensor(inputs) and inputs.dim() == 5 and inputs.size(0) == 1:
            inputs = inputs.squeeze(0)
        return inputs

    @staticmethod
    def _prepare_support_tensor(data_samples, device):
        if data_samples is None:
            return None

        samples = data_samples if isinstance(data_samples, (list, tuple)) else [data_samples]
        support_tensors = []
        for sample in samples:
            support_imgs = getattr(sample, 'support_imgs', None)
            if support_imgs is None:
                continue
            if isinstance(support_imgs, (list, tuple)):
                support_imgs = torch.stack([
                    item if torch.is_tensor(item) else torch.as_tensor(item)
                    for item in support_imgs
                ], dim=0)
            elif not torch.is_tensor(support_imgs):
                support_imgs = torch.as_tensor(support_imgs)
            support_tensors.append(support_imgs.float())

        if not support_tensors:
            return None

        if len(support_tensors) == 1:
            support_imgs = support_tensors[0]
        else:
            support_imgs = torch.stack(support_tensors, dim=0)

        if support_imgs.dim() == 6 and support_imgs.size(0) == 1:
            support_imgs = support_imgs.squeeze(0)
        if support_imgs.dim() == 4:
            support_imgs = support_imgs.unsqueeze(0)
        return support_imgs.to(device)

    def forward(self, inputs, data_samples=None, mode='tensor'):
        inputs = self._prepare_inputs(inputs)

        if mode == 'loss':
            return self._forward_train(inputs, data_samples)
        elif mode == 'predict':
            return self._forward_predict(inputs, data_samples)
        else:
            return self.extract_features(inputs)

    def _forward_train(self, inputs, data_samples):
        """Training forward pass.

        Uses BYOL-style symmetric cosine loss:
        loss = CosLoss(p1, z2)/2 + CosLoss(p2, z1)/2
        """
        if not torch.is_tensor(inputs) or inputs.dim() != 4:
            raise ValueError(
                f'RegAD training expects inputs with shape (B,C,H,W), got {type(inputs)!r} '
                f'and shape {tuple(inputs.shape) if torch.is_tensor(inputs) else None}.'
            )

        support_imgs = self._prepare_support_tensor(data_samples, inputs.device)
        if support_imgs is None:
            raise ValueError('RegAD training requires `support_imgs` in each data sample.')
        if support_imgs.dim() != 5:
            raise ValueError(f'Expected support_imgs with shape (B,K,C,H,W), got {tuple(support_imgs.shape)}')
        if support_imgs.size(0) != inputs.size(0):
            raise ValueError(
                f'Batch mismatch between query inputs ({inputs.size(0)}) '
                f'and support_imgs ({support_imgs.size(0)}).'
            )
        self._cached_support_images = support_imgs.detach().cpu().reshape(-1, *support_imgs.shape[-3:])

        query_feat = self.extract_features(inputs)['final_feat']

        batch_size, shot, channels, height, width = support_imgs.shape
        support_feats = self.extract_features(
            support_imgs.view(batch_size * shot, channels, height, width)
        )['final_feat']
        _, feat_channels, feat_h, feat_w = support_feats.shape
        support_feat = support_feats.view(batch_size, shot, feat_channels, feat_h, feat_w).mean(dim=1)

        z1 = self.encoder(query_feat)
        z2 = self.encoder(support_feat)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        loss = cos_loss(p1, z2, mean=True) / 2 + cos_loss(p2, z1, mean=True) / 2
        return {'loss': loss}

    def _compute_mahalanobis_map(self, inputs):
        if self.embedding_mean is None or self.embedding_cov_inv is None:
            raise RuntimeError(
                'RegAD predict() requires build_support_bank_from_images()/fit() before inference.'
            )

        query_feats = self.extract_features(inputs)
        layer_names = [f'layer{int(layer)}' for layer in self.layers]
        embedding_vectors = query_feats[layer_names[0]]
        for layer_name in layer_names[1:]:
            embedding_vectors = embedding_concat(embedding_vectors, query_feats[layer_name])

        batch_size, channels, height, width = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(batch_size, channels, height * width)

        device = inputs.device
        embedding_mean = self.embedding_mean.to(device)
        embedding_cov_inv = self.embedding_cov_inv.to(device)

        dist_list = []
        for i in range(height * width):
            mean = embedding_mean[:, i]
            cov_inv = embedding_cov_inv[:, :, i]
            dists = torch.stack([
                mahalanobis_distance(sample[:, i], mean, cov_inv)
                for sample in embedding_vectors
            ], dim=0)
            dist_list.append(dists)

        dist_tensor = torch.stack(dist_list, dim=1).reshape(batch_size, height, width)
        anomaly_map = F.interpolate(
            dist_tensor.unsqueeze(1),
            size=inputs.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        if self.blur is not None:
            anomaly_map = self.blur(anomaly_map)
        return anomaly_map

    @torch.no_grad()
    def predict_raw_maps(self, inputs):
        """Return the raw anomaly map and image score before result packing."""
        inputs = self._prepare_inputs(inputs)
        if not torch.is_tensor(inputs) or inputs.dim() != 4:
            raise ValueError(f'Expected inputs with shape (B,C,H,W), got {type(inputs)!r}')
        anomaly_map = self._compute_mahalanobis_map(inputs)
        img_scores = anomaly_map.view(inputs.size(0), -1).max(dim=1).values
        return anomaly_map, img_scores

    def _forward_predict(self, inputs, data_samples):
        """Inference forward pass.

        Uses Mahalanobis distance on concatenated embeddings.
        """
        anomaly_map, img_scores = self.predict_raw_maps(inputs)
        return build_predict_results(data_samples, img_scores, anomaly_map)

    @torch.no_grad()
    def build_support_bank_from_images(self, support_images):
        """Build support statistics from fixed support images."""
        if not torch.is_tensor(support_images):
            support_images = torch.as_tensor(support_images)
        if support_images.dim() == 5 and support_images.size(0) == 1:
            support_images = support_images.squeeze(0)
        if support_images.dim() != 4:
            raise ValueError(
                f'support_images must have shape (K,C,H,W), got {tuple(support_images.shape)}'
            )

        device = next(self.parameters()).device
        support_images = support_images.detach().cpu().float()

        if self.few_shot is not None and support_images.size(0) > self.few_shot:
            generator = torch.Generator().manual_seed(42)
            indices = torch.randperm(support_images.size(0), generator=generator)[:self.few_shot]
            support_images = support_images[indices]

        support_images = augment_support_set(support_images)
        if support_images.size(0) < 2:
            raise ValueError('RegAD requires at least two augmented support images to estimate covariance.')

        batch_size = 8
        layer_names = [f'layer{int(layer)}' for layer in self.layers]
        all_layers = {layer_name: [] for layer_name in layer_names}
        all_final = []

        was_training = self.training
        self.eval()
        for start in range(0, support_images.size(0), batch_size):
            end = min(start + batch_size, support_images.size(0))
            batch = support_images[start:end].to(device)
            feats = self.backbone(batch)
            for layer_name in layer_names:
                all_layers[layer_name].append(feats[layer_name])
            all_final.append(feats['final_feat'])

        if was_training:
            self.train()

        final_feats = torch.cat(all_final, dim=0)
        self.support_feat = final_feats.mean(dim=0, keepdim=True)

        layer_feats = {
            layer_name: torch.cat(layer_tensors, dim=0)
            for layer_name, layer_tensors in all_layers.items()
        }
        embedding_vectors = layer_feats[layer_names[0]]
        for layer_name in layer_names[1:]:
            embedding_vectors = embedding_concat(embedding_vectors, layer_feats[layer_name])
        batch_size, channels, height, width = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(batch_size, channels, height * width)

        mean = embedding_vectors.mean(dim=0)
        centered = embedding_vectors - mean.unsqueeze(0)
        eye = torch.eye(channels, device=device)
        chunk_size = 512
        cov_inv_chunks = []
        denom = max(batch_size - 1, 1)

        for start in range(0, height * width, chunk_size):
            end = min(start + chunk_size, height * width)
            chunk = centered[:, :, start:end]
            chunk_t = chunk.transpose(0, 1)
            cov_chunk = torch.einsum('cbk,dbk->cdk', chunk_t, chunk_t) / denom
            cov_chunk = cov_chunk + 0.01 * eye.unsqueeze(-1)
            cov_chunk = cov_chunk.permute(2, 0, 1)
            cov_inv_chunk = torch.linalg.inv(cov_chunk).permute(1, 2, 0).contiguous()
            cov_inv_chunks.append(cov_inv_chunk)

        self.embedding_mean = mean
        self.embedding_cov_inv = torch.cat(cov_inv_chunks, dim=-1)

    def fit(self, dataloader=None, support_data_root=None, target_cls=None, support_images=None):
        """Build support set statistics after training.

        Called by MemoryBankHook.
        Uses full 24 augmentations as in original RegAD:
        - 8 small rotations: [-π/4, -3π/16, -π/8, -π/16, π/16, π/8, 3π/16, π/4]
        - 8 translations: [(±0.2, ±0.2), (±0.1, ±0.1)]
        - 1 horizontal flip
        - 1 grayscale
        - 3 rotations (90°, 180°, 270°)

        For cross-category training (RegAD's original strategy):
        - Training dataloader contains images from OTHER categories
        - Support set must be built from TARGET category's training data
        - This is handled via support_data_root and target_cls parameters

        Args:
            dataloader: Training dataloader to extract support images from.
                       For cross-category training, this contains OTHER categories' data,
                       so we use support_data_root/target_cls instead.
            support_data_root: Data root for support images (for cross-category training).
            target_cls: Target category name (for cross-category training).
        """
        if support_images is not None:
            self.build_support_bank_from_images(support_images)
            return

        if support_data_root is not None and target_cls is not None:
            support_images = self._load_support_images(support_data_root, target_cls)
            if support_images is None or len(support_images) == 0:
                return
        elif dataloader is not None:
            support_images = []
            for data_batch in dataloader:
                if isinstance(data_batch, dict):
                    imgs = data_batch.get('inputs')
                elif isinstance(data_batch, (list, tuple)):
                    imgs = data_batch[0]
                else:
                    imgs = data_batch

                if isinstance(imgs, (list, tuple)):
                    imgs = torch.stack(imgs)
                if not isinstance(imgs, torch.Tensor):
                    continue
                support_images.append(imgs.detach().cpu())
            if not support_images:
                return
            support_images = torch.cat(support_images, dim=0)
        elif self._cached_support_images is not None:
            support_images = self._cached_support_images
        else:
            return

        self.build_support_bank_from_images(support_images)

    def build_memory_bank(self, dataloader=None):
        """Alias for MemoryBankHook compatibility."""
        # For cross-category training, we need to load support images from
        # the target category's training data, not from the training dataloader
        self.fit(
            dataloader=dataloader,
            support_data_root=self.data_root,
            target_cls=self.target_cls,
        )

    def _load_support_images(self, data_root, target_cls):
        """Load support images from the target category's training data.

        For cross-category training, the support set must come from the
        target category's training data, not from the training dataloader
        which contains images from OTHER categories.

        Args:
            data_root: Dataset root directory.
            target_cls: Target category name.

        Returns:
            Tensor of support images (N, C, H, W) on CPU.
        """
        import os
        from PIL import Image
        import torchvision.transforms as T

        # Build path to target category's training images
        train_dir = os.path.join(data_root, target_cls, 'train', 'good')
        if not os.path.isdir(train_dir):
            logger.warning(f"Support directory not found: {train_dir}")
            return None

        # Get list of training images
        img_files = sorted([f for f in os.listdir(train_dir)
                           if f.lower().endswith(('.png', '.jpg', '.bmp'))])

        if not img_files:
            logger.warning(f"No support images found in {train_dir}")
            return None

        logger.info(f"Loading support images from {target_cls}/train/good: {len(img_files)} images")

        # Build transform (same as dataset pipeline)
        transform = T.Compose([
            T.Resize(self.img_size),
            T.ToTensor(),
        ])

        # Load images
        images = []
        for img_file in img_files:
            img_path = os.path.join(train_dir, img_file)
            try:
                with Image.open(img_path) as _img:
                    img = _img.convert('RGB')
                img = transform(img)
                images.append(img)
            except Exception as e:
                logger.warning(f"Failed to load {img_path}: {e}")
                continue

        if not images:
            return None

        # Stack into tensor (N, C, H, W)
        support_images = torch.stack(images, dim=0)
        logger.info(f"Loaded {support_images.size(0)} support images, shape: {support_images.shape}")

        return support_images

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self
