"""RD++ (Reverse Distillation++) anomaly detector (CVPR 2023).

This module keeps the historical in-detector noise path for legacy configs and
adds a strict path aligned to the original official implementation:
- dataset-side simplex noise generation
- split projection/distillation optimizers
- official anomaly-map interpolation + Gaussian smoothing
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.optim import OptimWrapperDict
from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS
from baoiad.models.base_ad_model import KnowledgeDistillationADModel


# ========== Utility convolutions ==========
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def deconv2x2(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=stride,
                              groups=groups, bias=False, dilation=dilation)


# ========== Bottleneck for OCBE ==========
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


# ========== DeBottleneck for Student Decoder ==========
class DeBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, upsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        if stride == 2:
            self.conv2 = deconv2x2(width, width, stride, groups, dilation)
        else:
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.upsample is not None:
            identity = self.upsample(x)
        out += identity
        return self.relu(out)


# ========== Student Decoder (reversed WRN50-2) ==========
class StudentDecoder(nn.Module):
    def __init__(self, block=DeBottleneck, layers=(3, 4, 6), width_per_group=128,
                 zero_init_residual=False, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.groups = 1
        self.base_width = width_per_group
        self.inplanes = 512 * block.expansion  # 2048
        self.dilation = 1

        self.layer1 = self._make_layer(block, 256, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, DeBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        upsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                deconv2x2(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, upsample, self.groups,
                        self.base_width, 1, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        f3 = self.layer1(x)
        f2 = self.layer2(f3)
        f1 = self.layer3(f2)
        return [f1, f2, f3]


# ========== OCBE (MFF_OCE) ==========
class OCBE(nn.Module):
    def __init__(self, block=Bottleneck, layers=3, width_per_group=64, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.base_width = width_per_group

        self.conv1 = conv3x3(64 * block.expansion, 128 * block.expansion, 2)
        self.bn1 = norm_layer(128 * block.expansion)
        self.conv2 = conv3x3(128 * block.expansion, 256 * block.expansion, 2)
        self.bn2 = norm_layer(256 * block.expansion)
        self.conv3 = conv3x3(128 * block.expansion, 256 * block.expansion, 2)
        self.bn3 = norm_layer(256 * block.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.inplanes = 256 * block.expansion
        self.dilation = 1
        self.bn_layer = self._make_layer(block, 512, layers, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes * 3, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = [block(self.inplanes * 3, planes, stride, downsample,
                        base_width=self.base_width, dilation=previous_dilation,
                        norm_layer=norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, base_width=self.base_width,
                                dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, feats):
        f1, f2, f3 = feats
        l1 = self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(f1))))))
        l2 = self.relu(self.bn3(self.conv3(f2)))
        feature = torch.cat([l1, l2, f3], dim=1)
        return self.bn_layer(feature).contiguous()


# ========== Projection Layers ==========
class ProjLayer(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_c, in_c // 2, 3, 1, 1), nn.InstanceNorm2d(in_c // 2), nn.LeakyReLU(),
            nn.Conv2d(in_c // 2, in_c // 4, 3, 1, 1), nn.InstanceNorm2d(in_c // 4), nn.LeakyReLU(),
            nn.Conv2d(in_c // 4, in_c // 2, 3, 1, 1), nn.InstanceNorm2d(in_c // 2), nn.LeakyReLU(),
            nn.Conv2d(in_c // 2, out_c, 3, 1, 1), nn.InstanceNorm2d(out_c), nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.proj(x)


class MultiProjectionLayer(nn.Module):
    def __init__(self, base=64):
        super().__init__()
        self.proj_a = ProjLayer(base * 4, base * 4)    # 256 -> 256
        self.proj_b = ProjLayer(base * 8, base * 8)    # 512 -> 512
        self.proj_c = ProjLayer(base * 16, base * 16)  # 1024 -> 1024

    def forward(self, features, features_noise=False):
        if features_noise is not False:
            return (
                [self.proj_a(features_noise[0]), self.proj_b(features_noise[1]), self.proj_c(features_noise[2])],
                [self.proj_a(features[0]), self.proj_b(features[1]), self.proj_c(features[2])],
            )
        else:
            return [self.proj_a(features[0]), self.proj_b(features[1]), self.proj_c(features[2])]


# ========== Loss Functions ==========
class CosineReconstruct(nn.Module):
    def forward(self, x, y):
        return torch.mean(1 - F.cosine_similarity(x, y, dim=1))


class Revisit_RDLoss(nn.Module):
    """Three-part loss: SSOT + Cosine Reconstruct + Cosine Embedding Contrast.

    Weights: SSOT=1.0, Reconstruct=0.01, Contrast=0.1
    Total normalized by 1.11
    """

    def __init__(self, allow_mse_fallback: bool = True):
        super().__init__()
        self.allow_mse_fallback = bool(allow_mse_fallback)
        try:
            import geomloss
            self.sinkhorn = geomloss.SamplesLoss(
                loss='sinkhorn', p=2, blur=0.05, reach=None, diameter=10000000,
                scaling=0.95, truncate=10, cost=None, kernel=None, cluster_scale=None,
                debias=True, potentials=False, verbose=False, backend='auto',
            )
        except Exception:
            import warnings
            if self.allow_mse_fallback:
                message = (
                    'geomloss is not installed. RD++ SSOT loss will fall back to MSE, '
                    'which significantly degrades performance. Install with: pip install geomloss'
                )
            else:
                message = (
                    'geomloss is not installed. Strict RD++ requires geomloss for the official '
                    'Sinkhorn SSOT loss.'
                )
            warnings.warn(
                message,
                UserWarning, stacklevel=2,
            )
            self.sinkhorn = None
        self.reconstruct = CosineReconstruct()
        self.contrast = nn.CosineEmbeddingLoss(margin=0.5)

    def forward(self, noised_feature, projected_noised_feature, projected_normal_feature):
        """
        Args:
            noised_feature: teacher features from noised image [f1, f2, f3]
            projected_noised_feature: projection(noised_feature)
            projected_normal_feature: projection(normal_feature)
        """
        bs = projected_normal_feature[0].shape[0]
        device = projected_normal_feature[0].device
        target = -torch.ones(bs, device=device)

        normal_proj1, normal_proj2, normal_proj3 = projected_normal_feature
        abnormal_proj1, abnormal_proj2, abnormal_proj3 = projected_noised_feature
        noised_feature1, noised_feature2, noised_feature3 = noised_feature

        # Shuffle for pair-wise SSOT
        shuffle_idx = torch.randperm(bs)
        shuffle_1 = normal_proj1[shuffle_idx]
        shuffle_2 = normal_proj2[shuffle_idx]
        shuffle_3 = normal_proj3[shuffle_idx]

        # SSOT loss
        if self.sinkhorn is not None:
            loss_ssot = (
                self.sinkhorn(
                    torch.softmax(normal_proj1.view(bs, -1), -1),
                    torch.softmax(shuffle_1.view(bs, -1), -1)) +
                self.sinkhorn(
                    torch.softmax(normal_proj2.view(bs, -1), -1),
                    torch.softmax(shuffle_2.view(bs, -1), -1)) +
                self.sinkhorn(
                    torch.softmax(normal_proj3.view(bs, -1), -1),
                    torch.softmax(shuffle_3.view(bs, -1), -1))
            )
        else:
            if not self.allow_mse_fallback:
                raise RuntimeError(
                    'Strict RD++ requires geomloss for the official Sinkhorn SSOT loss.'
                )
            # Fallback: MSE between shuffled pairs (significantly weaker than SSOT)
            loss_ssot = (
                F.mse_loss(normal_proj1, shuffle_1) +
                F.mse_loss(normal_proj2, shuffle_2) +
                F.mse_loss(normal_proj3, shuffle_3)
            )

        # Cosine Reconstruct loss
        loss_reconstruct = (
            self.reconstruct(abnormal_proj1, normal_proj1) +
            self.reconstruct(abnormal_proj2, normal_proj2) +
            self.reconstruct(abnormal_proj3, normal_proj3)
        )

        # Cosine Embedding Contrast loss
        loss_contrast = (
            self.contrast(noised_feature1.view(bs, -1), normal_proj1.view(bs, -1), target=target) +
            self.contrast(noised_feature2.view(bs, -1), normal_proj2.view(bs, -1), target=target) +
            self.contrast(noised_feature3.view(bs, -1), normal_proj3.view(bs, -1), target=target)
        )

        return (loss_ssot + 0.01 * loss_reconstruct + 0.1 * loss_contrast) / 1.11


# ========== Simplex Noise for Feature Jittering ==========
def _generate_simplex_noise_patch(size, h_noise, w_noise):
    """Generate a random noise patch (simplified version of opensimplex).

    The original RD++ uses opensimplex 3D octave noise. Here we use
    a multi-octave Perlin-like approximation for portability.
    """
    noise = np.zeros((3, h_noise, w_noise), dtype=np.float32)
    for octave in range(6):
        freq = 2 ** octave
        amp = 0.6 ** octave
        # Random noise at this frequency, upsampled
        oh, ow = max(1, h_noise // freq), max(1, w_noise // freq)
        octave_noise = np.random.randn(3, oh, ow).astype(np.float32)
        if oh != h_noise or ow != w_noise:
            # Simple bilinear upscale
            import torch.nn.functional as _F
            t = torch.from_numpy(octave_noise).unsqueeze(0)
            t = _F.interpolate(t, size=(h_noise, w_noise), mode='bilinear', align_corners=False)
            octave_noise = t.squeeze(0).numpy()
        noise += amp * octave_noise
    return noise


def generate_noised_image(img_tensor):
    """Add simplex-like noise patch to image tensor for feature jittering.

    Args:
        img_tensor: (B, 3, H, W) normalized image tensor
    Returns:
        noised image tensor of same shape
    """
    B, C, H, W = img_tensor.shape
    img_noise = img_tensor.clone()
    for b in range(B):
        h_noise = np.random.randint(10, max(11, H // 8))
        w_noise = np.random.randint(10, max(11, W // 8))
        start_h = np.random.randint(0, max(1, H - h_noise))
        start_w = np.random.randint(0, max(1, W - w_noise))
        noise_patch = _generate_simplex_noise_patch(None, h_noise, w_noise)
        noise_patch = torch.from_numpy(noise_patch * 0.2).to(img_tensor.device, img_tensor.dtype)
        img_noise[b, :, start_h:start_h + h_noise, start_w:start_w + w_noise] += noise_patch
    return img_noise


def _gaussian_blur_bchw(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply scipy gaussian_filter per sample/channel, matching official RD++."""
    if sigma <= 0:
        return x
    from scipy.ndimage import gaussian_filter

    x_np = x.detach().float().cpu().numpy()
    for batch_index in range(x_np.shape[0]):
        for channel_index in range(x_np.shape[1]):
            x_np[batch_index, channel_index] = gaussian_filter(
                x_np[batch_index, channel_index], sigma=float(sigma))
    return torch.from_numpy(x_np).to(device=x.device, dtype=x.dtype)


def _official_rdpp_loss(feats_t, feats_s) -> torch.Tensor:
    """Match the official flattened cosine distillation loss."""
    loss = feats_t[0].new_tensor(0.0)
    for ft, fs in zip(feats_t, feats_s):
        ft_flat = ft.contiguous().view(ft.shape[0], -1)
        fs_flat = fs.contiguous().view(fs.shape[0], -1)
        loss = loss + (1 - F.cosine_similarity(ft_flat, fs_flat, dim=1)).mean()
    return loss


@MODELS.register_module(force=True)
class RDPPDetector(KnowledgeDistillationADModel):
    """RD++ anomaly detector (CVPR 2023).

    Improvements over RD:
    - MultiProjectionLayer with InstanceNorm + LeakyReLU
    - Feature jittering via noised image through teacher
    - Revisit_RDLoss: SSOT + Cosine Reconstruct + Cosine Embedding Contrast
    """

    def __init__(self, backbone: str = 'wide_resnet50_2',
                 recon_loss=dict(type='CosineDistanceLoss'),
                 smooth_sigma=4.0, smooth_kernel_size=21,
                 strict: bool = False,
                 proj_loss_weight: float = 0.2,
                 data_preprocessor=None, init_cfg=None, **kwargs):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.strict = bool(strict)
        self.proj_loss_weight = float(proj_loss_weight)
        self.smooth_sigma = smooth_sigma
        self.smooth_kernel_size = smooth_kernel_size
        # backbone built via MODELS.build(dict(type='RawBackbone', ...))
        if isinstance(backbone, str):
            wrn = MODELS.build(dict(type='RawBackbone', backbone_name=backbone))
        else:
            wrn = MODELS.build(backbone)
        self.teacher_conv1 = wrn.conv1
        self.teacher_bn1 = wrn.bn1
        self.teacher_relu = wrn.relu
        self.teacher_maxpool = wrn.maxpool
        self.teacher_layer1 = wrn.layer1
        self.teacher_layer2 = wrn.layer2
        self.teacher_layer3 = wrn.layer3

        self.ocbe = OCBE(Bottleneck, layers=3, width_per_group=64)
        self.proj_layer = MultiProjectionLayer(base=64)
        self.student = StudentDecoder(DeBottleneck, layers=(3, 4, 6), width_per_group=128)
        self.proj_loss = Revisit_RDLoss(allow_mse_fallback=not self.strict)
        self.recon_loss = MODELS.build(recon_loss) if recon_loss is not None else None

        # Freeze teacher
        for name, param in self.named_parameters():
            if name.startswith('teacher_'):
                param.requires_grad = False

    def extract_teacher_feats(self, x):
        with torch.no_grad():
            x = self.teacher_maxpool(self.teacher_relu(self.teacher_bn1(self.teacher_conv1(x))))
            f1 = self.teacher_layer1(x)
            f2 = self.teacher_layer2(f1)
            f3 = self.teacher_layer3(f2)
        return [f1, f2, f3]

    @staticmethod
    def _stack_strict_noise_inputs(data_samples, device: torch.device) -> torch.Tensor:
        if data_samples is None:
            raise ValueError('Strict RD++ training requires data_samples with img_noise metadata.')
        noises = []
        for sample in data_samples:
            if not hasattr(sample, 'img_noise'):
                raise ValueError('Strict RD++ training requires img_noise in data_samples metainfo.')
            noises.append(sample.img_noise.to(device))
        return torch.stack(noises)

    def _legacy_loss(self, inputs):
        feats_t = self.extract_teacher_feats(inputs)
        img_noise = generate_noised_image(inputs)
        feats_noise = self.extract_teacher_feats(img_noise)
        projected_noised, projected_normal = self.proj_layer(feats_t, features_noise=feats_noise)
        loss_proj = self.proj_loss(feats_noise, projected_noised, projected_normal)
        feats_s = self.student(self.ocbe(projected_normal))
        loss_distill = _official_rdpp_loss(feats_t, feats_s)
        return {
            'loss': loss_distill + self.proj_loss_weight * loss_proj,
            'loss_distill': loss_distill,
            'loss_proj': loss_proj,
        }

    def _strict_loss(self, inputs, data_samples):
        feats_t = self.extract_teacher_feats(inputs)
        img_noise = self._stack_strict_noise_inputs(data_samples, inputs.device)
        feats_noise = self.extract_teacher_feats(img_noise)
        projected_noised, projected_normal = self.proj_layer(feats_t, features_noise=feats_noise)
        loss_proj = self.proj_loss(feats_noise, projected_noised, projected_normal)
        feats_s = self.student(self.ocbe(projected_normal))
        loss_distill = _official_rdpp_loss(feats_t, feats_s)
        return {
            'loss': loss_distill + self.proj_loss_weight * loss_proj,
            'loss_distill': loss_distill,
            'loss_proj': loss_proj,
        }

    def _legacy_predict(self, inputs, data_samples):
        feats_t = self.extract_teacher_feats(inputs)
        projected = self.proj_layer(feats_t)
        feats_s = self.student(self.ocbe(projected))

        scores = []
        for ft, fs in zip(feats_t, feats_s):
            ft = F.normalize(ft, dim=1)
            fs = F.normalize(fs, dim=1)
            dist = 1 - (ft * fs).sum(dim=1)
            dist = F.interpolate(
                dist.unsqueeze(1),
                size=inputs.shape[-2:],
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)
            scores.append(dist)
        score_map = sum(scores)
        if self.smooth_sigma > 0:
            from torchvision.transforms.functional import gaussian_blur
            score_map = gaussian_blur(
                score_map.unsqueeze(1),
                kernel_size=[self.smooth_kernel_size] * 2,
                sigma=[self.smooth_sigma] * 2,
            ).squeeze(1)
        img_scores = score_map.view(score_map.shape[0], -1).max(dim=1).values
        return build_predict_results(data_samples, img_scores, score_map)

    def _strict_predict(self, inputs, data_samples):
        feats_t = self.extract_teacher_feats(inputs)
        projected = self.proj_layer(feats_t)
        feats_s = self.student(self.ocbe(projected))

        scores = []
        for ft, fs in zip(feats_t, feats_s):
            dist = 1 - F.cosine_similarity(ft, fs, dim=1)
            dist = F.interpolate(
                dist.unsqueeze(1),
                size=inputs.shape[-2:],
                mode='bilinear',
                align_corners=True,
            ).squeeze(1)
            scores.append(dist)
        score_map = sum(scores)
        if self.smooth_sigma > 0:
            score_map = _gaussian_blur_bchw(score_map.unsqueeze(1), self.smooth_sigma).squeeze(1)
        img_scores = score_map.view(score_map.shape[0], -1).max(dim=1).values
        return build_predict_results(data_samples, img_scores, score_map)

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        if mode == 'loss':
            return self._strict_loss(inputs, data_samples) if self.strict else self._legacy_loss(inputs)

        if mode == 'predict':
            return self._strict_predict(inputs, data_samples) if self.strict else self._legacy_predict(inputs, data_samples)

        feats_t = self.extract_teacher_feats(inputs)
        projected = self.proj_layer(feats_t)
        return self.student(self.ocbe(projected))

    def train_step(self, data, optim_wrapper):
        if not self.strict:
            return super().train_step(data, optim_wrapper)

        if (
            not isinstance(optim_wrapper, OptimWrapperDict)
            or 'projection' not in optim_wrapper
            or 'distillation' not in optim_wrapper
        ):
            raise TypeError(
                'Strict RD++ training requires an OptimWrapperDict with projection and distillation optimizers.'
            )

        data = self.data_preprocessor(data, True)
        inputs = data['inputs']
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)
        data_samples = data.get('data_samples', None)

        projection_optim = optim_wrapper['projection']
        distillation_optim = optim_wrapper['distillation']

        if projection_optim._inner_count % projection_optim._accumulative_counts == 0:
            projection_optim.zero_grad()
        if distillation_optim._inner_count % distillation_optim._accumulative_counts == 0:
            distillation_optim.zero_grad()

        losses = self(inputs, data_samples, mode='loss')
        scaled_loss = distillation_optim.scale_loss(losses['loss'])
        distillation_optim.backward(scaled_loss)
        projection_optim._inner_count += 1

        if projection_optim.should_update():
            projection_optim.step()
            projection_optim.zero_grad()
        if distillation_optim.should_update():
            distillation_optim.step()
            distillation_optim.zero_grad()

        detached = {}
        for key, value in losses.items():
            detached[key] = value.detach() if torch.is_tensor(value) else value
        return detached

    def train(self, mode=True):
        super().train(mode)
        self.teacher_conv1.eval()
        self.teacher_bn1.eval()
        self.teacher_layer1.eval()
        self.teacher_layer2.eval()
        self.teacher_layer3.eval()
        return self
