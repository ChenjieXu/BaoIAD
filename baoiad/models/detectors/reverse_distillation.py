"""Reverse Distillation anomaly detector (CVPR 2022).

Strict alignment targets the original RD4AD repository:
https://github.com/hq-deng/RD4AD
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from baoiad.models.base_ad_model import KnowledgeDistillationADModel
from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS


# ========== Utility convolutions ==========
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def deconv2x2(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=stride,
                              groups=groups, bias=False, dilation=dilation)


# ========== Bottleneck for OCBE (standard ResNet Bottleneck) ==========
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
        self.stride = stride

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
    """Bottleneck block with ConvTranspose2d for upsampling (reversed ResNet)."""
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
        self.stride = stride

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
    """Reversed Wide ResNet50-2 decoder using DeBottleneck blocks."""

    def __init__(self, block=DeBottleneck, layers=(3, 4, 6), width_per_group=128,
                 zero_init_residual=False, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.groups = 1
        self.base_width = width_per_group  # 128 for wide_resnet50_2
        # Start from 2048 (512 * expansion=4)
        self.inplanes = 512 * block.expansion  # 2048
        self.dilation = 1

        # layer1: 2048 -> 1024 (256*4), stride=2 upsample
        self.layer1 = self._make_layer(block, 256, layers[0], stride=2)
        # layer2: 1024 -> 512 (128*4), stride=2 upsample
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # layer3: 512 -> 256 (64*4), stride=2 upsample
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        # Weight initialization
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
        f3 = self.layer1(x)    # 2048 -> 1024
        f2 = self.layer2(f3)   # 1024 -> 512
        f1 = self.layer3(f2)   # 512 -> 256
        return [f1, f2, f3]


# ========== OCBE (MFF_OCE) ==========
class OCBE(nn.Module):
    """One-Class Bottleneck Embedding (Multi-scale Feature Fusion + OCE).

    Uses Bottleneck residual blocks for the reduce layer, and conv3x3 with
    stride=2 for spatial alignment of multi-scale features.
    """

    def __init__(self, block=Bottleneck, layers=3, width_per_group=64, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.base_width = width_per_group

        # Spatial alignment convolutions (conv3x3 with stride 2)
        # f1: 256 channels -> downsample 2x -> 512 -> downsample 2x -> 1024
        self.conv1 = conv3x3(64 * block.expansion, 128 * block.expansion, 2)  # 256->512
        self.bn1 = norm_layer(128 * block.expansion)
        self.conv2 = conv3x3(128 * block.expansion, 256 * block.expansion, 2)  # 512->1024
        self.bn2 = norm_layer(256 * block.expansion)
        # f2: 512 channels -> downsample 2x -> 1024
        self.conv3 = conv3x3(128 * block.expansion, 256 * block.expansion, 2)  # 512->1024
        self.bn3 = norm_layer(256 * block.expansion)
        self.relu = nn.ReLU(inplace=True)

        # Reduce: concat [1024, 1024, 1024] = 3072 -> 2048 using Bottleneck layers
        self.inplanes = 256 * block.expansion  # Will be multiplied by 3 in _make_layer
        self.dilation = 1
        self.bn_layer = self._make_layer(block, 512, layers, stride=2)

        # Weight initialization
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
        f1, f2, f3 = feats  # 256xHxW, 512xH/2xW/2, 1024xH/4xW/4
        l1 = self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(f1))))))
        l2 = self.relu(self.bn3(self.conv3(f2)))
        feature = torch.cat([l1, l2, f3], dim=1)  # 3072 channels
        output = self.bn_layer(feature)  # -> 2048
        return output.contiguous()


@MODELS.register_module()
class ReverseDistillation(KnowledgeDistillationADModel):
    """Reverse Distillation for anomaly detection (CVPR 2022).

    Teacher (frozen WRN50-2) extracts features at layer1/2/3.
    OCBE fuses multi-scale features into bottleneck embedding.
    Student decoder (reversed WRN50-2) reconstructs features.
    Anomaly score: cosine distance between teacher and student features.
    """

    def __init__(self, backbone='wide_resnet50_2', loss=dict(type='CosineDistanceLoss'),
                 smooth_sigma=4.0, smooth_kernel_size=None,
                 anomaly_map_mode='add', smoothing_backend='scipy',
                 data_preprocessor=None, init_cfg=None, **kwargs):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.smooth_sigma = smooth_sigma
        mode_aliases = {
            'a': 'add',
            'add': 'add',
            'sum': 'add',
            'mul': 'multiply',
            'multiply': 'multiply',
        }
        if anomaly_map_mode not in mode_aliases:
            raise ValueError(
                f'Unsupported anomaly_map_mode={anomaly_map_mode!r}. '
                "Use 'add' or 'multiply'.")
        self.anomaly_map_mode = mode_aliases[anomaly_map_mode]
        if smoothing_backend not in {'scipy', 'torchvision'}:
            raise ValueError(
                f'Unsupported smoothing_backend={smoothing_backend!r}. '
                "Use 'scipy' or 'torchvision'.")
        self.smoothing_backend = smoothing_backend
        # Auto-compute kernel size from sigma (matching anomalib)
        if smooth_kernel_size is None:
            self.smooth_kernel_size = 2 * int(4.0 * smooth_sigma + 0.5) + 1
        else:
            self.smooth_kernel_size = smooth_kernel_size
        # Backward compat: str -> config dict
        if isinstance(backbone, str):
            backbone = dict(type='FeatureExtractor', backbone_name=backbone,
                            out_indices=(1, 2, 3), frozen=True)
        self.teacher = MODELS.build(backbone)

        self.ocbe = OCBE(Bottleneck, layers=3, width_per_group=64)
        self.student = StudentDecoder(DeBottleneck, layers=(3, 4, 6), width_per_group=128)
        self.loss_fn = MODELS.build(loss)

    def extract_teacher_feats(self, x):
        with torch.no_grad():
            return self.teacher(x)

    @staticmethod
    def _flatten_cosine_loss(
        teacher_features: list[torch.Tensor],
        student_features: list[torch.Tensor],
    ) -> torch.Tensor:
        """Match the original RD4AD flattened cosine loss."""
        cos_loss = nn.CosineSimilarity()
        loss = 0
        for teacher_feature, student_feature in zip(teacher_features, student_features, strict=True):
            loss += torch.mean(
                1 - cos_loss(
                    teacher_feature.view(teacher_feature.shape[0], -1),
                    student_feature.view(student_feature.shape[0], -1),
                )
            )
        return loss

    def _compute_anomaly_map(
        self,
        teacher_features: list[torch.Tensor],
        student_features: list[torch.Tensor],
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        """Compute the multi-scale anomaly map following RD4AD test.py."""
        if self.anomaly_map_mode == 'multiply':
            anomaly_map = torch.ones(
                (teacher_features[0].shape[0], *image_size),
                device=teacher_features[0].device,
                dtype=teacher_features[0].dtype,
            )
        else:
            anomaly_map = torch.zeros(
                (teacher_features[0].shape[0], *image_size),
                device=teacher_features[0].device,
                dtype=teacher_features[0].dtype,
            )

        for teacher_feature, student_feature in zip(teacher_features, student_features, strict=True):
            distance_map = 1 - F.cosine_similarity(teacher_feature, student_feature)
            distance_map = F.interpolate(
                distance_map.unsqueeze(1),
                size=image_size,
                mode='bilinear',
                align_corners=True,
            ).squeeze(1)
            if self.anomaly_map_mode == 'multiply':
                anomaly_map *= distance_map
            else:
                anomaly_map += distance_map
        return anomaly_map

    def _smooth_anomaly_map(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian smoothing using the strict-reference backend."""
        if self.smooth_sigma <= 0:
            return anomaly_map

        if self.smoothing_backend == 'scipy':
            from scipy.ndimage import gaussian_filter

            smoothed = []
            anomaly_map_np = anomaly_map.detach().cpu().numpy()
            for sample in anomaly_map_np:
                smoothed.append(
                    gaussian_filter(
                        sample,
                        sigma=self.smooth_sigma,
                        truncate=4.0,
                    ))
            return torch.from_numpy(np.stack(smoothed, axis=0)).to(
                device=anomaly_map.device,
                dtype=anomaly_map.dtype,
            )

        from torchvision.transforms.functional import gaussian_blur

        anomaly_map_4d = anomaly_map.unsqueeze(1)
        anomaly_map_4d = gaussian_blur(
            anomaly_map_4d,
            kernel_size=[self.smooth_kernel_size] * 2,
            sigma=[self.smooth_sigma] * 2,
        )
        return anomaly_map_4d.squeeze(1)

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)
        feats_t = self.extract_teacher_feats(inputs)
        bottleneck = self.ocbe(feats_t)
        feats_s = self.student(bottleneck)

        if mode == 'loss':
            return {'loss': self._flatten_cosine_loss(feats_t, feats_s)}
        elif mode == 'predict':
            score_map = self._compute_anomaly_map(
                feats_t,
                feats_s,
                image_size=inputs.shape[-2:],
            )
            score_map = self._smooth_anomaly_map(score_map)
            img_scores = score_map.view(score_map.shape[0], -1).max(dim=1).values

            return build_predict_results(data_samples, img_scores, score_map)
        return feats_s

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self
