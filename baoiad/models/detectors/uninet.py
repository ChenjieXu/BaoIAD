"""UniNet: Unified Anomaly Detection via Teacher-Student with Multi-Teacher Feature Aggregation.

Reimplementation aligned with anomalib reference:
- Dual teacher (source frozen + target learnable) using WRN-50-2 feature extraction
- Attention bottleneck with dual-branch (3x3 + 7x7 → merged to 3x3) processing
- Student decoder (de_resnet style with DeBottleneck and ConvTranspose2d upsampling)
- Domain-Related Feature Selection (DFS)
- Weighted decision mechanism for anomaly scoring
- Contrastive + cosine + margin loss for distillation
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Callable
from baoiad.models.predict_utils import build_predict_results
from baoiad.models.base_ad_model import BaseADModel

# backbone built via MODELS.build(dict(type='FeatureExtractor', ...))
from baoiad.registry import MODELS


def _normalize_teacher_backbone_cfg(backbone: str | dict) -> dict:
    """Normalize UniNet teacher backbone config to a FeatureExtractor spec."""
    if isinstance(backbone, dict):
        cfg = copy.deepcopy(backbone)
        backbone_type = cfg.get('type', 'FeatureExtractor')
        if backbone_type == 'RawBackbone':
            cfg = dict(
                type='FeatureExtractor',
                backbone_name=cfg.get('backbone_name', 'wide_resnet50_2'),
                pretrained=cfg.get('pretrained', True),
                out_indices=cfg.get('out_indices', (1, 2, 3)),
                frozen=cfg.get('frozen', False),
            )
        else:
            cfg.setdefault('type', 'FeatureExtractor')
            cfg.setdefault('pretrained', True)
            cfg.setdefault('out_indices', (1, 2, 3))
            cfg.setdefault('frozen', False)
        return cfg

    return dict(
        type='FeatureExtractor',
        backbone_name=backbone,
        pretrained=True,
        out_indices=(1, 2, 3),
        frozen=False,
    )


# ─── Loss ───────────────────────────────────────────────────────────────────

class UniNetLoss(nn.Module):
    """Contrastive + cosine + margin loss for teacher-student distillation."""

    def __init__(self, lambda_weight: float = 0.7, temperature: float = 0.1):
        super().__init__()
        self.lambda_weight = lambda_weight
        self.temperature = temperature

    def forward(self, student_features, teacher_features, margin=1, mask=None, stop_gradient=False):
        loss = 0.0
        for idx in range(len(student_features)):
            sf = student_features[idx]
            tf = teacher_features[idx].detach() if stop_gradient else teacher_features[idx]

            n, c, h, w = sf.shape
            sf = sf.view(n, c, -1).transpose(1, 2)
            tf = tf.view(n, c, -1).transpose(1, 2)

            sf_norm = F.normalize(sf, p=2, dim=2)
            tf_norm = F.normalize(tf, p=2, dim=2)

            cosine_loss = (1 - F.cosine_similarity(sf_norm, tf_norm, dim=2)).mean()

            similarity = torch.matmul(sf_norm, tf_norm.transpose(1, 2)) / self.temperature
            similarity = torch.exp(similarity)
            similarity = similarity / (similarity.sum(dim=2, keepdim=True) + 1e-8)
            diag_sum = torch.diagonal(similarity, dim1=1, dim2=2)

            margin_loss_a = 0.0
            if mask is None:
                contrastive_loss = -torch.log(diag_sum + 1e-8).mean()
                margin_loss_n = F.relu(margin - diag_sum).mean()
            else:
                if len(mask.shape) < 3:
                    normal_mask = mask == 0
                    abnormal_mask = mask == 1
                else:
                    mask_ = F.interpolate(mask, size=(h, w), mode='nearest').squeeze(1)
                    mask_flat = mask_.view(mask_.size(0), -1)
                    normal_mask = mask_flat == 0
                    abnormal_mask = mask_flat == 1

                contrastive_loss = torch.tensor(0.0, device=sf.device)
                margin_loss_n = torch.tensor(0.0, device=sf.device)
                if normal_mask.sum() > 0:
                    diag_n = diag_sum[normal_mask]
                    contrastive_loss = -torch.log(diag_n + 1e-8).mean()
                    margin_loss_n = F.relu(margin - diag_n).mean()
                if abnormal_mask.sum() > 0:
                    diag_a = diag_sum[abnormal_mask]
                    margin_loss_a = F.relu(diag_a - margin / 2).mean()

            loss += cosine_loss * self.lambda_weight + contrastive_loss * (1 - self.lambda_weight) + margin_loss_n + margin_loss_a
        return loss


# ─── Gaussian Blur ──────────────────────────────────────────────────────────

class GaussianBlur2d(nn.Module):
    def __init__(self, sigma: float = 4.0, kernel_size: tuple = (5, 5), channels: int = 1):
        super().__init__()
        ks = kernel_size[0]
        self.padding = ks // 2
        x = torch.arange(ks, dtype=torch.float32) - ks // 2
        gauss = torch.exp(-0.5 * x ** 2 / sigma ** 2)
        kernel_1d = gauss / gauss.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel_2d = kernel_2d.expand(channels, 1, -1, -1).contiguous()
        self.register_buffer('kernel', kernel_2d)
        self.channels = channels

    def forward(self, x):
        return F.conv2d(x, self.kernel, padding=self.padding, groups=self.channels)


# ─── Utility convolutions ──────────────────────────────────────────────────

def conv3x3(in_p, out_p, stride=1):
    return nn.Conv2d(in_p, out_p, 3, stride=stride, padding=1, bias=False)


def conv1x1(in_p, out_p, stride=1):
    return nn.Conv2d(in_p, out_p, 1, stride=stride, bias=False)


def deconv2x2(in_p, out_p, stride=1):
    return nn.ConvTranspose2d(in_p, out_p, kernel_size=2, stride=stride, bias=False)


def fuse_bn(conv, bn):
    """Fuse convolution and batch normalization layers."""
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


# ─── Attention Bottleneck ───────────────────────────────────────────────────

class AttentionBottleneck(nn.Module):
    """Dual-branch attention bottleneck (3x3 + 7x7, merged via fuse_bn at init)."""
    channel_expansion: int = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, norm_layer=None, attention=True, halve=1):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.halve = halve
        k = 7
        p = 3

        self.bn2 = norm_layer(width // halve)
        self.conv3 = conv1x1(width, planes * self.channel_expansion)
        self.bn3 = norm_layer(planes * self.channel_expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.bn4 = norm_layer(width // 2)
        self.bn5 = norm_layer(width // 2)
        self.bn6 = norm_layer(width // 2)
        self.bn7 = norm_layer(width)
        self.conv3x3 = nn.Conv2d(inplanes // 2, width // 2, 3, stride=stride, padding=1, bias=False)
        self.conv3x3_ = nn.Conv2d(width // 2, width // 2, 3, 1, 1, bias=False)
        self.conv7x7 = nn.Conv2d(inplanes // 2, width // 2, k, stride=stride, padding=p, bias=False)
        self.conv7x7_ = nn.Conv2d(width // 2, width // 2, k, 1, p, bias=False)

    def get_same_kernel_bias(self):
        k1, b1 = fuse_bn(self.conv3x3, self.bn2)
        k2, b2 = fuse_bn(self.conv3x3_, self.bn6)
        return k1, b1, k2, b2

    def merge_kernel(self):
        """Merge 3x3 conv+BN weights into 7x7 branch (reparameterization init)."""
        k1, b1, k2, b2 = self.get_same_kernel_bias()
        self.conv7x7 = nn.Conv2d(
            self.conv3x3.in_channels, self.conv3x3.out_channels,
            self.conv3x3.kernel_size, self.conv3x3.stride,
            self.conv3x3.padding, self.conv3x3.dilation, self.conv3x3.groups,
        )
        self.conv7x7_ = nn.Conv2d(
            self.conv3x3_.in_channels, self.conv3x3_.out_channels,
            self.conv3x3_.kernel_size, self.conv3x3_.stride,
            self.conv3x3_.padding, self.conv3x3_.dilation, self.conv3x3_.groups,
        )
        self.conv7x7.weight.data = k1
        self.conv7x7.bias = nn.Parameter(b1)
        self.conv7x7_.weight.data = k2
        self.conv7x7_.bias = nn.Parameter(b2)

    @staticmethod
    def _process_branch(x, conv1, bn1, conv2, bn2, relu):
        out = relu(bn1(conv1(x)))
        return bn2(conv2(out))

    def forward(self, x):
        identity = x

        if self.halve == 1:
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
        else:
            num_ch = x.shape[1]
            x1, x2 = torch.split(x, [num_ch // 2, num_ch // 2], dim=1)

            out1 = self._process_branch(x1, self.conv3x3, self.bn2, self.conv3x3_, self.bn5, self.relu)
            out2 = self._process_branch(x2, self.conv7x7, self.bn4, self.conv7x7_, self.bn6, self.relu)
            out = torch.cat([out1, out2], dim=1)
            out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class BottleneckLayer(nn.Module):
    """Bottleneck layer aggregating multi-scale teacher features."""

    def __init__(self, block=AttentionBottleneck, layers=3, halve=2):
        super().__init__()
        norm_layer = nn.BatchNorm2d
        self.inplanes = 256 * block.channel_expansion
        self.halve = halve

        self.bn_layer = nn.Sequential(self._make_layer(block, 512, layers, stride=2, norm_layer=norm_layer, halve=halve))
        self.conv1 = conv3x3(64 * block.channel_expansion, 128 * block.channel_expansion, 2)
        self.bn1 = norm_layer(128 * block.channel_expansion)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(128 * block.channel_expansion, 256 * block.channel_expansion, 2)
        self.bn2 = norm_layer(256 * block.channel_expansion)
        self.conv3 = conv3x3(128 * block.channel_expansion, 256 * block.channel_expansion, 2)
        self.bn3 = norm_layer(256 * block.channel_expansion)

        self.conv4 = conv1x1(1024 * block.channel_expansion, 512 * block.channel_expansion, 1)
        self.bn4_ = norm_layer(512 * block.channel_expansion)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # merge_kernel after weight init (matches anomalib)
        if self.halve == 2:
            for m in self.modules():
                if hasattr(m, 'merge_kernel'):
                    m.merge_kernel()

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None, halve=2):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.channel_expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes * 3, planes * block.channel_expansion, stride),
                norm_layer(planes * block.channel_expansion),
            )
        layers = [block(self.inplanes * 3, planes, stride, downsample, halve=halve)]
        self.inplanes = planes * block.channel_expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, halve=halve))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x is list of 3 tensors: [layer1, layer2, layer3] (concatenated source+target)
        l1 = self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x[0]))))))
        l2 = self.relu(self.bn3(self.conv3(x[1])))
        feature = torch.cat([l1, l2, x[2]], 1)
        return self.bn_layer(feature).contiguous()


# ─── De-Bottleneck for Student Decoder (de_resnet style) ───────────────────

class DeBottleneck(nn.Module):
    """Bottleneck block with ConvTranspose2d for upsampling (reversed ResNet)."""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, upsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        if stride == 2:
            self.conv2 = deconv2x2(width, width, stride)
        else:
            self.conv2 = nn.Conv2d(width, width, 3, stride=stride, padding=1, bias=False)
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
        return self.relu(out + identity)


class StudentDecoder(nn.Module):
    """Reversed Wide ResNet50-2 decoder using DeBottleneck blocks (de_resnet style).

    Takes 2048-channel bottleneck output and decodes to multi-scale features:
    - layer1: 2048 → 1024 (stride=2 upsample)
    - layer2: 1024 → 512 (stride=2 upsample)
    - layer3: 512 → 256 (stride=2 upsample)

    Returns [f1(256ch), f2(512ch), f3(1024ch)] matching teacher layer1/2/3.
    """

    def __init__(self, block=DeBottleneck, layers=(3, 4, 6), width_per_group=128,
                 norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.groups = 1
        self.base_width = width_per_group  # 128 for wide_resnet50_2
        self.inplanes = 512 * block.expansion  # 2048
        self.dilation = 1

        # layer1: 2048 -> 1024 (256*4), stride=2 upsample
        self.layer1 = self._make_layer(block, 256, layers[0], stride=2)
        # layer2: 1024 -> 512 (128*4), stride=2 upsample
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # layer3: 512 -> 256 (64*4), stride=2 upsample
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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


# ─── Domain Related Feature Selection ───────────────────────────────────────

class DomainRelatedFeatureSelection(nn.Module):
    def __init__(self, num_channels: int = 256, learnable: bool = True):
        super().__init__()
        self.num_channels = num_channels
        self.learnable = learnable
        self.theta1 = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.theta2 = nn.Parameter(torch.zeros(1, num_channels * 2, 1, 1))
        self.theta3 = nn.Parameter(torch.zeros(1, num_channels * 4, 1, 1))

    def _get_theta(self, idx):
        return [self.theta1, self.theta2, self.theta3][idx - 1]

    def forward(self, source_features, target_features, maximize=True):
        features = []
        for idx, (sf, tf) in enumerate(zip(source_features, target_features)):
            theta = 1
            if self.learnable:
                if idx < 3:
                    theta = torch.clamp(torch.sigmoid(self._get_theta(idx + 1)) * 1.0 + 0.5, max=1)
                else:
                    theta = torch.clamp(torch.sigmoid(self._get_theta(idx - 2)) * 1.0 + 0.5, max=1)

            b, c, h, w = sf.shape
            prior_flat = tf.view(b, c, -1)
            if maximize:
                prior_flat = prior_flat - prior_flat.max(dim=-1, keepdim=True)[0]
            weights = F.softmax(prior_flat, dim=-1).view(b, c, h, w)
            global_inf = tf.mean(dim=(-2, -1), keepdim=True)
            features.append(sf * weights * (theta + global_inf))
        return features


# ─── Teachers ───────────────────────────────────────────────────────────────

class Teachers(nn.Module):
    """Dual teacher: frozen source + learnable target."""

    def __init__(self, backbone: str | dict = 'wide_resnet50_2'):
        super().__init__()
        self.backbone_cfg = _normalize_teacher_backbone_cfg(backbone)
        self.source_teacher = self._get_teacher(self.backbone_cfg)
        self.target_teacher = self._get_teacher(self.backbone_cfg)
        # Freeze source
        self.source_teacher.eval()
        for p in self.source_teacher.parameters():
            p.requires_grad = False

    @staticmethod
    def _get_teacher(backbone_cfg):
        return MODELS.build(backbone_cfg)

    def forward(self, images):
        with torch.no_grad():
            source_features = self.source_teacher(images)
        target_features = self.target_teacher(images)

        bottleneck_inputs = [
            torch.cat([a, b], dim=0)
            for a, b in zip(target_features, source_features)
        ]
        return source_features + target_features, bottleneck_inputs


# ─── Weighted Decision Mechanism ────────────────────────────────────────────

def weighted_decision_mechanism(batch_size, output_list, alpha=0.01, beta=3e-5, output_size=(256, 256)):
    """Compute anomaly map and score from cosine distance outputs."""
    device = output_list[0].device
    gaussian_blur = GaussianBlur2d(sigma=4.0, kernel_size=(5, 5), channels=1).to(device)

    total_weights = torch.zeros(batch_size, device=device)
    for i in range(batch_size):
        max_values = torch.tensor([torch.max(o[i]) for o in output_list], device=device)
        probs = F.softmax(max_values, dim=0)
        mask = probs > probs.mean()
        if mask.any():
            weight = max(max_values[mask].mean().item() * alpha, beta)
        else:
            weight = beta
        total_weights[i] = weight

    processed = torch.zeros(batch_size, *output_size, device=device)
    for o in output_list:
        resized = F.interpolate(o.unsqueeze(1), output_size, mode='bilinear', align_corners=True).squeeze(1)
        processed += resized

    scores = torch.zeros(batch_size, device=device)
    for i in range(batch_size):
        top_k = max(int(output_size[0] * output_size[1] * total_weights[i]), 1)
        smoothed = gaussian_blur(processed[i].unsqueeze(0).unsqueeze(0)).squeeze()
        scores[i] = smoothed.view(-1).topk(top_k).values[0].detach()

    return scores.unsqueeze(1), processed.detach()


# ─── UniNet Detector ────────────────────────────────────────────────────────

@MODELS.register_module(force=True)
class UniNetDetector(BaseADModel):
    """UniNet anomaly detector with multi-teacher distillation.

    Args:
        teacher_backbone (str): Teacher backbone. Default 'wide_resnet50_2'.
        lambda_weight (float): Loss balance parameter. Default 0.7.
        temperature (float): Contrastive loss temperature. Default 0.1.
    """

    def __init__(
        self,
        teacher_backbone: str | dict = 'wide_resnet50_2',
        lambda_weight: float = 0.7,
        temperature: float = 0.1,
        data_preprocessor=None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        from baoiad.models.backbones.raw_backbone import _CHANNEL_DIMS
        teacher_backbone_cfg = _normalize_teacher_backbone_cfg(teacher_backbone)
        teacher_backbone_name = teacher_backbone_cfg.get('backbone_name', 'wide_resnet50_2')
        ch = _CHANNEL_DIMS[teacher_backbone_name]
        self.teacher_backbone_cfg = teacher_backbone_cfg
        self.teacher_backbone_name = teacher_backbone_name
        self.teachers = Teachers(teacher_backbone_cfg)
        self.bottleneck = BottleneckLayer(block=AttentionBottleneck, layers=3, halve=2)
        # de_resnet style student decoder (wide_resnet50_2 reversed)
        self.student = StudentDecoder(DeBottleneck, layers=(3, 4, 6), width_per_group=128)
        self.dfs = DomainRelatedFeatureSelection(num_channels=ch[0], learnable=True)
        self.loss_fn = UniNetLoss(lambda_weight=lambda_weight, temperature=temperature)
        self.bce_loss = MODELS.build(dict(type='BCEWithLogitsLoss'))

        # Classification head (from student_features[0] which has ch[0] channels)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(ch[0], 1)

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        source_target_features, bottleneck_inputs = self.teachers(inputs)
        bottleneck_output = self.bottleneck(bottleneck_inputs)
        student_features_raw = self.student(bottleneck_output)

        # Classification prediction from first student feature (256 channels)
        pred = self.avgpool(student_features_raw[0])
        pred = torch.flatten(pred, 1)
        pred = self.fc(pred).squeeze(-1)
        predictions = pred.chunk(dim=0, chunks=2) if pred.shape[0] > 1 else (pred, pred)

        # Split student features into source/target halves
        # student_features_raw[i] has batch dim = 2*B (source concat target from bottleneck)
        # source_target_features: [src_l3, src_l2, src_l1, tgt_l3, tgt_l2, tgt_l1]
        student_features = []
        for sf in student_features_raw:
            chunks = sf.chunk(dim=0, chunks=2) if sf.shape[0] > 1 else (sf, sf)
            student_features.append(chunks[0])
        for sf in student_features_raw:
            chunks = sf.chunk(dim=0, chunks=2) if sf.shape[0] > 1 else (sf, sf)
            student_features.append(chunks[1])

        if mode == 'loss':
            B = inputs.shape[0]
            device = inputs.device
            if data_samples is not None:
                labels = torch.tensor([
                    getattr(ds, 'gt_label', 0) for ds in data_samples
                ], dtype=torch.float32, device=device)
                masks = torch.stack([
                    getattr(ds, 'gt_mask', torch.zeros(inputs.shape[-2:], device=device)).squeeze()
                    for ds in data_samples
                ]).to(device)
            else:
                labels = torch.zeros(B, dtype=torch.float32, device=device)
                masks = torch.zeros(B, *inputs.shape[-2:], device=device)

            # Feature selection
            selected = self.dfs(student_features, source_target_features, maximize=True)

            mask_ = masks.float().unsqueeze(1)  # Bx1xHxW
            loss = self.loss_fn(selected, source_target_features, mask=mask_)
            loss += self.bce_loss(predictions[0], labels) + self.bce_loss(predictions[1], labels)
            return {'loss': loss}

        elif mode == 'predict':
            output_list = []
            for tf, sf in zip(source_target_features, student_features):
                if sf.shape[-2:] != tf.shape[-2:]:
                    sf = F.interpolate(sf, size=tf.shape[-2:], mode='bilinear', align_corners=False)
                output = 1 - F.cosine_similarity(tf, sf)
                output_list.append(output)

            anomaly_score, anomaly_map = weighted_decision_mechanism(
                batch_size=inputs.shape[0],
                output_list=output_list,
                alpha=0.01,
                beta=3e-5,
                output_size=inputs.shape[-2:],
            )

            B = inputs.shape[0]
            return build_predict_results(data_samples, anomaly_score, anomaly_map)

        else:  # tensor
            return student_features

    def train(self, mode=True):
        super().train(mode)
        self.teachers.source_teacher.eval()
        return self
