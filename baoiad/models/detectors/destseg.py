"""Official-style DeSTSeg detector."""

from __future__ import annotations

import logging
import math
import os
import urllib.request
from typing import Sequence

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.optim import OptimWrapperDict

from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS
from baoiad.models.base_ad_model import ReconstructionADModel

logger = logging.getLogger(__name__)
_LEGACY_RESNET18_URL = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
_LEGACY_RESNET18_PATH = os.path.join('pretrained', 'resnet18-5c106cde.pth')


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def make_layer(block, inplanes, planes, blocks, stride=1, norm_layer=None):
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            conv1x1(inplanes, planes * block.expansion, stride),
            norm_layer(planes * block.expansion),
        )

    layers = [block(inplanes, planes, stride, downsample, norm_layer=norm_layer)]
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(inplanes, planes, norm_layer=norm_layer))
    return nn.Sequential(*layers)


def l2_normalize(tensor, dim=1, eps=1e-12):
    denom = torch.sqrt(torch.sum(tensor**2, dim=dim, keepdim=True))
    return tensor / (denom + eps)


class BasicBlock(nn.Module):
    """Residual block copied from the official implementation."""

    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class ConvNormAct2d(nn.Module):
    """Small conv-norm-activation helper."""

    def __init__(self, in_channels, out_channels, kernel_size, padding='same', dilation=1):
        super().__init__()
        if padding == 'same':
            padding = 0 if kernel_size == 1 else dilation
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ASPP(nn.Module):
    """Official ASPP module used in the segmentation head."""

    def __init__(self, input_channels, output_channels, atrous_rates):
        super().__init__()
        modules = [
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                ConvNormAct2d(input_channels, output_channels, kernel_size=1),
            )
        ]
        for rate in atrous_rates:
            modules.append(
                ConvNormAct2d(
                    input_channels,
                    output_channels,
                    kernel_size=1 if rate == 1 else 3,
                    padding=0 if rate == 1 else rate,
                    dilation=rate,
                )
            )
        self.extractors = nn.ModuleList(modules)
        self.fusion = ConvNormAct2d((1 + len(atrous_rates)) * output_channels, output_channels, kernel_size=3)

    def forward(self, x):
        feats = [extractor(x) for extractor in self.extractors]
        feats[0] = F.interpolate(feats[0], size=x.shape[2:], mode='bilinear', align_corners=False)
        return self.fusion(torch.cat(feats, dim=1))


class TeacherNet(nn.Module):
    """Frozen ResNet-18 teacher network."""

    def __init__(self, pretrained=True, checkpoint_path: str | None = None):
        super().__init__()
        self.encoder = timm.create_model(
            'resnet18',
            pretrained=pretrained and not checkpoint_path,
            features_only=True,
            out_indices=[1, 2, 3],
        )
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            missing, unexpected = self.encoder.load_state_dict(state_dict, strict=False)
            if missing:
                logger.warning('DeSTSeg teacher checkpoint missing keys: %s', missing[:10])
            if unexpected:
                logger.info('DeSTSeg teacher checkpoint ignored %d unexpected keys.', len(unexpected))
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        self.eval()
        x1, x2, x3 = self.encoder(x)
        return x1, x2, x3


class StudentNet(nn.Module):
    """Student encoder-decoder from the official release."""

    def __init__(self, ed=True):
        super().__init__()
        self.ed = ed
        if self.ed:
            self.decoder_layer4 = make_layer(BasicBlock, 512, 512, 2)
            self.decoder_layer3 = make_layer(BasicBlock, 512, 256, 2)
            self.decoder_layer2 = make_layer(BasicBlock, 256, 128, 2)
            self.decoder_layer1 = make_layer(BasicBlock, 128, 64, 2)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        self.encoder = timm.create_model(
            'resnet18',
            pretrained=False,
            features_only=True,
            out_indices=[1, 2, 3, 4],
        )

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        if not self.ed:
            return x1, x2, x3
        b4 = self.decoder_layer4(x4)
        b3 = F.interpolate(b4, size=x3.shape[2:], mode='bilinear', align_corners=False)
        b3 = self.decoder_layer3(b3)
        b2 = F.interpolate(b3, size=x2.shape[2:], mode='bilinear', align_corners=False)
        b2 = self.decoder_layer2(b2)
        b1 = F.interpolate(b2, size=x1.shape[2:], mode='bilinear', align_corners=False)
        b1 = self.decoder_layer1(b1)
        return b1, b2, b3


class SegmentationNet(nn.Module):
    """Official ASPP-based segmentation branch."""

    def __init__(self, inplanes=448):
        super().__init__()
        self.res = make_layer(BasicBlock, inplanes, 256, 2)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        self.head = nn.Sequential(
            ASPP(256, 256, [6, 12, 18]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1),
        )

    def forward(self, x):
        x = self.res(x)
        x = self.head(x)
        return torch.sigmoid(x)


def _cosine_similarity_loss(output_de_st_list):
    loss = 0.0
    for instance in output_de_st_list:
        _, _, h, w = instance.shape
        loss = loss + torch.sum(instance) / (h * w)
    return loss


def _focal_loss(inputs, targets, alpha=-1, gamma=4, reduction='mean'):
    ce_loss = F.binary_cross_entropy(inputs.float(), targets.float(), reduction='none')
    p_t = inputs * targets + (1 - inputs) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    return loss


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for param in module.parameters():
        param.requires_grad = enabled


def _stack_sample_tensors(
    data_samples: Sequence,
    key: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensors = []
    for sample in data_samples:
        value = getattr(sample, key, None)
        if value is None:
            raise RuntimeError(
                f'DeSTSeg loss expects data_samples with `{key}` from PackDeSTSegInputs.'
            )
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        tensors.append(value.to(device=device, dtype=dtype))
    return torch.stack(tensors)


@MODELS.register_module()
class DeSTSegDetector(ReconstructionADModel):
    """Official-style DeSTSeg detector with step-based two-stage training."""

    def __init__(
        self,
        backbone='resnet18',
        teacher_pretrained=False,
        teacher_checkpoint_path: str | None = None,
        phase_ratio=0.2,
        de_st_steps=None,
        top_k_score=100,
        dtd_path='auto',
        gamma=4.0,
        seg_mid_channels=64,
        data_preprocessor=None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        del dtd_path, seg_mid_channels, kwargs

        backbone_name = backbone if isinstance(backbone, str) else backbone.get('backbone_name') or backbone.get('model_name')
        if backbone_name and backbone_name != 'resnet18':
            logger.warning(
                'DeSTSeg official implementation uses resnet18. Overriding requested backbone %s to resnet18.',
                backbone_name,
            )

        self.phase_ratio = float(phase_ratio)
        self.de_st_steps = None if de_st_steps is None else int(de_st_steps)
        self.top_k_score = int(top_k_score)
        self.gamma = float(gamma)
        self._phase = 'student'

        if teacher_checkpoint_path == 'auto':
            teacher_checkpoint_path = self._ensure_legacy_teacher_checkpoint()

        self.teacher_net = TeacherNet(
            pretrained=teacher_pretrained,
            checkpoint_path=teacher_checkpoint_path,
        )
        self.student_net = StudentNet(ed=True)
        self.segmentation_net = SegmentationNet(inplanes=448)
        self._set_phase_trainability()

    @staticmethod
    def _ensure_legacy_teacher_checkpoint() -> str:
        os.makedirs(os.path.dirname(_LEGACY_RESNET18_PATH), exist_ok=True)
        if not os.path.exists(_LEGACY_RESNET18_PATH):
            logger.info('Downloading legacy ResNet-18 teacher checkpoint for DeSTSeg...')
            urllib.request.urlretrieve(_LEGACY_RESNET18_URL, _LEGACY_RESNET18_PATH)
        return _LEGACY_RESNET18_PATH

    def _set_phase_trainability(self):
        student_phase = self._phase == 'student'
        _set_requires_grad(self.teacher_net, False)
        _set_requires_grad(self.student_net, student_phase)
        _set_requires_grad(self.segmentation_net, not student_phase)

    def _set_phase(self, phase: str) -> None:
        if phase != self._phase:
            self._phase = phase
            self._set_phase_trainability()
        self._apply_phase_mode(self.training)

    def _apply_phase_mode(self, mode):
        super().train(mode)
        self.teacher_net.eval()
        if self._phase == 'student':
            self.student_net.train(mode)
            self.segmentation_net.eval()
        else:
            self.student_net.eval()
            self.segmentation_net.train(mode)
        return self

    def train(self, mode=True):
        return self._apply_phase_mode(mode)

    def set_epoch_info(self, epoch, max_epochs):
        student_epochs = max(1, int(math.ceil(max_epochs * self.phase_ratio)))
        self._set_phase('student' if epoch < student_epochs else 'segmentation')

    def set_iter_info(self, iter, max_iters):
        if self.de_st_steps is not None:
            student_iters = max(1, self.de_st_steps)
        else:
            student_iters = max(1, int(math.ceil(max_iters * self.phase_ratio)))
        self._set_phase('student' if iter < student_iters else 'segmentation')

    def _forward_destseg(self, img_aug, img_origin=None):
        if img_origin is None:
            img_origin = img_aug.clone()

        outputs_teacher_aug = [
            l2_normalize(output_t.detach()) for output_t in self.teacher_net(img_aug)
        ]
        outputs_student_aug = [
            l2_normalize(output_s) for output_s in self.student_net(img_aug)
        ]
        output = torch.cat(
            [
                F.interpolate(
                    -output_t * output_s,
                    size=outputs_student_aug[0].shape[2:],
                    mode='bilinear',
                    align_corners=False,
                )
                for output_t, output_s in zip(outputs_teacher_aug, outputs_student_aug)
            ],
            dim=1,
        )
        output_segmentation = self.segmentation_net(output)

        outputs_teacher = [
            l2_normalize(output_t.detach()) for output_t in self.teacher_net(img_origin)
        ]
        output_de_st_list = []
        for output_t, output_s in zip(outputs_teacher, outputs_student_aug):
            a_map = 1 - torch.sum(output_s * output_t, dim=1, keepdim=True)
            output_de_st_list.append(a_map)
        output_de_st = torch.cat(
            [
                F.interpolate(
                    output_de_st_instance,
                    size=outputs_student_aug[0].shape[2:],
                    mode='bilinear',
                    align_corners=False,
                )
                for output_de_st_instance in output_de_st_list
            ],
            dim=1,
        )
        output_de_st = torch.prod(output_de_st, dim=1, keepdim=True)
        return output_segmentation, output_de_st, output_de_st_list

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        if mode == 'loss':
            if not data_samples:
                raise RuntimeError('DeSTSeg loss requires PackDeSTSegInputs-style data_samples.')
            img_origin = _stack_sample_tensors(
                data_samples,
                'img_origin',
                device=inputs.device,
                dtype=inputs.dtype,
            )
            gt_masks = _stack_sample_tensors(
                data_samples,
                'gt_mask',
                device=inputs.device,
                dtype=inputs.dtype,
            )
            if gt_masks.ndim == 3:
                gt_masks = gt_masks.unsqueeze(1)

            output_segmentation, _, output_de_st_list = self._forward_destseg(inputs, img_origin)
            gt_masks = F.interpolate(
                gt_masks,
                size=output_segmentation.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
            gt_masks = torch.where(gt_masks < 0.5, torch.zeros_like(gt_masks), torch.ones_like(gt_masks))

            if self._phase == 'student':
                return {'loss': _cosine_similarity_loss(output_de_st_list)}

            loss_focal = _focal_loss(output_segmentation, gt_masks, gamma=self.gamma)
            loss_l1 = F.l1_loss(output_segmentation, gt_masks)
            return {'loss': loss_focal + loss_l1}

        if mode == 'predict':
            output_segmentation, _, _ = self._forward_destseg(inputs)
            score_map = F.interpolate(
                output_segmentation,
                size=inputs.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
            flat = score_map.flatten(1)
            k = min(self.top_k_score, flat.shape[1])
            topk = torch.topk(flat, k=k, dim=1).values
            img_scores = topk.mean(dim=1)
            return build_predict_results(data_samples, img_scores, score_map)

        if mode == 'tensor':
            return self._forward_destseg(inputs)

        raise RuntimeError(f'Invalid mode "{mode}".')

    def train_step(self, data, optim_wrapper):
        if (
            not isinstance(optim_wrapper, OptimWrapperDict)
            or 'student' not in optim_wrapper
            or 'segmentation' not in optim_wrapper
        ):
            raise TypeError(
                'Strict DeSTSeg training requires an OptimWrapperDict with student and segmentation optimizers.'
            )

        data = self.data_preprocessor(data, True)
        inputs = data['inputs']
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)
        data_samples = data.get('data_samples', None)

        student_optim = optim_wrapper['student']
        segmentation_optim = optim_wrapper['segmentation']
        active_optim = student_optim if self._phase == 'student' else segmentation_optim
        inactive_optim = segmentation_optim if self._phase == 'student' else student_optim

        if active_optim._inner_count % active_optim._accumulative_counts == 0:
            active_optim.zero_grad()
        inactive_optim.zero_grad()

        losses = self(inputs, data_samples, mode='loss')
        active_optim.update_params(losses['loss'])

        detached = {}
        for key, value in losses.items():
            detached[key] = value.detach() if torch.is_tensor(value) else value
        return detached
