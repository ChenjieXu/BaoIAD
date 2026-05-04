"""Official-style MemAE detector adapted to image anomaly detection.

This implementation follows the original `donggong1/memae-anomaly-detection`
architecture and loss/scoring semantics as closely as possible:

- 3D convolutional encoder/decoder
- memory unit with softmax + hard shrinkage
- MSE reconstruction loss + entropy sparsity penalty
- test-time score based on channel-L2 reconstruction error averaged over the
  spatiotemporal error volume

The only intentional adaptation is the image-to-clip bridge for MVTec-style
single images: each input image is repeated along the temporal axis.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter

from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS
from baoiad.models.base_ad_model import ReconstructionADModel


def hard_shrink_relu(input_tensor: Tensor, lambd: float = 0.0, epsilon: float = 1e-12) -> Tensor:
    """Official ReLU-based hard shrinkage used by MemAE."""
    return (F.relu(input_tensor - lambd) * input_tensor) / (torch.abs(input_tensor - lambd) + epsilon)


class MemoryUnit(nn.Module):
    """Official MemAE memory unit."""

    def __init__(self, mem_dim: int, fea_dim: int, shrink_thres: float = 0.0025) -> None:
        super().__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.weight = Parameter(torch.empty(self.mem_dim, self.fea_dim))
        self.bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs: Tensor) -> dict[str, Tensor]:
        att_weight = F.linear(inputs, self.weight)
        att_weight = F.softmax(att_weight, dim=1)
        if self.shrink_thres > 0:
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1)

        mem_transposed = self.weight.permute(1, 0)
        output = F.linear(att_weight, mem_transposed)
        return {'output': output, 'att': att_weight}


class MemModule(nn.Module):
    """Official MemAE memory wrapper supporting 3D/4D/5D tensors."""

    def __init__(self, mem_dim: int, fea_dim: int, shrink_thres: float = 0.0025) -> None:
        super().__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.memory = MemoryUnit(mem_dim, fea_dim, shrink_thres)

    def forward(self, inputs: Tensor) -> dict[str, Tensor]:
        shape = inputs.shape
        dims = len(shape)

        if dims == 3:
            flattened = inputs.permute(0, 2, 1)
        elif dims == 4:
            flattened = inputs.permute(0, 2, 3, 1)
        elif dims == 5:
            flattened = inputs.permute(0, 2, 3, 4, 1)
        else:
            raise ValueError(f'Unsupported feature map rank for MemModule: {dims}')

        flattened = flattened.contiguous().view(-1, shape[1])
        outputs = self.memory(flattened)
        feature_out = outputs['output']
        att = outputs['att']

        if dims == 3:
            feature_out = feature_out.view(shape[0], shape[2], shape[1]).permute(0, 2, 1)
            att = att.view(shape[0], shape[2], self.mem_dim).permute(0, 2, 1)
        elif dims == 4:
            feature_out = feature_out.view(shape[0], shape[2], shape[3], shape[1]).permute(0, 3, 1, 2)
            att = att.view(shape[0], shape[2], shape[3], self.mem_dim).permute(0, 3, 1, 2)
        else:
            feature_out = feature_out.view(shape[0], shape[2], shape[3], shape[4], shape[1]).permute(0, 4, 1, 2, 3)
            att = att.view(shape[0], shape[2], shape[3], shape[4], self.mem_dim).permute(0, 4, 1, 2, 3)

        return {'output': feature_out, 'att': att}


class EntropyLoss(nn.Module):
    """Official entropy loss used on the attention volume."""

    def __init__(self, eps: float = 1e-12) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, inputs: Tensor) -> Tensor:
        entropy = inputs * torch.log(inputs + self.eps)
        entropy = -entropy.sum(dim=1)
        return entropy.mean()


class EntropyLossEncap(nn.Module):
    """Apply entropy loss after moving feature channels to the last dim."""

    def __init__(self, eps: float = 1e-12) -> None:
        super().__init__()
        self.entropy_loss = EntropyLoss(eps)

    def forward(self, inputs: Tensor) -> Tensor:
        shape = inputs.shape
        dims = len(shape)
        if dims == 2:
            flattened = inputs
        elif dims == 3:
            flattened = inputs.permute(0, 2, 1)
        elif dims == 4:
            flattened = inputs.permute(0, 2, 3, 1)
        elif dims == 5:
            flattened = inputs.permute(0, 2, 3, 4, 1)
        else:
            raise ValueError(f'Unsupported attention rank for entropy loss: {dims}')
        flattened = flattened.contiguous().view(-1, shape[1])
        return self.entropy_loss(flattened)


class AutoEncoderCov3DMem(nn.Module):
    """Official MemAE 3D convolutional autoencoder."""

    def __init__(self, in_channels: int, mem_dim: int, shrink_thres: float = 0.0025) -> None:
        super().__init__()
        feature_num = 128
        feature_num_2 = 96
        feature_num_x2 = 256

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, feature_num_2, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(feature_num_2, feature_num, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(feature_num, feature_num_x2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(feature_num_x2, feature_num_x2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mem_rep = MemModule(mem_dim=mem_dim, fea_dim=feature_num_x2, shrink_thres=shrink_thres)

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(
                feature_num_x2,
                feature_num_x2,
                (3, 3, 3),
                stride=(2, 2, 2),
                padding=(1, 1, 1),
                output_padding=(1, 1, 1),
            ),
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(
                feature_num_x2,
                feature_num,
                (3, 3, 3),
                stride=(2, 2, 2),
                padding=(1, 1, 1),
                output_padding=(1, 1, 1),
            ),
            nn.BatchNorm3d(feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(
                feature_num,
                feature_num_2,
                (3, 3, 3),
                stride=(2, 2, 2),
                padding=(1, 1, 1),
                output_padding=(1, 1, 1),
            ),
            nn.BatchNorm3d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(
                feature_num_2,
                in_channels,
                (3, 3, 3),
                stride=(1, 2, 2),
                padding=(1, 1, 1),
                output_padding=(0, 1, 1),
            ),
        )

    def forward(self, inputs: Tensor) -> dict[str, Tensor]:
        features = self.encoder(inputs)
        memory_outputs = self.mem_rep(features)
        features = memory_outputs['output']
        att = memory_outputs['att']
        outputs = self.decoder(features)
        return {'output': outputs, 'att': att}


def _official_weights_init(module: nn.Module) -> None:
    """Reproduce the weight initialization used in the official repository."""
    classname = module.__class__.__name__
    if 'Conv' in classname:
        if hasattr(module, 'weight') and module.weight is not None:
            module.weight.data.normal_(0.0, 0.02)
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.data.zero_()
    elif 'BatchNorm' in classname:
        if hasattr(module, 'weight') and module.weight is not None:
            module.weight.data.normal_(1.0, 0.02)
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.data.zero_()


@MODELS.register_module(force=True)
class MemAEDetector(ReconstructionADModel):
    """Official-style MemAE for image anomaly detection via repeated clips."""

    def __init__(
        self,
        in_channels: int = 3,
        frame_num: int = 16,
        clip_mode: str = 'repeat_image',
        mem_dim: int = 2000,
        shrink_thres: float = 0.0025,
        entropy_loss_weight: float = 0.0002,
        clip_jitter_strength: float = 0.03,
        clip_pingpong_shift: int = 1,
        clip_schedule_strength: float = 0.08,
        clip_schedule_blur_kernel_size: int = 5,
        clip_window_scale_min: float = 0.76,
        clip_window_translation_max: float = 0.16,
        clip_neighbor_window_radius: int = 7,
        temporal_reduce_mode: str = 'mean',
        image_score_mode: str = 'spatiotemporal_mean',
        topk_ratio: float = 0.01,
        loss: dict = dict(type='MSELoss'),
        data_preprocessor: Optional[dict] = None,
        init_cfg: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if frame_num <= 0:
            raise ValueError('frame_num must be positive')
        if clip_mode not in {
            'repeat_image',
            'repeat_image_with_small_jitter',
            'two_frame_pingpong',
            'multi_shift_pingpong',
            'multi_view_intensity_schedule',
            'local_motion_window',
            'centered_local_motion_window',
            'progressive_crop_window',
            'adjacent_filename_window',
            'train_good_reference_window',
            'test_good_reference_window',
        }:
            raise ValueError(f'Unsupported clip_mode: {clip_mode}')
        if clip_jitter_strength < 0:
            raise ValueError('clip_jitter_strength must be >= 0.')
        if clip_pingpong_shift < 0:
            raise ValueError('clip_pingpong_shift must be >= 0.')
        if clip_schedule_strength < 0:
            raise ValueError('clip_schedule_strength must be >= 0.')
        if clip_schedule_blur_kernel_size < 1 or clip_schedule_blur_kernel_size % 2 == 0:
            raise ValueError('clip_schedule_blur_kernel_size must be a positive odd integer.')
        if not 0 < clip_window_scale_min <= 1:
            raise ValueError('clip_window_scale_min must be in (0, 1].')
        if clip_window_translation_max < 0:
            raise ValueError('clip_window_translation_max must be >= 0.')
        if clip_neighbor_window_radius < 0:
            raise ValueError('clip_neighbor_window_radius must be >= 0.')
        if temporal_reduce_mode not in {'mean', 'max'}:
            raise ValueError(f'Unsupported temporal_reduce_mode: {temporal_reduce_mode}')
        if image_score_mode not in {'spatiotemporal_mean', 'map_mean', 'map_max', 'map_p95', 'map_topk_mean'}:
            raise ValueError(f'Unsupported image_score_mode: {image_score_mode}')
        if not 0 < topk_ratio <= 1:
            raise ValueError('topk_ratio must be in (0, 1].')

        self.in_channels = in_channels
        self.frame_num = frame_num
        self.clip_mode = clip_mode
        self.mem_dim = mem_dim
        self.shrink_thres = shrink_thres
        self.entropy_loss_weight = entropy_loss_weight
        self.clip_jitter_strength = clip_jitter_strength
        self.clip_pingpong_shift = clip_pingpong_shift
        self.clip_schedule_strength = clip_schedule_strength
        self.clip_schedule_blur_kernel_size = clip_schedule_blur_kernel_size
        self.clip_window_scale_min = clip_window_scale_min
        self.clip_window_translation_max = clip_window_translation_max
        self.clip_neighbor_window_radius = clip_neighbor_window_radius
        self.temporal_reduce_mode = temporal_reduce_mode
        self.image_score_mode = image_score_mode
        self.topk_ratio = topk_ratio
        self._adjacent_frame_cache: dict[tuple[str, tuple[int, int]], Tensor] = {}

        self.model = AutoEncoderCov3DMem(
            in_channels=in_channels,
            mem_dim=mem_dim,
            shrink_thres=shrink_thres,
        )
        self.model.apply(_official_weights_init)
        self.loss_fn = MODELS.build(loss)
        self.entropy_loss_fn = EntropyLossEncap()

    @staticmethod
    def _stack_inputs(inputs: Union[Tensor, Sequence[Tensor]]) -> Tensor:
        if isinstance(inputs, (list, tuple)):
            return torch.stack(list(inputs))
        return inputs

    def _build_repeat_clip(self, inputs: Tensor) -> Tensor:
        return inputs.unsqueeze(2).repeat(1, 1, self.frame_num, 1, 1)

    def _build_jitter_clip(self, inputs: Tensor) -> Tensor:
        clip = self._build_repeat_clip(inputs)
        if self.frame_num == 1 or self.clip_jitter_strength == 0:
            return clip

        # Diagnose-only bridge: inject a tiny deterministic brightness ramp so
        # temporal operators see non-zero variation without changing semantics
        # as aggressively as a spatial shift.
        ramp = torch.linspace(
            -self.clip_jitter_strength,
            self.clip_jitter_strength,
            steps=self.frame_num,
            device=clip.device,
            dtype=clip.dtype,
        ).view(1, 1, self.frame_num, 1, 1)
        return torch.clamp(clip + ramp, min=-1.0, max=1.0)

    def _build_pingpong_clip(self, inputs: Tensor) -> Tensor:
        clip = self._build_repeat_clip(inputs)
        if self.frame_num == 1 or self.clip_pingpong_shift == 0:
            return clip

        shifted = torch.roll(inputs, shifts=self.clip_pingpong_shift, dims=-1)
        for frame_index in range(self.frame_num):
            if frame_index % 2 == 1:
                clip[:, :, frame_index] = shifted
        return clip

    @staticmethod
    def _shift_image(inputs: Tensor, shift_h: int = 0, shift_w: int = 0) -> Tensor:
        shifted = inputs
        if shift_h:
            shifted = torch.roll(shifted, shifts=shift_h, dims=-2)
        if shift_w:
            shifted = torch.roll(shifted, shifts=shift_w, dims=-1)
        return shifted

    def _build_multi_shift_pingpong_clip(self, inputs: Tensor) -> Tensor:
        clip = self._build_repeat_clip(inputs)
        if self.frame_num == 1 or self.clip_pingpong_shift == 0:
            return clip

        shift = self.clip_pingpong_shift
        variants = [
            inputs,
            self._shift_image(inputs, shift_w=shift),
            inputs,
            self._shift_image(inputs, shift_w=-shift),
            inputs,
            self._shift_image(inputs, shift_h=shift),
            inputs,
            self._shift_image(inputs, shift_h=-shift),
        ]
        for frame_index in range(self.frame_num):
            clip[:, :, frame_index] = variants[frame_index % len(variants)]
        return clip

    def _build_multi_view_intensity_schedule_clip(self, inputs: Tensor) -> Tensor:
        clip = self._build_repeat_clip(inputs)
        if self.frame_num == 1:
            return clip

        strength = self.clip_schedule_strength
        blur_kernel = self.clip_schedule_blur_kernel_size
        blurred = F.avg_pool2d(inputs, kernel_size=blur_kernel, stride=1, padding=blur_kernel // 2)

        views = [
            inputs,
            torch.clamp(inputs + strength, min=-1.0, max=1.0),
            torch.clamp(inputs - strength, min=-1.0, max=1.0),
            torch.clamp(inputs * (1.0 + strength), min=-1.0, max=1.0),
            torch.clamp(inputs * max(0.0, 1.0 - strength), min=-1.0, max=1.0),
            torch.clamp(blurred, min=-1.0, max=1.0),
        ]
        for frame_index in range(self.frame_num):
            clip[:, :, frame_index] = views[frame_index % len(views)]
        return clip

    def _affine_sample(self, inputs: Tensor, scale: float, translate_x: float, translate_y: float) -> Tensor:
        batch_size, channels, height, width = inputs.shape
        theta = inputs.new_zeros((batch_size, 2, 3))
        theta[:, 0, 0] = scale
        theta[:, 1, 1] = scale
        theta[:, 0, 2] = translate_x
        theta[:, 1, 2] = translate_y
        grid = F.affine_grid(theta, size=(batch_size, channels, height, width), align_corners=False)
        return F.grid_sample(inputs, grid, mode='bilinear', padding_mode='border', align_corners=False)

    def _build_local_motion_window_clip(self, inputs: Tensor) -> Tensor:
        clip = self._build_repeat_clip(inputs)
        if self.frame_num == 1:
            return clip

        tx_values = torch.linspace(
            -self.clip_window_translation_max,
            self.clip_window_translation_max,
            steps=self.frame_num,
            device=inputs.device,
            dtype=inputs.dtype,
        )
        ty_values = torch.linspace(
            self.clip_window_translation_max * 0.5,
            -self.clip_window_translation_max * 0.5,
            steps=self.frame_num,
            device=inputs.device,
            dtype=inputs.dtype,
        )
        scale_values = torch.linspace(
            1.0,
            self.clip_window_scale_min,
            steps=self.frame_num,
            device=inputs.device,
            dtype=inputs.dtype,
        )
        for frame_index in range(self.frame_num):
            clip[:, :, frame_index] = self._affine_sample(
                inputs,
                float(scale_values[frame_index].item()),
                float(tx_values[frame_index].item()),
                float(ty_values[frame_index].item()),
            )
        return clip

    def _build_centered_local_motion_window_clip(self, inputs: Tensor) -> Tensor:
        clip = self._build_repeat_clip(inputs)
        if self.frame_num == 1:
            return clip

        center_index = self.frame_num // 2
        max_offset = max(center_index, self.frame_num - center_index - 1, 1)
        for frame_index in range(self.frame_num):
            relative = frame_index - center_index
            if relative == 0:
                clip[:, :, frame_index] = inputs
                continue
            ratio = abs(relative) / float(max_offset)
            scale = 1.0 - ratio * (1.0 - self.clip_window_scale_min)
            translate_x = (relative / float(max_offset)) * self.clip_window_translation_max
            translate_y = -(relative / float(max_offset)) * (self.clip_window_translation_max * 0.5)
            clip[:, :, frame_index] = self._affine_sample(inputs, scale, translate_x, translate_y)
        return clip

    def _build_progressive_crop_window_clip(self, inputs: Tensor) -> Tensor:
        clip = self._build_repeat_clip(inputs)
        if self.frame_num == 1:
            return clip

        scale_values = torch.linspace(
            1.0,
            self.clip_window_scale_min,
            steps=self.frame_num,
            device=inputs.device,
            dtype=inputs.dtype,
        )
        for frame_index in range(self.frame_num):
            clip[:, :, frame_index] = self._affine_sample(
                inputs,
                float(scale_values[frame_index].item()),
                0.0,
                0.0,
            )
        return clip

    def _resolve_adjacent_offsets(self) -> list[int]:
        if self.frame_num == 1:
            return [0]
        radius = max(self.clip_neighbor_window_radius, 1)
        offsets = torch.linspace(-radius, radius, steps=self.frame_num)
        return [int(round(value.item())) for value in offsets]

    @staticmethod
    def _list_image_siblings(directory: Path, suffix: str) -> list[Path]:
        return sorted(directory.glob(suffix and f'*{suffix}' or '*'))

    def _load_adjacent_frame(self, path: Path, target_hw: tuple[int, int], device: torch.device, dtype: torch.dtype) -> Tensor:
        cache_key = (str(path), target_hw)
        if cache_key not in self._adjacent_frame_cache:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f'Failed to load adjacent frame: {path}')
            if self.in_channels == 1:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)
                tensor = torch.from_numpy(image).unsqueeze(0).float()
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)
                tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
            self._adjacent_frame_cache[cache_key] = ((tensor - 127.5) / 127.5).contiguous()
        return self._adjacent_frame_cache[cache_key].to(device=device, dtype=dtype)

    def _build_adjacent_filename_window_clip(
        self,
        inputs: Tensor,
        data_samples: Sequence[object] | None,
    ) -> Tensor:
        if data_samples is None or len(data_samples) != inputs.shape[0]:
            raise ValueError('adjacent_filename_window requires data_samples with img_path for every input.')

        clip = self._build_repeat_clip(inputs)
        offsets = self._resolve_adjacent_offsets()
        target_hw = (inputs.shape[-2], inputs.shape[-1])
        device = inputs.device
        dtype = inputs.dtype

        for batch_index, sample in enumerate(data_samples):
            img_path = getattr(sample, 'img_path', '')
            if not img_path:
                raise ValueError('adjacent_filename_window requires each data sample to expose img_path.')
            current_path = Path(img_path)
            siblings = self._list_image_siblings(current_path.parent, current_path.suffix)
            sibling_names = [path.name for path in siblings]
            if current_path.name not in sibling_names:
                raise ValueError(f'Current path {current_path} not found in sibling list.')
            current_index = sibling_names.index(current_path.name)

            for frame_index, offset in enumerate(offsets):
                neighbor_index = min(max(current_index + offset, 0), len(siblings) - 1)
                neighbor_path = siblings[neighbor_index]
                if neighbor_path == current_path:
                    clip[batch_index, :, frame_index] = inputs[batch_index]
                else:
                    clip[batch_index, :, frame_index] = self._load_adjacent_frame(
                        neighbor_path,
                        target_hw,
                        device,
                        dtype,
                    )
        return clip

    def _resolve_reference_good_path(self, current_path: Path) -> tuple[list[Path], int]:
        current_siblings = self._list_image_siblings(current_path.parent, current_path.suffix)
        current_names = [path.name for path in current_siblings]
        if current_path.name not in current_names:
            raise ValueError(f'Current path {current_path} not found in sibling list.')
        current_index = current_names.index(current_path.name)

        cls_root = current_path.parents[2]
        reference_dir = cls_root / 'train' / 'good'
        reference_siblings = self._list_image_siblings(reference_dir, current_path.suffix)
        if not reference_siblings:
            raise ValueError(f'No reference good frames found under {reference_dir}.')

        if current_path.parent == reference_dir:
            return reference_siblings, current_index

        if len(current_siblings) <= 1:
            anchor = 0
        else:
            anchor = int(round(current_index / float(len(current_siblings) - 1) * (len(reference_siblings) - 1)))
        anchor = min(max(anchor, 0), len(reference_siblings) - 1)
        return reference_siblings, anchor

    def _resolve_reference_test_good_path(self, current_path: Path) -> tuple[list[Path], int]:
        current_siblings = self._list_image_siblings(current_path.parent, current_path.suffix)
        current_names = [path.name for path in current_siblings]
        if current_path.name not in current_names:
            raise ValueError(f'Current path {current_path} not found in sibling list.')
        current_index = current_names.index(current_path.name)

        cls_root = current_path.parents[2]
        reference_dir = cls_root / 'test' / 'good'
        reference_siblings = self._list_image_siblings(reference_dir, current_path.suffix)
        if not reference_siblings:
            raise ValueError(f'No test-good reference frames found under {reference_dir}.')

        if current_path.parent == reference_dir:
            return reference_siblings, current_index

        anchor = min(max(current_index, 0), len(reference_siblings) - 1)
        return reference_siblings, anchor

    def _build_train_good_reference_window_clip(
        self,
        inputs: Tensor,
        data_samples: Sequence[object] | None,
    ) -> Tensor:
        if data_samples is None or len(data_samples) != inputs.shape[0]:
            raise ValueError('train_good_reference_window requires data_samples with img_path for every input.')

        clip = self._build_repeat_clip(inputs)
        offsets = self._resolve_adjacent_offsets()
        center_index = self.frame_num // 2
        target_hw = (inputs.shape[-2], inputs.shape[-1])
        device = inputs.device
        dtype = inputs.dtype

        for batch_index, sample in enumerate(data_samples):
            img_path = getattr(sample, 'img_path', '')
            if not img_path:
                raise ValueError('train_good_reference_window requires each data sample to expose img_path.')
            current_path = Path(img_path)
            reference_siblings, anchor = self._resolve_reference_good_path(current_path)

            for frame_index, offset in enumerate(offsets):
                if frame_index == center_index:
                    clip[batch_index, :, frame_index] = inputs[batch_index]
                    continue
                neighbor_index = min(max(anchor + offset, 0), len(reference_siblings) - 1)
                neighbor_path = reference_siblings[neighbor_index]
                clip[batch_index, :, frame_index] = self._load_adjacent_frame(
                    neighbor_path,
                    target_hw,
                    device,
                    dtype,
                )
        return clip

    def _build_test_good_reference_window_clip(
        self,
        inputs: Tensor,
        data_samples: Sequence[object] | None,
    ) -> Tensor:
        if data_samples is None or len(data_samples) != inputs.shape[0]:
            raise ValueError('test_good_reference_window requires data_samples with img_path for every input.')

        clip = self._build_repeat_clip(inputs)
        offsets = self._resolve_adjacent_offsets()
        center_index = self.frame_num // 2
        target_hw = (inputs.shape[-2], inputs.shape[-1])
        device = inputs.device
        dtype = inputs.dtype

        for batch_index, sample in enumerate(data_samples):
            img_path = getattr(sample, 'img_path', '')
            if not img_path:
                raise ValueError('test_good_reference_window requires each data sample to expose img_path.')
            current_path = Path(img_path)
            reference_siblings, anchor = self._resolve_reference_test_good_path(current_path)

            for frame_index, offset in enumerate(offsets):
                if frame_index == center_index:
                    clip[batch_index, :, frame_index] = inputs[batch_index]
                    continue
                neighbor_index = min(max(anchor + offset, 0), len(reference_siblings) - 1)
                neighbor_path = reference_siblings[neighbor_index]
                clip[batch_index, :, frame_index] = self._load_adjacent_frame(
                    neighbor_path,
                    target_hw,
                    device,
                    dtype,
                )
        return clip

    def _to_clip(self, inputs: Tensor, data_samples: Sequence[object] | None = None) -> Tensor:
        if inputs.ndim == 5:
            return inputs
        if inputs.ndim != 4:
            raise ValueError(f'MemAEDetector expects 4D or 5D inputs, got shape {tuple(inputs.shape)}')
        if self.clip_mode == 'repeat_image':
            return self._build_repeat_clip(inputs)
        if self.clip_mode == 'repeat_image_with_small_jitter':
            return self._build_jitter_clip(inputs)
        if self.clip_mode == 'two_frame_pingpong':
            return self._build_pingpong_clip(inputs)
        if self.clip_mode == 'multi_shift_pingpong':
            return self._build_multi_shift_pingpong_clip(inputs)
        if self.clip_mode == 'multi_view_intensity_schedule':
            return self._build_multi_view_intensity_schedule_clip(inputs)
        if self.clip_mode == 'local_motion_window':
            return self._build_local_motion_window_clip(inputs)
        if self.clip_mode == 'centered_local_motion_window':
            return self._build_centered_local_motion_window_clip(inputs)
        if self.clip_mode == 'progressive_crop_window':
            return self._build_progressive_crop_window_clip(inputs)
        if self.clip_mode == 'adjacent_filename_window':
            return self._build_adjacent_filename_window_clip(inputs, data_samples)
        if self.clip_mode == 'train_good_reference_window':
            return self._build_train_good_reference_window_clip(inputs, data_samples)
        if self.clip_mode == 'test_good_reference_window':
            return self._build_test_good_reference_window_clip(inputs, data_samples)
        raise ValueError(f'Unsupported clip_mode: {self.clip_mode}')

    @staticmethod
    def _resize_if_needed(reconstruction: Tensor, target: Tensor) -> Tensor:
        if reconstruction.shape == target.shape:
            return reconstruction
        return F.interpolate(
            reconstruction,
            size=target.shape[-3:],
            mode='trilinear',
            align_corners=False,
        )

    def _reduce_temporal_map(self, spatiotemporal_map: Tensor) -> Tensor:
        if self.temporal_reduce_mode == 'mean':
            return spatiotemporal_map.mean(dim=1, keepdim=True)
        if self.temporal_reduce_mode == 'max':
            return spatiotemporal_map.max(dim=1, keepdim=True).values
        raise ValueError(f'Unsupported temporal_reduce_mode: {self.temporal_reduce_mode}')

    def _aggregate_image_score(self, spatiotemporal_map: Tensor, anomaly_map: Tensor) -> Tensor:
        if self.image_score_mode == 'spatiotemporal_mean':
            return spatiotemporal_map.flatten(1).mean(dim=1)

        flat_map = anomaly_map.flatten(1)
        if self.image_score_mode == 'map_mean':
            return flat_map.mean(dim=1)
        if self.image_score_mode == 'map_max':
            return flat_map.max(dim=1).values
        if self.image_score_mode == 'map_p95':
            return torch.quantile(flat_map, 0.95, dim=1)
        if self.image_score_mode == 'map_topk_mean':
            k = max(1, math.ceil(flat_map.shape[1] * self.topk_ratio))
            return torch.topk(flat_map, k=k, dim=1).values.mean(dim=1)
        raise ValueError(f'Unsupported image_score_mode: {self.image_score_mode}')

    def _compute_scores(self, reconstruction: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        residual = reconstruction - target
        spatiotemporal_map = torch.sum(residual.pow(2), dim=1).sqrt()
        anomaly_map = self._reduce_temporal_map(spatiotemporal_map)
        img_scores = self._aggregate_image_score(spatiotemporal_map, anomaly_map)
        return img_scores, anomaly_map

    def _forward_clip_details(self, clip_inputs: Tensor) -> dict[str, Tensor]:
        encoded = self.model.encoder(clip_inputs)
        memory_outputs = self.model.mem_rep(encoded)
        memory_features = memory_outputs['output']
        attention = memory_outputs['att']
        reconstruction = self.model.decoder(memory_features)
        reconstruction = self._resize_if_needed(reconstruction, clip_inputs)
        residual = reconstruction - clip_inputs
        spatiotemporal_map = torch.sum(residual.pow(2), dim=1).sqrt()
        anomaly_map = self._reduce_temporal_map(spatiotemporal_map)
        img_scores = self._aggregate_image_score(spatiotemporal_map, anomaly_map)
        return {
            'clip_inputs': clip_inputs,
            'encoded': encoded,
            'memory_features': memory_features,
            'attention': attention,
            'reconstruction': reconstruction,
            'residual': residual,
            'spatiotemporal_map': spatiotemporal_map,
            'anomaly_map': anomaly_map,
            'img_scores': img_scores,
        }

    def forward(
        self,
        inputs: Union[Tensor, Sequence[Tensor]],
        data_samples=None,
        mode: str = 'tensor',
    ):
        inputs = self._stack_inputs(inputs)
        clip_inputs = self._to_clip(inputs, data_samples)
        details = self._forward_clip_details(clip_inputs)
        reconstruction = details['reconstruction']
        attention = details['attention']

        if mode == 'tensor':
            return reconstruction
        if mode == 'loss':
            recon_loss = self.loss_fn(reconstruction, clip_inputs)
            entropy_loss = self.entropy_loss_fn(attention)
            loss = recon_loss + self.entropy_loss_weight * entropy_loss
            return {
                'loss': loss,
                'recon_loss': recon_loss,
                'entropy_loss': entropy_loss,
            }
        if mode == 'predict':
            img_scores = details['img_scores']
            anomaly_map = details['anomaly_map']
            return build_predict_results(data_samples, img_scores, anomaly_map)
        raise RuntimeError(f'Invalid mode "{mode}".')
