"""AST: Asymmetric Student-Teacher Networks for Industrial Anomaly Detection.

Official reference:
https://github.com/marco-rudolph/AST

This implementation keeps two execution modes:
- ``joint`` for the historical BaoIAD single-run AST path
- ``teacher`` / ``student`` for strict official-compatible two-stage training
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS
from baoiad.models.base_ad_model import BaseADModel


def positional_encoding_2d(dim: int, height: int, width: int) -> torch.Tensor:
    """Build the 2D sinusoidal positional encoding used by official AST."""
    if dim % 4 != 0:
        raise ValueError(f'pos_enc_dim must be divisible by 4, got {dim}.')

    pos = torch.zeros(dim, height, width)
    half_dim = dim // 2
    div_term = torch.exp(torch.arange(0.0, half_dim, 2) * -(np.log(1e4) / half_dim))
    pos_w = torch.arange(0.0, width).unsqueeze(1)
    pos_h = torch.arange(0.0, height).unsqueeze(1)
    pos[0:half_dim:2, :, :] = (
        torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    )
    pos[1:half_dim:2, :, :] = (
        torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    )
    pos[half_dim::2, :, :] = (
        torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    )
    pos[half_dim + 1::2, :, :] = (
        torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    )
    return pos.unsqueeze(0)


class PermutationLayer(nn.Module):
    """Fixed random channel permutation."""

    def __init__(self, channels: int, seed: int = 0):
        super().__init__()
        rng = np.random.RandomState(seed)
        perm = rng.permutation(channels)
        perm_inv = np.zeros_like(perm)
        for idx, value in enumerate(perm):
            perm_inv[value] = idx
        self.register_buffer('perm', torch.as_tensor(perm, dtype=torch.long))
        self.register_buffer('perm_inv', torch.as_tensor(perm_inv, dtype=torch.long))

    def forward(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        if reverse:
            return x[:, self.perm_inv]
        return x[:, self.perm]


class ConvSubnet(nn.Module):
    """Two-layer conv subnet used inside the coupling layers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels_hidden: int = 64,
        kernel_size: int = 3,
        use_gamma: bool = True,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(
            in_channels,
            channels_hidden,
            kernel_size=kernel_size,
            padding=pad,
            padding_mode='replicate',
        )
        self.conv2 = nn.Conv2d(
            channels_hidden,
            out_channels,
            kernel_size=kernel_size,
            padding=pad,
            padding_mode='replicate',
        )
        self.relu = nn.ReLU(inplace=False)
        self.use_gamma = use_gamma
        if use_gamma:
            self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        if self.use_gamma:
            out = out * self.gamma
        return out


class ConditionalGlowCouplingBlock(nn.Module):
    """Glow-style affine coupling with optional positional conditioning."""

    def __init__(
        self,
        channels: int,
        channels_hidden: int = 64,
        kernel_size: int = 3,
        clamp: float = 1.9,
        cond_dim: int = 0,
        use_gamma: bool = True,
    ) -> None:
        super().__init__()
        self.split_len1 = channels // 2
        self.split_len2 = channels - self.split_len1
        self.clamp = clamp
        self.cond_dim = int(cond_dim)

        self.s1 = ConvSubnet(
            self.split_len1 + self.cond_dim,
            self.split_len2 * 2,
            channels_hidden=channels_hidden,
            kernel_size=kernel_size,
            use_gamma=use_gamma,
        )
        self.s2 = ConvSubnet(
            self.split_len2 + self.cond_dim,
            self.split_len1 * 2,
            channels_hidden=channels_hidden,
            kernel_size=kernel_size,
            use_gamma=use_gamma,
        )

    def _log_e(self, s: torch.Tensor) -> torch.Tensor:
        if self.clamp > 0:
            return self.clamp * 0.636 * torch.atan(s / self.clamp)
        return s

    def _e(self, s: torch.Tensor) -> torch.Tensor:
        return torch.exp(self._log_e(s))

    def _cat_cond(self, x: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        if self.cond_dim <= 0:
            return x
        if cond is None:
            raise ValueError('Condition tensor is required when cond_dim > 0.')
        return torch.cat([x, cond], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = x[:, :self.split_len1]
        x2 = x[:, self.split_len1:]

        if not reverse:
            r2 = self.s2(self._cat_cond(x2, cond))
            s2, t2 = torch.split(r2, self.split_len1, dim=1)
            y1 = self._e(s2) * x1 + t2

            r1 = self.s1(self._cat_cond(y1, cond))
            s1, t1 = torch.split(r1, self.split_len2, dim=1)
            y2 = self._e(s1) * x2 + t1
        else:
            r1 = self.s1(self._cat_cond(x1, cond))
            s1, t1 = torch.split(r1, self.split_len2, dim=1)
            y2 = (x2 - t1) / self._e(s1)

            r2 = self.s2(self._cat_cond(y2, cond))
            s2, t2 = torch.split(r2, self.split_len1, dim=1)
            y1 = (x1 - t2) / self._e(s2)

        y = torch.cat([y1, y2], dim=1)
        y = torch.clamp(y, -1e6, 1e6)

        jac = torch.sum(self._log_e(s1), dim=1) + torch.sum(self._log_e(s2), dim=1)
        if reverse:
            jac = -jac
        return y, jac


class TeacherNF(nn.Module):
    """Teacher normalizing flow from official AST."""

    def __init__(
        self,
        n_feat: int,
        n_coupling_blocks: int = 4,
        channels_hidden: int = 64,
        clamp: float = 1.9,
        kernel_sizes: Optional[list[int]] = None,
        pos_enc_dim: int = 0,
        use_gamma: bool = True,
    ) -> None:
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3] * (n_coupling_blocks - 1) + [5]

        self.permutations = nn.ModuleList(
            [PermutationLayer(n_feat, seed=index) for index in range(n_coupling_blocks)]
        )
        self.blocks = nn.ModuleList([
            ConditionalGlowCouplingBlock(
                channels=n_feat,
                channels_hidden=channels_hidden,
                kernel_size=kernel_sizes[index],
                clamp=clamp,
                cond_dim=pos_enc_dim,
                use_gamma=use_gamma,
            )
            for index in range(n_coupling_blocks)
        ])

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = x
        jacobian = torch.zeros_like(x[:, 0], dtype=x.dtype, device=x.device)
        for permute, block in zip(self.permutations, self.blocks):
            z = permute(z)
            z, block_jac = block(z, cond=cond)
            jacobian = jacobian + block_jac
        return z, jacobian


class ResidualBlock(nn.Module):
    """Residual block used by the student network."""

    def __init__(self, channels: int):
        super().__init__()
        self.l1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.l2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.l1(x)))
        out = self.act(self.bn2(self.l2(out)))
        return out + x


class StudentCNN(nn.Module):
    """Student CNN from official AST."""

    def __init__(
        self,
        n_feat: int,
        channels_hidden: int = 1024,
        n_blocks: int = 4,
        pos_enc_dim: int = 0,
    ) -> None:
        super().__init__()
        in_channels = n_feat + int(pos_enc_dim)
        self.conv1 = nn.Conv2d(in_channels, channels_hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels_hidden, n_feat, kernel_size=3, padding=1)
        self.res_blocks = nn.ModuleList([ResidualBlock(channels_hidden) for _ in range(n_blocks)])
        self.gamma = nn.Parameter(torch.zeros(1))
        self.act = nn.LeakyReLU()

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        if cond is not None:
            x = torch.cat([cond, x], dim=1)
        out = self.act(self.conv1(x))
        for block in self.res_blocks:
            out = block(out)
        return self.conv2(out)


@MODELS.register_module(force=True)
class ASTDetector(BaseADModel):
    """AST detector with official two-stage and legacy joint modes."""

    VALID_PHASES = {'joint', 'teacher', 'student'}
    VALID_IMAGE_SCORE_MODES = {'mean', 'max'}

    def __init__(
        self,
        backbone='tf_efficientnet_b5',
        extract_layer: int = 35,
        n_feat: int = 304,
        map_len: int = 24,
        n_coupling_blocks: int = 4,
        channels_hidden_teacher: int = 64,
        channels_hidden_student: int = 1024,
        n_student_blocks: int = 4,
        clamp: float = 1.9,
        kernel_sizes: Optional[list[int]] = None,
        teacher_weight: float = 1.0,
        student_weight: float = 1.0,
        img_size: int = 768,
        pos_enc: bool = True,
        pos_enc_dim: int = 32,
        use_gamma: bool = True,
        training_phase: str = 'joint',
        teacher_checkpoint: Optional[str] = None,
        score_map_size: Optional[int] = None,
        image_score_mode: str = 'max',
        loss=None,
        data_preprocessor=None,
        init_cfg=None,
        **kwargs,
    ) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        del loss, kwargs

        if training_phase not in self.VALID_PHASES:
            raise ValueError(
                f'Invalid training_phase {training_phase!r}. '
                f'Expected one of {sorted(self.VALID_PHASES)}.')
        if image_score_mode not in self.VALID_IMAGE_SCORE_MODES:
            raise ValueError(
                f'Invalid image_score_mode {image_score_mode!r}. '
                f'Expected one of {sorted(self.VALID_IMAGE_SCORE_MODES)}.')

        if kernel_sizes is None:
            kernel_sizes = [3] * (n_coupling_blocks - 1) + [5]

        self.n_feat = int(n_feat)
        self.map_len = int(map_len)
        self.img_size = int(img_size)
        self.score_map_size = int(score_map_size or (img_size // 4))
        self.teacher_weight = float(teacher_weight)
        self.student_weight = float(student_weight)
        self.training_phase = training_phase
        self.use_pos_enc = bool(pos_enc)
        self.pos_enc_dim = int(pos_enc_dim if pos_enc else 0)
        self.image_score_mode = image_score_mode

        backbone_name = backbone.get('model_name', str(backbone)) if isinstance(backbone, dict) else str(backbone)
        backbone_pretrained = backbone.get('pretrained', True) if isinstance(backbone, dict) else True
        if 'efficientnet' in backbone_name.lower() or 'effnet' in backbone_name.lower():
            self.feature_extractor = MODELS.build(dict(
                type='EfficientNetLayerExtractor',
                model_name=backbone_name,
                extract_layer=extract_layer,
                pretrained=backbone_pretrained,
            ))
        else:
            self.feature_extractor = MODELS.build(dict(
                type='GenericFeatureExtractor',
                model_name=backbone_name,
                out_index=-1,
                pretrained=backbone_pretrained,
            ))

        self.teacher = TeacherNF(
            n_feat=self.n_feat,
            n_coupling_blocks=n_coupling_blocks,
            channels_hidden=channels_hidden_teacher,
            clamp=clamp,
            kernel_sizes=kernel_sizes,
            pos_enc_dim=self.pos_enc_dim,
            use_gamma=use_gamma,
        )
        self.student = StudentCNN(
            n_feat=self.n_feat,
            channels_hidden=channels_hidden_student,
            n_blocks=n_student_blocks,
            pos_enc_dim=self.pos_enc_dim,
        )

        if self.use_pos_enc:
            pos = positional_encoding_2d(self.pos_enc_dim, self.map_len, self.map_len)
            self.register_buffer('pos_enc_buffer', pos, persistent=False)
        else:
            self.register_buffer('pos_enc_buffer', torch.empty(0), persistent=False)

        if teacher_checkpoint:
            self.load_teacher_checkpoint(teacher_checkpoint)

        self._configure_training_phase()

    def _configure_training_phase(self) -> None:
        teacher_trainable = self.training_phase in {'joint', 'teacher'}
        student_trainable = self.training_phase in {'joint', 'student'}

        for parameter in self.teacher.parameters():
            parameter.requires_grad = teacher_trainable
        for parameter in self.student.parameters():
            parameter.requires_grad = student_trainable
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False

    def load_teacher_checkpoint(self, checkpoint_path: str) -> None:
        """Load teacher weights from a raw teacher checkpoint or mmengine checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f'Teacher checkpoint not found: {checkpoint_path}')

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)

        teacher_state = {
            key[len('teacher.'):]: value
            for key, value in state_dict.items()
            if key.startswith('teacher.')
        }
        if not teacher_state:
            teacher_state = state_dict

        missing, unexpected = self.teacher.load_state_dict(teacher_state, strict=False)
        if unexpected:
            raise RuntimeError(
                f'Unexpected keys while loading AST teacher checkpoint {checkpoint_path}: {unexpected}')
        if missing:
            missing_str = ', '.join(sorted(missing))
            raise RuntimeError(
                f'Missing keys while loading AST teacher checkpoint {checkpoint_path}: {missing_str}')

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract frozen EfficientNet features."""
        with torch.no_grad():
            return self.feature_extractor(x)

    def _get_pos_enc(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if not self.use_pos_enc:
            return None
        return self.pos_enc_buffer.to(device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

    @staticmethod
    def _teacher_loss_map(z_teacher: torch.Tensor, jacobian: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(z_teacher ** 2, dim=1) - jacobian

    @staticmethod
    def _student_loss_map(z_teacher: torch.Tensor, z_student: torch.Tensor) -> torch.Tensor:
        return torch.mean((z_teacher - z_student) ** 2, dim=1)

    def _teacher_forward(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        cond = self._get_pos_enc(feats.shape[0], feats.device, feats.dtype)
        z_teacher, jacobian = self.teacher(feats, cond=cond)
        nll_map = self._teacher_loss_map(z_teacher, jacobian)
        return z_teacher, jacobian, nll_map, cond

    def _build_results(self, data_samples, score_map: torch.Tensor):
        score_map = F.interpolate(
            score_map[:, None],
            size=(self.score_map_size, self.score_map_size),
            mode='bicubic',
            align_corners=False,
        )
        score_map = score_map.squeeze(1)
        img_score_mean = score_map.mean(dim=(1, 2))
        img_score_max = score_map.amax(dim=(1, 2))
        img_scores = img_score_mean if self.image_score_mode == 'mean' else img_score_max
        return build_predict_results(
            data_samples,
            img_scores,
            score_map,
            extra_scores={
                'pred_score_mean': img_score_mean,
                'pred_score_max': img_score_max,
            },
        )

    def forward(self, inputs, data_samples=None, mode: str = 'tensor'):
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        feats = self.extract_features(inputs)

        if mode == 'loss':
            z_teacher, _, teacher_map, cond = self._teacher_forward(feats)

            if self.training_phase == 'teacher':
                loss_teacher = teacher_map.mean()
                return {
                    'loss': self.teacher_weight * loss_teacher,
                    'loss_teacher': loss_teacher.detach(),
                }

            z_student = self.student(feats, cond=cond if self.use_pos_enc else None)
            loss_student = self._student_loss_map(z_teacher.detach(), z_student).mean()

            if self.training_phase == 'student':
                return {
                    'loss': self.student_weight * loss_student,
                    'loss_student': loss_student.detach(),
                }

            loss_teacher = teacher_map.mean()
            loss = self.teacher_weight * loss_teacher + self.student_weight * loss_student
            return {
                'loss': loss,
                'loss_teacher': loss_teacher.detach(),
                'loss_student': loss_student.detach(),
            }

        if mode == 'predict':
            with torch.no_grad():
                z_teacher, _, teacher_map, cond = self._teacher_forward(feats)
                if self.training_phase == 'teacher':
                    return self._build_results(data_samples, teacher_map)

                z_student = self.student(feats, cond=cond if self.use_pos_enc else None)
                student_map = self._student_loss_map(z_teacher, z_student)
                return self._build_results(data_samples, student_map)

        z_teacher, jacobian, _, cond = self._teacher_forward(feats)
        if self.training_phase == 'teacher':
            return z_teacher, jacobian
        z_student = self.student(feats, cond=cond if self.use_pos_enc else None)
        return z_teacher, z_student

    def train(self, mode: bool = True):
        super().train(mode)
        self.feature_extractor.eval()
        if self.training_phase == 'teacher':
            self.student.eval()
        elif self.training_phase == 'student':
            self.teacher.eval()
        return self
