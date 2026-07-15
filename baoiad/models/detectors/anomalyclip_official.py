"""Official AnomalyCLIP wrapper for strict VisA->MVTec alignment.

This detector keeps the official AnomalyCLIP code path intact for the
alignment-critical path:

- official `AnomalyCLIP_lib.load`
- official `AnomalyCLIP_PromptLearner`
- official DPAM attention surgery
- official text/image similarity and anomaly-map computation

The repo-specific differences are intentionally small:

- inputs are restored from ImageNet stats to CLIP stats inside the detector
- outputs are wrapped with ``build_predict_results`` for BaoIAD APIs
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from baoiad.checkpoint import load_checkpoint as load_baoiad_checkpoint
from baoiad.models.base_ad_model import VisionLanguageADModel

from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))


def _resolve_reference_root(reference_root: str) -> str:
    if os.path.isabs(reference_root):
        return reference_root
    return os.path.abspath(os.path.join(_project_root(), reference_root))


def _gaussian_blur_bchw(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    radius = max(int(round(4 * sigma)), 1)
    kernel_size = 2 * radius + 1
    coord = torch.arange(kernel_size, device=x.device, dtype=x.dtype) - radius
    kernel_1d = torch.exp(-(coord ** 2) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(
        x.shape[1], 1, 1, 1)
    return F.conv2d(x, kernel, padding=radius, groups=x.shape[1])


def _tensor_to_scalar_label(value) -> int:
    if torch.is_tensor(value):
        return int(value.detach().cpu().item())
    return int(value)


def _mask_from_sample(sample, image_size: int, device: torch.device) -> torch.Tensor:
    mask = getattr(sample, 'gt_mask', None)
    if mask is None:
        return torch.zeros(1, image_size, image_size, device=device)
    if not torch.is_tensor(mask):
        mask = torch.tensor(mask)
    mask = mask.to(device=device, dtype=torch.float32)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    elif mask.ndim == 3 and mask.shape[0] != 1:
        mask = mask[:1]
    elif mask.ndim != 3:
        raise ValueError(f'Unsupported gt_mask shape: {tuple(mask.shape)}')
    if tuple(mask.shape[-2:]) != (image_size, image_size):
        mask = F.interpolate(
            mask.unsqueeze(0),
            size=(image_size, image_size),
            mode='nearest',
        ).squeeze(0)
    mask = (mask > 0.5).float()
    return mask


def _labels_from_samples(
    data_samples: Optional[Sequence],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if data_samples is None:
        return torch.zeros(batch_size, dtype=torch.long, device=device)
    labels = []
    for sample in data_samples[:batch_size]:
        labels.append(_tensor_to_scalar_label(getattr(sample, 'gt_label', 0)))
    if len(labels) < batch_size:
        labels.extend([0] * (batch_size - len(labels)))
    return torch.tensor(labels[:batch_size], dtype=torch.long, device=device)


def _masks_from_samples(
    data_samples: Optional[Sequence],
    batch_size: int,
    image_size: int,
    device: torch.device,
) -> torch.Tensor:
    if data_samples is None:
        return torch.zeros(
            batch_size, 1, image_size, image_size, dtype=torch.float32,
            device=device)
    masks = [
        _mask_from_sample(sample, image_size=image_size, device=device)
        for sample in data_samples[:batch_size]
    ]
    if len(masks) < batch_size:
        zeros = torch.zeros(1, image_size, image_size, device=device)
        masks.extend([zeros.clone() for _ in range(batch_size - len(masks))])
    return torch.stack(masks[:batch_size], dim=0)


@MODELS.register_module(force=True)
class AnomalyCLIPOfficialDetector(VisionLanguageADModel):
    """Official AnomalyCLIP detector for strict alignment work."""

    def __init__(
        self,
        clip_model: str = 'ViT-L/14@336px',
        image_size: int = 518,
        features_list: Optional[Sequence[int]] = None,
        feature_map_layer: Optional[Sequence[int]] = None,
        prompt_depth: int = 9,
        prompt_length: int = 12,
        prompt_text_length: int = 4,
        temperature: float = 0.07,
        gaussian_sigma: float = 4.0,
        dpam_layer: int = 20,
        official_checkpoint: Optional[str] = None,
        reference_root: str = '.refs/AnomalyCLIP',
        require_official_assets: bool = True,
        freeze_prompt_learner: bool = False,
        enable_train_loss: bool = True,
        download_root: Optional[str] = None,
        data_preprocessor=None,
        init_cfg=None,
        **kwargs,
    ) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        del kwargs

        self.clip_model = clip_model
        self.image_size = image_size
        self.features_list = list(features_list or [24])
        self.feature_map_layer = list(feature_map_layer or [0])
        self.prompt_depth = prompt_depth
        self.prompt_length = prompt_length
        self.prompt_text_length = prompt_text_length
        self.temperature = temperature
        self.gaussian_sigma = gaussian_sigma
        self.dpam_layer = dpam_layer
        self.reference_root = _resolve_reference_root(reference_root)
        self.require_official_assets = require_official_assets
        self.freeze_prompt_learner = freeze_prompt_learner
        self.enable_train_loss = enable_train_loss
        self.download_root = download_root

        self.official_checkpoint = None
        if official_checkpoint is not None:
            if os.path.isabs(official_checkpoint):
                self.official_checkpoint = official_checkpoint
            else:
                self.official_checkpoint = os.path.abspath(
                    os.path.join(_project_root(), official_checkpoint))

        self._validate_assets()
        anomalyclip_lib, prompt_cls, focal_cls, dice_cls = self._import_reference()
        self._anomalyclip_lib = anomalyclip_lib

        from baoiad.runtime import is_offline_mode, require_network

        if is_offline_mode():
            load_globals = getattr(anomalyclip_lib.load, '__globals__', {})
            model_url = load_globals.get('_MODELS', {}).get(self.clip_model)
            download_root = os.path.expanduser(self.download_root or '~/.cache/clip')
            cached_model = os.path.join(download_root, os.path.basename(model_url)) if model_url else None
            if not cached_model or not os.path.isfile(cached_model):
                require_network('download AnomalyCLIP backbone weights', url=model_url)

        design_details = {
            'Prompt_length': self.prompt_length,
            'learnabel_text_embedding_depth': self.prompt_depth,
            'learnabel_text_embedding_length': self.prompt_text_length,
        }
        clip_model_obj, _ = anomalyclip_lib.load(
            self.clip_model,
            device='cpu',
            design_details=design_details,
            download_root=self.download_root,
        )
        clip_model_obj.eval()
        for param in clip_model_obj.parameters():
            param.requires_grad = False
        self.clip = clip_model_obj
        self._resize_visual_positional_embedding(self.image_size)

        self.prompt_learner = prompt_cls(self.clip.to('cpu'), design_details)
        if self.official_checkpoint is not None:
            self._load_prompt_checkpoint(self.official_checkpoint)

        if self.freeze_prompt_learner:
            for param in self.prompt_learner.parameters():
                param.requires_grad = False

        self.clip.visual.DAPM_replace(DPAM_layer=self.dpam_layer)
        self.loss_focal = focal_cls()
        self.loss_dice = dice_cls()

        self.register_buffer(
            '_imagenet_mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            '_imagenet_std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            '_clip_mean',
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            '_clip_std',
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
            persistent=False,
        )

    def _validate_assets(self) -> None:
        root_ok = os.path.isdir(self.reference_root)
        lib_ok = os.path.isdir(os.path.join(self.reference_root, 'AnomalyCLIP_lib'))
        prompt_ok = os.path.isfile(os.path.join(self.reference_root, 'prompt_ensemble.py'))
        if self.require_official_assets and (not root_ok or not lib_ok or not prompt_ok):
            raise FileNotFoundError(
                f'AnomalyCLIP reference assets missing under {self.reference_root}. '
                'Expected official repo with `AnomalyCLIP_lib/` and `prompt_ensemble.py`.')

        if self.official_checkpoint is not None:
            ckpt_ok = os.path.isfile(self.official_checkpoint)
            if self.require_official_assets and not ckpt_ok:
                raise FileNotFoundError(
                    f'Official AnomalyCLIP checkpoint not found: {self.official_checkpoint}')

    def _import_reference(self):
        if self.reference_root not in sys.path:
            sys.path.insert(0, self.reference_root)
        anomalyclip_lib = importlib.import_module('AnomalyCLIP_lib')
        prompt_module = importlib.import_module('prompt_ensemble')
        loss_module = importlib.import_module('loss')
        return (
            anomalyclip_lib,
            prompt_module.AnomalyCLIP_PromptLearner,
            loss_module.FocalLoss,
            loss_module.BinaryDiceLoss,
        )

    def _resize_visual_positional_embedding(self, image_size: int) -> None:
        visual = getattr(self.clip, 'visual', None)
        pos_embedding = getattr(visual, 'positional_embedding', None)
        conv1 = getattr(visual, 'conv1', None)
        if (
            visual is None
            or not isinstance(pos_embedding, nn.Parameter)
            or conv1 is None
        ):
            return

        patch_size = conv1.kernel_size[0] if isinstance(conv1.kernel_size, tuple) else conv1.kernel_size
        if patch_size is None or patch_size <= 0:
            return

        side = int(round((pos_embedding.shape[0] - 1) ** 0.5))
        new_side = int(image_size // patch_size)
        if side == new_side:
            if hasattr(visual, 'input_resolution'):
                visual.input_resolution = image_size
            return

        dtype = pos_embedding.dtype
        device = pos_embedding.device
        new_pos = pos_embedding[1:, :].reshape(-1, side, side, pos_embedding.shape[-1])
        new_pos = new_pos.permute(0, 3, 1, 2).float()
        new_pos = F.interpolate(new_pos, size=(new_side, new_side), mode='bilinear', align_corners=False)
        new_pos = new_pos.reshape(-1, pos_embedding.shape[-1], new_side * new_side).transpose(1, 2)
        resized = torch.cat([pos_embedding[:1, :].float(), new_pos[0]], dim=0).to(device=device, dtype=dtype)
        visual.positional_embedding = nn.Parameter(resized)
        if hasattr(visual, 'input_resolution'):
            visual.input_resolution = image_size

    def _load_prompt_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = load_baoiad_checkpoint(
            checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('prompt_learner', checkpoint)
        self.prompt_learner.load_state_dict(state_dict, strict=True)

    def _normalize_for_clip(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self._imagenet_std + self._imagenet_mean
        return (x - self._clip_mean) / self._clip_std

    def _prepare_inputs(self, inputs) -> torch.Tensor:
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)
        x = self._normalize_for_clip(inputs)
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False,
            )
        return x

    def _encode_text_features(self) -> torch.Tensor:
        prompts, tokenized_prompts, compound_prompts_text = self.prompt_learner(
            cls_id=None)
        text_features = self.clip.encode_text_learn(
            prompts,
            tokenized_prompts,
            compound_prompts_text,
        ).float()
        text_features = torch.stack(
            torch.chunk(text_features, dim=0, chunks=2),
            dim=1,
        )
        return text_features / text_features.norm(dim=-1, keepdim=True)

    def _encode_image(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        image_features, patch_features = self.clip.encode_image(
            x,
            self.features_list,
            DPAM_layer=self.dpam_layer,
        )
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features, patch_features

    def _similarity_maps(
        self,
        patch_features: Iterable[torch.Tensor],
        text_features: torch.Tensor,
    ) -> List[torch.Tensor]:
        maps = []
        for idx, patch_feature in enumerate(patch_features):
            if idx < self.feature_map_layer[0]:
                continue
            patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
            similarity, _ = self._anomalyclip_lib.compute_similarity(
                patch_feature,
                text_features[0],
            )
            similarity_map = self._anomalyclip_lib.get_similarity_map(
                similarity[:, 1:, :],
                self.image_size,
            )
            maps.append(similarity_map)
        return maps

    def train(self, mode: bool = True):
        super().train(mode)
        # The official training loop always keeps the CLIP backbone in eval
        # mode and only optimizes the prompt learner.
        self.clip.eval()
        self.prompt_learner.train(mode and not self.freeze_prompt_learner)
        return self

    def forward(self, inputs, data_samples=None, mode='tensor'):
        x = self._prepare_inputs(inputs)
        batch_size = x.shape[0]

        if mode == 'tensor':
            image_features, patch_features = self._encode_image(x)
            return dict(image_features=image_features, patch_features=patch_features)

        text_features = self._encode_text_features()
        image_features, patch_features = self._encode_image(x)

        if mode == 'loss':
            if not self.enable_train_loss:
                return {'loss': image_features.sum() * 0}

            labels = _labels_from_samples(data_samples, batch_size, x.device)
            gt_masks = _masks_from_samples(
                data_samples,
                batch_size=batch_size,
                image_size=self.image_size,
                device=x.device,
            )
            gt_binary = gt_masks.squeeze(1)

            text_logits = image_features.unsqueeze(1) @ text_features.permute(0, 2, 1)
            text_logits = text_logits[:, 0, ...] / self.temperature
            loss_cls = F.cross_entropy(text_logits, labels)

            similarity_maps = self._similarity_maps(patch_features, text_features)
            if not similarity_maps:
                loss_seg = loss_cls * 0.0
            else:
                seg_loss = image_features.new_tensor(0.0)
                for similarity_map in similarity_maps:
                    similarity_map = similarity_map.permute(0, 3, 1, 2)
                    seg_loss = seg_loss + self.loss_focal(similarity_map, gt_masks)
                    seg_loss = seg_loss + self.loss_dice(
                        similarity_map[:, 1, :, :],
                        gt_binary,
                    )
                    seg_loss = seg_loss + self.loss_dice(
                        similarity_map[:, 0, :, :],
                        1 - gt_binary,
                    )
                loss_seg = 4.0 * seg_loss

            return {
                'loss': loss_cls + loss_seg,
                'loss_cls': loss_cls,
                'loss_seg': loss_seg,
            }

        if mode == 'predict':
            text_probs = image_features.unsqueeze(1) @ text_features.permute(0, 2, 1)
            text_probs = (text_probs / self.temperature).softmax(-1)
            img_scores = text_probs[:, 0, 1]

            similarity_maps = self._similarity_maps(patch_features, text_features)
            if similarity_maps:
                anomaly_maps = []
                for similarity_map in similarity_maps:
                    anomaly_map = (
                        similarity_map[..., 1] + 1 - similarity_map[..., 0]
                    ) / 2.0
                    anomaly_maps.append(anomaly_map)
                score_map = torch.stack(anomaly_maps, dim=0).sum(dim=0).unsqueeze(1)
            else:
                score_map = torch.zeros(
                    batch_size,
                    1,
                    self.image_size,
                    self.image_size,
                    device=x.device,
                    dtype=x.dtype,
                )

            if self.gaussian_sigma > 0:
                score_map = _gaussian_blur_bchw(score_map, self.gaussian_sigma)

            if isinstance(inputs, (list, tuple)):
                out_h, out_w = inputs[0].shape[-2:]
            else:
                out_h, out_w = inputs.shape[-2:]
            if tuple(score_map.shape[-2:]) != (out_h, out_w):
                score_map = F.interpolate(
                    score_map,
                    size=(out_h, out_w),
                    mode='bilinear',
                    align_corners=False,
                )
            return build_predict_results(data_samples, img_scores, score_map)

        raise ValueError(f'Unsupported mode: {mode}')
