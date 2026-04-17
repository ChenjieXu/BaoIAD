"""AA-CLIP detector aligned to the official two-stage implementation.

Reference:
    Mwxinnn/AA-CLIP, commit ``53db195f230442aa118c246876c94ba1c76139cc``

This wrapper keeps the official model structure and stage semantics:

- ``text``: train the text adapter with the separate DPAM surgery encoder
- ``image``: train the image adapter while freezing the text adapter
- ``inference``: load adapter checkpoints and evaluate
- ``none``: raw CLIP baseline without adapters
"""

from __future__ import annotations

import copy
import importlib
import importlib.util
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS
from baoiad.models.base_ad_model import VisionLanguageADModel
from baoiad.models.losses.dice_loss import BinaryDiceLoss


def _project_root() -> str:
    return os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
        )
    )


def _resolve_reference_root(reference_root: str) -> str:
    if os.path.isabs(reference_root):
        return reference_root
    return os.path.abspath(os.path.join(_project_root(), reference_root))


def _resolve_optional_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(_project_root(), path))


def _purge_conflicting_reference_modules(reference_root: str) -> None:
    prefixes = (
        'dataset',
        'dataset.constants',
        'model',
        'model.clip',
        'model.model',
        'model.openai',
        'model.tokenizer',
        'model.transformer',
        'utils',
    )
    for name in list(sys.modules):
        if not any(name == prefix or name.startswith(prefix + '.') for prefix in prefixes):
            continue
        module = sys.modules.get(name)
        module_file = os.path.abspath(getattr(module, '__file__', '') or '')
        if module_file and module_file.startswith(reference_root):
            continue
        sys.modules.pop(name, None)


def _import_reference_api(reference_root: str):
    _purge_conflicting_reference_modules(reference_root)
    if reference_root not in sys.path:
        sys.path.insert(0, reference_root)
    importlib.invalidate_caches()

    constants_path = os.path.join(reference_root, 'dataset', 'constants.py')
    spec = importlib.util.spec_from_file_location('aaclip_reference_constants', constants_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Failed to load AA-CLIP constants from {constants_path}')
    constants_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants_module)
    tokenizer_module = importlib.import_module('model.tokenizer')
    clip_module = importlib.import_module('model.clip')
    openai_module = importlib.import_module('model.openai')
    model_module = importlib.import_module('model.model')

    model_name = 'ViT-L-14-336'
    current_ckpt = Path(clip_module._MODEL_CKPT_PATHS[model_name])
    if not current_ckpt.exists():
        fallback_candidates = [
            Path(os.environ.get('AACLIP_OPENAI_MODEL_PATH', '')) if os.environ.get('AACLIP_OPENAI_MODEL_PATH') else None,
            Path(os.path.join(os.environ.get('BAOIAD_CACHE_DIR', ''), 'aaclip-cache', 'ViT-L-14-336px.pt')) if os.environ.get('BAOIAD_CACHE_DIR') else None,
            Path(reference_root) / 'model' / 'ViT-L-14-336px.pt',
            Path(_project_root()).parent / 'projects' / 'baseline' / 'AA-CLIP' / 'model' / 'ViT-L-14-336px.pt',
        ]
        for candidate in fallback_candidates:
            if candidate is None:
                continue
            if candidate.exists():
                clip_module._MODEL_CKPT_PATHS[model_name] = candidate
                break

    return dict(
        class_names=constants_module.CLASS_NAMES,
        domains=constants_module.DOMAINS,
        prompts=constants_module.PROMPTS,
        real_names=constants_module.REAL_NAMES,
        tokenize=tokenizer_module.tokenize,
        create_model=clip_module.create_model,
        checkpoint_paths=dict(clip_module._MODEL_CKPT_PATHS),
        load_openai_model=openai_module.load_openai_model,
        resize_pos_embed=model_module.resize_pos_embed,
    )


from baoiad.utils.score_utils import safe_l2_normalize as _safe_l2_normalize


def _gaussian_blur_bchw(
    x: torch.Tensor,
    *,
    sigma: float,
    kernel_size: int,
) -> torch.Tensor:
    radius = kernel_size // 2
    coord = torch.arange(kernel_size, device=x.device, dtype=x.dtype) - radius
    kernel_1d = torch.exp(-(coord ** 2) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(
        x.shape[1], 1, 1, 1
    )
    return F.conv2d(x, kernel, padding=radius, groups=x.shape[1])


class FocalLoss(nn.Module):
    """Official AA-CLIP focal loss."""

    def __init__(
        self,
        apply_nonlin=None,
        alpha=None,
        gamma: float = 2.0,
        balance_index: int = 0,
        smooth: float = 1e-5,
        size_average: bool = True,
    ) -> None:
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

    def forward(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))

        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)

        alpha = self.alpha
        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, tuple)):
            alpha = torch.tensor(alpha, dtype=torch.float32).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha
        else:
            raise TypeError('Unsupported alpha type.')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.long()
        one_hot_key = torch.zeros(target.size(0), num_class, device=logit.device)
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key,
                self.smooth / (num_class - 1),
                1.0 - self.smooth,
            )

        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()
        alpha = alpha[idx].squeeze()
        loss = -1 * alpha * torch.pow(1 - pt, self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()


class SimpleAdapter(nn.Module):
    """Official AA-CLIP adapter block."""

    def __init__(self, c_in: int, c_out: int = 768) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_out, bias=False),
            nn.LeakyReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_dtype = x.dtype
        weight_dtype = self.fc[0].weight.dtype
        if x.dtype != weight_dtype:
            x = x.to(dtype=weight_dtype)
        out = self.fc(x)
        if out.dtype != output_dtype:
            out = out.to(dtype=output_dtype)
        return out


class SimpleProj(nn.Module):
    """Official AA-CLIP projection head."""

    def __init__(self, c_in: int, c_out: int = 768, relu: bool = True) -> None:
        super().__init__()
        if relu:
            self.fc = nn.Sequential(
                nn.Linear(c_in, c_out, bias=False),
                nn.LeakyReLU(),
            )
        else:
            self.fc = nn.Linear(c_in, c_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_dtype = x.dtype
        if isinstance(self.fc, nn.Sequential):
            weight_dtype = self.fc[0].weight.dtype
        else:
            weight_dtype = self.fc.weight.dtype
        if x.dtype != weight_dtype:
            x = x.to(dtype=weight_dtype)
        out = self.fc(x)
        if out.dtype != output_dtype:
            out = out.to(dtype=output_dtype)
        return out


class AdaptedCLIP(nn.Module):
    """Official AA-CLIP adapter wrapper."""

    def __init__(
        self,
        clip_model: nn.Module,
        text_adapt_weight: float = 0.1,
        image_adapt_weight: float = 0.1,
        text_adapt_until: int = 3,
        image_adapt_until: int = 6,
        levels: Sequence[int] = (6, 12, 18, 24),
        relu: bool = True,
    ) -> None:
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.text_adapt_until = int(text_adapt_until)
        self.image_adapt_until = int(image_adapt_until)
        self.t_w = float(text_adapt_weight)
        self.i_w = float(image_adapt_weight)
        self.levels = list(levels)

        layer_adapters = nn.ModuleList(
            [SimpleAdapter(1024, 1024) for _ in range(self.image_adapt_until)]
        )
        seg_proj = nn.ModuleList(
            [SimpleProj(1024, 768, relu) for _ in range(len(self.levels))]
        )
        det_proj = SimpleProj(1024, 768, relu)
        self.image_adapter = nn.ModuleDict(
            dict(
                layer_adapters=layer_adapters,
                seg_proj=seg_proj,
                det_proj=det_proj,
            )
        )
        self.text_adapter = nn.ModuleList(
            [SimpleAdapter(768, 768) for _ in range(self.text_adapt_until)]
            + [SimpleProj(768, 768, relu=True)]
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for parameter in self.image_adapter.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
        for parameter in self.text_adapter.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward_original(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        cls_features, patch_features = self.clipmodel.encode_image(x, [24])
        patch_features = [
            self.clipmodel.visual._global_pool(feature)[1]
            for feature in patch_features
        ]
        patch_features = [self.clipmodel.visual.ln_post(feature) for feature in patch_features]
        patch_features = [feature @ self.clipmodel.visual.proj for feature in patch_features]
        return patch_features, cls_features

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        x = self.image_encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [
                self.image_encoder.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0],
                    1,
                    x.shape[-1],
                    dtype=x.dtype,
                    device=x.device,
                ),
                x,
            ],
            dim=1,
        )
        x = x + self.image_encoder.positional_embedding.to(x.dtype)
        x = self.image_encoder.patch_dropout(x)
        x = self.image_encoder.ln_pre(x)
        x = x.permute(1, 0, 2)

        tokens = []
        for index in range(24):
            x, _ = self.image_encoder.transformer.resblocks[index](x, attn_mask=None)
            if index < self.image_adapt_until:
                adapt_out = self.image_adapter['layer_adapters'][index](x)
                adapt_out = adapt_out * x.norm(dim=-1, keepdim=True) / adapt_out.norm(
                    dim=-1,
                    keepdim=True,
                )
                x = self.i_w * adapt_out + (1 - self.i_w) * x
            if index + 1 in self.levels:
                tokens.append(x[1:, :, :])

        x = x.permute(1, 0, 2)
        tokens = [token.permute(1, 0, 2) for token in tokens]
        tokens = [self.image_encoder.ln_post(token) for token in tokens]
        seg_tokens = [
            self.image_adapter['seg_proj'][i](token)
            for i, token in enumerate(tokens)
        ]
        seg_tokens = [F.normalize(token, dim=-1) for token in seg_tokens]
        det_token = self.image_adapter['det_proj'](tokens[-1])
        det_token = F.normalize(det_token, dim=-1).mean(1)
        return seg_tokens, det_token

    def encode_text(self, text: torch.Tensor, adapt_text: bool = True) -> torch.Tensor:
        if not adapt_text:
            return self.clipmodel.encode_text(text)

        cast_dtype = self.clipmodel.transformer.get_cast_dtype()
        x = self.clipmodel.token_embedding(text).to(cast_dtype)
        x = x + self.clipmodel.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)

        for index in range(12):
            x, _ = self.clipmodel.transformer.resblocks[index](
                x,
                attn_mask=self.clipmodel.attn_mask,
            )
            if index < self.text_adapt_until:
                adapt_out = self.text_adapter[index](x)
                adapt_out = adapt_out * x.norm(dim=-1, keepdim=True) / adapt_out.norm(
                    dim=-1,
                    keepdim=True,
                )
                x = self.t_w * adapt_out + (1 - self.t_w) * x

        x = x.permute(1, 0, 2)
        x = self.clipmodel.ln_final(x)
        x = self.text_adapter[-1](x[torch.arange(x.shape[0]), text.argmax(dim=-1)])
        return x


@MODELS.register_module()
class AACLIPDetector(VisionLanguageADModel):
    """AA-CLIP detector with official stage semantics."""

    def __init__(
        self,
        clip_model: str = 'ViT-L-14-336',
        pretrained: str = 'openai',
        image_size: int = 518,
        training_stage: str = 'inference',
        reference_root: str = '.refs/AA-CLIP',
        text_adapter_ckpt: Optional[str] = None,
        image_adapter_ckpt: Optional[str] = None,
        model_name: Optional[str] = None,
        text_norm_weight: float = 0.1,
        text_adapt_weight: float = 0.1,
        image_adapt_weight: float = 0.1,
        text_adapt_until: int = 3,
        image_adapt_until: int = 6,
        surgery_until_layer: int = 20,
        levels: Sequence[int] = (6, 12, 18, 24),
        relu: bool = False,
        default_dataset_name: str = 'MVTec',
        temperature: float = 0.07,
        use_fast_build: bool = False,
        require_official_assets: bool = True,
        data_preprocessor=None,
        init_cfg=None,
        **kwargs,
    ) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        del kwargs

        if pretrained.lower() != 'openai':
            raise ValueError('AACLIPDetector currently only supports `pretrained="openai"`.')

        self.model_name = model_name or clip_model
        self.image_size = int(image_size)
        self.training_stage = str(training_stage)
        self.reference_root = _resolve_reference_root(reference_root)
        self.text_adapter_ckpt = _resolve_optional_path(text_adapter_ckpt)
        self.image_adapter_ckpt = _resolve_optional_path(image_adapter_ckpt)
        self.text_norm_weight = float(text_norm_weight)
        self.text_adapt_weight = float(text_adapt_weight)
        self.image_adapt_weight = float(image_adapt_weight)
        self.text_adapt_until = int(text_adapt_until)
        self.image_adapt_until = int(image_adapt_until)
        self.surgery_until_layer = int(surgery_until_layer)
        self.levels = list(levels)
        self.relu = bool(relu)
        self.default_dataset_name = str(default_dataset_name)
        self.temperature = float(temperature)
        self.use_fast_build = bool(use_fast_build)
        self.require_official_assets = bool(require_official_assets)

        if self.require_official_assets and not os.path.isdir(self.reference_root):
            raise FileNotFoundError(
                f'AA-CLIP reference assets missing under {self.reference_root}.'
            )

        api = _import_reference_api(self.reference_root)
        self._class_names = dict(api['class_names'])
        self._domains = dict(api['domains'])
        self._prompts = dict(api['prompts'])
        self._real_names = dict(api['real_names'])
        self._tokenize = api['tokenize']
        self._create_model = api['create_model']
        self._checkpoint_paths = dict(api.get('checkpoint_paths', {}))
        self._load_openai_model = api.get('load_openai_model', None)
        self._resize_pos_embed = api.get('resize_pos_embed', None)

        raw_clip_model = self._build_clip_model()
        raw_clip_model.eval()
        for parameter in raw_clip_model.parameters():
            parameter.requires_grad = False
        self.clip_model = raw_clip_model

        if self.training_stage == 'text':
            surgery_model = copy.deepcopy(raw_clip_model)
            surgery_model.eval()
            surgery_model.visual.DAPM_replace(DPAM_layer=self.surgery_until_layer)
            for parameter in surgery_model.parameters():
                parameter.requires_grad = False
            self.clip_surgery = surgery_model
        else:
            self.clip_surgery = None

        self.adapted_model = AdaptedCLIP(
            clip_model=self.clip_model,
            text_adapt_weight=self.text_adapt_weight,
            image_adapt_weight=self.image_adapt_weight,
            text_adapt_until=self.text_adapt_until,
            image_adapt_until=self.image_adapt_until,
            levels=self.levels,
            relu=self.relu,
        )
        self.loss_focal = FocalLoss()
        self.loss_dice = BinaryDiceLoss()

        if self.text_adapter_ckpt:
            self._load_adapter_weights(self.text_adapter_ckpt, target='text')
        if self.image_adapter_ckpt:
            self._load_adapter_weights(self.image_adapter_ckpt, target='image')

        self._freeze_non_trainable_params()

    def _build_clip_model(self) -> nn.Module:
        checkpoint_path = self._checkpoint_paths.get(self.model_name)
        build_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if (
            not self.use_fast_build
            or os.environ.get('AACLIP_USE_FAST_BUILD', '').lower() in {'0', 'false', 'no'}
        ):
            return self._create_model(
                model_name=self.model_name,
                img_size=self.image_size,
                pretrained='openai',
                device=build_device,
                require_pretrained=True,
            )

        if (
            self._load_openai_model is None
            or self._resize_pos_embed is None
            or checkpoint_path is None
        ):
            return self._create_model(
                model_name=self.model_name,
                img_size=self.image_size,
                pretrained='openai',
                device='cpu',
                require_pretrained=True,
            )

        build_precision = None if build_device == 'cuda' else 'fp32'
        model = self._load_openai_model(
            str(checkpoint_path),
            precision=build_precision,
            device=build_device,
            jit=True,
        )
        visual = model.visual
        patch_size = getattr(visual, 'patch_size', None)
        grid_size = getattr(visual, 'grid_size', None)
        if patch_size is None or grid_size is None:
            return model

        if isinstance(patch_size, tuple):
            patch_h, patch_w = int(patch_size[0]), int(patch_size[1])
        else:
            patch_h = patch_w = int(patch_size)
        new_grid = (self.image_size // patch_h, self.image_size // patch_w)
        if tuple(int(v) for v in grid_size) != new_grid:
            state_dict = {
                'visual.positional_embedding': visual.positional_embedding.detach().clone()
            }
            visual.image_size = (self.image_size, self.image_size)
            visual.grid_size = new_grid
            self._resize_pos_embed(state_dict, model)
            with torch.no_grad():
                visual.positional_embedding = nn.Parameter(
                    state_dict['visual.positional_embedding']
                )

        model.visual.image_mean = (0.48145466, 0.4578275, 0.40821073)
        model.visual.image_std = (0.26862954, 0.26130258, 0.27577711)
        return model

    def _freeze_non_trainable_params(self) -> None:
        for parameter in self.clip_model.parameters():
            parameter.requires_grad = False
        if self.clip_surgery is not None:
            for parameter in self.clip_surgery.parameters():
                parameter.requires_grad = False

        for parameter in self.adapted_model.text_adapter.parameters():
            parameter.requires_grad = self.training_stage == 'text'
        for parameter in self.adapted_model.image_adapter.parameters():
            parameter.requires_grad = self.training_stage == 'image'

    def train(self, mode: bool = True):
        super().train(mode)
        self.adapted_model.train(mode)
        # The official pipeline always keeps the CLIP backbones in eval mode so
        # patch dropout and other training-time behaviors stay disabled.
        self.clip_model.eval()
        self.adapted_model.clipmodel.eval()
        self.adapted_model.image_encoder.eval()
        if self.clip_surgery is not None:
            self.clip_surgery.eval()
        if self.training_stage != 'text':
            self.adapted_model.text_adapter.eval()
        if self.training_stage != 'image':
            self.adapted_model.image_adapter.eval()
        return self

    def _resolve_dataset_name(self, class_name: str) -> str:
        if class_name == 'object':
            return self.default_dataset_name
        for dataset_name, class_names in self._class_names.items():
            if class_name in class_names:
                return dataset_name
        return self.default_dataset_name

    def _get_real_name(self, dataset_name: str, class_name: str) -> str:
        if class_name == 'object':
            return class_name
        return self._real_names[dataset_name][class_name]

    def _get_single_class_text_embedding(
        self,
        class_name: str,
        *,
        adapt_text: bool,
        device: torch.device,
    ) -> torch.Tensor:
        dataset_name = self._resolve_dataset_name(class_name)
        real_name = self._get_real_name(dataset_name, class_name)
        prompt_normal = self._prompts['prompt_normal']
        prompt_abnormal = self._prompts['prompt_abnormal']
        prompt_templates = self._prompts['prompt_templates']

        text_features = []
        for prompt_state in (prompt_normal, prompt_abnormal):
            prompted_state = [state.format(real_name) for state in prompt_state]
            prompted_sentence = []
            for sentence in prompted_state:
                for template in prompt_templates:
                    prompted_sentence.append(template.format(sentence))
            tokenized = self._tokenize(prompted_sentence).to(device)
            class_embeddings = self.adapted_model.encode_text(
                tokenized,
                adapt_text=adapt_text,
            )
            class_embeddings = _safe_l2_normalize(class_embeddings, dim=-1)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding = _safe_l2_normalize(class_embedding, dim=0)
            text_features.append(class_embedding)
        return torch.stack(text_features, dim=1).to(device)

    def _get_batch_text_embeddings(
        self,
        class_names: Sequence[str],
        *,
        adapt_text: bool,
        device: torch.device,
    ) -> torch.Tensor:
        text_feature_dict: Dict[str, torch.Tensor] = {}
        for class_name in sorted(set(class_names)):
            text_feature_dict[class_name] = self._get_single_class_text_embedding(
                class_name,
                adapt_text=adapt_text,
                device=device,
            )
        return torch.stack(
            [text_feature_dict[class_name] for class_name in class_names],
            dim=0,
        )

    def _prepare_inputs(self, inputs) -> torch.Tensor:
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)
        if inputs.shape[-2:] != (self.image_size, self.image_size):
            inputs = F.interpolate(
                inputs,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False,
            )
        target_dtype = self.adapted_model.image_encoder.conv1.weight.dtype
        if inputs.dtype != target_dtype:
            inputs = inputs.to(dtype=target_dtype)
        return inputs

    def _labels_from_samples(
        self,
        data_samples: Optional[Sequence],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        labels = []
        if data_samples is not None:
            for sample in data_samples[:batch_size]:
                labels.append(int(getattr(sample, 'gt_label', 0)))
        if len(labels) < batch_size:
            labels.extend([0] * (batch_size - len(labels)))
        return torch.tensor(labels[:batch_size], dtype=torch.long, device=device)

    def _masks_from_samples(
        self,
        data_samples: Optional[Sequence],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        masks = []
        if data_samples is not None:
            for sample in data_samples[:batch_size]:
                mask = getattr(sample, 'gt_mask', None)
                if mask is None:
                    mask = torch.zeros(self.image_size, self.image_size)
                if not torch.is_tensor(mask):
                    mask = torch.tensor(mask, dtype=torch.float32)
                mask = mask.to(device=device, dtype=torch.float32)
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0)
                elif mask.ndim == 3 and mask.shape[0] != 1:
                    mask = mask[:1]
                if mask.shape[-2:] != (self.image_size, self.image_size):
                    mask = F.interpolate(
                        mask.unsqueeze(0),
                        size=(self.image_size, self.image_size),
                        mode='nearest',
                    ).squeeze(0)
                mask = (mask > 0).float()
                masks.append(mask)
        while len(masks) < batch_size:
            masks.append(torch.zeros(1, self.image_size, self.image_size, device=device))
        return torch.stack(masks[:batch_size], dim=0)

    def _class_names_from_samples(
        self,
        data_samples: Optional[Sequence],
        batch_size: int,
    ) -> List[str]:
        class_names = []
        if data_samples is not None:
            for sample in data_samples[:batch_size]:
                class_names.append(str(getattr(sample, 'cls_name', 'object')))
        if len(class_names) < batch_size:
            class_names.extend(['object'] * (batch_size - len(class_names)))
        return class_names[:batch_size]

    def _calculate_similarity_map(
        self,
        patch_features: torch.Tensor,
        text_features: torch.Tensor,
        *,
        test: bool,
        dataset_name: str,
    ) -> torch.Tensor:
        patch_anomaly_scores = 100.0 * torch.matmul(patch_features, text_features)
        batch_size, num_tokens, num_channels = patch_anomaly_scores.shape
        spatial_size = int(math.sqrt(num_tokens))
        patch_pred = patch_anomaly_scores.permute(0, 2, 1).view(
            batch_size,
            num_channels,
            spatial_size,
            spatial_size,
        )
        if test:
            sigma = 1.0 if self._domains[dataset_name] == 'Industrial' else 1.5
            kernel_size = 7 if self._domains[dataset_name] == 'Industrial' else 9
            patch_pred = (patch_pred[:, 1] + 1 - patch_pred[:, 0]) / 2
            patch_pred = _gaussian_blur_bchw(
                patch_pred.unsqueeze(1),
                sigma=sigma,
                kernel_size=kernel_size,
            )
        patch_preds = F.interpolate(
            patch_pred,
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=True,
        )
        if not test and num_channels > 1:
            patch_preds = torch.softmax(patch_preds, dim=1)
        return patch_preds

    def _calculate_seg_loss(
        self,
        patch_preds: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        loss = self.loss_focal(patch_preds, mask)
        loss = loss + self.loss_dice(patch_preds[:, 0, :, :], 1 - mask)
        loss = loss + self.loss_dice(patch_preds[:, 1, :, :], mask)
        return loss

    def _forward_text_stage_loss(
        self,
        x: torch.Tensor,
        data_samples: Sequence,
    ) -> Dict[str, torch.Tensor]:
        if self.clip_surgery is None:
            raise RuntimeError('Text stage requires the surgery encoder.')

        batch_size = x.shape[0]
        device = x.device
        class_names = self._class_names_from_samples(data_samples, batch_size)
        masks = self._masks_from_samples(data_samples, batch_size, device)

        epoch_text_feature = self._get_batch_text_embeddings(
            class_names,
            adapt_text=True,
            device=device,
        )

        with torch.no_grad():
            _, patch_features = self.clip_surgery.encode_image(x, [6, 12, 18, 24])
            cls_token, _ = self.adapted_model.clipmodel.encode_image(x, [])
            cls_token = _safe_l2_normalize(cls_token, dim=-1)
            patch_features = [
                self.clip_surgery.visual.ln_post(feature[:, 1:, :])
                for feature in patch_features
            ]
            patch_features = [
                feature @ self.clip_surgery.visual.proj
                for feature in patch_features
            ]
            patch_features = [
                _safe_l2_normalize(feature, dim=-1)
                for feature in patch_features
            ]
            patch_features = [
                feature + cls_token.unsqueeze(1)
                for feature in patch_features
            ]

        loss = None
        loss_seg = None
        loss_orth = None
        dataset_name = self._resolve_dataset_name(class_names[0])
        for feature in patch_features:
            patch_preds = self._calculate_similarity_map(
                feature,
                epoch_text_feature,
                test=False,
                dataset_name=dataset_name,
            )
            loss_seg = self._calculate_seg_loss(patch_preds, masks)
            loss_orth = (
                (epoch_text_feature[:, :, 0] * epoch_text_feature[:, :, 1])
                .sum(1)
                .mean()
            ) ** 2
            loss = loss_seg + loss_orth * self.text_norm_weight

        assert loss is not None
        return dict(loss=loss, loss_seg=loss_seg, loss_orth=loss_orth)

    def _forward_image_stage_loss(
        self,
        x: torch.Tensor,
        data_samples: Sequence,
    ) -> Dict[str, torch.Tensor]:
        batch_size = x.shape[0]
        device = x.device
        class_names = self._class_names_from_samples(data_samples, batch_size)
        masks = self._masks_from_samples(data_samples, batch_size, device)
        labels = self._labels_from_samples(data_samples, batch_size, device)

        with torch.no_grad():
            epoch_text_feature = self._get_batch_text_embeddings(
                class_names,
                adapt_text=self.text_adapter_ckpt is not None,
                device=device,
            )

        patch_features, det_feature = self.adapted_model(x)
        det_feature = det_feature.unsqueeze(1)
        cls_preds = torch.matmul(det_feature, epoch_text_feature)[:, 0]
        loss_cls = F.cross_entropy(cls_preds, labels)

        loss = loss_cls
        loss_seg = None
        dataset_name = self._resolve_dataset_name(class_names[0])
        for feature in patch_features:
            patch_preds = self._calculate_similarity_map(
                feature,
                epoch_text_feature,
                test=False,
                dataset_name=dataset_name,
            )
            current_seg_loss = self._calculate_seg_loss(patch_preds, masks)
            loss = loss + current_seg_loss
            loss_seg = current_seg_loss if loss_seg is None else loss_seg + current_seg_loss

        return dict(loss=loss, loss_cls=loss_cls, loss_seg=loss_seg)

    def _predict_with_model(
        self,
        model_outputs: Tuple[List[torch.Tensor], torch.Tensor],
        *,
        class_names: Sequence[str],
        adapt_text: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        patch_features, det_feature = model_outputs
        device = det_feature.device
        epoch_text_feature = self._get_batch_text_embeddings(
            class_names,
            adapt_text=adapt_text,
            device=device,
        )

        det_scores = (torch.matmul(det_feature.unsqueeze(1), epoch_text_feature)[:, 0, 1] + 1) / 2
        dataset_name = self._resolve_dataset_name(class_names[0])
        patch_preds = []
        for feature in patch_features:
            patch_pred = self._calculate_similarity_map(
                feature,
                epoch_text_feature,
                test=True,
                dataset_name=dataset_name,
            )
            patch_preds.append(patch_pred)
        anomaly_map = torch.cat(patch_preds, dim=1).sum(1)
        return det_scores, anomaly_map

    def _forward_predict(self, x: torch.Tensor, data_samples: Optional[Sequence]):
        class_names = self._class_names_from_samples(data_samples, x.shape[0])
        if self.training_stage == 'none':
            model_outputs = self.adapted_model.forward_original(x)
            det_scores, anomaly_map = self._predict_with_model(
                model_outputs,
                class_names=class_names,
                adapt_text=False,
            )
        else:
            model_outputs = self.adapted_model(x)
            det_scores, anomaly_map = self._predict_with_model(
                model_outputs,
                class_names=class_names,
                adapt_text=self.text_adapter_ckpt is not None,
            )
        return build_predict_results(
            data_samples,
            img_scores=det_scores,
            score_maps=anomaly_map,
        )

    def _strip_prefix(
        self,
        state_dict: Dict[str, torch.Tensor],
        prefix: str,
    ) -> Dict[str, torch.Tensor]:
        return {
            key[len(prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }

    def _extract_state_dict(self, checkpoint) -> Dict[str, torch.Tensor]:
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            return checkpoint['state_dict']
        if isinstance(checkpoint, dict):
            return checkpoint
        raise TypeError('Unsupported checkpoint format.')

    def _remap_linear_keys(
        self,
        state_dict: Dict[str, torch.Tensor],
        target_state_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if not state_dict:
            return state_dict
        remapped = dict(state_dict)
        for key in list(state_dict):
            if '.fc.weight' in key:
                seq_key = key.replace('.fc.weight', '.fc.0.weight')
                if seq_key in target_state_dict and seq_key not in remapped:
                    remapped[seq_key] = remapped.pop(key)
            elif '.fc.0.weight' in key:
                linear_key = key.replace('.fc.0.weight', '.fc.weight')
                if linear_key in target_state_dict and linear_key not in remapped:
                    remapped[linear_key] = remapped.pop(key)
        return remapped

    def _load_adapter_weights(self, checkpoint_path: str, *, target: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = self._extract_state_dict(checkpoint)

        if target == 'text':
            module = self.adapted_model.text_adapter
            candidates = []
            if isinstance(checkpoint, dict) and 'text_adapter' in checkpoint:
                candidates.append(checkpoint['text_adapter'])
            candidates.extend([
                self._strip_prefix(state_dict, 'text_adapter.'),
                self._strip_prefix(state_dict, 'adapted_model.text_adapter.'),
                self._strip_prefix(state_dict, 'model.text_adapter.'),
                self._strip_prefix(state_dict, 'module.text_adapter.'),
                self._strip_prefix(state_dict, 'module.adapted_model.text_adapter.'),
            ])
        elif target == 'image':
            module = self.adapted_model.image_adapter
            candidates = []
            if isinstance(checkpoint, dict) and 'image_adapter' in checkpoint:
                candidates.append(checkpoint['image_adapter'])
            candidates.extend([
                self._strip_prefix(state_dict, 'image_adapter.'),
                self._strip_prefix(state_dict, 'adapted_model.image_adapter.'),
                self._strip_prefix(state_dict, 'model.image_adapter.'),
                self._strip_prefix(state_dict, 'module.image_adapter.'),
                self._strip_prefix(state_dict, 'module.adapted_model.image_adapter.'),
            ])
        else:
            raise ValueError(f'Unsupported adapter target: {target}')

        expected = module.state_dict()
        for candidate in candidates:
            if not candidate:
                continue
            candidate = self._remap_linear_keys(candidate, expected)
            try:
                module.load_state_dict(candidate, strict=True)
                return
            except RuntimeError:
                continue

        raise RuntimeError(
            f'Failed to load {target} adapter weights from checkpoint: {checkpoint_path}'
        )

    def forward(self, inputs, data_samples=None, mode: str = 'tensor'):
        x = self._prepare_inputs(inputs)
        if mode == 'tensor':
            if self.training_stage == 'none':
                patch_features, det_feature = self.adapted_model.forward_original(x)
            else:
                patch_features, det_feature = self.adapted_model(x)
            return dict(patch_features=patch_features, det_feature=det_feature)

        if mode == 'loss':
            if self.training_stage == 'text':
                return self._forward_text_stage_loss(x, data_samples)
            if self.training_stage == 'image':
                return self._forward_image_stage_loss(x, data_samples)
            return {'loss': x.sum() * 0}

        if mode == 'predict':
            return self._forward_predict(x, data_samples)

        raise ValueError(f'Unsupported forward mode: {mode}')
