"""Generic timm backbone wrapper registered in MODELS registry.

Supports timm models both in ``features_only`` mode for multi-scale feature
extraction and in full-model mode for pre-logits extraction.
"""
import os
from typing import Sequence

import torch
from mmengine.model import BaseModule

from baoiad.checkpoint import load_checkpoint as load_baoiad_checkpoint
from baoiad.registry import MODELS


@MODELS.register_module(force=True)
class TIMMBackbone(BaseModule):
    """Wrapper around timm.create_model registered in MODELS registry.

    Args:
        model_name (str): timm model name. Default 'resnet18'.
        pretrained (bool): Load pretrained weights. Default True.
        features_only (bool): Return intermediate features. Default True.
        out_indices (tuple[int]): Feature level indices to return. Default (1, 2, 3).
        frozen (bool): Freeze all parameters. Default True.
        frozen_names (Sequence[str] | None): Named submodules to freeze while
            leaving the rest of the backbone trainable. Applied only when
            ``frozen`` is False.
        frozen_names_eval (bool): Keep ``frozen_names`` in eval mode during
            training. Set False when the reference freezes parameters but still
            updates BatchNorm running stats in those layers.
    """

    def __init__(self, model_name='resnet18', pretrained=True,
                 features_only=True, out_indices=(1, 2, 3), frozen=True,
                 frozen_names: Sequence[str] | None = None,
                 frozen_names_eval: bool = True,
                 checkpoint_path='', strict=False, allow_legacy_fallback=True, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.features_only = features_only
        self.out_indices = tuple(out_indices)
        self._frozen_module_names = tuple(str(name) for name in (frozen_names or ()))
        self._frozen_module_eval = bool(frozen_names_eval)
        import timm
        effective_checkpoint_path = checkpoint_path
        use_pretrained = pretrained

        if effective_checkpoint_path:
            use_pretrained = False
        elif pretrained:
            cached_checkpoint = self._resolve_cached_pretrained_path(
                timm,
                model_name,
                allow_legacy_fallback=allow_legacy_fallback,
            )
            if cached_checkpoint:
                effective_checkpoint_path = cached_checkpoint
                use_pretrained = False

        model_kwargs = dict(
            model_name=model_name,
            pretrained=use_pretrained,
            features_only=features_only,
        )
        if features_only:
            model_kwargs['out_indices'] = list(out_indices)

        self.net = timm.create_model(**model_kwargs)
        if effective_checkpoint_path:
            if os.path.exists(effective_checkpoint_path):
                self._load_checkpoint(effective_checkpoint_path, strict)
            else:
                print(f'[TIMMBackbone] Warning: checkpoint not found: {effective_checkpoint_path}')
        if hasattr(self.net, 'feature_info'):
            feature_info = self.net.feature_info
            if hasattr(feature_info, 'channels') and hasattr(feature_info, 'reduction'):
                self.out_channels = tuple(feature_info.channels())
                self.reduction = tuple(feature_info.reduction())
            elif isinstance(feature_info, list):
                self.out_channels = tuple(int(item['num_chs']) for item in feature_info)
                self.reduction = tuple(int(item['reduction']) for item in feature_info)
            else:
                self.out_channels = None
                self.reduction = None
        else:
            num_features = getattr(self.net, 'num_features', None)
            self.out_channels = (int(num_features),) if num_features is not None else None
            self.reduction = None
        self.num_features = getattr(self.net, 'num_features', None)

        if frozen:
            self.eval()
            for p in self.parameters():
                p.requires_grad = False
            self._frozen_module_names = tuple(name for name, _ in self.net.named_children())
            self._frozen_module_eval = True
        elif self._frozen_module_names:
            self._freeze_named_modules(self._frozen_module_names)

    def forward(self, x):
        return self.net(x)

    def forward_intermediates(self, x, indices=None):
        if self.features_only:
            feats = self.net(x)
            if indices is None:
                return feats
            index_map = {idx: pos for pos, idx in enumerate(self.out_indices)}
            selected = []
            for idx in indices:
                if idx not in index_map:
                    raise IndexError(f'Requested out index {idx} not available in {self.out_indices}.')
                selected.append(feats[index_map[idx]])
            return selected

        if not hasattr(self.net, 'forward_intermediates'):
            raise AttributeError('Wrapped timm model does not expose forward_intermediates().')

        outputs = self.net.forward_intermediates(x, indices=indices)
        if isinstance(outputs, tuple) and len(outputs) == 2:
            return outputs[1]
        return outputs

    def forward_features(self, x):
        if not hasattr(self.net, 'forward_features'):
            raise AttributeError('Wrapped timm model does not expose forward_features().')
        return self.net.forward_features(x)

    def forward_pre_logits(self, x):
        features = self.forward_features(x)
        if hasattr(self.net, 'forward_head'):
            pre_logits = self.net.forward_head(features, pre_logits=True)
        else:
            pre_logits = features
        if pre_logits.ndim > 2:
            pre_logits = torch.nn.functional.adaptive_avg_pool2d(pre_logits, 1).flatten(1)
        return pre_logits

    @staticmethod
    def _resolve_cached_pretrained_path(
        timm_module,
        model_name: str,
        allow_legacy_fallback: bool = True,
    ) -> str:
        fallback_filenames = {
            'wide_resnet50_2.tv_in1k': 'wide_resnet50_2-95faca4d.pth',
            'wide_resnet50_2.tv2_in1k': 'wide_resnet50_2-9ba9bcbe.pth',
            'tf_efficientnet_b4': 'tf_efficientnet_b4_aa-818f208c.pth',
            'resnet18': 'resnet18-f37072fd.pth',
            'alexnet': 'alexnet-owt-7be5be79.pth',
        }
        if allow_legacy_fallback:
            fallback_filenames['wide_resnet50_2'] = 'wide_resnet50_2-95faca4d.pth'
        fallback_filename = fallback_filenames.get(model_name, '')
        if fallback_filename:
            fallback_candidate = os.path.join(torch.hub.get_dir(), 'checkpoints', fallback_filename)
            if os.path.exists(fallback_candidate):
                return fallback_candidate

        get_pretrained_cfg = getattr(timm_module, 'get_pretrained_cfg', None)
        if get_pretrained_cfg is None:
            return ''
        try:
            pretrained_cfg = get_pretrained_cfg(model_name)
        except Exception:
            return ''

        if pretrained_cfg is None:
            return ''

        if isinstance(pretrained_cfg, dict):
            pretrained_url = pretrained_cfg.get('url', '')
        else:
            pretrained_url = getattr(pretrained_cfg, 'url', '')
        if not pretrained_url:
            return ''

        filename = os.path.basename(pretrained_url)
        if not filename:
            return ''

        candidate = os.path.join(torch.hub.get_dir(), 'checkpoints', filename)
        if os.path.exists(candidate):
            return candidate

        hf_hub_id = (
            pretrained_cfg.get('hf_hub_id', '')
            if isinstance(pretrained_cfg, dict)
            else getattr(pretrained_cfg, 'hf_hub_id', '')
        )
        if not hf_hub_id:
            return ''

        hf_filename = (
            pretrained_cfg.get('hf_hub_filename', '')
            if isinstance(pretrained_cfg, dict)
            else getattr(pretrained_cfg, 'hf_hub_filename', '')
        ) or 'model.safetensors'

        try:
            from huggingface_hub import try_to_load_from_cache

            cached_path = try_to_load_from_cache(hf_hub_id, hf_filename)
            if cached_path and os.path.exists(cached_path):
                return cached_path
        except Exception:
            pass
        return ''

    def _load_checkpoint(self, checkpoint_path: str, strict: bool) -> None:
        checkpoint = load_baoiad_checkpoint(
            checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if isinstance(state_dict, dict) and state_dict:
            first_key = next(iter(state_dict))
            if first_key.startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
        self.net.load_state_dict(state_dict, strict=strict)

    def _freeze_named_modules(self, module_names: Sequence[str]) -> None:
        for module_name in module_names:
            try:
                module = self.net.get_submodule(module_name)
            except AttributeError as exc:
                raise AttributeError(
                    f'TIMMBackbone: failed to find submodule {module_name!r} on {type(self.net).__name__}.'
                ) from exc
            if self._frozen_module_eval:
                module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False

    def train(self, mode=True):
        """Keep frozen layers in eval mode."""
        if mode and not any(p.requires_grad for p in self.parameters()):
            return super().train(False)

        result = super().train(mode)
        if mode and self._frozen_module_eval:
            for module_name in self._frozen_module_names:
                try:
                    self.net.get_submodule(module_name).eval()
                except AttributeError:
                    continue
        return result
