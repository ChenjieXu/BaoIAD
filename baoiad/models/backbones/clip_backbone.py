"""OpenCLIP backbone wrapper registered in MODELS registry.

Wraps open_clip models for CLIP-based anomaly detection methods.
"""
import contextlib
import importlib
import os
import sys

import torch

from mmengine.model import BaseModule

from baoiad.checkpoint import load_checkpoint as load_baoiad_checkpoint
from baoiad.registry import MODELS


def _local_clip_roots():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(this_dir)))
    return [
        os.path.join(project_root, '.refs', 'local reference_ref', 'models'),
        os.path.join(project_root, '.refs', 'local reference', 'models'),
    ]


def _import_local_clip():
    """Import local reference's local CLIP fork when available."""
    for root in _local_clip_roots():
        clip_dir = os.path.join(root, 'clip')
        if not os.path.isdir(clip_dir):
            continue
        if root not in sys.path:
            sys.path.insert(0, root)
        return importlib.import_module('clip')
    return None


def _import_open_clip(prefer_local_reference=False):
    """Import open_clip or fall back to a local reference implementation."""
    if prefer_local_reference:
        local_clip = _import_local_clip()
        if local_clip is not None:
            return local_clip

    try:
        return importlib.import_module('open_clip')
    except ImportError:
        local_clip = _import_local_clip()
        if local_clip is not None:
            return local_clip
        raise


@MODELS.register_module(force=True)
class OpenCLIPBackbone(BaseModule):
    """Wrapper around open_clip for config-driven CLIP model construction.

    Args:
        model_name (str): OpenCLIP model name. Default 'ViT-L-14-336'.
        pretrained (str): Pretrained weights source. Default 'openai'.
        frozen (bool): Freeze all parameters. Default True.
        force_quick_gelu (bool): Force use of QuickGELU activation. Default False.
            Set to True for OpenAI pretrained models (ViT-L-14-336, etc.).
        cache_dir (str | None): Optional cache directory for open_clip downloads.
        pretrained_image_path (str | None): Optional image tower checkpoint path.
        pretrained_text_path (str | None): Optional text tower checkpoint path.
        load_weights (bool): Whether create_model should load pretrained weights.
        prefer_local_reference (bool): Prefer local reference's local CLIP fork when
            available. Required by methods that depend on the modified
            `encode_image(image, out_layers)` API.
    """

    def __init__(self, model_name='ViT-L-14-336', pretrained='openai', frozen=True,
                 force_quick_gelu=False, image_size=None,
                 cache_dir=None, pretrained_image_path=None,
                 pretrained_text_path=None, load_weights=True,
                 hf_endpoint=None,
                 prefer_local_reference=False, init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        @contextlib.contextmanager
        def _maybe_hf_endpoint():
            if not hf_endpoint:
                yield
                return
            old_endpoint = os.environ.get('HF_ENDPOINT')
            os.environ['HF_ENDPOINT'] = hf_endpoint
            try:
                yield
            finally:
                if old_endpoint is None:
                    os.environ.pop('HF_ENDPOINT', None)
                else:
                    os.environ['HF_ENDPOINT'] = old_endpoint

        # Try offline-compatible pretrained resolution before calling open_clip.
        effective_pretrained = self._resolve_cached_pretrained(
            model_name, pretrained, cache_dir)
        # If resolved to a local file, create model without weights then
        # load the checkpoint manually to avoid open_clip's weights_only
        # issue with TorchScript archives in PyTorch >= 2.6.
        manual_load = (effective_pretrained != pretrained
                       and os.path.isfile(effective_pretrained))

        with _maybe_hf_endpoint():
            open_clip = _import_open_clip(prefer_local_reference=prefer_local_reference)
            self.model_name = model_name
            self.image_size = image_size
            self.pretrained = effective_pretrained
            self.cache_dir = cache_dir
            self.hf_endpoint = hf_endpoint
            create_fn = open_clip.create_model_and_transforms

            # When loading a local file, create model without weights then
            # load manually to handle TorchScript archives in PyTorch >= 2.6.
            if manual_load:
                if open_clip.__name__ == 'clip':
                    effective_image_size = image_size
                    if effective_image_size is None:
                        effective_image_size = 336 if '336' in model_name else 224
                    self.model, _, self.preprocess = create_fn(
                        model_name, effective_image_size,
                        pretrained='',
                        force_quick_gelu=force_quick_gelu
                    )
                else:
                    create_kwargs = dict(
                        pretrained='',
                        load_weights=False,
                        force_quick_gelu=force_quick_gelu,
                    )
                    if image_size is not None:
                        create_kwargs['force_image_size'] = image_size
                    self.model, _, self.preprocess = create_fn(
                        model_name, **create_kwargs
                    )
                # Manually load checkpoint with TorchScript handling
                ckpt = load_baoiad_checkpoint(
                    effective_pretrained, map_location='cpu')
                if hasattr(ckpt, 'state_dict'):
                    sd = ckpt.state_dict()
                elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
                    sd = ckpt['state_dict']
                elif isinstance(ckpt, dict):
                    sd = ckpt
                else:
                    sd = {}
                sd = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}
                if sd and next(iter(sd.items()))[0].startswith('module'):
                    sd = {k[7:]: v for k, v in sd.items()}
                # Resize positional embeddings if needed (e.g. different image sizes)
                if hasattr(open_clip, 'model') and hasattr(open_clip.model, 'resize_pos_embed'):
                    open_clip.model.resize_pos_embed(sd, self.model)
                self.model.load_state_dict(sd, strict=False)
            else:
                if open_clip.__name__ == 'clip':
                    effective_image_size = image_size
                    if effective_image_size is None:
                        effective_image_size = 336 if '336' in model_name else 224
                    self.model, _, self.preprocess = create_fn(
                        model_name, effective_image_size,
                        pretrained=effective_pretrained,
                        force_quick_gelu=force_quick_gelu
                    )
                else:
                    create_kwargs = dict(
                        pretrained=effective_pretrained,
                        load_weights=load_weights,
                        force_quick_gelu=force_quick_gelu,
                    )
                    if image_size is not None:
                        create_kwargs['force_image_size'] = image_size
                    if cache_dir is not None:
                        create_kwargs['cache_dir'] = cache_dir
                    if pretrained_image_path is not None:
                        create_kwargs['pretrained_image_path'] = pretrained_image_path
                    if pretrained_text_path is not None:
                        create_kwargs['pretrained_text_path'] = pretrained_text_path
                    self.model, _, self.preprocess = create_fn(
                        model_name, **create_kwargs
                    )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self._tokenize = open_clip.tokenize

        if frozen:
            self.eval()
            for p in self.parameters():
                p.requires_grad = False

    @staticmethod
    def _resolve_cached_pretrained(model_name, pretrained, cache_dir=None):
        """Resolve pretrained tag to a local file path when offline.

        Checks ``~/.cache/clip/`` and common locations for cached CLIP
        weights.  Returns the *pretrained* argument unchanged when the
        tag should be forwarded to open_clip directly (e.g. it is already
        a file path, or we are not in an offline-like situation).
        """

        # If pretrained is already a file path, keep it.
        if pretrained and os.path.isfile(pretrained):
            return pretrained

        # Known mapping: (model_name, pretrained_tag) -> local filename
        _cache_map = {
            ('ViT-L-14-336', 'openai'): 'ViT-L-14-336px.pt',
            ('ViT-L-14', 'openai'): 'ViT-L-14.pt',
            ('ViT-B-16', 'openai'): 'ViT-B-16.pt',
        }

        candidate_filename = _cache_map.get((model_name, pretrained), '')
        if candidate_filename:
            search_dirs = []
            if cache_dir:
                search_dirs.append(cache_dir)
            search_dirs.append(os.path.expanduser('~/.cache/clip'))
            for d in search_dirs:
                candidate = os.path.join(d, candidate_filename)
                if os.path.isfile(candidate):
                    return candidate

        # Fuzzy match: check ~/.cache/clip/ for files matching model_name
        clip_cache = os.path.expanduser('~/.cache/clip')
        if os.path.isdir(clip_cache):
            # Normalize model name for fuzzy matching
            name_norm = model_name.lower().replace('-', '').replace('.', '')
            for f in sorted(os.listdir(clip_cache)):
                if not f.endswith('.pt'):
                    continue
                fname_norm = f.lower().replace('-', '').replace('.', '').replace('px', '')
                if name_norm in fname_norm:
                    return os.path.join(clip_cache, f)

        return pretrained

    @property
    def visual(self):
        """Access the visual encoder (for methods needing internal access)."""
        return self.model.visual

    @property
    def grid_size(self):
        """Grid size of the visual encoder's patch embedding."""
        return self.model.visual.grid_size

    def tokenize(self, texts):
        """Tokenize text inputs using the model's tokenizer."""
        return self._tokenize(texts)

    def encode_image(self, x):
        return self.model.encode_image(x)

    def encode_text(self, t):
        return self.model.encode_text(t)

    def forward(self, x):
        return self.encode_image(x)

    def train(self, mode=True):
        if mode and not any(p.requires_grad for p in self.parameters()):
            return super().train(False)
        return super().train(mode)
