"""MuSc CLIP Backbone with Multi-layer Feature Extraction.

This module provides CLIP backbone support for MuSc using either:
1. The modified open_clip from ref/MuSc (preferred for exact alignment)
2. Standard open_clip_torch with a wrapper for multi-layer extraction (fallback)

The fallback approach extracts intermediate features by running hooks during forward.
"""
import sys
import os
import math
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from baoiad.registry import MODELS


def _get_musc_ref_path():
    """Get the path to the local MuSc reference repo."""
    # Navigate from this file to project root, then prefer .refs/MuSc.
    # this_dir = baoiad/models/backbones/
    # project_root = BaoIAD/
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(this_dir)))
    for rel_path in (os.path.join('.refs', 'MuSc'), os.path.join('ref', 'MuSc')):
        candidate = os.path.join(project_root, rel_path)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(project_root, '.refs', 'MuSc')


def _try_import_musc_open_clip():
    """Try to import the modified open_clip from ref/MuSc.

    Returns:
        Tuple of (open_clip module, True, None) if successful,
        (None, False, error_message) otherwise.
    """
    ref_path = _get_musc_ref_path()
    if not os.path.exists(ref_path):
        return None, False, f'MuSc reference repo not found at {ref_path}'

    # Add to path if not already there
    if ref_path not in sys.path:
        sys.path.insert(0, ref_path)

    try:
        import models.backbone.open_clip as open_clip
        return open_clip, True, None
    except ImportError as exc:
        return None, False, f'Failed to import MuSc reference open_clip from {ref_path}: {exc}'


class MultiLayerHook:
    """Hook to extract intermediate layer features from ViT."""

    def __init__(self, layer_indices: List[int]):
        """
        Args:
            layer_indices: List of 1-indexed layer numbers to extract.
        """
        self.layer_indices = [i - 1 for i in layer_indices]  # Convert to 0-indexed
        self.features: Dict[int, torch.Tensor] = {}

    def __call__(self, module, inp, out):
        """Hook function to capture layer output."""
        # Find which layer this is
        for idx in self.layer_indices:
            if hasattr(module, '_layer_idx') and module._layer_idx == idx:
                self.features[idx] = out[0] if isinstance(out, tuple) else out
                break


@MODELS.register_module(force=True)
class MuScCLIPBackbone(BaseModule):
    """CLIP backbone with multi-layer feature extraction for MuSc.

    Uses either:
    1. Modified open_clip from ref/MuSc (if available) - supports out_layers parameter
    2. Standard open_clip_torch with hooks for multi-layer extraction (fallback)

    Args:
        model_name: CLIP model name (e.g., 'ViT-L-14-336').
        pretrained: Pretrained weights source (e.g., 'openai', 'laion400m_e31').
        feature_layers: Reference config layer ids. For CLIP MuSc this matches the
            YAML values from the official repo; runtime extraction applies
            `feature_layer_offset` before querying the transformer blocks.
        image_size: Input image size. Default 518.
        frozen: Whether to freeze backbone weights. Default True.
        use_ref_open_clip: Whether to prefer ref/MuSc open_clip. Default True.
            Falls back to standard open_clip_torch if not available.
        require_ref_open_clip: Whether to fail fast when the modified MuSc
            `open_clip` cannot be imported.
        feature_layer_offset: Offset applied to the reference config layer ids
            before CLIP extraction. The official MuSc CLIP path uses `+1`.
    """

    def __init__(
        self,
        model_name: str = 'ViT-L-14-336',
        pretrained: str = 'openai',
        feature_layers: List[int] = None,
        image_size: int = 518,
        frozen: bool = True,
        use_ref_open_clip: bool = True,
        require_ref_open_clip: bool = False,
        feature_layer_offset: int = 1,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)

        if feature_layers is None:
            feature_layers = [5, 11, 17, 23]

        self.model_name = model_name
        self.image_size = image_size
        self.feature_layers = feature_layers
        self.feature_layer_offset = feature_layer_offset
        self.resolved_feature_layers = self._resolve_feature_layers(feature_layers)
        self.frozen = frozen
        self.reference_repo_path = _get_musc_ref_path()

        # Try to use modified open_clip from reference first
        self._use_ref_open_clip = False
        ref_import_error = None
        if use_ref_open_clip:
            open_clip, success, ref_import_error = _try_import_musc_open_clip()
            if success:
                self._use_ref_open_clip = True
                self._build_ref_open_clip(open_clip, model_name, image_size, pretrained)
        if require_ref_open_clip and not self._use_ref_open_clip:
            message = ref_import_error or (
                f'MuSc reference open_clip is required but unavailable at {self.reference_repo_path}.'
            )
            raise RuntimeError(message)

        # Fall back to standard open_clip_torch
        if not self._use_ref_open_clip:
            self._build_standard_open_clip(model_name, image_size, pretrained)

        # Get feature dimension from model config
        if 'ViT-L-14' in model_name:
            self.embed_dim = 768  # output projection dim
            self.width = 1024  # internal width
            self.num_layers = 24
        elif 'ViT-B-16' in model_name or 'ViT-B-32' in model_name:
            self.embed_dim = 512
            self.width = 768
            self.num_layers = 12
        else:
            # Default to ViT-L-14 dims
            self.embed_dim = 768
            self.width = 1024
            self.num_layers = 24

        if frozen:
            self._freeze_backbone()

    def _resolve_feature_layers(self, feature_layers: List[int]) -> List[int]:
        """Map reference config layer ids to runtime extraction layer ids."""
        return [int(layer) + self.feature_layer_offset for layer in feature_layers]

    def _build_ref_open_clip(self, open_clip, model_name, image_size, pretrained):
        """Build model using modified open_clip from ref/MuSc."""
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, image_size, pretrained=pretrained
        )
        self._encode_image = self._encode_image_ref

    def _build_standard_open_clip(self, model_name, image_size, pretrained):
        """Build model using standard open_clip_torch with hooks."""
        import open_clip

        # Map pretrained names
        if pretrained == 'openai':
            # For ViT-L-14-336, openai weights are available
            pretrained_name = 'openai'
        else:
            pretrained_name = pretrained

        # Get original image size from model name
        original_size = 336 if '336' in model_name else 224

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained_name
        )
        self._encode_image = self._encode_image_standard
        self._original_image_size = original_size

        # If image_size differs from original, we need to resize positional embeddings
        if image_size != original_size:
            self._resize_pos_embed(image_size)

        # Register hooks for multi-layer extraction
        self._setup_hooks()

    def _resize_pos_embed(self, new_size):
        """Resize positional embeddings to support different image sizes."""
        visual = self.model.visual

        # Get original positional embedding
        pos_embed = visual.positional_embedding
        if pos_embed is None:
            return

        # pos_embed shape: (num_patches + 1, embed_dim)
        # +1 for CLS token
        cls_token = pos_embed[0:1, :]
        patch_pos_embed = pos_embed[1:, :]

        # Calculate original grid size
        original_grid = int(math.sqrt(patch_pos_embed.shape[0]))
        patch_size = visual.patch_size
        if isinstance(patch_size, tuple):
            patch_size = patch_size[0]
        new_grid = new_size // patch_size

        if original_grid == new_grid:
            return

        # Reshape to 2D spatial format
        embed_dim = pos_embed.shape[1]
        patch_pos_embed = patch_pos_embed.reshape(1, original_grid, original_grid, embed_dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)  # (1, C, H, W)

        # Interpolate to new size
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(new_grid, new_grid),
            mode='bicubic',
            align_corners=False
        )

        # Reshape back
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1)  # (1, H, W, C)
        patch_pos_embed = patch_pos_embed.reshape(1, new_grid * new_grid, embed_dim)
        patch_pos_embed = patch_pos_embed.squeeze(0)  # (new_grid*new_grid, C)

        # Concatenate with CLS token
        new_pos_embed = torch.cat([cls_token, patch_pos_embed], dim=0)

        # Update the model
        visual.positional_embedding = nn.Parameter(new_pos_embed)

        # Update grid_size attribute
        visual.grid_size = (new_grid, new_grid)

    def _setup_hooks(self):
        """Setup forward hooks to extract intermediate layer features."""
        self._hooks = []
        self._hook_features = {}

        # Get transformer blocks
        visual = self.model.visual
        if hasattr(visual, 'transformer'):
            blocks = visual.transformer.resblocks
        else:
            raise ValueError("Cannot find transformer blocks in visual encoder")

        # Register hooks on target layers
        layer_indices_0 = [layer - 1 for layer in self.resolved_feature_layers]  # 0-indexed

        for idx, block in enumerate(blocks):
            if idx in layer_indices_0:
                hook = block.register_forward_hook(self._make_hook_fn(idx))
                self._hooks.append(hook)

    def _make_hook_fn(self, layer_idx):
        """Create a hook function for a specific layer."""
        def hook_fn(module, inp, out):
            # out is the output of the transformer block
            # For ViT, it's (L, B, C) before permutation
            if isinstance(out, tuple):
                out = out[0]
            self._hook_features[layer_idx] = out
        return hook_fn

    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.model.parameters():
            param.requires_grad = False

    def _encode_image_ref(self, x, out_layers):
        """Encode using ref/MuSc modified open_clip."""
        resolved_layers = self._resolve_feature_layers(out_layers)
        image_features, patch_tokens = self.model.encode_image(x, resolved_layers)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        patch_tokens = [patch_tokens[layer_idx].cpu() for layer_idx in range(len(resolved_layers))]
        return image_features, patch_tokens

    def _encode_image_standard(self, x, out_layers):
        """Encode using standard open_clip with hooks."""
        # Clear hook features
        self._hook_features.clear()
        resolved_layers = self._resolve_feature_layers(out_layers)

        # Forward pass (hooks will capture intermediate features)
        # Get the class token and patch embeddings
        # This follows the standard ViT forward but we intercept intermediate outputs
        with torch.no_grad():
            # Run full forward - hooks capture intermediate outputs
            image_features = self.model.encode_image(x, normalize=True)

        # Extract patch tokens from hooked features
        patch_tokens = []
        layer_indices_0 = [layer - 1 for layer in resolved_layers]  # 0-indexed

        for idx in layer_indices_0:
            if idx in self._hook_features:
                feat = self._hook_features[idx]
                # feat shape: (L, B, C) after transformer block
                # Permute to (B, L, C)
                if feat.dim() == 3 and feat.shape[0] > feat.shape[1]:
                    # Assume (L, B, C) format
                    feat = feat.permute(1, 0, 2)
                patch_tokens.append(feat.cpu())
            else:
                # Fallback: create zero tensor
                B = x.shape[0]
                L = (self.image_size // 14) ** 2 + 1  # patch_size=14 for ViT-L-14
                patch_tokens.append(torch.zeros(B, L, self.width))

        return image_features, patch_tokens

    def encode_image(
        self,
        x: torch.Tensor,
        out_layers: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Encode image and extract multi-layer features.

        Args:
            x: Input images (B, 3, H, W).
            out_layers: List of 1-indexed layers to extract. Uses feature_layers if None.

        Returns:
            image_features: (B, D) normalized CLS token features.
            patch_tokens: List of (B, L+1, C) tensors for each layer.
        """
        if out_layers is None:
            out_layers = self.feature_layers

        return self._encode_image(x, out_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass extracting multi-layer features.

        Args:
            x: Input images (B, 3, H, W).

        Returns:
            Tuple of (image_features, patch_tokens_list).
        """
        return self.encode_image(x, self.feature_layers)

    def train(self, mode: bool = True):
        """Override train to keep backbone frozen in eval mode."""
        super().train(mode)
        if self.frozen:
            self.model.eval()
        return self


@MODELS.register_module(force=True)
class MuScDINOv2Backbone(BaseModule):
    """DINOv2 backbone with multi-layer feature extraction for MuSc.

    Alternative to CLIP backbone using DINOv2.

    Args:
        model_name: DINOv2 model name (e.g., 'dinov2_vitb14', 'dinov2_vitl14').
        feature_layers: List of 1-indexed layer numbers to extract features from.
        frozen: Whether to freeze backbone weights. Default True.
    """

    def __init__(
        self,
        model_name: str = 'dinov2_vitb14',
        feature_layers: List[int] = None,
        frozen: bool = True,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)

        if feature_layers is None:
            # Default layers depend on model size
            if 'vitl' in model_name.lower():
                feature_layers = [5, 11, 17, 23]
            else:
                feature_layers = [3, 6, 9]

        self.model_name = model_name
        self.feature_layers = feature_layers
        self.frozen = frozen

        # Load DINOv2 from torch hub
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)

        # Get dimensions
        if 'vitl' in model_name.lower():
            self.width = 1024
        elif 'vitb' in model_name.lower():
            self.width = 768
        else:
            self.width = 384

        if frozen:
            self._freeze_backbone()

    def _freeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass extracting multi-layer features.

        Returns:
            Tuple of (image_features, patch_tokens_list).
        """
        # Get intermediate layers (0-indexed)
        layer_indices = [layer - 1 for layer in self.feature_layers]
        patch_tokens = self.model.get_intermediate_layers(
            x, n=layer_indices, return_class_token=False
        )

        # Get final CLS token as image features
        image_features = self.model(x)

        # Add fake CLS token to patch tokens for consistency with CLIP
        patch_tokens_out = []
        for pt in patch_tokens:
            fake_cls = torch.zeros_like(pt)[:, 0:1, :]
            patch_tokens_out.append(torch.cat([fake_cls, pt], dim=1).cpu())

        # Normalize image features
        image_features = F.normalize(image_features, dim=-1)

        return image_features, patch_tokens_out

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.model.eval()
        return self
