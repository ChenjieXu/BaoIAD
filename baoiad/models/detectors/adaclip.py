"""AdaCLIP: Adapting CLIP with Hybrid Learnable Prompts for Zero-Shot Anomaly Detection.

Reference: Cao et al., "AdaCLIP: Adapting CLIP with Hybrid Learnable Prompts for
Zero-Shot Anomaly Detection", ECCV 2024.

Official implementation: https://github.com/caoyunkang/AdaCLIP

Key components:
1. PromptLayer: Static and dynamic prompt injection for text/visual branches
2. ProjectLayer: Projects visual tokens to text embedding space
3. HybridSemanticFusion (HSF): K-means clustering for robust image-level scoring
4. Multi-hierarchy feature extraction from layers [6, 12, 18, 24]

This implementation follows the official reference closely for alignment.
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS
from baoiad.models.base_ad_model import VisionLanguageADModel
from baoiad.models.losses.dice_loss import BinaryDiceLoss as AdaCLIPBinaryDiceLoss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Normalization constants
# ---------------------------------------------------------------------------

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])
IMAGENET_MEAN_PIXEL = IMAGENET_MEAN * 255.0
IMAGENET_STD_PIXEL = IMAGENET_STD * 255.0
CLIP_MEAN_PIXEL = CLIP_MEAN * 255.0
CLIP_STD_PIXEL = CLIP_STD * 255.0
OFFICIAL_TRAINABLE_NAME_FRAGMENTS = (
    'text_prompter',
    'visual_prompter',
    'patch_token_layer',
    'cls_token_layer',
    'dynamic_visual_prompt_generator',
    'dynamic_text_prompt_generator',
)


# ---------------------------------------------------------------------------
# Prompt Templates (from official implementation)
# ---------------------------------------------------------------------------

PROMPT_NORMAL = [
    "{}",
    "flawless {}",
    "perfect {}",
    "unblemished {}",
    "{} without flaw",
    "{} without defect",
    "{} without damage",
]

PROMPT_ABNORMAL = [
    "damaged {}",
    "broken {}",
    "{} with flaw",
    "{} with defect",
    "{} with damage",
]

PROMPT_TEMPLATES = [
    "a bad photo of a {}.",
    "a low resolution photo of the {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
]


from baoiad.utils.score_utils import (
    normalize_class_name as _normalize_class_name,
    safe_l2_normalize as _safe_l2_normalize,
)


def _tensor_debug_summary(tensor: torch.Tensor) -> dict:
    """Build a compact tensor summary for debugging."""
    data = tensor.detach().float().cpu()
    payload = {
        'shape': list(data.shape),
        'dtype': str(tensor.dtype),
        'numel': int(data.numel()),
        'finite': bool(torch.isfinite(data).all().item()),
    }
    if data.numel() > 0:
        payload.update({
            'min': float(data.min().item()),
            'max': float(data.max().item()),
            'mean': float(data.mean().item()),
            'std': float(data.std(unbiased=False).item()),
        })
    return payload


def _emit_debug_payload(path: Optional[str], payload: dict) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')


def _raise_if_not_finite(
    name: str,
    tensor: torch.Tensor,
    *,
    context: Optional[dict] = None,
    debug_output_path: Optional[str] = None,
) -> None:
    """Raise a detailed error when a tensor contains NaN/Inf values."""
    if torch.isfinite(tensor.detach().float()).all():
        return
    payload = {
        'name': name,
        'context': context or {},
        'summary': _tensor_debug_summary(tensor),
    }
    _emit_debug_payload(debug_output_path, payload)
    raise ValueError(f'Non-finite values detected in `{name}`.')


# ---------------------------------------------------------------------------
# ProjectLayer: Multi-layer projection for visual tokens
# ---------------------------------------------------------------------------

class ProjectLayer(nn.Module):
    """Project visual tokens to text embedding space.

    Official implementation uses ModuleList for multiple output layers,
    handling each feature layer separately.

    Args:
        input_dim: Input visual feature dimension.
        output_dim: Output text embedding dimension.
        num_replicas: Number of output layers (one linear per layer).
        stack: Whether to stack outputs.
        is_array: Whether input is a list of tensors.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_replicas: int,
        stack: bool = False,
        is_array: bool = True,
    ):
        super().__init__()
        self.head = nn.ModuleList([
            nn.Linear(input_dim, output_dim) for _ in range(num_replicas)
        ])
        self.num_replicas = num_replicas
        self.stack = stack
        self.is_array = is_array

    def forward(self, tokens):
        """Project tokens from visual to text space.

        Args:
            tokens: List of tensors (B, N, D) if is_array, else single tensor.

        Returns:
            List or stacked tensor of projected tokens.
        """
        out_tokens = []
        for i in range(self.num_replicas):
            if self.is_array:
                # For ViT, exclude class token (index 0) and only use patch tokens
                temp = self.head[i](tokens[i][:, 1:, :])
            else:
                temp = self.head[i](tokens)
            out_tokens.append(temp)

        if self.stack:
            out_tokens = torch.stack(out_tokens, dim=1)

        return out_tokens


# ---------------------------------------------------------------------------
# PromptLayer: Static and Dynamic Prompt Injection
# ---------------------------------------------------------------------------

class PromptLayer(nn.Module):
    """Prompt layer for injecting static and dynamic prompts.

    Follows official implementation:
    - Visual branch: Layer 0 appends prompts, layers 1 to depth-1 replace last N tokens
    - Text branch: Replace tokens after first (prefix) token
    - Uses LND (Length, Batch, Dim) format internally

    Args:
        channel: Embedding dimension.
        length: Number of prompt tokens.
        depth: Number of layers to inject prompts.
        is_text: Whether this is for text branch.
        prompting_type: 'S', 'D', or 'SD'.
        enabled: Whether prompting is enabled.
    """

    def __init__(
        self,
        channel: int,
        length: int,
        depth: int,
        is_text: bool,
        prompting_type: str,
        enabled: bool = True,
    ):
        super().__init__()
        self.channel = channel
        self.length = length
        self.depth = depth
        self.is_text = is_text
        self.enabled = enabled
        self.prompting_type = prompting_type

        if self.enabled:
            if "S" in prompting_type:
                # Static prompts: learnable parameters, one per layer
                self.static_prompts = nn.ParameterList([
                    nn.Parameter(torch.empty(self.length, self.channel))
                    for _ in range(self.depth)
                ])
                for single_para in self.static_prompts:
                    nn.init.normal_(single_para, std=0.02)

            if "D" in prompting_type:
                # Dynamic prompts: placeholder, set externally
                self.dynamic_prompts = [0.0]

    def set_dynamic_prompts(self, dynamic_prompts):
        """Set dynamic prompts from generator."""
        self.dynamic_prompts = dynamic_prompts

    def _call_resblock(self, resblock, x, k_x=None, v_x=None, attn_mask=None):
        """Support both the official custom block API and standard OpenCLIP blocks."""
        batch_first = bool(getattr(getattr(resblock, 'attn', None), 'batch_first', False))
        if batch_first:
            x = x.permute(1, 0, 2)
            if k_x is not None:
                k_x = k_x.permute(1, 0, 2)
            if v_x is not None:
                v_x = v_x.permute(1, 0, 2)

        try:
            out = resblock(q_x=x, k_x=k_x, v_x=v_x, attn_mask=attn_mask)
        except TypeError:
            try:
                out = resblock(x, attn_mask=attn_mask)
            except TypeError:
                out = resblock(x)

        if not batch_first:
            return out

        if isinstance(out, (tuple, list)):
            out = list(out)
            if len(out) > 0 and torch.is_tensor(out[0]) and out[0].ndim == 3:
                out[0] = out[0].permute(1, 0, 2)
            return tuple(out)

        if torch.is_tensor(out) and out.ndim == 3:
            return out.permute(1, 0, 2)
        return out

    def forward_text(self, resblock, indx, x, k_x=None, v_x=None, attn_mask=None):
        """Forward for text branch.

        Args:
            resblock: Transformer residual block.
            indx: Current layer index.
            x: Input tensor in LND format.
            k_x, v_x: Optional key/value tensors.
            attn_mask: Attention mask.

        Returns:
            Output tensor and attention.
        """
        if self.enabled:
            length = self.length

            # Only prompt the first J layers
            if indx < self.depth:
                if "S" in self.prompting_type and "D" in self.prompting_type:
                    static_prompts = self.static_prompts[indx].unsqueeze(0).expand(x.shape[1], -1, -1)
                    textual_context = self.dynamic_prompts + static_prompts
                elif "S" in self.prompting_type:
                    static_prompts = self.static_prompts[indx].unsqueeze(0).expand(x.shape[1], -1, -1)
                    textual_context = static_prompts
                elif "D" in self.prompting_type:
                    textual_context = self.dynamic_prompts
                else:
                    raise NotImplementedError(
                        "You should at least choose one type of prompts when prompting branches are not none."
                    )

            if indx == 0:
                # First layer: no modification
                x = x
            else:
                if indx < self.depth:
                    prefix = x[:1, :, :]
                    suffix = x[1 + length:, :, :]
                    textual_context = textual_context.permute(1, 0, 2).to(x.dtype)
                    x = torch.cat([prefix, textual_context, suffix], dim=0)
                # else: keep the same

        out = self._call_resblock(resblock, x, k_x=k_x, v_x=v_x, attn_mask=attn_mask)
        if isinstance(out, (tuple, list)):
            x, attn_tmp = out[0], out[1] if len(out) > 1 else None
        else:
            x, attn_tmp = out, None
        return x, attn_tmp

    def forward_visual(self, resblock, indx, x, k_x=None, v_x=None, attn_mask=None):
        """Forward for visual branch.

        Args:
            resblock: Transformer residual block.
            indx: Current layer index.
            x: Input tensor in LND format.
            k_x, v_x: Optional key/value tensors.
            attn_mask: Attention mask.

        Returns:
            Output tensor, tokens (without prompts), and attention.
        """
        if self.enabled:
            length = self.length

            # Only prompt the first J layers
            if indx < self.depth:
                if "S" in self.prompting_type and "D" in self.prompting_type:
                    static_prompts = self.static_prompts[indx].unsqueeze(0).expand(x.shape[1], -1, -1)
                    visual_context = self.dynamic_prompts + static_prompts
                elif "S" in self.prompting_type:
                    static_prompts = self.static_prompts[indx].unsqueeze(0).expand(x.shape[1], -1, -1)
                    visual_context = static_prompts
                elif "D" in self.prompting_type:
                    visual_context = self.dynamic_prompts
                else:
                    raise NotImplementedError(
                        "You should at least choose one type of prompts when prompting branches are not none."
                    )

            if indx == 0:
                # First layer: append prompts at the end
                visual_context = visual_context.permute(1, 0, 2).to(x.dtype)
                x = torch.cat([x, visual_context], dim=0)
            else:
                if indx < self.depth:
                    # Replace last N tokens with prompts
                    prefix = x[0:x.shape[0] - length, :, :]
                    visual_context = visual_context.permute(1, 0, 2).to(x.dtype)
                    x = torch.cat([prefix, visual_context], dim=0)
                # else: keep the same

        out = self._call_resblock(resblock, x, k_x=k_x, v_x=v_x, attn_mask=attn_mask)
        if isinstance(out, (tuple, list)):
            x, attn_tmp = out[0], out[1] if len(out) > 1 else None
        else:
            x, attn_tmp = out, None

        if self.enabled:
            # Return tokens without prompts
            tokens = x[:x.shape[0] - self.length, :, :]
        else:
            tokens = x

        return x, tokens, attn_tmp

    def forward(self, resblock, indx, x, k_x=None, v_x=None, attn_mask=None):
        """Forward pass."""
        if self.is_text:
            return self.forward_text(resblock, indx, x, k_x, v_x, attn_mask)
        else:
            return self.forward_visual(resblock, indx, x, k_x, v_x, attn_mask)


# ---------------------------------------------------------------------------
# TextEmbeddingLayer: Text Feature Encoding with Ensemble
# ---------------------------------------------------------------------------

class TextEmbeddingLayer(nn.Module):
    """Text embedding layer with prompt ensemble.

    Follows official implementation for text encoding with prompt ensemble.
    """

    def __init__(self, clip_model, tokenizer=None, fixed=True):
        super().__init__()
        self.clip = clip_model
        self._tokenize = tokenizer
        self.fixed = fixed
        self.ensemble_text_features = {}

        self.prompt_normal = PROMPT_NORMAL
        self.prompt_abnormal = PROMPT_ABNORMAL
        self.prompt_state = [self.prompt_normal, self.prompt_abnormal]
        self.prompt_templates = PROMPT_TEMPLATES

    def tokenize(self, texts, context_length=77):
        """Tokenize text using CLIP tokenizer."""
        if isinstance(texts, str):
            texts = [texts]
        if self._tokenize is None:
            raise RuntimeError('AdaCLIP tokenizer is not initialized.')
        try:
            return self._tokenize(texts, context_length=context_length)
        except TypeError:
            return self._tokenize(texts)

    def encode_text(self, model, text, device):
        """Encode text with prompt ensemble.

        Args:
            model: CLIP model.
            text: Class name.
            device: Device.

        Returns:
            Stacked text features (D, 2) for normal and abnormal.
        """
        text_features = []
        for i in range(len(self.prompt_state)):
            text = text.replace("-", " ")
            prompted_state = [state.format(text) for state in self.prompt_state[i]]
            prompted_sentence = []
            for s in prompted_state:
                for template in self.prompt_templates:
                    prompted_sentence.append(template.format(s))
            prompted_sentence = self.tokenize(prompted_sentence, context_length=77).to(device)

            class_embeddings = model.encode_text(prompted_sentence)
            class_embeddings = _safe_l2_normalize(class_embeddings, dim=-1)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding = _safe_l2_normalize(class_embedding, dim=0)
            text_features.append(class_embedding)

        text_features = torch.stack(text_features, dim=1)  # (D, 2)
        return text_features

    def forward(self, model, texts, device):
        """Forward pass.

        Args:
            model: CLIP model.
            texts: List of class names.
            device: Device.

        Returns:
            Stacked text features (B, D, 2).
        """
        text_feature_list = []

        for text in texts:
            if self.fixed:
                if self.ensemble_text_features.get(text) is None:
                    text_features = self.encode_text(model, text, device)
                    self.ensemble_text_features[text] = text_features
                else:
                    text_features = self.ensemble_text_features[text]
            else:
                text_features = self.encode_text(model, text, device)
                self.ensemble_text_features[text] = text_features

            text_feature_list.append(text_features)

        text_features = torch.stack(text_feature_list, dim=0)  # (B, D, 2)
        text_features = _safe_l2_normalize(text_features, dim=1)

        return text_features


# ---------------------------------------------------------------------------
# HybridSemanticFusion: K-means Clustering for Image Scoring
# ---------------------------------------------------------------------------

class HybridSemanticFusion(nn.Module):
    """Hybrid Semantic Fusion for robust image-level scoring.

    Uses K-means clustering on top-k abnormal patch tokens to aggregate
    features for more robust image-level anomaly scoring.

    Args:
        k_clusters: Number of K-means clusters.
    """

    def __init__(self, k_clusters: int = 20):
        super().__init__()
        self.k_clusters = k_clusters
        self.n_aggregate_patch_tokens = k_clusters * 5
        self.cluster_performer = KMeans(n_clusters=self.k_clusters, n_init="auto")
        self.debug_finite_checks = False
        self.debug_output_path = None
        self.debug_context = {}

    def forward(self, patch_tokens: list, anomaly_maps: list):
        """Compute fused image-level features.

        Args:
            patch_tokens: List of (B, N, D) patch token features per layer.
            anomaly_maps: List of (B, 2, N) anomaly logits per layer.

        Returns:
            Fused image-level features (B, D).
        """
        # Average anomaly maps across layers
        anomaly_map = torch.mean(torch.stack(anomaly_maps, dim=1), dim=1)  # (B, 2, N)
        _raise_if_not_finite(
            'hsf.anomaly_map_logits',
            anomaly_map,
            context=self.debug_context,
            debug_output_path=self.debug_output_path if self.debug_finite_checks else None,
        )
        anomaly_map = torch.softmax(anomaly_map, dim=2)[:, :, 1]  # (B, N) - probability of abnormal
        _raise_if_not_finite(
            'hsf.anomaly_map_probs',
            anomaly_map,
            context=self.debug_context,
            debug_output_path=self.debug_output_path if self.debug_finite_checks else None,
        )

        # Extract most abnormal features
        selected_abnormal_tokens = []
        k = min(anomaly_map.shape[1], self.n_aggregate_patch_tokens)
        top_k_indices = torch.topk(anomaly_map, k=k, dim=1).indices

        for layer in range(len(patch_tokens)):
            selected_tokens = patch_tokens[layer].gather(
                dim=1,
                index=top_k_indices.unsqueeze(-1).expand(-1, -1, patch_tokens[layer].shape[-1])
            )
            _raise_if_not_finite(
                f'hsf.selected_tokens.layer{layer}',
                selected_tokens,
                context=self.debug_context,
                debug_output_path=self.debug_output_path if self.debug_finite_checks else None,
            )
            selected_abnormal_tokens.append(selected_tokens)

        # Stack and perform K-means clustering
        stacked_data = torch.cat(selected_abnormal_tokens, dim=2).float()  # (B, k, D*layers)
        _raise_if_not_finite(
            'hsf.stacked_data',
            stacked_data,
            context=self.debug_context,
            debug_output_path=self.debug_output_path if self.debug_finite_checks else None,
        )

        batch_cluster_centers = []
        for b in range(stacked_data.shape[0]):
            cluster_labels = self.cluster_performer.fit_predict(
                stacked_data[b, :, :].detach().cpu().numpy()
            )
            unique_labels = sorted({int(label) for label in cluster_labels.tolist()})

            cluster_centers = []
            cluster_sizes = {}
            for cluster_id in unique_labels:
                collected_cluster_data = []
                for abnormal_tokens in selected_abnormal_tokens:
                    cluster_data = abnormal_tokens[b, :, :][cluster_labels == cluster_id]
                    if cluster_data.numel() > 0:
                        collected_cluster_data.append(cluster_data)
                cluster_sizes[str(cluster_id)] = int(sum(item.shape[0] for item in collected_cluster_data))
                if not collected_cluster_data:
                    continue
                collected_cluster_data = torch.cat(collected_cluster_data, dim=0)
                cluster_center = torch.mean(collected_cluster_data, dim=0, keepdim=True)
                cluster_centers.append(cluster_center)

            if self.debug_finite_checks and len(unique_labels) < self.k_clusters:
                payload = {
                    'name': 'hsf.reduced_clusters',
                    'context': self.debug_context,
                    'cluster_count': len(unique_labels),
                    'requested_clusters': int(self.k_clusters),
                    'cluster_sizes': cluster_sizes,
                    'stacked_data': _tensor_debug_summary(stacked_data[b, :, :]),
                }
                _emit_debug_payload(self.debug_output_path, payload)

            if not cluster_centers:
                payload = {
                    'name': 'hsf.empty_cluster_set',
                    'context': self.debug_context,
                    'cluster_count': len(unique_labels),
                    'requested_clusters': int(self.k_clusters),
                    'cluster_sizes': cluster_sizes,
                    'stacked_data': _tensor_debug_summary(stacked_data[b, :, :]),
                }
                _emit_debug_payload(self.debug_output_path if self.debug_finite_checks else None, payload)
                raise ValueError('HSF produced no non-empty clusters during KMeans aggregation.')

            cluster_centers = torch.cat(cluster_centers, dim=0)
            cluster_centers = torch.mean(cluster_centers, dim=0)
            batch_cluster_centers.append(cluster_centers)

        batch_cluster_centers = torch.stack(batch_cluster_centers, dim=0)
        _raise_if_not_finite(
            'hsf.cluster_centers',
            batch_cluster_centers,
            context=self.debug_context,
            debug_output_path=self.debug_output_path if self.debug_finite_checks else None,
        )
        batch_cluster_centers = _safe_l2_normalize(batch_cluster_centers, dim=1)

        return batch_cluster_centers


class AdaCLIPFocalLoss(nn.Module):
    """Focal loss that consumes probabilities, matching official AdaCLIP."""

    def __init__(
        self,
        alpha=None,
        gamma: float = 2.0,
        balance_index: int = 0,
        smooth: float = 1e-5,
        size_average: bool = True,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

    def forward(self, probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_class = probs.shape[1]
        if probs.dim() > 2:
            probs = probs.view(probs.size(0), probs.size(1), -1)
            probs = probs.permute(0, 2, 1).contiguous()
            probs = probs.view(-1, probs.size(-1))

        target = torch.squeeze(target, 1).view(-1, 1)
        alpha = self.alpha
        if alpha is None:
            alpha = torch.ones(num_class, 1, device=probs.device, dtype=probs.dtype)
        elif isinstance(alpha, (list, tuple)):
            alpha = torch.tensor(alpha, dtype=probs.dtype, device=probs.device).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1, dtype=probs.dtype, device=probs.device)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha
        else:
            raise TypeError('Unsupported alpha type for AdaCLIP focal loss')

        idx = target.long()
        one_hot = torch.zeros(target.size(0), num_class, device=probs.device, dtype=probs.dtype)
        one_hot = one_hot.scatter_(1, idx, 1)
        if self.smooth:
            one_hot = torch.clamp(one_hot, self.smooth / (num_class - 1), 1.0 - self.smooth)

        pt = (one_hot * probs).sum(1) + self.smooth
        logpt = pt.log()
        alpha = alpha[idx].squeeze()
        loss = -alpha * torch.pow(1 - pt, self.gamma) * logpt
        return loss.mean() if self.size_average else loss.sum()


# ---------------------------------------------------------------------------
# AdaCLIP: Main Model Class (following official implementation)
# ---------------------------------------------------------------------------

class AdaCLIP(nn.Module):
    """AdaCLIP model following official implementation.

    This wraps the frozen CLIP model with learnable prompts.
    """

    def __init__(
        self,
        freeze_clip,
        tokenizer,
        text_channel: int,
        visual_channel: int,
        prompting_length: int,
        prompting_depth: int,
        prompting_branch: str,
        prompting_type: str,
        use_hsf: bool,
        k_clusters: int,
        output_layers: list,
        device: str,
        image_size: int,
    ):
        super().__init__()
        self.freeze_clip = freeze_clip
        self._tokenize = tokenizer

        # Extract CLIP components
        self.visual = self.freeze_clip.visual
        self.transformer = self.freeze_clip.transformer
        self.token_embedding = self.freeze_clip.token_embedding
        self.positional_embedding = self.freeze_clip.positional_embedding
        self.ln_final = self.freeze_clip.ln_final
        self.text_projection = self.freeze_clip.text_projection
        self.attn_mask = self.freeze_clip.attn_mask

        self.output_layers = output_layers
        self.prompting_branch = prompting_branch
        self.prompting_type = prompting_type
        self.prompting_depth = prompting_depth
        self.prompting_length = prompting_length
        self.use_hsf = use_hsf
        self.k_clusters = k_clusters
        self.image_size = image_size
        self._device = device  # Store as private, use property for dynamic resolution
        self.debug_finite_checks = False
        self.debug_output_path = None
        self.debug_context = {}

        # Enable prompting
        if "L" in self.prompting_branch:
            self.enable_text_prompt = True
        else:
            self.enable_text_prompt = False

        if "V" in self.prompting_branch:
            self.enable_visual_prompt = True
        else:
            self.enable_visual_prompt = False

        # Text embedding layer
        self.text_embedding_layer = TextEmbeddingLayer(
            self.freeze_clip, tokenizer=self._tokenize, fixed=(not self.enable_text_prompt)
        )

        # Prompt layers
        self.text_prompter = PromptLayer(
            text_channel, prompting_length, prompting_depth,
            is_text=True, prompting_type=prompting_type,
            enabled=self.enable_text_prompt
        )
        self.visual_prompter = PromptLayer(
            visual_channel, prompting_length, prompting_depth,
            is_text=False, prompting_type=prompting_type,
            enabled=self.enable_visual_prompt
        )

        # Projection layers
        self.patch_token_layer = ProjectLayer(
            visual_channel, text_channel,
            len(output_layers), stack=False, is_array=True
        )
        self.cls_token_layer = ProjectLayer(
            text_channel, text_channel,
            1, stack=False, is_array=False
        )

        # Dynamic prompt generators
        if "D" in self.prompting_type:
            self.dynamic_visual_prompt_generator = ProjectLayer(
                text_channel, visual_channel,
                prompting_length, stack=True, is_array=False
            )
            self.dynamic_text_prompt_generator = ProjectLayer(
                text_channel, text_channel,
                prompting_length, stack=True, is_array=False
            )

        # HSF
        if self.use_hsf:
            self.HSF = HybridSemanticFusion(k_clusters)

    @property
    def device(self):
        """Get the current device of the model."""
        # Try to get device from freeze_clip parameters
        try:
            return next(self.freeze_clip.parameters()).device
        except StopIteration:
            # Fallback to stored device
            return self._device

    def generate_and_set_dynamic_prompts(self, image):
        """Generate and set dynamic prompts from image features.

        This method extracts image features WITHOUT prompts first, then generates
        dynamic prompts from those features. This follows the reference implementation
        which uses self.visual.forward(image, out_layers) that doesn't use prompts.
        """
        with torch.no_grad(), torch.autocast('cuda', enabled=False):
            # Extract image features WITHOUT prompts (like reference's visual.forward)
            pooled = self._extract_image_features_no_prompts(image.float())
        _raise_if_not_finite(
            'adaclip.dynamic_prompt_seed',
            pooled,
            context=self.debug_context,
            debug_output_path=self.debug_output_path if self.debug_finite_checks else None,
        )

        dynamic_visual_prompts = self.dynamic_visual_prompt_generator(pooled)
        dynamic_text_prompts = self.dynamic_text_prompt_generator(pooled)

        self.visual_prompter.set_dynamic_prompts(dynamic_visual_prompts)
        self.text_prompter.set_dynamic_prompts(dynamic_text_prompts)

    def _extract_image_features_no_prompts(self, image):
        """Extract image features without using prompts.

        This is used by generate_and_set_dynamic_prompts to get image features
        before prompts are set. Follows reference implementation's visual.forward().
        """
        x = self.visual.conv1(image)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = torch.cat([
            self.visual.class_embedding.to(x.dtype)
            + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x,
        ], dim=1)
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.patch_dropout(x)
        x = self.visual.ln_pre(x)
        x = x.permute(1, 0, 2)

        for resblock in self.visual.transformer.resblocks:
            out = self.visual_prompter._call_resblock(
                resblock,
                x,
                k_x=None,
                v_x=None,
                attn_mask=None,
            )
            if isinstance(out, (tuple, list)):
                x = out[0]
            else:
                x = out

        x = x.permute(1, 0, 2)
        if self.visual.attn_pool is not None:
            x = self.visual.attn_pool(x)
            x = self.visual.ln_post(x)
            pooled, _ = self.visual._global_pool(x)
        else:
            pooled, _ = self.visual._global_pool(x)
            pooled = self.visual.ln_post(pooled)
        if self.visual.proj is not None:
            pooled = pooled @ self.visual.proj

        _raise_if_not_finite(
            'adaclip.image_features_no_prompts',
            pooled,
            context=self.debug_context,
            debug_output_path=self.debug_output_path if self.debug_finite_checks else None,
        )
        return pooled

    def encode_image(self, image):
        """Encode image with visual prompts.

        This method is adapted to work with open_clip's VisionTransformer,
        which has a different interface than the reference implementation.

        Args:
            image: Input image tensor (B, C, H, W).

        Returns:
            pooled: Pooled image features (B, D).
            patch_tokens: List of patch token tensors per output layer.
            patch_embedding: Initial patch embedding.
        """
        # open_clip's VisionTransformer doesn't have input_patchnorm
        # Use the standard conv1 path
        x = self.visual.conv1(image)  # (B, width, grid, grid)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (B, width, grid**2)
        x = x.permute(0, 2, 1)  # (B, grid**2, width)

        # Class embeddings and positional embeddings
        x = torch.cat([
            self.visual.class_embedding.to(x.dtype) +
            torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)
        x = x + self.visual.positional_embedding.to(x.dtype)

        # Patch dropout and pre-LN
        x = self.visual.patch_dropout(x)
        x = self.visual.ln_pre(x)

        patch_embedding = x

        # Convert to LND format for transformer
        x = x.permute(1, 0, 2)

        patch_tokens = []

        # Transformer blocks with prompt injection
        for indx, r in enumerate(self.visual.transformer.resblocks):
            x, tokens, attn_tmp = self.visual_prompter(r, indx, x, k_x=None, v_x=None, attn_mask=None)

            if (indx + 1) in self.output_layers:
                patch_tokens.append(tokens)

        # Convert back to NLD
        x = x.permute(1, 0, 2)
        patch_tokens = [pt.permute(1, 0, 2) for pt in patch_tokens]

        # Global pooling
        if self.visual.attn_pool is not None:
            x = self.visual.attn_pool(x)
            x = self.visual.ln_post(x)
            pooled, tokens = self.visual._global_pool(x)
        else:
            pooled, tokens = self.visual._global_pool(x)
            pooled = self.visual.ln_post(pooled)

        if self.visual.proj is not None:
            pooled = pooled @ self.visual.proj
        _raise_if_not_finite(
            'adaclip.image_features',
            pooled,
            context=self.debug_context,
            debug_output_path=self.debug_output_path if self.debug_finite_checks else None,
        )

        return pooled, patch_tokens, patch_embedding

    def proj_visual_tokens(self, image_features, patch_tokens):
        """Project visual tokens to text space.

        Args:
            image_features: CLS token features.
            patch_tokens: List of patch token tensors.

        Returns:
            proj_cls_tokens: Projected CLS tokens.
            proj_patch_tokens: List of projected patch tokens.
        """
        # Project patch tokens
        proj_patch_tokens = self.patch_token_layer(patch_tokens)
        for layer in range(len(proj_patch_tokens)):
            proj_patch_tokens[layer] = _safe_l2_normalize(proj_patch_tokens[layer], dim=-1)
            _raise_if_not_finite(
                f'adaclip.proj_patch_tokens.layer{layer}',
                proj_patch_tokens[layer],
                context=self.debug_context,
                debug_output_path=self.debug_output_path if self.debug_finite_checks else None,
            )

        # Project CLS tokens
        proj_cls_tokens = self.cls_token_layer(image_features)[0]
        proj_cls_tokens = _safe_l2_normalize(proj_cls_tokens, dim=-1)
        _raise_if_not_finite(
            'adaclip.proj_cls_tokens',
            proj_cls_tokens,
            context=self.debug_context,
            debug_output_path=self.debug_output_path if self.debug_finite_checks else None,
        )

        return proj_cls_tokens, proj_patch_tokens

    def encode_text(self, text):
        """Encode text with prompt injection.

        Args:
            text: Tokenized text (B, L).

        Returns:
            Text features (B, D).
        """
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)
        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        attn_mask = self.attn_mask.to(x.device) if self.attn_mask is not None else None

        for indx, r in enumerate(self.transformer.resblocks):
            x, attn_tmp = self.text_prompter(r, indx, x, k_x=None, v_x=None, attn_mask=attn_mask)

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # Take features from EOT embedding
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def visual_text_similarity(self, image_feature, patch_token, text_feature, aggregation):
        """Compute visual-text similarity for anomaly detection.

        Args:
            image_feature: CLS token features (B, D).
            patch_token: List of patch token tensors (B, N, D).
            text_feature: Text features (B, D, 2).
            aggregation: Whether to aggregate across layers.

        Returns:
            anomaly_map: Anomaly map tensor.
            anomaly_score: Image-level anomaly score.
        """
        anomaly_maps = []

        for layer in range(len(patch_token)):
            anomaly_map = 100.0 * patch_token[layer] @ text_feature
            _raise_if_not_finite(
                f'adaclip.anomaly_logits.layer{layer}',
                anomaly_map,
                context=self.debug_context,
                debug_output_path=self.debug_output_path if self.debug_finite_checks else None,
            )
            anomaly_maps.append(anomaly_map)

        if self.use_hsf:
            alpha = 0.2
            self.HSF.debug_finite_checks = self.debug_finite_checks
            self.HSF.debug_output_path = self.debug_output_path
            self.HSF.debug_context = dict(self.debug_context)
            clustered_feature = self.HSF.forward(patch_token, anomaly_maps)
            cur_image_feature = alpha * clustered_feature + (1 - alpha) * image_feature
            cur_image_feature = _safe_l2_normalize(cur_image_feature, dim=1)
        else:
            cur_image_feature = image_feature

        anomaly_score = 100.0 * cur_image_feature.unsqueeze(1) @ text_feature
        anomaly_score = anomaly_score.squeeze(1)
        anomaly_score = torch.softmax(anomaly_score, dim=1)
        _raise_if_not_finite(
            'adaclip.anomaly_score',
            anomaly_score,
            context=self.debug_context,
            debug_output_path=self.debug_output_path if self.debug_finite_checks else None,
        )

        # Reshape anomaly maps to spatial
        for i in range(len(anomaly_maps)):
            B, L, C = anomaly_maps[i].shape
            H = int(np.sqrt(L))
            anomaly_maps[i] = anomaly_maps[i].permute(0, 2, 1).view(B, 2, H, H)
            anomaly_maps[i] = F.interpolate(
                anomaly_maps[i], size=self.image_size, mode="bilinear", align_corners=True
            )

        if aggregation:
            # Aggregate across layers first, then softmax (matches official)
            anomaly_map = torch.mean(torch.stack(anomaly_maps, dim=1), dim=1)
            anomaly_map = torch.softmax(anomaly_map, dim=1)
            # Final anomaly map formula from official
            anomaly_map = (anomaly_map[:, 1:, :, :] + 1 - anomaly_map[:, 0:1, :, :]) / 2.0
            anomaly_score = anomaly_score[:, 1]
            return anomaly_map, anomaly_score
        else:
            # Per-layer softmax
            for i in range(len(anomaly_maps)):
                anomaly_maps[i] = torch.softmax(anomaly_maps[i], dim=1)
            return anomaly_maps, anomaly_score

    def extract_feat(self, image, cls_name):
        """Extract features for image and text.

        Args:
            image: Input image tensor.
            cls_name: List of class names.

        Returns:
            proj_cls_tokens: Projected CLS tokens.
            proj_patch_tokens: Projected patch tokens.
            text_features: Text features.
        """
        if "D" in self.prompting_type:
            self.generate_and_set_dynamic_prompts(image)

        if self.enable_visual_prompt:
            image_features, patch_tokens, _ = self.encode_image(image)
        else:
            with torch.no_grad():
                image_features, patch_tokens, _ = self.encode_image(image)

        if self.enable_text_prompt:
            text_features = self.text_embedding_layer(self, cls_name, self.device)
        else:
            with torch.no_grad():
                text_features = self.text_embedding_layer(self, cls_name, self.device)

        proj_cls_tokens, proj_patch_tokens = self.proj_visual_tokens(image_features, patch_tokens)

        return proj_cls_tokens, proj_patch_tokens, text_features

    @torch.amp.autocast('cuda')
    def forward(self, image, cls_name, aggregation=True):
        """Forward pass.

        Args:
            image: Input image tensor.
            cls_name: List of class names.
            aggregation: Whether to aggregate across layers.

        Returns:
            anomaly_map: Anomaly map.
            anomaly_score: Image-level score.
        """
        image_features, patch_tokens, text_features = self.extract_feat(image, cls_name)
        anomaly_map, anomaly_score = self.visual_text_similarity(
            image_features, patch_tokens, text_features, aggregation
        )

        if aggregation:
            anomaly_map = anomaly_map.squeeze(1)
            return anomaly_map, anomaly_score
        else:
            return anomaly_map, anomaly_score


# ---------------------------------------------------------------------------
# AdaCLIPDetector: BaoIAD Detector Wrapper
# ---------------------------------------------------------------------------

@MODELS.register_module()
class AdaCLIPDetector(VisionLanguageADModel):
    """AdaCLIP: Adapting CLIP with Hybrid Learnable Prompts for Zero-Shot AD.

    This is the BaoIAD wrapper for the AdaCLIP model, following MMEngine conventions.

    Key features:
    1. Static prompts: Learnable parameters shared across all images
    2. Dynamic prompts: Generated per-image using a generator network
    3. Hybrid Semantic Fusion (HSF): K-means clustering for robust scoring
    4. Multi-hierarchy feature extraction from layers [6, 12, 18, 24]

    For zero-shot mode (no training), use pretrained='openai' and official_checkpoint=None.
    For few-shot mode, train the prompt parameters on a few normal images.

    Args:
        clip_model: CLIP model name (default 'ViT-L-14-336').
        pretrained: Pretrained weights source (default 'openai').
        class_name: Object class name for prompts.
        image_size: Input image size (default 518, official default).
        features_list: List of layer indices for multi-scale features.
        prompting_depth: Number of layers for prompt injection (default 4).
        prompting_length: Number of prompt tokens per layer (default 5).
        prompting_branch: 'V' (visual), 'L' (language), or 'VL' (both).
        prompting_type: 'S' (static), 'D' (dynamic), or 'SD' (both).
        use_hsf: Whether to use Hybrid Semantic Fusion.
        k_clusters: Number of K-means clusters for HSF.
        temperature: Temperature for softmax.
        gaussian_sigma: Sigma for Gaussian blur on anomaly maps.
        official_checkpoint: Path to official pretrained checkpoint.
    """

    def __init__(
        self,
        clip_model: str = "ViT-L-14-336",
        pretrained: str = "openai",
        class_name: str = "object",
        image_size: int = 518,
        features_list: List[int] = None,
        prompting_depth: int = 4,
        prompting_length: int = 5,
        prompting_branch: str = "VL",
        prompting_type: str = "SD",
        use_hsf: bool = True,
        k_clusters: int = 20,
        temperature: float = 0.07,
        gaussian_sigma: float = 4.0,
        official_checkpoint: str = None,
        require_official_checkpoint: bool = False,
        enable_train_loss: bool = False,
        hf_endpoint: str = None,
        debug_finite_checks: bool = False,
        debug_output_path: str = None,
        data_preprocessor=None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        # Store config
        self.clip_model_name = clip_model
        self.class_name = _normalize_class_name(class_name)
        self.image_size = image_size
        self.prompting_depth = prompting_depth
        self.prompting_length = prompting_length
        self.prompting_branch = prompting_branch
        self.prompting_type = prompting_type
        self.use_hsf = use_hsf
        self.k_clusters = k_clusters
        self.temperature = temperature
        self.gaussian_sigma = gaussian_sigma
        self.official_checkpoint = official_checkpoint
        self.require_official_checkpoint = require_official_checkpoint
        self.enable_train_loss = enable_train_loss
        self.hf_endpoint = hf_endpoint
        self.debug_finite_checks = debug_finite_checks
        self.debug_output_path = debug_output_path
        self._debug_step = 0

        # Default feature layers (official: [n_layers//4, n_layers//2, 3*n_layers//4, n_layers])
        if features_list is None:
            features_list = [6, 12, 18, 24]  # For ViT-L-14-336 with 24 layers
        self.features_list = features_list

        # Build CLIP backbone using OpenCLIP
        # force_quick_gelu=True is required for OpenAI pretrained models
        clip_backbone = MODELS.build(
            dict(
                type="OpenCLIPBackbone",
                model_name=clip_model,
                pretrained=pretrained,
                frozen=True,
                force_quick_gelu=True,
                hf_endpoint=hf_endpoint,
            )
        )
        self.clip = clip_backbone.model
        self._tokenize = clip_backbone.tokenize

        # Get dimensions
        self.visual_dim = self._get_visual_dim()
        self.text_dim = self._get_text_dim()

        # Adapt positional embedding for different image size
        self._adapt_positional_embedding(image_size)

        # Build AdaCLIP model
        self.adaclip = AdaCLIP(
            freeze_clip=self.clip,
            tokenizer=self._tokenize,
            text_channel=self.text_dim,
            visual_channel=self.visual_dim,
            prompting_length=prompting_length,
            prompting_depth=prompting_depth,
            prompting_branch=prompting_branch,
            prompting_type=prompting_type,
            use_hsf=use_hsf,
            k_clusters=k_clusters,
            output_layers=features_list,
            device="cuda" if torch.cuda.is_available() else "cpu",
            image_size=image_size,
        )
        self.adaclip.debug_finite_checks = self.debug_finite_checks
        self.adaclip.debug_output_path = self.debug_output_path

        # Normalization buffers
        self.register_buffer(
            "_imagenet_mean",
            IMAGENET_MEAN_PIXEL.view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_imagenet_std",
            IMAGENET_STD_PIXEL.view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_clip_mean",
            CLIP_MEAN_PIXEL.view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_clip_std",
            CLIP_STD_PIXEL.view(1, 3, 1, 1),
            persistent=False,
        )
        self.loss_focal = AdaCLIPFocalLoss()
        self.loss_dice = AdaCLIPBinaryDiceLoss()
        self._freeze_non_prompt_parameters()

        # Load official checkpoint if provided
        if official_checkpoint is not None:
            self._load_official_checkpoint(official_checkpoint)

    def _get_visual_dim(self) -> int:
        """Get visual embedding dimension."""
        if hasattr(self.clip.visual, "transformer"):
            return self.clip.visual.transformer.width
        return 1024  # Default for ViT-L-14

    def _get_text_dim(self) -> int:
        """Get text embedding dimension."""
        if hasattr(self.clip, "text_projection") and self.clip.text_projection is not None:
            return self.clip.text_projection.shape[0]
        if hasattr(self.clip, "transformer"):
            return self.clip.transformer.width
        return 512  # Default for ViT-B-16

    def _adapt_positional_embedding(self, input_size: int):
        """Adapt positional embedding for different input size."""
        visual = self.clip.visual
        if not hasattr(visual, "positional_embedding"):
            return

        patch_size = visual.conv1.kernel_size[0] if hasattr(visual, "conv1") else None
        if patch_size is None:
            return

        old_grid = visual.grid_size
        new_grid_h = input_size // patch_size
        new_grid_w = input_size // patch_size

        if old_grid[0] == new_grid_h and old_grid[1] == new_grid_w:
            return

        pe = visual.positional_embedding
        cls_pe = pe[:1]
        patch_pe = pe[1:]
        dim = pe.shape[1]
        patch_pe = patch_pe.reshape(1, old_grid[0], old_grid[1], dim).permute(0, 3, 1, 2)
        patch_pe = F.interpolate(
            patch_pe, size=(new_grid_h, new_grid_w), mode="bicubic", align_corners=False
        )
        patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(new_grid_h * new_grid_w, dim)
        visual.positional_embedding = nn.Parameter(
            torch.cat([cls_pe, patch_pe], dim=0),
            requires_grad=bool(pe.requires_grad),
        )
        visual.grid_size = (new_grid_h, new_grid_w)

    def _freeze_non_prompt_parameters(self) -> None:
        """Match the official trainer by updating prompt-related layers only."""
        for name, param in self.named_parameters():
            is_official_trainable = any(
                fragment in name for fragment in OFFICIAL_TRAINABLE_NAME_FRAGMENTS
            )
            param.requires_grad = is_official_trainable

    def get_official_trainable_parameter_names(self) -> List[str]:
        """Return parameter names optimized by the official AdaCLIP trainer."""
        return [name for name, param in self.named_parameters() if param.requires_grad]

    def _load_official_checkpoint(self, checkpoint_path: str):
        """Load official AdaCLIP checkpoint.

        Args:
            checkpoint_path: Path to official .pth checkpoint.
        """
        if not os.path.exists(checkpoint_path):
            message = f"Checkpoint not found: {checkpoint_path}"
            if self.require_official_checkpoint:
                raise FileNotFoundError(message)
            logger.warning(f"{message}. Continuing without loading official weights.")
            return

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        # The official trainer saves only learnable prompt-related parameters from
        # `clip_model.state_dict()`. Load them directly into the inner AdaCLIP
        # model instead of remapping onto the outer detector wrapper.
        state_dict = {}
        for key, value in checkpoint.items():
            if key.startswith("module."):
                key = key[len("module."):]
            if key.startswith("clip_model."):
                key = key[len("clip_model."):]
            elif key.startswith("adaclip."):
                key = key[len("adaclip."):]
            state_dict[key] = value

        # Load state dict (strict=False to allow partial loading)
        missing, unexpected = self.adaclip.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys: {missing[:5]}...")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected[:5]}...")

        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def _normalize_for_clip(self, x: torch.Tensor) -> torch.Tensor:
        """Convert 0-255 ImageNet normalization to CLIP normalization."""
        # `NormalizeAD` uses ImageNet statistics in the original 0-255 pixel
        # space. Recover raw pixels first, then apply CLIP's 0-255-equivalent
        # normalization so the result matches CLIP preprocess semantics.
        x = x * self._imagenet_std + self._imagenet_mean
        x = (x - self._clip_mean) / self._clip_std
        return x

    def _gaussian_blur(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply Gaussian blur to anomaly maps."""
        if sigma <= 0:
            return x

        radius = max(int(round(4 * sigma)), 1)
        kernel_size = 2 * radius + 1
        coord = torch.arange(kernel_size, device=x.device, dtype=x.dtype) - radius
        kernel_1d = torch.exp(-(coord ** 2) / (2 * sigma * sigma))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel = kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(x.shape[1], 1, 1, 1)

        return F.conv2d(x, kernel, padding=radius, groups=x.shape[1])

    def _resolve_batch_classes(self, data_samples, batch_size: int) -> List[str]:
        """Resolve class names from data samples."""
        if data_samples is None:
            return [self.class_name] * batch_size

        classes = []
        for sample in data_samples:
            cls_name = getattr(sample, "cls_name", None)
            if cls_name is None:
                cls_name = getattr(sample, "class_name", None)
            classes.append(
                _normalize_class_name(cls_name) if cls_name else self.class_name
            )

        if len(classes) < batch_size:
            classes.extend([self.class_name] * (batch_size - len(classes)))

        return classes[:batch_size]

    def _stack_gt_labels(self, data_samples, device: torch.device) -> torch.Tensor:
        if data_samples is None:
            raise ValueError('AdaCLIP training requires `data_samples` with gt_label.')

        labels = []
        for sample in data_samples:
            label = getattr(sample, 'gt_label', None)
            if label is None:
                raise ValueError('AdaCLIP training requires each sample to provide gt_label.')
            labels.append(torch.as_tensor(label, device=device, dtype=torch.float32))
        return torch.stack(labels, dim=0).view(-1, 1)

    def _stack_gt_masks(
        self,
        data_samples,
        device: torch.device,
        target_size,
    ) -> torch.Tensor:
        if data_samples is None:
            raise ValueError('AdaCLIP training requires `data_samples` with gt_mask.')

        masks = []
        for sample in data_samples:
            mask = getattr(sample, 'gt_mask', None)
            if mask is None:
                raise ValueError('AdaCLIP training requires each sample to provide gt_mask.')
            mask_tensor = torch.as_tensor(mask, device=device, dtype=torch.float32)
            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
            elif mask_tensor.ndim != 3:
                raise ValueError(f'Unexpected AdaCLIP gt_mask shape: {tuple(mask_tensor.shape)}')
            masks.append(mask_tensor)

        masks = torch.stack(masks, dim=0)
        masks = (masks > 0.5).float()
        if masks.shape[-2:] != target_size:
            masks = F.interpolate(masks, size=target_size, mode='nearest')
        return masks

    def _compute_training_loss(
        self,
        x: torch.Tensor,
        data_samples,
        class_names: List[str],
    ) -> dict:
        self._debug_step += 1
        labels = self._stack_gt_labels(data_samples, device=x.device)
        gt_masks = self._stack_gt_masks(data_samples, device=x.device, target_size=(self.image_size, self.image_size))

        anomaly_maps, anomaly_score = self.adaclip(x, class_names, aggregation=False)
        if not isinstance(anomaly_maps, list):
            anomaly_maps = [anomaly_maps]

        loss_cls = self.loss_focal(anomaly_score, labels)
        gt_masks_flat = gt_masks.squeeze(1)

        loss_seg = anomaly_score.new_tensor(0.0)
        for anomaly_map in anomaly_maps:
            loss_seg = loss_seg + self.loss_focal(anomaly_map, gt_masks)
            loss_seg = loss_seg + self.loss_dice(anomaly_map[:, 1, :, :], gt_masks_flat)
            loss_seg = loss_seg + self.loss_dice(anomaly_map[:, 0, :, :], 1 - gt_masks_flat)

        return {
            'loss': loss_cls + loss_seg,
            'loss_cls': loss_cls,
            'loss_seg': loss_seg,
        }

    def _build_debug_context(self, data_samples, class_names: List[str], mode: str) -> dict:
        context = {
            'mode': mode,
            'classes': class_names,
            'step': int(self._debug_step),
        }
        if data_samples is not None:
            context['defect_types'] = [getattr(sample, 'defect_type', None) for sample in data_samples]
            context['img_paths'] = [getattr(sample, 'img_path', None) for sample in data_samples]
        return context

    def forward(
        self,
        inputs: Union[torch.Tensor, List[torch.Tensor]],
        data_samples=None,
        mode: str = "tensor",
    ):
        """Forward pass.

        Args:
            inputs: Input images (B, C, H, W) or list of tensors.
            data_samples: Data samples with class info.
            mode: 'loss', 'predict', or 'tensor'.

        Returns:
            For 'loss': dict of classification and segmentation losses.
            For 'predict': list of ADDataSample predictions.
            For 'tensor': tuple of (cls_features, patch_features).
        """
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        # Normalize for CLIP
        x = self._normalize_for_clip(inputs)

        # Resize if needed
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(
                x, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False
            )

        B = inputs.shape[0]
        class_names = self._resolve_batch_classes(data_samples, B)
        self.adaclip.debug_finite_checks = self.debug_finite_checks
        self.adaclip.debug_output_path = self.debug_output_path
        self.adaclip.debug_context = self._build_debug_context(data_samples, class_names, mode)

        if mode == "loss":
            if not self.enable_train_loss:
                return {"loss": torch.tensor(0.0, device=x.device, requires_grad=True)}
            return self._compute_training_loss(x, data_samples, class_names)

        # Forward through AdaCLIP
        anomaly_map, anomaly_score = self.adaclip(x, class_names, aggregation=True)

        # Reshape anomaly map
        anomaly_map = anomaly_map.unsqueeze(1)  # (B, 1, H, W)

        # Upsample to input size if needed
        if anomaly_map.shape[-2:] != inputs.shape[-2:]:
            anomaly_map = F.interpolate(
                anomaly_map,
                size=inputs.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # Apply Gaussian blur
        if self.gaussian_sigma > 0:
            anomaly_map = self._gaussian_blur(anomaly_map, self.gaussian_sigma)

        if mode == "predict":
            return build_predict_results(data_samples, anomaly_score, anomaly_map)

        # Tensor mode
        return anomaly_score, anomaly_map

    def train(self, mode: bool = True):
        """Override train to keep CLIP frozen."""
        super().train(mode)
        self.clip.eval()
        return self
