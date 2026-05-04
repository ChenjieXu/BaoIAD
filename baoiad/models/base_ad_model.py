"""Base anomaly detection model and sub-class hierarchy.

Architecture follows mmdetection pattern:
  BaseADModel (backbone → neck → head, 3-mode forward)
    ├── MemoryBankADModel (fit/build_memory_bank lifecycle)
    ├── KnowledgeDistillationADModel (teacher-student with frozen teacher)
    ├── FlowBasedADModel (normalizing flows on backbone features)
    ├── ReconstructionADModel (autoencoder/reconstruction-based)
    ├── VisionLanguageADModel (CLIP-based zero/few-shot)
    └── DiscriminatorADModel (feature discrimination with noise)
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine.model import BaseModel
from torch import Tensor

from baoiad.registry import MODELS

logger = logging.getLogger(__name__)


class BaseADModel(BaseModel):
    """Base class for all anomaly detection models.

    Follows the backbone → neck → head architecture.
    Backbone is frozen by default (common in AD methods).

    Args:
        backbone: Backbone config dict.
        neck: Neck config dict (optional).
        head: Head config dict.
        freeze_backbone: Whether to freeze backbone parameters.
        data_preprocessor: Data preprocessor config dict.
        init_cfg: Initialization config.
    """

    def __init__(
        self,
        backbone: Optional[dict] = None,
        neck: Optional[dict] = None,
        head: Optional[dict] = None,
        freeze_backbone: bool = True,
        data_preprocessor: Optional[dict] = None,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
        )
        if backbone is not None:
            self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck) if neck else None
        self.head = MODELS.build(head) if head else None
        self.freeze_backbone = freeze_backbone

        if freeze_backbone and hasattr(self, 'backbone'):
            self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _freeze_module(self, module: nn.Module) -> None:
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def extract_feat(self, batch_inputs: Union[Tensor, List[Tensor]]) -> Tuple[Tensor, ...]:
        if isinstance(batch_inputs, (list, tuple)):
            batch_inputs = torch.stack(batch_inputs)
        ctx = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
        with ctx:
            feats = self.backbone(batch_inputs)
        if isinstance(feats, Tensor):
            feats = (feats,)
        if self.neck is not None:
            feats = self.neck(feats)
        return feats

    def forward(
        self,
        inputs: Union[Tensor, List[Tensor]],
        data_samples: Optional[List] = None,
        mode: str = 'tensor',
    ) -> Union[Dict[str, Tensor], List, Tuple[Tensor, ...]]:
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)
        feats = self.extract_feat(inputs)

        if mode == 'loss':
            if self.head is None:
                return dict()
            return self.head.loss(feats, data_samples)
        elif mode == 'predict':
            if self.head is None:
                return list(feats)
            return self.head.predict(feats, data_samples)
        elif mode == 'tensor':
            return feats
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    @staticmethod
    def _stack_inputs(inputs: Union[Tensor, List[Tensor]]) -> Tensor:
        if isinstance(inputs, (list, tuple)):
            return torch.stack(inputs)
        return inputs

    def build_memory_bank(self) -> None:
        pass

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone and hasattr(self, 'backbone'):
            self.backbone.eval()
        return self


class MemoryBankADModel(BaseADModel):
    """Base for memory-bank-based detectors (PatchCore, PaDiM, DFKDE, DFM).

    Provides: feature collection during training, fit() after training,
    build_memory_bank() lifecycle hook, and frozen backbone with eval-mode override.
    """

    def __init__(self, backbone: Optional[dict] = None, freeze_backbone: bool = True,
                 **kwargs):
        super().__init__(backbone=backbone, freeze_backbone=freeze_backbone, **kwargs)
        self._memory_bank: List[Tensor] = []

    def _collect_features(self, feats: Tuple[Tensor, ...]) -> None:
        for f in feats:
            self._memory_bank.append(f.cpu())

    def _clear_memory_bank(self) -> None:
        self._memory_bank.clear()

    # Canonical entry point invoked by MemoryBankHook after training. Subclasses
    # override this. `fit()` is provided as a backward-compatible alias and must
    # NOT be overridden to call back into `build_memory_bank` to avoid recursion.
    def build_memory_bank(self) -> None:
        return None

    def fit(self) -> None:
        self.build_memory_bank()


class KnowledgeDistillationADModel(BaseADModel):
    """Base for teacher-student detectors (RD, EfficientAD, RD++, Dinomaly).

    Provides: teacher (frozen) + student (trainable) pattern,
    extract_teacher_feats() with no_grad, train() override.
    """

    def __init__(self, backbone: Optional[dict] = None, freeze_backbone: bool = True,
                 **kwargs):
        super().__init__(backbone=backbone, freeze_backbone=freeze_backbone, **kwargs)

    def extract_teacher_feats(self, batch_inputs: Tensor) -> Tuple[Tensor, ...]:
        with torch.no_grad():
            feats = self.backbone(batch_inputs)
        if isinstance(feats, Tensor):
            feats = (feats,)
        return feats

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, 'backbone') and self.freeze_backbone:
            self.backbone.eval()
        if hasattr(self, 'teacher'):
            self.teacher.eval()
        return self


class FlowBasedADModel(BaseADModel):
    """Base for normalizing-flow detectors (FastFlow, CFlow, UFlow, DifferNet).

    Provides: frozen backbone + trainable flows, NLL loss computation,
    z-score anomaly map extraction.
    """

    def __init__(self, backbone: Optional[dict] = None, freeze_backbone: bool = True,
                 **kwargs):
        super().__init__(backbone=backbone, freeze_backbone=freeze_backbone, **kwargs)

    def compute_flow_loss(self, outputs: List[Tensor]) -> Tensor:
        nll = 0.0
        for output in outputs:
            nll += 0.5 * output.pow(2).mean()
        return nll

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, 'backbone') and self.freeze_backbone:
            self.backbone.eval()
        return self


class ReconstructionADModel(BaseADModel):
    """Base for reconstruction-based detectors (DRAEM, MemSeg, DeSTSeg, autoencoder baselines).

    Provides: reconstruction + comparison pattern, optional anomaly generation.
    """

    def __init__(self, backbone: Optional[dict] = None, freeze_backbone: bool = False,
                 **kwargs):
        super().__init__(backbone=backbone, freeze_backbone=freeze_backbone, **kwargs)


class VisionLanguageADModel(BaseADModel):
    """Base for CLIP-based detectors (WinCLIP, AnomalyCLIP, AnoVL, MuSc, AdaCLIP, AACLIP).

    Provides: CLIP normalization, text embedding caching, eval-mode override.
    """

    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, backbone: Optional[dict] = None, freeze_backbone: bool = True,
                 **kwargs):
        super().__init__(backbone=backbone, freeze_backbone=freeze_backbone, **kwargs)

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, 'backbone'):
            self.backbone.eval()
        return self


class DiscriminatorADModel(BaseADModel):
    """Base for discriminator-based detectors (SimpleNet, SuperSimpleNet, CFA).

    Provides: frozen backbone + trainable discriminator, feature preprocessing.
    """

    def __init__(self, backbone: Optional[dict] = None, freeze_backbone: bool = True,
                 **kwargs):
        super().__init__(backbone=backbone, freeze_backbone=freeze_backbone, **kwargs)

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, 'backbone') and self.freeze_backbone:
            self.backbone.eval()
        return self
