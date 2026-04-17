"""PatchCore anomaly detector."""

from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from baoiad.models.base_ad_model import BaseADModel
from baoiad.registry import MODELS


@MODELS.register_module()
class PatchCore(BaseADModel):
    """PatchCore: Towards Total Recall in Industrial Anomaly Detection.

    PatchCore uses a pre-trained backbone to extract patch features,
    builds a coreset memory bank from normal training data, and
    detects anomalies via kNN distance at test time.

    No gradient-based training is needed.

    Args:
        backbone: Backbone config.
        neck: Neck config (typically MultiScalePooling).
        head: Head config (typically MemoryBankHead).
        freeze_backbone: Always True for PatchCore.
        init_cfg: Initialization config.
    """

    def __init__(
        self,
        backbone: dict,
        neck: Optional[dict] = None,
        head: Optional[dict] = None,
        freeze_backbone: bool = True,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            freeze_backbone=True,  # Always freeze for PatchCore
            init_cfg=init_cfg,
        )

    def forward(
        self,
        inputs: Tensor,
        data_samples: Optional[List] = None,
        mode: str = 'tensor',
    ) -> Union[Dict[str, Tensor], List, Tuple[Tensor, ...]]:
        """Forward pass.

        For PatchCore:
        - 'loss' mode: extracts features and collects them (no actual loss).
        - 'predict' mode: computes anomaly scores via kNN.
        - 'tensor' mode: returns extracted features.
        """
        feats = self.extract_feat(inputs)

        if mode == 'loss':
            return self.head.loss(feats, data_samples)
        elif mode == 'predict':
            return self.head.predict(feats, data_samples)
        elif mode == 'tensor':
            return feats
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def build_memory_bank(self) -> None:
        """Build the coreset memory bank after feature collection."""
        self.head.build_memory_bank()
