"""PNI (Position and Neighborhood Information) anomaly detector.

Implements PNI from:
"Position and Neighborhood Information for Anomaly Detection" (ICCV 2023)
https://arxiv.org/abs/2211.12634

PNI improves upon PatchCore by incorporating:
1. Position-specific feature distributions
2. Neighborhood conditional probability via MLP
3. Dual coreset for distance and probability estimation
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from baoiad.models.base_ad_model import BaseADModel
from baoiad.registry import MODELS


@MODELS.register_module(force=True)
class PNI(BaseADModel):
    """PNI: Position and Neighborhood Information detector.

    PNI extends PatchCore with position histograms and a neighborhood MLP
    to model p(c_dist | position) and p(c_dist | neighborhood), achieving
    improved anomaly detection performance.

    No gradient-based training is needed for the detector itself;
    only the internal MLP is trained during memory bank building.

    Args:
        backbone: Backbone config.
        neck: Neck config (typically MultiScalePooling).
        head: Head config (PNIHead).
        freeze_backbone: Always True for PNI.
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
            freeze_backbone=True,  # Always freeze for PNI
            init_cfg=init_cfg,
        )

    def forward(
        self,
        inputs: Tensor,
        data_samples: Optional[List] = None,
        mode: str = 'tensor',
    ) -> Union[Dict[str, Tensor], List, Tuple[Tensor, ...]]:
        """Forward pass.

        For PNI:
        - 'loss' mode: extracts features and collects them (no actual loss).
        - 'predict' mode: computes anomaly scores using PNI scoring.
        - 'tensor' mode: returns extracted features.

        Args:
            inputs: Input images (B, C, H, W).
            data_samples: Optional list of data samples.
            mode: One of 'loss', 'predict', 'tensor'.

        Returns:
            - 'loss': dict with dummy loss.
            - 'predict': list of data samples with predictions.
            - 'tensor': tuple of feature tensors.
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
        """Build dual coreset, position histograms, and train MLP."""
        self.head.build_memory_bank()
