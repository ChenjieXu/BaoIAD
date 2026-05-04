"""ComposeAD: Modular trick composition detector.

Demonstrates BaoIAD's modularity by combining tricks from different
anomaly detection methods via a swappable ScoringHead.

Usage in config:
    model = dict(
        type='ComposeAD',
        backbone=dict(type='TIMMBackbone', ...),
        neck=dict(type='MultiScalePooling', output_size=28),  # optional
        scoring_head=dict(type='KNNScoringHead', ...),        # swap this!
    )
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from baoiad.models.base_ad_model import BaseADModel
from baoiad.registry import MODELS


@MODELS.register_module(force=True)
class ComposeAD(BaseADModel):
    """Composable anomaly detector with swappable scoring heads.

    The scoring_head encapsulates feature processing, normal modeling,
    and anomaly scoring — all configurable via MMEngine config.

    Args:
        backbone: Backbone config dict.
        neck: Neck config dict (optional, e.g. MultiScalePooling).
        scoring_head: ScoringHead config dict (required).
        freeze_backbone: Whether to freeze backbone. Default True.
    """

    def __init__(
        self,
        backbone: dict,
        neck: Optional[dict] = None,
        scoring_head: Optional[dict] = None,
        freeze_backbone: bool = True,
        data_preprocessor=None,
        init_cfg=None,
    ):
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=None,  # We use scoring_head instead
            freeze_backbone=freeze_backbone,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
        )
        if scoring_head is None:
            raise ValueError('ComposeAD requires a scoring_head config.')
        self.scoring_head = MODELS.build(scoring_head)

    def forward(
        self,
        inputs: Union[Tensor, List[Tensor]],
        data_samples: Optional[List] = None,
        mode: str = 'tensor',
    ) -> Union[Dict[str, Tensor], List, Tuple[Tensor, ...]]:
        """Unified forward with three modes.

        Args:
            inputs: Batch images (B, C, H, W).
            data_samples: List of ADDataSample.
            mode: 'loss', 'predict', or 'tensor'.

        Returns:
            - 'loss': dict of loss tensors (delegated to scoring_head).
            - 'predict': list of predictions with anomaly scores.
            - 'tensor': raw feature tensors.
        """
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        feats = self.extract_feat(inputs)
        if isinstance(feats, Tensor):
            feats = (feats,)

        if mode == 'loss':
            return self.scoring_head.loss(feats, data_samples)
        elif mode == 'predict':
            return self.scoring_head.predict(feats, data_samples)
        elif mode == 'tensor':
            return feats
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def build_memory_bank(self) -> None:
        """Build scoring model after training (called by MemoryBankHook)."""
        self.scoring_head.fit()
