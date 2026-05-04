"""Registered loss modules for BaoIAD (mmdetection-style)."""
from baoiad.models.losses.mse_loss import MSELoss  # noqa: F401
from baoiad.models.losses.focal_loss import BinaryFocalLoss, FocalLoss, sigmoid_focal_loss  # noqa: F401
from baoiad.models.losses.cosine_loss import CosineDistanceLoss  # noqa: F401
from baoiad.models.losses.ssim_loss import SSIMLoss  # noqa: F401
from baoiad.models.losses.bce_loss import BCEWithLogitsLoss, BCELoss  # noqa: F401
from baoiad.models.losses.l1_loss import L1Loss, SmoothL1Loss  # noqa: F401
from baoiad.models.losses.cross_entropy_loss import CrossEntropyLoss  # noqa: F401
from baoiad.models.losses.dice_loss import BinaryDiceLoss  # noqa: F401

__all__ = [
    'MSELoss', 'BinaryFocalLoss', 'FocalLoss', 'sigmoid_focal_loss',
    'CosineDistanceLoss',
    'SSIMLoss', 'BCEWithLogitsLoss', 'BCELoss', 'L1Loss', 'SmoothL1Loss',
    'CrossEntropyLoss',
    'BinaryDiceLoss',
]
