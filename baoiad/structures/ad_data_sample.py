"""Anomaly-detection specific data sample."""

from typing import TYPE_CHECKING

from mmengine.structures import BaseDataElement
from torch import Tensor


class ADDataSample(BaseDataElement):
    """Data sample for anomaly detection tasks.

    This class extends ``BaseDataElement`` with anomaly detection specific
    fields including ground truth labels/masks and predictions.

    **Meta fields** (set via ``sample.set_metainfo({...})``):
        - cls_name (str): Category name, e.g., 'bottle', 'cable'.
        - img_path (str): Path to the image file.
        - defect_type (str): Defect type name, e.g., 'broken', 'good'.

    **Data fields** (set via ``sample.gt_label = value``):
        - gt_label (int): Ground truth label (0=normal, 1=anomaly).
        - gt_mask (Tensor): Ground truth segmentation mask (H, W).
        - pred_score (float): Predicted anomaly score.
        - pred_anomaly_map (Tensor): Predicted anomaly map (1, H, W).

    Examples:
        >>> sample = ADDataSample()
        >>> sample.set_metainfo({'cls_name': 'bottle', 'img_path': '/data/001.png'})
        >>> sample.gt_label = 1
        >>> sample.pred_score = 0.85
        >>> print(sample.cls_name)
        'bottle'
    """

    # Type hints for IDE autocompletion and type checking.
    # Actual storage is handled by BaseDataElement's dynamic mechanism.
    if TYPE_CHECKING:
        gt_label: int
        gt_mask: Tensor
        pred_score: float
        pred_score_mean: float
        pred_score_max: float
        pred_nfa_score: float
        pred_anomaly_map: Tensor
        pred_anomaly_map_raw: Tensor
        pred_nfa_anomaly_map: Tensor
        cls_name: str
        img_path: str
        defect_type: str
