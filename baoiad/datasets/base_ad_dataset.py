"""Base anomaly detection dataset."""

from typing import Dict, List, Optional, Sequence

from mmengine.dataset import BaseDataset

from baoiad.registry import DATASETS


class BaseADDataset(BaseDataset):
    """Base class for anomaly detection datasets.

    Supports both multi-class (all categories in one dataset) and
    single-class (one category per dataset instance) modes.

    Args:
        data_root: Root directory of the dataset.
        split: 'train' or 'test'.
        cls_names: List of category names to include. None means all.
        multi_class: If True, load all categories together.
        pipeline: Data processing pipeline configs.
        **kwargs: Additional arguments for BaseDataset.
    """

    METAINFO: dict = dict(task='anomaly_detection')

    # Subclasses should override this with the full list of categories.
    ALL_CATEGORIES: Sequence[str] = ()

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        cls_names: Optional[List[str]] = None,
        multi_class: bool = True,
        pipeline: Optional[List[dict]] = None,
        **kwargs,
    ) -> None:
        self.split = split
        self.multi_class = multi_class
        if cls_names is not None:
            self.cls_names = cls_names
        elif multi_class:
            self.cls_names = list(self.ALL_CATEGORIES)
        else:
            raise ValueError(
                'cls_names must be provided when multi_class=False. '
                'Specify one or more category names explicitly.'
            )

        pipeline = pipeline or []
        # Only pass kwargs that BaseDataset actually accepts
        import inspect
        from mmengine.dataset import BaseDataset
        _bd_params = set(inspect.signature(BaseDataset.__init__).parameters) - {'self'}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in _bd_params}
        super().__init__(
            data_root=data_root,
            pipeline=pipeline,
            **filtered_kwargs,
        )

    def load_data_list(self) -> List[Dict]:
        """Load annotation list. Must be implemented by subclass."""
        raise NotImplementedError
