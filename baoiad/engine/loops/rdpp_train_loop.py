"""Official RD++ train loop helpers."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

from mmengine.registry import LOOPS as MMENGINE_LOOPS
from mmengine.runner import EpochBasedTrainLoop

from baoiad.registry import LOOPS


DEFAULT_RDPP_CATEGORY_EPOCHS: Dict[str, int] = {
    'bottle': 200,
    'cable': 240,
    'capsule': 300,
    'carpet': 10,
    'grid': 260,
    'hazelnut': 160,
    'leather': 10,
    'metal_nut': 160,
    'pill': 200,
    'screw': 280,
    'tile': 260,
    'toothbrush': 280,
    'transistor': 300,
    'wood': 100,
    'zipper': 300,
}


@LOOPS.register_module(force=True)
@MMENGINE_LOOPS.register_module(force=True)
class RDPPTrainLoop(EpochBasedTrainLoop):
    """Match the official RD++ per-category epoch schedule."""

    def __init__(
        self,
        runner,
        dataloader,
        max_epochs: int,
        val_begin: int = 1,
        val_interval: int = 1,
        dynamic_intervals=None,
        category_epochs: Optional[Dict[str, int]] = None,
    ) -> None:
        self.category_epochs = dict(DEFAULT_RDPP_CATEGORY_EPOCHS)
        if category_epochs is not None:
            self.category_epochs.update({str(k): int(v) for k, v in category_epochs.items()})
        super().__init__(
            runner,
            dataloader,
            max_epochs=max_epochs,
            val_begin=val_begin,
            val_interval=val_interval,
            dynamic_intervals=dynamic_intervals,
        )

    @staticmethod
    def resolve_max_epochs(
        cls_names: Optional[Iterable[str]],
        category_epochs: Dict[str, int],
        fallback_max_epochs: int,
    ) -> int:
        names = list(cls_names or [])
        if len(names) != 1:
            return int(fallback_max_epochs)
        return int(category_epochs.get(names[0], fallback_max_epochs))

    def run(self):
        dataset = getattr(self.dataloader, 'dataset', None)
        cls_names = getattr(dataset, 'cls_names', None)
        self._max_epochs = self.resolve_max_epochs(cls_names, self.category_epochs, self._max_epochs)
        self._max_iters = self._max_epochs * len(self.dataloader)
        return super().run()
