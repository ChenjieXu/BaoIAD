"""Official-style MemSeg train loop."""

from __future__ import annotations

import bisect
import copy
import logging
import math
from typing import Any, Dict

from mmengine.dataset import pseudo_collate
from mmengine.logging import print_log
from mmengine.registry import LOOPS as MMENGINE_LOOPS
from mmengine.runner import BaseLoop
from mmengine.runner.loops import calc_dynamic_intervals
from torch.utils.data import DataLoader

from baoiad.registry import DATASETS, LOOPS


def _build_memseg_dataset(dataset_cfg: Dict[str, Any]):
    dataset_cfg = copy.deepcopy(dataset_cfg)
    dataset = DATASETS.build(dataset_cfg)
    if hasattr(dataset, 'full_init'):
        dataset.full_init()
    return dataset


def _build_memseg_official_dataloader(dataloader_cfg: Dict[str, Any]) -> DataLoader:
    """Build a raw PyTorch DataLoader like the frozen reference."""
    dataloader_cfg = copy.deepcopy(dataloader_cfg)
    dataset = _build_memseg_dataset(dataloader_cfg.pop('dataset'))
    dataloader_cfg.pop('sampler', None)
    dataloader_cfg.pop('batch_sampler', None)
    dataloader_cfg.pop('collate_fn', None)
    dataloader_cfg.pop('worker_init_fn', None)

    batch_size = int(dataloader_cfg.pop('batch_size'))
    num_workers = int(dataloader_cfg.pop('num_workers', 0))
    drop_last = bool(dataloader_cfg.pop('drop_last', False))
    pin_memory = bool(dataloader_cfg.pop('pin_memory', False))
    persistent_workers = bool(dataloader_cfg.pop('persistent_workers', False)) and num_workers > 0

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=pseudo_collate,
    )


@LOOPS.register_module()
@MMENGINE_LOOPS.register_module()
class MemSegOfficialTrainLoop(BaseLoop):
    """Mirror the frozen MemSeg ``while + for trainloader`` iteration rhythm."""

    def __init__(
        self,
        runner,
        dataloader,
        max_iters: int,
        val_begin: int = 1,
        val_interval: int = 1,
        dynamic_intervals=None,
    ) -> None:
        self._runner = runner
        self._dataloader_cfg = copy.deepcopy(dataloader) if isinstance(dataloader, dict) else dataloader
        self._dataloader = None
        self._max_iters = int(max_iters)
        self._iter = 0
        self._epoch = 0
        self._max_epochs = 0
        self.val_begin = int(val_begin)
        self.val_interval = int(val_interval)
        self.stop_training = False
        self.dynamic_milestones, self.dynamic_intervals = calc_dynamic_intervals(
            self.val_interval, dynamic_intervals)

    @property
    def runner(self):
        return self._runner

    @property
    def dataloader(self):
        if self._dataloader is None:
            if isinstance(self._dataloader_cfg, dict):
                self._dataloader = _build_memseg_official_dataloader(self._dataloader_cfg)
            else:
                self._dataloader = self._dataloader_cfg

            if hasattr(self._dataloader.dataset, 'metainfo'):
                self.runner.visualizer.dataset_meta = self._dataloader.dataset.metainfo
            else:
                print_log(
                    f'Dataset {self._dataloader.dataset.__class__.__name__} has no metainfo. '
                    '``dataset_meta`` in visualizer will be None.',
                    logger='current',
                    level=logging.WARNING,
                )
            self._max_epochs = max(1, math.ceil(self._max_iters / max(len(self._dataloader), 1)))
        return self._dataloader

    @property
    def max_iters(self):
        return self._max_iters

    @property
    def max_epochs(self):
        _ = self.dataloader
        return self._max_epochs

    @property
    def epoch(self):
        return self._epoch

    @property
    def iter(self):
        return self._iter

    def run(self):
        _ = self.dataloader
        self.runner.call_hook('before_train')
        self.runner.call_hook('before_train_epoch')

        while self._iter < self._max_iters and not self.stop_training:
            self.runner.model.train()
            for batch_idx, data_batch in enumerate(self.dataloader):
                if self._iter >= self._max_iters or self.stop_training:
                    break
                self.run_iter(batch_idx, data_batch)
                self._decide_current_val_interval()
                if (
                    self.runner.val_loop is not None
                    and self._iter >= self.val_begin
                    and (self._iter % self.val_interval == 0 or self._iter == self._max_iters)
                ):
                    self.runner.val_loop.run()
            self._epoch += 1

        self.runner.call_hook('after_train_epoch')
        self.runner.call_hook('after_train')
        return self.runner.model

    def run_iter(self, batch_idx, data_batch):
        self.runner.call_hook('before_train_iter', batch_idx=batch_idx, data_batch=data_batch)
        outputs = self.runner.model.train_step(data_batch, optim_wrapper=self.runner.optim_wrapper)
        self.runner.call_hook(
            'after_train_iter',
            batch_idx=batch_idx,
            data_batch=data_batch,
            outputs=outputs,
        )
        self._iter += 1

    def _decide_current_val_interval(self) -> None:
        step = bisect.bisect(self.dynamic_milestones, (self.iter + 1))
        self.val_interval = self.dynamic_intervals[step - 1]
