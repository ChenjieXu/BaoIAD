"""Official-style ResAD train loop with staged optimization."""

from __future__ import annotations

import bisect
import copy
import logging
from typing import Any, Dict, Sequence

import torch
from mmengine.logging import print_log
from mmengine.registry import LOOPS as MMENGINE_LOOPS
from mmengine.runner import BaseLoop
from mmengine.runner.loops import calc_dynamic_intervals
from torch.utils.data import DataLoader

from baoiad.registry import DATASETS, LOOPS


def resad_official_collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate ResAD batches."""
    inputs = torch.stack([sample['inputs'] for sample in batch], dim=0)
    data_samples = [sample['data_samples'] for sample in batch]
    return {
        'inputs': inputs,
        'data_samples': data_samples,
    }


def _build_resad_dataset(dataset_cfg: Dict[str, Any]):
    dataset_cfg = copy.deepcopy(dataset_cfg)
    dataset = DATASETS.build(dataset_cfg)
    if hasattr(dataset, 'full_init'):
        dataset.full_init()
    return dataset


def _build_resad_official_dataloader(dataloader_cfg: Dict[str, Any]) -> DataLoader:
    dataloader_cfg = copy.deepcopy(dataloader_cfg)
    dataset = _build_resad_dataset(dataloader_cfg.pop('dataset'))
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
        collate_fn=resad_official_collate,
    )


@LOOPS.register_module(force=True)
@MMENGINE_LOOPS.register_module(force=True)
class ResADOfficialTrainLoop(BaseLoop):
    """Train loop for ResAD with staged optimization.

    ResAD official training has three separate optimizer steps per batch:
    1. VQ loss → backward → VQ optimizer step
    2. Constraintor forward + OCC loss → backward → constraintor optimizer step
    3. Flow loss on detached constraintor output → backward → flow optimizer step

    This loop lazily builds the DataLoader (like ViTADOfficialTrainLoop) and
    handles the staged training via a custom train_step on the model.
    """

    def __init__(
        self,
        runner,
        dataloader,
        max_epochs: int,
        val_begin: int = 1,
        val_interval: int = 1,
        dynamic_intervals=None,
        first_stage_epochs: int = 10,
        N_batch: int = 8192,
    ) -> None:
        self._runner = runner
        self._dataloader_cfg = copy.deepcopy(dataloader) if isinstance(dataloader, dict) else dataloader
        self._dataloader = None
        self._max_epochs = int(max_epochs)
        assert self._max_epochs == max_epochs, (
            f'`max_epochs` should be an integer, but got {max_epochs}.'
        )
        self._max_iters = 0
        self._epoch = 0
        self._iter = 0
        self.val_begin = int(val_begin)
        self.val_interval = int(val_interval)
        self.stop_training = False
        self.dynamic_milestones, self.dynamic_intervals = calc_dynamic_intervals(
            self.val_interval, dynamic_intervals)
        self.first_stage_epochs = first_stage_epochs
        self.N_batch = N_batch

    @property
    def runner(self):
        return self._runner

    @property
    def dataloader(self):
        if self._dataloader is None:
            if isinstance(self._dataloader_cfg, dict):
                self._dataloader = _build_resad_official_dataloader(self._dataloader_cfg)
            else:
                self._dataloader = self._dataloader_cfg

            self._max_iters = self._max_epochs * len(self._dataloader)
            if hasattr(self._dataloader.dataset, 'metainfo'):
                self.runner.visualizer.dataset_meta = self._dataloader.dataset.metainfo
            else:
                print_log(
                    f'Dataset {self._dataloader.dataset.__class__.__name__} has no metainfo. '
                    '``dataset_meta`` in visualizer will be None.',
                    logger='current',
                    level=logging.WARNING,
                )
        return self._dataloader

    @property
    def max_epochs(self):
        return self._max_epochs

    @property
    def max_iters(self):
        _ = self.dataloader
        return self._max_iters

    @property
    def epoch(self):
        return self._epoch

    @property
    def iter(self):
        return self._iter

    def run(self) -> torch.nn.Module:
        _ = self.dataloader
        self.runner.call_hook('before_train')

        while self._epoch < self._max_epochs and not self.stop_training:
            self.run_epoch()
            self._decide_current_val_interval()
            if (
                self.runner.val_loop is not None
                and self._epoch >= self.val_begin
                and (self._epoch % self.val_interval == 0 or self._epoch == self._max_epochs)
            ):
                self.runner.val_loop.run()

        self.runner.call_hook('after_train')
        return self.runner.model

    def run_epoch(self) -> None:
        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()
        # Update epoch info for stage switching
        if hasattr(self.runner.model, 'set_epoch_info'):
            self.runner.model.set_epoch_info(self._epoch, self._max_epochs)

        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        self.runner.call_hook('after_train_epoch')
        self._epoch += 1

    def run_iter(self, idx, data_batch) -> None:
        self.runner.call_hook('before_train_iter', batch_idx=idx, data_batch=data_batch)
        outputs = self.runner.model.train_step(
            data_batch,
            optim_wrapper=self.runner.optim_wrapper,
            epoch=self._epoch,
            first_stage_epochs=self.first_stage_epochs,
            N_batch=self.N_batch,
        )
        self.runner.call_hook('after_train_iter', batch_idx=idx, data_batch=data_batch, outputs=outputs)
        self._iter += 1

    def _decide_current_val_interval(self) -> None:
        step = bisect.bisect(self.dynamic_milestones, (self.epoch + 1))
        self.val_interval = self.dynamic_intervals[step - 1]
