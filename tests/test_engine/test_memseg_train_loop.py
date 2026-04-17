"""Tests for MemSeg official train loop."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import baoiad  # noqa: F401
from mmengine.dataset import pseudo_collate
from torch.utils.data import RandomSampler

from baoiad.engine.loops.memseg_train_loop import _build_memseg_official_dataloader


def test_build_memseg_official_dataloader_uses_raw_shuffle_loader(tmp_mvtec_dir):
    dataloader_cfg = dict(
        batch_size=2,
        num_workers=0,
        persistent_workers=False,
        sampler=dict(type='DefaultSampler', shuffle=True),
        dataset=dict(
            type='MVTecADDataset',
            data_root=str(tmp_mvtec_dir),
            split='train',
            cls_names=['bottle'],
            multi_class=False,
            pipeline=[],
        ),
    )

    loader = _build_memseg_official_dataloader(dataloader_cfg)

    assert loader.batch_size == 2
    assert loader.num_workers == 0
    assert loader.drop_last is False
    assert loader.collate_fn is pseudo_collate
    assert isinstance(loader.sampler, RandomSampler)


def test_memseg_official_loop_registered():
    from baoiad.registry import LOOPS

    assert LOOPS.get('MemSegOfficialTrainLoop') is not None


def test_memseg_official_loop_reports_max_epochs(tmp_mvtec_dir):
    from baoiad.engine.loops.memseg_train_loop import MemSegOfficialTrainLoop

    runner = SimpleNamespace(
        visualizer=SimpleNamespace(dataset_meta=None),
    )
    dataloader_cfg = dict(
        batch_size=2,
        num_workers=0,
        persistent_workers=False,
        sampler=dict(type='DefaultSampler', shuffle=True),
        dataset=dict(
            type='MVTecADDataset',
            data_root=str(tmp_mvtec_dir),
            split='train',
            cls_names=['bottle'],
            multi_class=False,
            pipeline=[],
        ),
    )
    loop = MemSegOfficialTrainLoop(runner, dataloader=dataloader_cfg, max_iters=5, val_interval=1)

    assert loop.max_iters == 5
    assert loop.max_epochs == 3
