"""Tests for RegAD strict support-set helpers."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from baoiad.utils.regad_strict import load_or_sample_support_rounds


def _write_rgb(path: Path, value: int) -> None:
    image = np.full((8, 8, 3), value, dtype=np.uint8)
    Image.fromarray(image, mode='RGB').save(path)


def test_load_or_sample_support_rounds_prefers_official_file(tmp_path):
    support_file = tmp_path / 'support_set' / 'bottle' / '4_10.pt'
    support_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        [
            torch.full((4, 3, 8, 8), 255, dtype=torch.uint8),
            torch.zeros((4, 3, 8, 8), dtype=torch.float32),
        ],
        support_file,
    )

    rounds, source, used_file = load_or_sample_support_rounds(
        data_root=str(tmp_path / 'mvtec'),
        target_cls='bottle',
        img_size=8,
        shot=4,
        inferences=10,
        seed=668,
        support_set_root=str(tmp_path / 'support_set'),
        allow_fallback=False,
    )

    assert source == 'official'
    assert used_file == support_file
    assert len(rounds) == 2
    assert rounds[0].dtype == torch.float32
    assert float(rounds[0].max().item()) == pytest.approx(1.0)
    assert float(rounds[1].min().item()) == pytest.approx(0.0)


def test_load_or_sample_support_rounds_requires_official_file_when_requested(tmp_path):
    with pytest.raises(FileNotFoundError, match='REGAD_SUPPORT_SET_ROOT'):
        load_or_sample_support_rounds(
            data_root=str(tmp_path / 'mvtec'),
            target_cls='bottle',
            img_size=8,
            shot=4,
            inferences=10,
            seed=668,
            support_set_root=str(tmp_path / 'support_set'),
            allow_fallback=False,
        )


def test_load_or_sample_support_rounds_rejects_empty_official_file(tmp_path):
    support_file = tmp_path / 'support_set' / 'bottle' / '4_10.pt'
    support_file.parent.mkdir(parents=True, exist_ok=True)
    support_file.touch()

    with pytest.raises(RuntimeError, match='unreadable or empty'):
        load_or_sample_support_rounds(
            data_root=str(tmp_path / 'mvtec'),
            target_cls='bottle',
            img_size=8,
            shot=4,
            inferences=10,
            seed=668,
            support_set_root=str(tmp_path / 'support_set'),
            allow_fallback=False,
        )


def test_load_or_sample_support_rounds_fallback_is_deterministic(tmp_path):
    train_dir = tmp_path / 'mvtec' / 'bottle' / 'train' / 'good'
    train_dir.mkdir(parents=True, exist_ok=True)
    _write_rgb(train_dir / '000.png', 32)
    _write_rgb(train_dir / '001.png', 96)

    lhs_rounds, lhs_source, _ = load_or_sample_support_rounds(
        data_root=str(tmp_path / 'mvtec'),
        target_cls='bottle',
        img_size=8,
        shot=2,
        inferences=3,
        seed=668,
        support_set_root=None,
        allow_fallback=True,
    )
    rhs_rounds, rhs_source, _ = load_or_sample_support_rounds(
        data_root=str(tmp_path / 'mvtec'),
        target_cls='bottle',
        img_size=8,
        shot=2,
        inferences=3,
        seed=668,
        support_set_root=None,
        allow_fallback=True,
    )

    assert lhs_source == 'fallback'
    assert rhs_source == 'fallback'
    assert len(lhs_rounds) == 3
    assert len(rhs_rounds) == 3
    for lhs_round, rhs_round in zip(lhs_rounds, rhs_rounds):
        assert torch.equal(lhs_round, rhs_round)
