"""Tests for the strict ViTAD train loop."""

import torch

from baoiad.engine.loops.vitad_train_loop import vitad_official_collate


def test_vitad_official_collate_stacks_inputs_and_keeps_data_samples():
    batch = [
        {'inputs': torch.ones(3, 4, 4), 'data_samples': 'a'},
        {'inputs': torch.zeros(3, 4, 4), 'data_samples': 'b'},
    ]
    collated = vitad_official_collate(batch)
    assert collated['inputs'].shape == (2, 3, 4, 4)
    assert collated['data_samples'] == ['a', 'b']
