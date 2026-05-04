"""Tests for dataset samplers."""

import torch
from torch.utils.data import RandomSampler

import baoiad  # noqa: F401
from baoiad.datasets.samplers import (
    ExplicitOrderSampler,
    OpenIADSubsetRandomSampler,
    PerEpochOrderSampler,
    PersistentShuffleSampler,
    PythonShuffleSampler,
)


class TestPersistentShuffleSampler:
    def test_matches_random_sampler_epoch_seed_stream(self):
        dataset = list(range(8))
        torch.manual_seed(42)
        expected = [list(iter(RandomSampler(dataset))) for _ in range(3)]

        torch.manual_seed(42)
        sampler = PersistentShuffleSampler(dataset, shuffle=True, seed=42, round_up=False)
        actual = [list(iter(sampler)) for _ in range(3)]

        assert actual == expected
        assert actual[0] != actual[1]

    def test_set_epoch_does_not_reseed(self):
        dataset = list(range(8))
        torch.manual_seed(7)
        sampler = PersistentShuffleSampler(dataset, shuffle=True, seed=7, round_up=False)

        epoch1 = list(iter(sampler))
        sampler.set_epoch(99)
        epoch2 = list(iter(sampler))

        torch.manual_seed(7)
        replay = PersistentShuffleSampler(dataset, shuffle=True, seed=7, round_up=False)
        assert epoch1 == list(iter(replay))
        replay.set_epoch(123)
        assert epoch2 == list(iter(replay))


class TestPythonShuffleSampler:
    def test_matches_python_random_shuffle(self):
        dataset = list(range(8))
        sampler = PythonShuffleSampler(dataset, shuffle=True, seed=66, round_up=False)
        actual = list(iter(sampler))

        expected = list(range(8))
        import random

        random.Random(66).shuffle(expected)
        assert actual == expected

    def test_set_epoch_keeps_fixed_shuffle(self):
        dataset = list(range(8))
        sampler = PythonShuffleSampler(dataset, shuffle=True, seed=7, round_up=False)
        first = list(iter(sampler))
        sampler.set_epoch(99)
        second = list(iter(sampler))
        assert first == second


class TestExplicitOrderSampler:
    def test_uses_inline_indices(self):
        dataset = list(range(8))
        sampler = ExplicitOrderSampler(dataset, indices=[3, 1, 4], round_up=False)
        assert list(iter(sampler)) == [3, 1, 4]

    def test_loads_indices_from_json_file(self, tmp_path):
        path = tmp_path / 'indices.json'
        path.write_text('{"indices": [2, 5, 1]}', encoding='utf-8')
        dataset = list(range(8))
        sampler = ExplicitOrderSampler(dataset, index_file=str(path), round_up=False)
        assert list(iter(sampler)) == [2, 5, 1]


class TestPerEpochOrderSampler:
    def test_switches_order_with_set_epoch(self):
        dataset = list(range(8))
        sampler = PerEpochOrderSampler(dataset, epoch_orders=[[0, 1, 2], [3, 4, 5]], round_up=False)
        assert list(iter(sampler)) == [0, 1, 2]
        sampler.set_epoch(1)
        assert list(iter(sampler)) == [3, 4, 5]
