"""Dataset samplers used by strict alignment configs."""

from __future__ import annotations

import math
import random
import json
from typing import Iterator, Optional, Sized

import torch
from mmengine.dist import get_dist_info, sync_random_seed
from mmengine.registry import DATA_SAMPLERS as MMENGINE_DATA_SAMPLERS
from torch.utils.data import Sampler

from baoiad.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module(force=True)
@MMENGINE_DATA_SAMPLERS.register_module(force=True)
class PersistentShuffleSampler(Sampler):
    """Sampler that mirrors PyTorch ``RandomSampler`` epoch semantics.

    ADer's non-distributed ViTAD training relies on ``DataLoader(shuffle=True)``,
    which uses ``RandomSampler``. That sampler does not reuse one generator
    directly for ``randperm``. Instead, every new iterator first samples an
    epoch seed from a persistent RNG stream, then builds a fresh generator for
    that epoch's permutation. MMEngine's ``DefaultSampler`` instead reseeds from
    ``seed + epoch``. This sampler mirrors the former behavior while retaining
    the same distributed slicing contract as ``DefaultSampler``.
    """

    def __init__(
        self,
        dataset: Sized,
        shuffle: bool = True,
        seed: Optional[int] = None,
        round_up: bool = True,
    ) -> None:
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size
        self.dataset = dataset
        self.shuffle = shuffle
        if seed is None:
            seed = sync_random_seed()
        self.seed = int(seed)
        self.round_up = round_up
        if self.round_up:
            self.num_samples = math.ceil(len(self.dataset) / world_size)
            self.total_size = self.num_samples * self.world_size
        else:
            self.num_samples = math.ceil((len(self.dataset) - rank) / world_size)
            self.total_size = len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            # Match torch.utils.data.RandomSampler(generator=None): sample an
            # epoch seed from the *current global* torch RNG state, then build
            # a fresh generator for this epoch's permutation.
            epoch_seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(epoch_seed)
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        if self.round_up:
            indices = (
                indices
                * int(self.total_size / len(indices) + 1)
            )[:self.total_size]

        indices = indices[self.rank:self.total_size:self.world_size]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Keep API compatibility with sampler hooks without reseeding.

        The epoch argument is intentionally ignored so the seed stream stays
        decoupled from MMEngine's ``seed + epoch`` policy.
        """
        del epoch


@DATA_SAMPLERS.register_module(force=True)
@MMENGINE_DATA_SAMPLERS.register_module(force=True)
class PythonShuffleSampler(Sampler):
    """Sampler that mirrors ``random.shuffle(indices)`` with a fixed seed."""

    def __init__(
        self,
        dataset: Sized,
        shuffle: bool = True,
        seed: Optional[int] = None,
        round_up: bool = True,
    ) -> None:
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size
        self.dataset = dataset
        self.shuffle = shuffle
        if seed is None:
            seed = sync_random_seed()
        self.seed = int(seed)
        self.round_up = round_up

        if self.round_up:
            self.num_samples = math.ceil(len(self.dataset) / world_size)
            self.total_size = self.num_samples * self.world_size
        else:
            self.num_samples = math.ceil((len(self.dataset) - rank) / world_size)
            self.total_size = len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.Random(self.seed).shuffle(indices)

        if self.round_up:
            indices = (indices * int(self.total_size / len(indices) + 1))[:self.total_size]

        indices = indices[self.rank:self.total_size:self.world_size]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        del epoch


@DATA_SAMPLERS.register_module(force=True)
@MMENGINE_DATA_SAMPLERS.register_module(force=True)
class OpenIADSubsetRandomSampler(Sampler):
    """Sampler matching open-iad's ``random.shuffle + SubsetRandomSampler``.

    The support-set data path first shuffles per-task indices with Python's
    ``random.shuffle`` under a fixed seed, then feeds that list into
    ``torch.utils.data.SubsetRandomSampler``. For the single-class MVTec
    benchmark used by BaoIAD, this is equivalent to applying those two
    permutations to the full dataset indices directly.
    """

    def __init__(
        self,
        dataset: Sized,
        shuffle: bool = True,
        seed: Optional[int] = None,
        round_up: bool = True,
    ) -> None:
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size
        self.dataset = dataset
        self.shuffle = shuffle
        if seed is None:
            seed = sync_random_seed()
        self.seed = int(seed)
        self.round_up = round_up

        if self.round_up:
            self.num_samples = math.ceil(len(self.dataset) / world_size)
            self.total_size = self.num_samples * self.world_size
        else:
            self.num_samples = math.ceil((len(self.dataset) - rank) / world_size)
            self.total_size = len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.Random(self.seed).shuffle(indices)
            generator = torch.Generator()
            generator.manual_seed(self.seed)
            order = torch.randperm(len(indices), generator=generator).tolist()
            indices = [indices[idx] for idx in order]

        if self.round_up:
            indices = (indices * int(self.total_size / len(indices) + 1))[:self.total_size]

        indices = indices[self.rank:self.total_size:self.world_size]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        del epoch


@DATA_SAMPLERS.register_module(force=True)
@MMENGINE_DATA_SAMPLERS.register_module(force=True)
class ExplicitOrderSampler(Sampler):
    """Sampler that yields a fixed index order loaded from config or JSON."""

    def __init__(
        self,
        dataset: Sized,
        indices: Optional[list[int]] = None,
        index_file: Optional[str] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
        round_up: bool = False,
    ) -> None:
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size
        self.dataset = dataset
        self.round_up = round_up
        self.shuffle = shuffle
        self.seed = seed

        if indices is None:
            if index_file is None:
                raise ValueError('ExplicitOrderSampler requires either indices or index_file.')
            with open(index_file, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            indices = payload['indices'] if isinstance(payload, dict) else payload

        self.indices = [int(i) for i in indices]
        if self.round_up:
            self.num_samples = math.ceil(len(self.indices) / world_size)
            self.total_size = self.num_samples * self.world_size
        else:
            self.num_samples = math.ceil((len(self.indices) - rank) / world_size)
            self.total_size = len(self.indices)

    def __iter__(self) -> Iterator[int]:
        indices = list(self.indices)
        if self.round_up:
            indices = (indices * int(self.total_size / len(indices) + 1))[:self.total_size]
        indices = indices[self.rank:self.total_size:self.world_size]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        del epoch


@DATA_SAMPLERS.register_module(force=True)
@MMENGINE_DATA_SAMPLERS.register_module(force=True)
class PerEpochOrderSampler(Sampler):
    """Sampler that replays a fixed list of indices for each epoch.

    Intended for strict alignment diagnosis when the exact upstream dataloader
    order has been dumped in advance.
    """

    def __init__(
        self,
        dataset: Sized,
        epoch_orders: Optional[list[list[int]]] = None,
        index_file: Optional[str] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
        round_up: bool = False,
    ) -> None:
        del shuffle, seed
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size
        self.dataset = dataset
        self.round_up = round_up
        self.epoch = 0

        if epoch_orders is None:
            if index_file is None:
                raise ValueError('PerEpochOrderSampler requires either epoch_orders or index_file.')
            with open(index_file, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            epoch_orders = payload['epoch_orders'] if isinstance(payload, dict) else payload

        self.epoch_orders = [[int(i) for i in order] for order in epoch_orders]
        if not self.epoch_orders:
            raise ValueError('PerEpochOrderSampler requires at least one epoch order.')

        current = self.epoch_orders[0]
        if self.round_up:
            self.num_samples = math.ceil(len(current) / world_size)
            self.total_size = self.num_samples * self.world_size
        else:
            self.num_samples = math.ceil((len(current) - rank) / world_size)
            self.total_size = len(current)

    def _current_indices(self) -> list[int]:
        order = self.epoch_orders[min(self.epoch, len(self.epoch_orders) - 1)]
        indices = list(order)
        if self.round_up:
            indices = (indices * int(self.total_size / len(indices) + 1))[:self.total_size]
        return indices[self.rank:self.total_size:self.world_size]

    def __iter__(self) -> Iterator[int]:
        return iter(self._current_indices())

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
