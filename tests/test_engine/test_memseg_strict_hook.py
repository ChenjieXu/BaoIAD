"""Tests for MemSeg strict runtime hook."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import baoiad  # noqa: F401
import torch

from baoiad.engine.hooks.memseg_strict_hook import MemSegStrictTrainHook


def test_memseg_strict_hook_matches_reference_runtime_flags():
    hook = MemSegStrictTrainHook()
    runner = SimpleNamespace(
        cfg=SimpleNamespace(env_cfg={'cudnn_benchmark': False}),
        logger=MagicMock(),
    )

    prev_cudnn_deterministic = torch.backends.cudnn.deterministic
    prev_cudnn_benchmark = torch.backends.cudnn.benchmark
    prev_det_algorithms = torch.are_deterministic_algorithms_enabled()

    try:
        torch.use_deterministic_algorithms(True)
        hook_prev_det_algorithms = torch.are_deterministic_algorithms_enabled()
        hook.before_run(runner)

        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
        assert torch.are_deterministic_algorithms_enabled() is False

        hook.after_run(runner)
        assert torch.backends.cudnn.deterministic is prev_cudnn_deterministic
        assert torch.backends.cudnn.benchmark is prev_cudnn_benchmark
        assert torch.are_deterministic_algorithms_enabled() is hook_prev_det_algorithms
    finally:
        torch.backends.cudnn.deterministic = prev_cudnn_deterministic
        torch.backends.cudnn.benchmark = prev_cudnn_benchmark
        torch.use_deterministic_algorithms(prev_det_algorithms)
