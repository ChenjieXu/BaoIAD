"""MemSeg strict runtime hook."""

import torch
from mmengine.hooks import Hook

from baoiad.registry import HOOKS


@HOOKS.register_module()
class MemSegStrictTrainHook(Hook):
    """Match the frozen MemSeg runtime without over-enabling determinism.

    The upstream reference sets ``torch.backends.cudnn.deterministic = True``
    and ``torch.backends.cudnn.benchmark = False`` after seeding, but it does
    not enable ``torch.use_deterministic_algorithms(True)``. MMEngine's
    ``randomness.deterministic=True`` would be stricter than the reference and
    currently breaks MemSeg training on CUDA because
    ``adaptive_avg_pool2d_backward`` has no deterministic implementation.
    """

    priority = 'VERY_HIGH'

    def __init__(self) -> None:
        self._prev_cudnn_deterministic = None
        self._prev_cudnn_benchmark = None
        self._prev_deterministic_algorithms = None

    def before_run(self, runner) -> None:
        self._prev_cudnn_deterministic = torch.backends.cudnn.deterministic
        self._prev_cudnn_benchmark = torch.backends.cudnn.benchmark
        self._prev_deterministic_algorithms = torch.are_deterministic_algorithms_enabled()

        env_cfg = getattr(runner.cfg, 'env_cfg', {}) or {}
        cudnn_benchmark = bool(env_cfg.get('cudnn_benchmark', False))

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = cudnn_benchmark
        if self._prev_deterministic_algorithms:
            torch.use_deterministic_algorithms(False)

        runner.logger.info(
            'MemSeg strict runtime: cudnn.deterministic=True, '
            'cudnn.benchmark=%s, deterministic_algorithms=False',
            cudnn_benchmark,
        )

    def after_run(self, runner) -> None:
        del runner
        if self._prev_cudnn_deterministic is not None:
            torch.backends.cudnn.deterministic = self._prev_cudnn_deterministic
        if self._prev_cudnn_benchmark is not None:
            torch.backends.cudnn.benchmark = self._prev_cudnn_benchmark
        if self._prev_deterministic_algorithms is not None:
            torch.use_deterministic_algorithms(self._prev_deterministic_algorithms)
