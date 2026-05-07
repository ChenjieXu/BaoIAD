# Add a Custom Hook

Hooks in BaoIAD extend the training and testing lifecycle. They can intervene at specific points (before training, after each iteration, etc.) to add custom behavior like logging, scheduling, or post-processing.

## Built-in Hooks

BaoIAD ships with these hooks in `baoiad/engine/hooks/`:

| Hook | Description |
|---|---|
| `MemoryBankHook` | Builds memory bank after training for PatchCore, PaDiM, etc. |
| `ADVisualizationHook` | Saves anomaly visualizations during testing |
| `MemSegStrictHook` | MemSeg-specific training lifecycle management |
| `UFlowStrictHook` | U-Flow-specific phase scheduling |
| `ViTADStrictHook` | ViTAD-specific training lifecycle |
| `MuScScoreHook` | MuSc score aggregation |

## Hook Interface

Custom hooks must:

1. Inherit from `mmengine.hooks.Hook`
2. Be registered with `@HOOKS.register_module()`
3. Override one or more hook point methods

## Available Hook Points

MMEngine provides these hook points on the `Hook` base class:

| Method | When it runs |
|---|---|
| `before_run(runner)` | Before the runner starts |
| `after_run(runner)` | After the runner finishes |
| `before_train(runner)` | Before training begins |
| `after_train(runner)` | After training ends |
| `before_train_epoch(runner)` | Before each training epoch |
| `after_train_epoch(runner, metrics=None)` | After each training epoch |
| `before_train_iter(runner, batch_idx, data_batch=None)` | Before each training iteration |
| `after_train_iter(runner, batch_idx, data_batch=None, outputs=None)` | After each training iteration |
| `before_val_epoch(runner)` | Before each validation epoch |
| `after_val_epoch(runner, metrics=None)` | After each validation epoch |
| `before_val_iter(runner, batch_idx, data_batch=None)` | Before each validation iteration |
| `after_val_iter(runner, batch_idx, data_batch=None, outputs=None)` | After each validation iteration |
| `before_test_epoch(runner)` | Before each test epoch |
| `after_test_epoch(runner, metrics=None)` | After each test epoch |
| `before_test_iter(runner, batch_idx, data_batch=None)` | Before each test iteration |
| `after_test_iter(runner, batch_idx, data_batch=None, outputs=None)` | After each test iteration |

## Example: Gradient Logging Hook

Create `baoiad/engine/hooks/gradient_log_hook.py`:

```python
"""Hook to log gradient statistics during training."""

from mmengine.hooks import Hook

from baoiad.registry import HOOKS


@HOOKS.register_module()
class GradientLogHook(Hook):
    """Log gradient norm statistics every N iterations.

    Helps detect vanishing/exploding gradients during training of
    reconstruction-based and flow-based methods.

    Args:
        interval: Log every N iterations.
        log_norm: Whether to log L2 gradient norm.
        log_max: Whether to log max absolute gradient.
    """

    priority = 'LOW'

    def __init__(
        self,
        interval: int = 100,
        log_norm: bool = True,
        log_max: bool = True,
    ):
        super().__init__()
        self.interval = interval
        self.log_norm = log_norm
        self.log_max = log_max

    def after_train_iter(
        self,
        runner,
        batch_idx: int,
        data_batch=None,
        outputs=None,
    ) -> None:
        """Log gradient stats after each training iteration."""
        if batch_idx % self.interval != 0:
            return

        model = runner.model
        # Unwrap DDP
        if hasattr(model, 'module'):
            model = model.module

        total_norm = 0.0
        max_grad = 0.0
        param_count = 0

        for p in model.parameters():
            if p.grad is not None:
                grad = p.grad.detach().data
                if self.log_norm:
                    total_norm += grad.norm().item() ** 2
                if self.log_max:
                    abs_max = grad.abs().max().item()
                    max_grad = max(max_grad, abs_max)
                param_count += 1

        if param_count == 0:
            return

        log_parts = []
        if self.log_norm:
            total_norm = total_norm ** 0.5
            log_parts.append(f'grad_norm={total_norm:.4f}')
        if self.log_max:
            log_parts.append(f'grad_max={max_grad:.6f}')

        if log_parts:
            runner.logger.info(
                f'[GradientLog] iter {runner.iter}  ' + '  '.join(log_parts))
```

## Register the Hook

Add to `baoiad/engine/hooks/__init__.py`:

```python
from .gradient_log_hook import GradientLogHook  # noqa: F401
```

## Configure in a Config

Hooks are configured under `custom_hooks` in the config:

```python
custom_hooks = [
    dict(type='GradientLogHook', interval=50, log_norm=True, log_max=True),
]
```

Or programmatically:

```python
cfg = Config.fromfile('configs/mydetector/mydetector_256_mvtec.py')
cfg.custom_hooks = [
    dict(type='GradientLogHook', interval=50),
]
runner = Runner.from_cfg(cfg)
runner.train()
```

## Example: MemoryBankHook Pattern

`MemoryBankHook` demonstrates a more complex hook that interacts with the model lifecycle:

```python
@HOOKS.register_module()
class MemoryBankHook(Hook):
    priority = 'LOW'

    def __init__(self):
        super().__init__()
        self._built = False

    def _get_model(self, runner):
        """Unwrap DDP model."""
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        return model

    def before_train(self, runner):
        """Pre-training setup (e.g., computing teacher stats)."""
        model = self._get_model(runner)
        if hasattr(model, 'pre_train_setup'):
            model.pre_train_setup(runner.train_dataloader)

    def after_train_epoch(self, runner, metrics=None):
        """Build memory bank after the last epoch."""
        model = self._get_model(runner)
        if runner.epoch == runner.max_epochs - 1:
            self._built = False
        self._ensure_memory_bank(runner)

    def after_train(self, runner):
        """Ensure memory bank is built after all training."""
        self._ensure_memory_bank(runner)

    def before_val_epoch(self, runner):
        """Ensure memory bank before validation."""
        self._ensure_memory_bank(runner)

    def before_test_epoch(self, runner):
        """Ensure memory bank before testing."""
        self._ensure_memory_bank(runner)
```

Key patterns in this hook:

- **`_get_model(runner)`**: Unwraps `DistributedDataParallel` to access the raw model. Always do this when accessing model attributes.
- **`priority = 'LOW'`**: Runs after other hooks at the same point. Use `priority` to control ordering.
- **`self._built` flag**: Ensures the memory bank is built exactly once (unless forced to rebuild).
- **Multiple hook points**: The same hook listens at `before_train`, `after_train_epoch`, `after_train`, `before_val_epoch`, and `before_test_epoch` to cover different training scenarios.

## Example: ADVisualizationHook Pattern

`ADVisualizationHook` shows how to process test outputs:

```python
@HOOKS.register_module()
class ADVisualizationHook(Hook):
    priority = 'LOW'

    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if not self.enable:
            return
        if batch_idx % self.interval != 0:
            return
        if outputs is None:
            return

        visualizer = runner.visualizer
        save_dir = self._get_save_dir(runner)

        for data_sample in outputs:
            # Extract fields from ADDataSample
            image = getattr(data_sample, 'img', None)
            anomaly_map = getattr(data_sample, 'pred_anomaly_map', None)
            pred_score = getattr(data_sample, 'pred_score', None)
            gt_mask = getattr(data_sample, 'gt_mask', None)

            visualizer.save_result(
                out_path, image, anomaly_map,
                gt_mask=gt_mask, pred_score=pred_score,
            )
```

## Priority Levels

Hooks have a priority attribute that determines execution order at the same hook point:

| Priority | Typical use |
|---|---|
| `'HIGHEST'` | Critical setup |
| `'HIGH'` | Important pre-processing |
| `'NORMAL'` | Default |
| `'LOW'` | Post-processing (most AD hooks) |
| `'VERY_LOW'` | Cleanup, final logging |

## Key Points

- **Always unwrap DDP**: Use `runner.model.module` to access the raw model inside hooks, since MMEngine wraps models in `DistributedDataParallel` during distributed training.
- **Check `hasattr` before access**: Model methods like `build_memory_bank()` or `pre_train_setup()` are optional. Guard with `hasattr(model, method_name)` before calling.
- **Use `runner.logger`**: Log through the runner's logger rather than `print()` for proper log routing and formatting.
- **Configurable via config**: Design hooks so all behavior is controlled by constructor arguments that can be set from config dicts.
