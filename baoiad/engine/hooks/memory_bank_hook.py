"""Hook to build memory bank after training for memory-bank-based methods."""

import inspect

from mmengine.hooks import Hook

from baoiad.registry import HOOKS


@HOOKS.register_module(force=True)
class MemoryBankHook(Hook):
    """Build memory bank after the last training epoch.

    For methods like PatchCore, PaDiM, SPADE that collect features
    during training and build a memory bank before evaluation.

    Also supports PyramidFlow which needs to build a latent template
    from the training/validation dataloader.

    Also supports EfficientAD which needs:
    - pre_train_setup(dataloader): compute teacher stats before training
    - compute_normalization_stats(dataloader): compute quantiles before validation

    Priority is set to LOW so it runs after other after_train hooks.
    """

    priority = 'LOW'

    def __init__(self) -> None:
        super().__init__()
        self._built = False

    def _get_model(self, runner):
        """Unwrap DDP model."""
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        return model

    def _ensure_memory_bank(self, runner) -> None:
        """Build memory bank if not already built."""
        if self._built:
            return

        model = self._get_model(runner)

        # Check if model needs dataloader for template building (e.g., PyramidFlow)
        if hasattr(model, 'build_template_from_dataloader'):
            split = getattr(model, 'template_dataloader_split', 'val')
            if split == 'train':
                dataloader = runner.train_dataloader
            else:
                try:
                    dataloader = runner.val_dataloader
                except (AttributeError, KeyError):
                    dataloader = runner.train_dataloader

            runner.logger.info(f'Building template from {split} dataloader...')

            device = next(model.parameters()).device
            model.build_template_from_dataloader(dataloader, device)
            self._built = True
            runner.logger.info('Template building completed.')
            return

        # Support multiple method names: build_memory_bank, fit
        for method_name in ['build_memory_bank', 'fit']:
            if hasattr(model, method_name):
                method = getattr(model, method_name)
                # Pass dataloader if the method accepts it
                sig = inspect.signature(method)
                params = list(sig.parameters.keys())
                if len(params) > 0:
                    runner.logger.info(f'Calling {method_name}(dataloader)...')
                    method(runner.train_dataloader)
                else:
                    runner.logger.info(f'Calling {method_name}()...')
                    method()
                self._built = True
                runner.logger.info(f'{method_name}() completed.')
                return

        # Also check head
        if hasattr(model, 'head'):
            for method_name in ['build_memory_bank', 'fit']:
                if hasattr(model.head, method_name):
                    runner.logger.info(f'Calling head.{method_name}()...')
                    getattr(model.head, method_name)()
                    self._built = True
                    runner.logger.info(f'head.{method_name}() completed.')
                    return

    def before_train(self, runner) -> None:
        """Pre-training setup (e.g., EfficientAD teacher stats)."""
        model = self._get_model(runner)
        if hasattr(model, 'pre_train_setup'):
            runner.logger.info('Running pre_train_setup()...')
            model.pre_train_setup(runner.train_dataloader)
            runner.logger.info('pre_train_setup() completed.')
            if getattr(model, 'pre_train_setup_builds_memory_bank', False):
                self._built = True
                runner.logger.info('Memory bank marked as built after pre_train_setup().')

    def before_train_epoch(self, runner) -> None:
        """Pass epoch info to model for phase switching (e.g., DSR, DeSTSeg)."""
        model = self._get_model(runner)
        if hasattr(model, 'set_epoch_info'):
            model.set_epoch_info(runner.epoch, runner.max_epochs)

    def before_train_iter(self, runner, batch_idx: int, data_batch=None) -> None:
        """Pass iteration info to model for phase switching (iteration-based training).

        For methods like CutPaste that use iteration-based training (by_epoch=False).
        """
        model = self._get_model(runner)
        if hasattr(model, 'set_iter_info'):
            model.set_iter_info(runner.iter, runner.max_iters)

    def after_train_epoch(self, runner, metrics=None) -> None:
        """Build memory bank after training epoch."""
        model = self._get_model(runner)
        # Always rebuild on the last epoch to use final trained weights
        if runner.epoch == runner.max_epochs - 1 and not getattr(
            model, 'pre_train_setup_builds_memory_bank', False
        ):
            self._built = False
        self._ensure_memory_bank(runner)

    def after_train(self, runner) -> None:
        """Ensure memory bank is built with final model weights after all training.

        For iteration-based methods, ``after_train_epoch`` may never have built the
        bank, so ``after_train`` still needs to do it once. For epoch-based
        memory-bank methods such as PatchCore/PaDiM/SPADE, re-fitting here would
        duplicate the final-epoch build and can rebuild from an already-cleared
        feature cache.

        Methods that truly need a final forced refit can opt in via
        ``refit_after_train = True``.
        """
        model = self._get_model(runner)
        if getattr(model, 'refit_after_train', False):
            self._built = False
        self._ensure_memory_bank(runner)

    def before_val_epoch(self, runner) -> None:
        """Ensure memory bank is built before validation starts.
        Also compute normalization stats if the model supports it.

        For models with ``always_refit = True`` (e.g. CutPaste), the memory
        bank / GDE is re-fitted at every validation so that intermediate
        evaluations reflect the current model weights.
        """
        model = self._get_model(runner)
        if getattr(model, 'always_refit', False):
            self._built = False
        self._ensure_memory_bank(runner)

        # Compute normalization stats (e.g., EfficientAD quantiles)
        model = self._get_model(runner)
        if hasattr(model, 'compute_normalization_stats'):
            runner.logger.info('Computing normalization stats...')
            model.compute_normalization_stats(runner.val_dataloader)
            runner.logger.info('Normalization stats computed.')

    def before_test_epoch(self, runner) -> None:
        """Ensure memory bank is built before test-only evaluation starts."""
        model = self._get_model(runner)
        if getattr(model, 'always_refit', False):
            self._built = False
        self._ensure_memory_bank(runner)

        if hasattr(model, 'compute_normalization_stats'):
            runner.logger.info('Computing normalization stats before test...')
            dataloader = getattr(runner, 'val_dataloader', None) or runner.test_dataloader
            model.compute_normalization_stats(dataloader)
            runner.logger.info('Normalization stats computed before test.')
