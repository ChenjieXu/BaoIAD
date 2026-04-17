"""Hook to initialize RealNet AFS channel indices before training."""

from mmengine.hooks import Hook

from baoiad.registry import HOOKS


@HOOKS.register_module()
class RealNetInitHook(Hook):
    """Run ``init_afs()`` once before the first RealNet training epoch."""

    priority = 'HIGH'

    @staticmethod
    def _get_model(runner):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        return model

    def before_train(self, runner) -> None:
        model = self._get_model(runner)
        if not hasattr(model, 'init_afs'):
            return
        if getattr(model, 'afs_initialized', False):
            return

        runner.logger.info('Initializing RealNet AFS indices before training...')
        model.init_afs(runner.train_dataloader)
        runner.logger.info('RealNet AFS initialization completed.')
