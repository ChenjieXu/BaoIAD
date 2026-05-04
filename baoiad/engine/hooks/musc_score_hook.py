"""Hook to perform mutual scoring for MuSc after all predictions."""

from mmengine.hooks import Hook

from baoiad.registry import HOOKS


@HOOKS.register_module(force=True)
class MuScScoreHook(Hook):
    """Call score_all() on MuSc detector after all test predictions.

    MuSc is a zero-shot method that requires all test images to be
    processed together for mutual nearest-neighbor scoring. This hook
    calls score_all() after the test epoch completes.

    Priority is set to VERY_LOW so it runs after metric computation setup.
    """

    priority = 'VERY_LOW'

    def __init__(self) -> None:
        super().__init__()
        self._scored = False

    def after_test_epoch(self, runner, metrics=None) -> None:
        """Call score_all() after test epoch to compute mutual scores."""
        if self._scored:
            return

        model = runner.model
        if hasattr(model, 'module'):
            model = model.module

        if hasattr(model, 'score_all'):
            runner.logger.info('Computing mutual scores for MuSc...')
            results = model.score_all()
            self._scored = True
            runner.logger.info(f'MuSc scoring complete. {len(results)} samples scored.')
            # Note: The results are updated in-place in the accumulated data_samples
            # The test loop should have already collected the placeholder results
            # We need to update the runner's results if possible

    def after_test(self, runner) -> None:
        """Ensure scoring is done after all testing completes."""
        self.after_test_epoch(runner, None)
