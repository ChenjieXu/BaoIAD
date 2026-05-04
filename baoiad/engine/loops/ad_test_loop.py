"""Anomaly detection test loop."""

from mmengine.registry import LOOPS as MMENGINE_LOOPS
from mmengine.runner import TestLoop

from baoiad.registry import LOOPS


@LOOPS.register_module(force=True)
@MMENGINE_LOOPS.register_module(force=True)
class ADTestLoop(TestLoop):
    """Test loop for anomaly detection.

    Adds deferred evaluator processing for methods that require dataset-level
    post-processing (e.g., MuSc mutual scoring).
    """

    def _get_inner_model(self):
        model = self.runner.model
        if hasattr(model, 'module'):
            model = model.module
        return model

    def _needs_deferred_postprocess(self) -> bool:
        model = self._get_inner_model()
        return bool(
            getattr(model, 'requires_full_test_postprocess', False)
            and callable(getattr(model, 'score_all', None))
        )

    def run(self) -> dict:
        """Launch test with optional deferred processing."""
        if not self._needs_deferred_postprocess():
            return super().run()

        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()

        deferred_results = []
        for idx, data_batch in enumerate(self.dataloader):
            self.runner.call_hook(
                'before_test_iter', batch_idx=idx, data_batch=data_batch)
            outputs = self.runner.model.test_step(data_batch)
            deferred_results.append((outputs, data_batch))
            self.runner.call_hook(
                'after_test_iter',
                batch_idx=idx,
                data_batch=data_batch,
                outputs=outputs)

        model = self._get_inner_model()
        if hasattr(model, 'score_all'):
            model.score_all()

        for outputs, data_batch in deferred_results:
            self.evaluator.process(data_samples=outputs, data_batch=data_batch)

        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        self.runner.call_hook('after_test_epoch', metrics=metrics)
        self.runner.call_hook('after_test')
        return metrics
