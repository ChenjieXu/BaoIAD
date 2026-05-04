"""Anomaly detection validation loop."""

from mmengine.registry import LOOPS as MMENGINE_LOOPS
from mmengine.runner import ValLoop

from baoiad.registry import LOOPS


@LOOPS.register_module(force=True)
@MMENGINE_LOOPS.register_module(force=True)
class ADValLoop(ValLoop):
    """Validation loop for anomaly detection.

    Adds deferred evaluator processing for methods that require dataset-level
    post-processing (e.g., MuSc mutual scoring).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        """Launch validation with optional deferred processing."""
        needs_deferred = self._needs_deferred_postprocess()
        if not needs_deferred:
            return super().run()

        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()

        deferred_results = []
        for idx, data_batch in enumerate(self.dataloader):
            self.runner.call_hook(
                'before_val_iter', batch_idx=idx, data_batch=data_batch)
            outputs = self.runner.model.val_step(data_batch)
            deferred_results.append((outputs, data_batch))
            self.runner.call_hook(
                'after_val_iter',
                batch_idx=idx,
                data_batch=data_batch,
                outputs=outputs)

        model = self._get_inner_model()
        if hasattr(model, 'score_all'):
            model.score_all()

        for outputs, data_batch in deferred_results:
            self.evaluator.process(data_samples=outputs, data_batch=data_batch)

        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        self.runner.call_hook('after_val_epoch', metrics=metrics)
        self.runner.call_hook('after_val')
        return metrics
