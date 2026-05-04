"""Visualization hook for anomaly detection test stage."""

import os
import os.path as osp
from typing import Optional, Sequence

import numpy as np
from mmengine.hooks import Hook

from baoiad.registry import HOOKS


@HOOKS.register_module(force=True)
class ADVisualizationHook(Hook):
    """Automatically save visualization results during the test stage.

    The hook calls :meth:`ADVisualizer.draw_result` for each sample and
    writes the composite image to ``save_dir``.

    Args:
        enable (bool): Master switch.  ``False`` by default so the hook
            is a no-op unless explicitly turned on in config.
        interval (int): Visualize every *interval* batches.
        max_samples (int): Maximum total images to save (0 = unlimited).
        show_gt (bool): Include ground truth mask in the visualization.
        score_threshold (float): Threshold for the binary prediction panel.
        save_dir (str | None): Output directory.  Defaults to
            ``<work_dir>/vis/``.
    """

    priority = 'LOW'

    def __init__(
        self,
        enable: bool = False,
        interval: int = 1,
        max_samples: int = 0,
        show_gt: bool = True,
        score_threshold: float = 0.5,
        save_dir: Optional[str] = None,
    ):
        super().__init__()
        self.enable = enable
        self.interval = interval
        self.max_samples = max_samples
        self.show_gt = show_gt
        self.score_threshold = score_threshold
        self._save_dir = save_dir
        self._count = 0

    def _get_save_dir(self, runner) -> str:
        if self._save_dir is not None:
            return self._save_dir
        return osp.join(runner.work_dir, 'vis')

    # ------------------------------------------------------------------
    # Hook entry point
    # ------------------------------------------------------------------

    def after_test_iter(
        self,
        runner,
        batch_idx: int,
        data_batch: Optional[dict] = None,
        outputs: Optional[Sequence] = None,
    ) -> None:
        """Called after every test iteration.

        Args:
            runner: The runner instance.
            batch_idx: Index of the current batch.
            data_batch: The raw data batch dict.
            outputs: List of :class:`DataSample` produced by the model.
        """
        if not self.enable:
            return
        if batch_idx % self.interval != 0:
            return
        if outputs is None:
            return

        visualizer = runner.visualizer
        save_dir = self._get_save_dir(runner)
        os.makedirs(save_dir, exist_ok=True)

        for data_sample in outputs:
            if 0 < self.max_samples <= self._count:
                return

            # --- extract fields from DataSample ---
            # image: try data_sample first, fall back to data_batch
            image = None
            if hasattr(data_sample, 'img') and data_sample.img is not None:
                image = data_sample.img
            elif data_batch is not None and 'inputs' in data_batch:
                # inputs is typically a list of tensors
                idx_in_batch = self._count % len(data_batch['inputs'])
                img_tensor = data_batch['inputs'][idx_in_batch]
                image = self._tensor_to_numpy(img_tensor)

            if image is None:
                continue

            # anomaly map
            anomaly_map = None
            if hasattr(data_sample, 'pred_anomaly_map'):
                anomaly_map = self._to_numpy(data_sample.pred_anomaly_map)
            elif hasattr(data_sample, 'anomaly_map'):
                anomaly_map = self._to_numpy(data_sample.anomaly_map)

            if anomaly_map is None:
                continue

            # squeeze extra dims (C, H, W) -> (H, W)
            if anomaly_map.ndim == 3:
                anomaly_map = anomaly_map.squeeze(0)

            # pred score
            pred_score = None
            if hasattr(data_sample, 'pred_score'):
                pred_score = float(self._to_numpy(data_sample.pred_score))
            elif hasattr(data_sample, 'pred_label'):
                pred_score = float(self._to_numpy(data_sample.pred_label))

            # gt
            gt_mask = None
            gt_label = None
            if self.show_gt:
                if hasattr(data_sample, 'gt_mask'):
                    gt_mask = self._to_numpy(data_sample.gt_mask)
                    if gt_mask is not None and gt_mask.ndim == 3:
                        gt_mask = gt_mask.squeeze(0)
                if hasattr(data_sample, 'gt_label'):
                    gt_label = int(self._to_numpy(data_sample.gt_label))

            # filename
            img_path = getattr(data_sample, 'img_path', None)
            if img_path:
                fname = osp.splitext(osp.basename(img_path))[0]
            else:
                fname = f'{self._count:06d}'

            out_path = osp.join(save_dir, f'{fname}.png')
            visualizer.save_result(
                out_path, image, anomaly_map,
                gt_mask=gt_mask,
                pred_score=pred_score,
                gt_label=gt_label,
                threshold=self.score_threshold,
            )
            self._count += 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numpy(x):
        """Convert tensor or array to numpy."""
        if x is None:
            return None
        if hasattr(x, 'cpu'):
            return x.detach().cpu().numpy()
        return x if isinstance(x, np.ndarray) else np.asarray(x)

    @staticmethod
    def _tensor_to_numpy(tensor) -> np.ndarray:
        """Convert a CHW float tensor to HWC uint8 numpy array."""
        if hasattr(tensor, 'cpu'):
            arr = tensor.detach().cpu().numpy()
        else:
            arr = np.asarray(tensor)
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = arr.transpose(1, 2, 0)
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        return arr
