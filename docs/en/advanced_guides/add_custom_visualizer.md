# Add a Custom Visualizer

BaoIAD provides `ADVisualizer` for rendering anomaly heatmaps, side-by-side comparisons, and batch grids. This tutorial shows how to create a custom visualizer for specialized output formats.

## Built-in Visualizer

`ADVisualizer` (`baoiad/visualization/ad_visualizer.py`) is registered in the `VISUALIZERS` registry and provides:

| Method | Description |
|---|---|
| `draw_anomaly_map(image, anomaly_map, gt_mask)` | Heatmap overlay on the image |
| `draw_result(image, anomaly_map, gt_mask, pred_score, ...)` | Side-by-side: original, GT mask, heatmap, thresholded prediction |
| `draw_batch(images, anomaly_maps, ...)` | Grid of per-sample visualizations |
| `save_result(output_path, ...)` | Render and save to file |
| `add_datasample(name, image, data_sample, ...)` | MMEngine-standard API called by hooks |

## Visualizer Interface

A custom visualizer must:

1. Inherit from `mmengine.visualization.Visualizer`
2. Be registered with `@VISUALIZERS.register_module()`
3. Implement `add_datasample()` for hook integration

## Example: Compact Visualizer

Create `baoiad/visualization/compact_visualizer.py`:

```python
"""Compact anomaly visualization with single-panel output."""

import os
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
from matplotlib import cm
from mmengine.visualization import Visualizer

from baoiad.registry import VISUALIZERS


@VISUALIZERS.register_module()
class CompactADVisualizer(Visualizer):
    """Compact single-panel anomaly visualizer.

    Instead of the 4-panel layout of ADVisualizer, this produces a
    single heatmap overlay with a color bar and score annotation.

    Args:
        name: Visualizer name.
        image: Optional default image.
        alpha: Heatmap blending factor.
    """

    def __init__(
        self,
        name: str = 'compact_ad_visualizer',
        image: Optional[np.ndarray] = None,
        alpha: float = 0.5,
        **kwargs,
    ):
        super().__init__(name=name, image=image, **kwargs)
        self.alpha = alpha

    @staticmethod
    def _normalize_map(amap: np.ndarray) -> np.ndarray:
        amap = amap.astype(np.float64)
        vmin, vmax = amap.min(), amap.max()
        if vmax - vmin > 1e-8:
            return (amap - vmin) / (vmax - vmin)
        return np.zeros_like(amap)

    def draw_compact(
        self,
        image: np.ndarray,
        anomaly_map: np.ndarray,
        pred_score: Optional[float] = None,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Draw a compact single-panel visualization.

        Args:
            image: Original image (H, W, 3) uint8.
            anomaly_map: Anomaly score map (H, W).
            pred_score: Image-level anomaly score.
            threshold: Score threshold for the label.

        Returns:
            Visualization image as (H, W, 3) uint8 BGR.
        """
        # Ensure uint8 BGR
        img = image.copy()
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.dtype != np.uint8:
            img = img.clip(0, 255).astype(np.uint8)

        # Heatmap overlay
        norm_map = self._normalize_map(anomaly_map)
        heatmap = (cm.jet(norm_map)[:, :, :3] * 255).astype(np.uint8)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
        overlay = (
            img.astype(np.float64) * (1 - self.alpha)
            + heatmap.astype(np.float64) * self.alpha
        )
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        # Score annotation
        if pred_score is not None:
            label = 'ANOMALY' if pred_score >= threshold else 'NORMAL'
            color = (0, 0, 255) if pred_score >= threshold else (0, 200, 0)
            text = f'{label} ({pred_score:.3f})'
            cv2.putText(overlay, text, (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                        cv2.LINE_AA)

        return overlay

    def add_datasample(
        self,
        name: str,
        image: np.ndarray,
        data_sample=None,
        draw_gt: bool = True,
        draw_pred: bool = True,
        show: bool = False,
        wait_time: float = 0,
        out_file: Optional[str] = None,
        step: int = 0,
    ) -> np.ndarray:
        """Standard MMEngine API for visualization hooks.

        Args:
            name: Visualization window name.
            image: Original image (H, W, 3).
            data_sample: ADDataSample with predictions.
            draw_gt: Whether to draw ground truth.
            draw_pred: Whether to draw predictions.
            show: Whether to display in a window.
            wait_time: Display wait time (ms).
            out_file: Path to save the visualization.
            step: Global step for logging.

        Returns:
            Visualization image.
        """
        pred_score = None
        anomaly_map = None

        if data_sample is not None and draw_pred:
            pred_score = getattr(data_sample, 'pred_score', None)
            amap = getattr(data_sample, 'pred_anomaly_map', None)
            if amap is not None:
                if isinstance(amap, torch.Tensor):
                    anomaly_map = amap.squeeze().cpu().numpy()
                else:
                    anomaly_map = np.asarray(amap).squeeze()

        if anomaly_map is not None:
            vis = self.draw_compact(image, anomaly_map, pred_score=pred_score)
        else:
            vis = image.copy()
            if vis.ndim == 2:
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        if show:
            self.show(vis, win_name=name, wait_time=wait_time)

        if out_file is not None:
            os.makedirs(os.path.dirname(out_file) or '.', exist_ok=True)
            cv2.imwrite(out_file, vis)

        self.set_image(vis)
        self.add_image(name, vis, step)
        return vis
```

## Register the Visualizer

Add to `baoiad/visualization/__init__.py`:

```python
from .compact_visualizer import CompactADVisualizer  # noqa: F401
```

## Configure in a Config

```python
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='CompactADVisualizer',
    vis_backends=vis_backends,
    name='compact_visualizer',
    alpha=0.5,
)
```

## How the Visualization Hook Works

`ADVisualizationHook` (`baoiad/visualization/ad_visualization_hook.py`) calls the visualizer during the test stage. After each test iteration, it:

1. Extracts `pred_anomaly_map`, `pred_score`, `gt_mask`, `gt_label` from each `ADDataSample`
2. Calls `visualizer.save_result()` for each sample
3. Writes the composite image to `<work_dir>/vis/`

The hook is controlled by config:

```python
custom_hooks = [
    dict(
        type='ADVisualizationHook',
        enable=True,           # master switch
        interval=1,            # visualize every N batches
        max_samples=0,         # max images to save (0=unlimited)
        show_gt=True,          # include GT mask panel
        score_threshold=0.5,   # threshold for binary prediction panel
        save_dir=None,         # defaults to <work_dir>/vis/
    ),
]
```

If you change the visualizer to a custom one, the hook will automatically call its `save_result()` or `add_datasample()` method.

## Key Points

- **`add_datasample()`** is the MMEngine-standard entry point. Hooks call this method, so implementing it ensures compatibility.
- **Tensor handling**: Convert `torch.Tensor` to numpy before rendering. Check `isinstance(x, torch.Tensor)` and call `.cpu().numpy()`.
- **Color space**: BaoIAD visualizers use BGR (OpenCV convention) internally. If you use matplotlib colormaps (`cm.jet`), convert from RGB to BGR.
- **Output directory**: Create parent directories with `os.makedirs(os.path.dirname(path) or '.', exist_ok=True)` before saving.
