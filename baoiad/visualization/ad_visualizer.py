"""Anomaly detection visualizer."""

import os
from typing import List, Optional, Sequence

import cv2
import numpy as np
import torch
from matplotlib import cm
from mmengine.visualization import Visualizer

from baoiad.registry import VISUALIZERS


@VISUALIZERS.register_module()
class ADVisualizer(Visualizer):
    """Visualizer for anomaly detection results.

    Provides anomaly heatmap overlay, side-by-side comparison, and batch
    grid visualization.

    Args:
        name: Visualizer name.
        image: Optional default image.
    """

    # Layout constants
    _PAD = 4            # pixels between panels
    _TITLE_H = 30       # pixels for title bar
    _CBAR_H = 20        # pixels for color bar
    _FONT = cv2.FONT_HERSHEY_SIMPLEX
    _FONT_SCALE = 0.45
    _FONT_THICK = 1
    _BG_COLOR = (40, 40, 40)       # dark grey background
    _TEXT_COLOR = (255, 255, 255)   # white text

    def __init__(self, name: str = 'ad_visualizer', image: Optional[np.ndarray] = None, **kwargs):
        super().__init__(name=name, image=image, **kwargs)

    # ------------------------------------------------------------------
    # Existing method
    # ------------------------------------------------------------------

    def draw_anomaly_map(
        self,
        image: np.ndarray,
        anomaly_map: np.ndarray,
        gt_mask: Optional[np.ndarray] = None,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Draw anomaly heatmap overlaid on the image.

        Args:
            image: Original image (H, W, 3), uint8.
            anomaly_map: Anomaly score map (H, W), float.
            gt_mask: Optional ground truth mask (H, W).
            alpha: Blending factor for heatmap overlay.

        Returns:
            Visualization image as numpy array.
        """
        amap = anomaly_map.astype(np.float64)
        vmin, vmax = amap.min(), amap.max()
        if vmax - vmin > 1e-8:
            amap = (amap - vmin) / (vmax - vmin)
        else:
            amap = np.zeros_like(amap)

        heatmap = (cm.jet(amap)[:, :, :3] * 255).astype(np.uint8)

        overlay = (image.astype(np.float64) * (1 - alpha) + heatmap.astype(np.float64) * alpha)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        if gt_mask is not None:
            mask_uint8 = (gt_mask > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)

        return overlay

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_map(amap: np.ndarray) -> np.ndarray:
        """Normalize a 2-D map to [0, 1]."""
        amap = amap.astype(np.float64)
        vmin, vmax = amap.min(), amap.max()
        if vmax - vmin > 1e-8:
            return (amap - vmin) / (vmax - vmin)
        return np.zeros_like(amap)

    @staticmethod
    def _apply_jet(norm_map: np.ndarray) -> np.ndarray:
        """Apply jet colormap → (H, W, 3) uint8 BGR."""
        rgb = (cm.jet(norm_map)[:, :, :3] * 255).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _make_colorbar(self, width: int) -> np.ndarray:
        """Create a horizontal jet colorbar of shape (_CBAR_H, width, 3)."""
        grad = np.linspace(0, 1, width).reshape(1, -1)
        grad = np.repeat(grad, self._CBAR_H, axis=0)
        bar = (cm.jet(grad)[:, :, :3] * 255).astype(np.uint8)
        return cv2.cvtColor(bar, cv2.COLOR_RGB2BGR)

    def _put_text_center(self, canvas: np.ndarray, text: str, y: int, x_start: int, x_end: int,
                         color=None, scale=None):
        """Put *text* horizontally centered between x_start..x_end at row y."""
        color = color or self._TEXT_COLOR
        scale = scale or self._FONT_SCALE
        (tw, th), _ = cv2.getTextSize(text, self._FONT, scale, self._FONT_THICK)
        x = x_start + (x_end - x_start - tw) // 2
        cv2.putText(canvas, text, (x, y + th), self._FONT, scale, color, self._FONT_THICK, cv2.LINE_AA)

    @staticmethod
    def _ensure_bgr_uint8(image: np.ndarray) -> np.ndarray:
        img = image.copy()
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).clip(0, 255).astype(np.uint8)
            else:
                img = img.clip(0, 255).astype(np.uint8)
        return img

    @staticmethod
    def _resize_to(image: np.ndarray, h: int, w: int) -> np.ndarray:
        if image.shape[:2] != (h, w):
            return cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        return image

    # ------------------------------------------------------------------
    # draw_result
    # ------------------------------------------------------------------

    def draw_result(
        self,
        image: np.ndarray,
        anomaly_map: np.ndarray,
        gt_mask: Optional[np.ndarray] = None,
        pred_score: Optional[float] = None,
        gt_label: Optional[int] = None,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Generate a side-by-side visualization result.

        Layout::

            [Title bar with score / label / judgement]
            [Original] [GT Mask] [Heatmap overlay] [Thresholded pred]
            [                 Color bar                              ]

        If *gt_mask* is ``None`` the GT panel is filled with a
        placeholder.

        Args:
            image: (H, W, 3) uint8 BGR original image.
            anomaly_map: (H, W) float anomaly scores.
            gt_mask: (H, W) binary ground truth mask (optional).
            pred_score: Image-level anomaly score (optional).
            gt_label: Ground truth label, 0=normal 1=anomaly (optional).
            threshold: Score threshold for binary prediction.

        Returns:
            Composite visualization as (H', W', 3) uint8 BGR.
        """
        img = self._ensure_bgr_uint8(image)
        h, w = img.shape[:2]
        pad = self._PAD
        n_panels = 4

        # --- panels ---
        # 1. original
        panel_orig = img.copy()

        # 2. GT mask
        if gt_mask is not None:
            gt_vis = np.zeros((h, w, 3), dtype=np.uint8)
            gt_vis[gt_mask > 0] = (0, 0, 255)  # red for defect
        else:
            gt_vis = np.full((h, w, 3), 60, dtype=np.uint8)
            self._put_text_center(gt_vis, 'No GT', h // 2 - 8, 0, w, (180, 180, 180))

        # 3. heatmap overlay
        panel_heat = self.draw_anomaly_map(img, anomaly_map, gt_mask=gt_mask)

        # 4. thresholded prediction
        norm_map = self._normalize_map(anomaly_map)
        pred_mask = (norm_map >= threshold).astype(np.uint8) * 255
        panel_pred = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)
        # tint anomaly regions red
        panel_pred[pred_mask > 0] = (0, 0, 255)

        panels = [panel_orig, gt_vis, panel_heat, panel_pred]
        labels = ['Original', 'GT Mask', 'Anomaly Heatmap', 'Prediction']

        # --- compose canvas ---
        total_w = n_panels * w + (n_panels + 1) * pad
        total_h = self._TITLE_H + h + self._CBAR_H + 3 * pad + 16  # 16 for panel labels
        canvas = np.full((total_h, total_w, 3), self._BG_COLOR, dtype=np.uint8)

        # Title
        parts = []
        if pred_score is not None:
            parts.append(f'Score: {pred_score:.3f}')
        if gt_label is not None:
            parts.append(f'GT: {"anomaly" if gt_label else "normal"}')
        if pred_score is not None:
            judgement = 'ANOMALY' if pred_score >= threshold else 'NORMAL'
            color = (0, 0, 255) if pred_score >= threshold else (0, 200, 0)
            parts.append(judgement)
        title_text = '  |  '.join(parts) if parts else ''
        if title_text:
            self._put_text_center(canvas, title_text, 6, 0, total_w, scale=0.55)

        # Panels + labels
        y0 = self._TITLE_H
        for i, (panel, label) in enumerate(zip(panels, labels)):
            x0 = pad + i * (w + pad)
            canvas[y0:y0 + h, x0:x0 + w] = panel
            self._put_text_center(canvas, label, y0 + h + 2, x0, x0 + w)

        # Color bar
        cbar_y = y0 + h + 18
        cbar = self._make_colorbar(total_w - 2 * pad)
        canvas[cbar_y:cbar_y + self._CBAR_H, pad:pad + cbar.shape[1]] = cbar
        # min/max annotations
        cv2.putText(canvas, '0', (pad, cbar_y + self._CBAR_H + 12),
                    self._FONT, 0.35, self._TEXT_COLOR, 1, cv2.LINE_AA)
        cv2.putText(canvas, '1', (total_w - pad - 8, cbar_y + self._CBAR_H + 12),
                    self._FONT, 0.35, self._TEXT_COLOR, 1, cv2.LINE_AA)

        return canvas

    # ------------------------------------------------------------------
    # draw_batch
    # ------------------------------------------------------------------

    def draw_batch(
        self,
        images: Sequence[np.ndarray],
        anomaly_maps: Sequence[np.ndarray],
        gt_masks: Optional[Sequence[Optional[np.ndarray]]] = None,
        pred_scores: Optional[Sequence[Optional[float]]] = None,
        gt_labels: Optional[Sequence[Optional[int]]] = None,
        max_images: int = 8,
        ncols: int = 4,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Draw a grid of per-sample visualizations.

        Args:
            images: List of (H, W, 3) uint8 images.
            anomaly_maps: List of (H, W) anomaly maps.
            gt_masks: Per-sample GT masks (optional per element).
            pred_scores: Per-sample anomaly scores.
            gt_labels: Per-sample GT labels.
            max_images: Cap on number of samples rendered.
            ncols: Columns in the grid.
            threshold: Score threshold.

        Returns:
            Grid visualization as uint8 BGR array.
        """
        n = min(len(images), max_images)
        gt_masks = gt_masks or [None] * n
        pred_scores = pred_scores or [None] * n
        gt_labels = gt_labels or [None] * n

        rows_list: List[np.ndarray] = []
        row_buf: List[np.ndarray] = []

        target_w: Optional[int] = None

        for i in range(n):
            vis = self.draw_result(
                images[i], anomaly_maps[i],
                gt_mask=gt_masks[i],
                pred_score=pred_scores[i],
                gt_label=gt_labels[i],
                threshold=threshold,
            )
            if target_w is None:
                target_w = vis.shape[1]
            row_buf.append(vis)

            if len(row_buf) == ncols:
                rows_list.append(np.concatenate(row_buf, axis=1))
                row_buf = []

        # pad last row
        if row_buf:
            dummy = np.full_like(row_buf[0], self._BG_COLOR[0], dtype=np.uint8)
            while len(row_buf) < ncols:
                row_buf.append(dummy)
            rows_list.append(np.concatenate(row_buf, axis=1))

        grid = np.concatenate(rows_list, axis=0)
        return grid

    # ------------------------------------------------------------------
    # save_result
    # ------------------------------------------------------------------

    def save_result(
        self,
        output_path: str,
        image: np.ndarray,
        anomaly_map: np.ndarray,
        gt_mask: Optional[np.ndarray] = None,
        pred_score: Optional[float] = None,
        gt_label: Optional[int] = None,
        threshold: float = 0.5,
    ) -> None:
        """Render and save a single result to *output_path*.

        Creates parent directories automatically.
        """
        vis = self.draw_result(image, anomaly_map, gt_mask, pred_score, gt_label, threshold)
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        cv2.imwrite(output_path, vis)

    # ------------------------------------------------------------------
    # add_datasample (standard mmengine Visualizer API)
    # ------------------------------------------------------------------

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
        """Visualize an anomaly detection data sample.

        This implements the standard ``add_datasample`` interface expected
        by mmengine hooks and runners.

        Args:
            name: Visualization window/storage name.
            image: Original image (H, W, 3) uint8.
            data_sample: ADDataSample with prediction and/or GT fields.
            draw_gt: Whether to draw ground truth mask.
            draw_pred: Whether to draw predictions.
            show: Whether to display in a window.
            wait_time: Display wait time (ms), 0 for blocking.
            out_file: Path to save the visualization.
            step: Global step for tensorboard/wandb logging.

        Returns:
            Visualization image as uint8 array.
        """
        gt_mask = None
        gt_label = None
        pred_score = None
        anomaly_map = None

        if data_sample is not None:
            if draw_gt:
                gt_mask_val = getattr(data_sample, 'gt_mask', None)
                if gt_mask_val is not None:
                    if isinstance(gt_mask_val, torch.Tensor):
                        gt_mask = gt_mask_val.cpu().numpy()
                    else:
                        gt_mask = np.asarray(gt_mask_val)
                gt_label = getattr(data_sample, 'gt_label', None)

            if draw_pred:
                pred_score = getattr(data_sample, 'pred_score', None)
                amap = getattr(data_sample, 'pred_anomaly_map', None)
                if amap is not None:
                    if isinstance(amap, torch.Tensor):
                        anomaly_map = amap.squeeze().cpu().numpy()
                    else:
                        anomaly_map = np.asarray(amap).squeeze()

        if anomaly_map is not None:
            vis = self.draw_result(
                image, anomaly_map, gt_mask=gt_mask,
                pred_score=pred_score, gt_label=gt_label,
            )
        else:
            vis = self._ensure_bgr_uint8(image)

        if show:
            self.show(vis, win_name=name, wait_time=wait_time)

        if out_file is not None:
            os.makedirs(os.path.dirname(out_file) or '.', exist_ok=True)
            cv2.imwrite(out_file, vis)

        self.set_image(vis)
        self.add_image(name, vis, step)
        return vis
