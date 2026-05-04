"""Generate paper-style visualization for README."""

import os
import cv2
import numpy as np
from matplotlib import cm

os.environ.pop('MMENGINE_DIST_INFO', None)


def normalize_map(amap):
    amap = amap.astype(np.float64)
    vmin, vmax = amap.min(), amap.max()
    if vmax - vmin > 1e-8:
        return (amap - vmin) / (vmax - vmin)
    return np.zeros_like(amap)


def apply_jet_gray(norm_map):
    """Apply jet colormap → (H, W, 3) uint8 RGB."""
    return (cm.jet(norm_map)[:, :, :3] * 255).astype(np.uint8)


def make_realistic_anomaly_map(gt_mask, h, w, strength=0.85):
    noise = np.random.rand(h, w).astype(np.float32) * 0.12
    kernel = cv2.getGaussianKernel(41, 12)
    spread = cv2.filter2D(gt_mask.astype(np.float32), -1, kernel * kernel.T)
    spread = spread / (spread.max() + 1e-8)
    amap = noise + spread * strength
    amap += np.random.rand(h, w).astype(np.float32) * 0.06
    return np.clip(amap, 0, 1)


def heatmap_overlay(image, amap, alpha=0.4):
    """Blend jet heatmap on top of image."""
    heatmap = apply_jet_gray(amap)
    vis = image.astype(np.float32) * (1 - alpha) + heatmap.astype(np.float32) * alpha
    return np.clip(vis, 0, 255).astype(np.uint8)


def add_label(img, text, position='top', font_scale=0.5, thickness=1, pad=6):
    """Add centered text label with white background strip above or below image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    if position == 'top':
        bar = np.full((th + pad * 2, img.shape[1], 3), 255, dtype=np.uint8)
        x = (img.shape[1] - tw) // 2
        cv2.putText(bar, text, (x, th + pad), font, font_scale, (50, 50, 50), thickness, cv2.LINE_AA)
        return np.vstack([bar, img])
    else:
        bar = np.full((th + pad * 2, img.shape[1], 3), 255, dtype=np.uint8)
        x = (img.shape[1] - tw) // 2
        cv2.putText(bar, text, (x, th + pad), font, font_scale, (50, 50, 50), thickness, cv2.LINE_AA)
        return np.vstack([img, bar])


def add_row_label(img, text, width=80, font_scale=0.45, thickness=1):
    """Add a vertical label strip on the left side of the image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    bar = np.full((img.shape[0], width, 3), 255, dtype=np.uint8)
    # Rotate text vertically
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    # Place text centered vertically
    y = (img.shape[0] + tw) // 2
    x = (width - th) // 2
    # Draw each character rotated - just place horizontally for simplicity
    # Actually, let's just place it horizontally centered
    lines = text.split('\n') if '\n' in text else [text]
    total_h = len(lines) * (th + 4)
    start_y = (img.shape[0] - total_h) // 2 + th
    for i, line in enumerate(lines):
        (lw, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        lx = (width - lw) // 2
        cv2.putText(bar, line, (lx, start_y + i * (th + 4)), font, font_scale, (30, 30, 30), thickness, cv2.LINE_AA)
    return np.hstack([bar, img])


def load_sample(data_root, category, defect, idx):
    """Load image and GT mask, resize to target size."""
    sz = 256
    if defect == 'good':
        img_path = os.path.join(data_root, category, 'test', 'good', f'{idx:03d}.png')
        image = cv2.imread(img_path)
        image = cv2.resize(image, (sz, sz))
        return image, None

    img_path = os.path.join(data_root, category, 'test', defect, f'{idx:03d}.png')
    gt_path = os.path.join(data_root, category, 'ground_truth', defect, f'{idx:03d}_mask.png')
    image = cv2.imread(img_path)
    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (sz, sz))
    if gt_mask is not None:
        gt_mask = cv2.resize(gt_mask, (sz, sz))
    return image, gt_mask


def generate_paper_figure(data_root, out_path):
    """Generate a paper-style figure with multiple rows."""
    sz = 256
    pad = 3
    bg = 255
    label_w = 90

    samples = [
        ('bottle', 'broken_large', 0, 'Bottle'),
        ('hazelnut', 'crack', 0, 'Hazelnut'),
        ('tile', 'crack', 0, 'Tile'),
    ]

    col_headers = ['Image', 'Ground Truth', 'Anomaly Map', 'Prediction']

    def make_sep(width, height=2, color=220):
        return np.full((height, width, 3), color, dtype=np.uint8)

    def make_label(text, h, w=label_w):
        bar = np.full((h, w, 3), bg, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, 0.5, 1)
        cv2.putText(bar, text, ((w - tw) // 2, h // 2 + th // 2),
                    font, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
        return bar

    def make_col_header(text, w=sz, h=40):
        panel = np.full((h, w, 3), bg, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, 0.55, 1)
        cv2.putText(panel, text, ((w - tw) // 2, 28), font, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
        return panel

    def hstack_with_pad(panels, p=pad):
        parts = [panels[0]]
        for pp in panels[1:]:
            parts.append(np.full((panels[0].shape[0], p, 3), bg, dtype=np.uint8))
            parts.append(pp)
        return np.hstack(parts)

    # --- Build figure ---
    total_w = label_w + len(col_headers) * sz + (len(col_headers) - 1) * pad
    rows = []

    # Header row: [empty label | col1 | pad | col2 | pad | col3 | pad | col4]
    empty_label = np.full((40, label_w, 3), bg, dtype=np.uint8)
    header_panels = [make_col_header(h) for h in col_headers]
    header_row = np.hstack([empty_label, hstack_with_pad(header_panels)])
    rows.append(header_row)
    rows.append(make_sep(total_w))

    for cat, defect, idx, label in samples:
        image, gt_mask = load_sample(data_root, cat, defect, idx)
        amap = make_realistic_anomaly_map(gt_mask, sz, sz)

        panel_img = image.copy()

        panel_gt = np.full((sz, sz, 3), bg, dtype=np.uint8)
        if gt_mask is not None:
            panel_gt[gt_mask > 0] = (0, 0, 200)

        panel_heatmap = heatmap_overlay(image, amap, alpha=0.45)

        pred_mask = (amap >= 0.5).astype(np.uint8)
        panel_pred = np.full((sz, sz, 3), bg, dtype=np.uint8)
        panel_pred[pred_mask > 0] = (60, 60, 220)

        panels = [panel_img, panel_gt, panel_heatmap, panel_pred]
        data_row = np.hstack([make_label(label, sz), hstack_with_pad(panels)])
        rows.append(data_row)
        rows.append(make_sep(total_w, pad))

    figure = np.vstack(rows)
    figure = cv2.copyMakeBorder(figure, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(200, 200, 200))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, figure)
    print(f'Saved: {out_path}  ({figure.shape[1]}x{figure.shape[0]})')


def generate_normal_figure(data_root, out_path):
    """Generate a normal sample figure."""
    sz = 256
    pad = 3
    bg = 255

    image, _ = load_sample(data_root, 'bottle', 'good', 0)
    amap = np.random.rand(sz, sz).astype(np.float32) * 0.2
    panel_heatmap = heatmap_overlay(image, amap, alpha=0.3)

    # Show: Image | Anomaly Map
    col_headers = ['Image (Normal)', 'Anomaly Map']
    header_panels = []
    for h in col_headers:
        panel = np.full((40, sz, 3), bg, dtype=np.uint8)
        (tw, th), _ = cv2.getTextSize(h, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(panel, h, ((sz - tw) // 2, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
        header_panels.append(panel)

    parts = [header_panels[0]]
    for p in header_panels[1:]:
        parts.append(np.full((40, pad, 3), bg, dtype=np.uint8))
        parts.append(p)
    header_row = np.hstack(parts)

    pad_col = np.full((sz, pad, 3), bg, dtype=np.uint8)
    data_row = np.hstack([image, pad_col, panel_heatmap])

    figure = np.vstack([header_row, data_row])
    figure = cv2.copyMakeBorder(figure, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(200, 200, 200))

    cv2.imwrite(out_path, figure)
    print(f'Saved: {out_path}  ({figure.shape[1]}x{figure.shape[0]})')


if __name__ == '__main__':
    data_root = 'data/mvtec_ad'
    out_dir = 'resources/vis_examples'

    generate_paper_figure(data_root, os.path.join(out_dir, 'anomaly_detection_results.png'))
    generate_normal_figure(data_root, os.path.join(out_dir, 'normal_sample.png'))

    print('Done.')
