"""Generate real PatchCore visualization — bypasses MMEngine runner."""

import os
import sys
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import cm

os.environ.pop('MMENGINE_DIST_INFO', None)

from baoiad.models.detectors.patchcore import PatchCore
from baoiad.datasets import MVTecADDataset
from baoiad.datasets.transforms import LoadImage, LoadMask, ResizeAD, ScaleNormalizeAD, PackADInputs


# ---------- visualization helpers ----------

def heatmap_overlay(image, amap, alpha=0.45):
    amap_n = amap.astype(np.float64)
    vmin, vmax = amap_n.min(), amap_n.max()
    if vmax - vmin > 1e-8:
        amap_n = (amap_n - vmin) / (vmax - vmin)
    else:
        amap_n = np.zeros_like(amap_n)
    heatmap = (cm.jet(amap_n)[:, :, :3] * 255).astype(np.uint8)
    vis = image.astype(np.float32) * (1 - alpha) + heatmap.astype(np.float32) * alpha
    return np.clip(vis, 0, 255).astype(np.uint8)


def hstack_with_pad(panels, pad=3):
    bg = 255
    parts = [panels[0]]
    for p in panels[1:]:
        parts.append(np.full((panels[0].shape[0], pad, 3), bg, dtype=np.uint8))
        parts.append(p)
    return np.hstack(parts)


def make_label(text, h, w=90):
    bg = 255
    bar = np.full((h, w, 3), bg, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.5, 1)
    cv2.putText(bar, text, ((w - tw) // 2, h // 2 + th // 2),
                font, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    return bar


def make_col_header(text, w=256, h=40):
    bg = 255
    panel = np.full((h, w, 3), bg, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.55, 1)
    cv2.putText(panel, text, ((w - tw) // 2, 28), font, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    return panel


# ---------- PatchCore inference ----------

def build_model(device='cuda:0'):
    model = PatchCore(
        backbone=dict(
            type='TIMMBackbone',
            model_name='wide_resnet50_2',
            pretrained=True,
            features_only=True,
            out_indices=(2, 3),
            frozen=True,
        ),
        neck=dict(type='MultiScalePooling', output_size=28),
        head=dict(
            type='MemoryBankHead',
            coreset_ratio=0.1,
            num_neighbors=9,
            distance='euclidean',
            input_size=(256, 256),
            blur_sigma=4.0,
            reweight_scores=False,
            image_score_source='postprocessed',
            patch_score_neighbors=1,
            patch_score_reduction='first',
            coreset_sampling_method='approx_greedy',
            coreset_projection_dim=128,
            coreset_starting_points=10,
            coreset_device='auto',
        ),
        freeze_backbone=True,
    ).to(device).eval()
    return model


def extract_train_features(model, data_root, category, device='cuda:0'):
    """Extract features from training (good) images."""
    from mmengine.dataset import pseudo_collate
    pipeline = [LoadImage(), ResizeAD(size=256), ScaleNormalizeAD(), PackADInputs()]
    ds = MVTecADDataset(
        data_root=data_root, split='train',
        cls_names=[category], pipeline=pipeline,
    )
    loader = DataLoader(ds, batch_size=8, num_workers=0, shuffle=False,
                        collate_fn=pseudo_collate)

    model.head._train_features = []
    with torch.no_grad():
        for batch in loader:
            # pseudo_collate returns list of dicts
            if isinstance(batch, list):
                inputs = torch.stack([b['inputs'] for b in batch]).to(device)
            else:
                inputs = batch['inputs']
                if isinstance(inputs, list):
                    inputs = torch.stack(inputs)
                inputs = inputs.to(device)
            feats = model.extract_feat(inputs)
            model.head.collect_features(feats)
    print(f'  Collected {sum(f.shape[0] for f in model.head._train_features)} patches from {len(ds)} train images')

    model.build_memory_bank()
    print(f'  Memory bank: {model.head.memory_bank.shape}')


def predict_one(model, image_np, device='cuda:0'):
    """Run PatchCore on a single image, return anomaly map and score."""
    # Prepare tensor
    img = image_np.astype(np.float32) / 255.0
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # Normalize (ImageNet)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)  # CHW
    tensor = torch.from_numpy(img).unsqueeze(0).float().to(device)

    with torch.no_grad():
        feats = model.extract_feat(tensor)
        # Create a fake data_sample
        from baoiad.structures import ADDataSample
        ds = ADDataSample()
        ds.set_metainfo(dict(img_path=''))
        results = model.head.predict(feats, [ds])

    pred = results[0]
    amap = pred.pred_anomaly_map.squeeze().cpu().numpy()
    score = float(pred.pred_score)
    return amap, score


def load_image(path, size=256):
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.resize(img, (size, size))


def load_gt_mask(path, size=256):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return cv2.resize(m, (size, size))


# ---------- main ----------

def generate_figure(data_root, out_path, categories, device='cuda:0'):
    """Generate paper-style figure with real PatchCore results."""
    sz = 256
    pad = 3
    bg = 255
    label_w = 90

    model = build_model(device)
    col_headers = ['Image', 'Ground Truth', 'Anomaly Map', 'Prediction']

    total_w = label_w + len(col_headers) * sz + (len(col_headers) - 1) * pad
    rows = []

    # Header
    empty_label = np.full((40, label_w, 3), bg, dtype=np.uint8)
    header_row = np.hstack([empty_label, hstack_with_pad([make_col_header(h) for h in col_headers])])
    rows.append(header_row)
    rows.append(np.full((2, total_w, 3), 220, dtype=np.uint8))

    for cat, defect, idx in categories:
        print(f'Processing {cat}/{defect}/{idx}...')
        # Build memory bank for this category
        extract_train_features(model, data_root, cat, device)

        # Load test image
        img_path = os.path.join(data_root, cat, 'test', defect, f'{idx:03d}.png')
        gt_path = os.path.join(data_root, cat, 'ground_truth', defect, f'{idx:03d}_mask.png')

        image = load_image(img_path, sz)
        gt_mask = load_gt_mask(gt_path, sz)
        if image is None:
            print(f'  SKIP: {img_path} not found')
            continue

        # Real PatchCore inference
        amap, score = predict_one(model, image, device)
        print(f'  Score: {score:.4f}')

        # Panels
        panel_img = image.copy()

        panel_gt = np.full((sz, sz, 3), bg, dtype=np.uint8)
        if gt_mask is not None:
            panel_gt[gt_mask > 0] = (0, 0, 200)

        panel_heatmap = heatmap_overlay(image, amap)

        pred_mask = (amap >= 0.5).astype(np.uint8)
        panel_pred = np.full((sz, sz, 3), bg, dtype=np.uint8)
        panel_pred[pred_mask > 0] = (60, 60, 220)

        data_row = np.hstack([
            make_label(cat.capitalize(), sz),
            hstack_with_pad([panel_img, panel_gt, panel_heatmap, panel_pred]),
        ])
        rows.append(data_row)
        rows.append(np.full((pad, total_w, 3), bg, dtype=np.uint8))

        # Reset for next category
        model.head._train_features = []
        model.head.memory_bank = None
        model.head._nn_index = None

    figure = np.vstack(rows)
    figure = cv2.copyMakeBorder(figure, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(200, 200, 200))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, figure)
    print(f'\nSaved: {out_path}  ({figure.shape[1]}x{figure.shape[0]})')


if __name__ == '__main__':
    data_root = 'data/mvtec_ad'
    out_path = 'resources/vis_examples/anomaly_detection_results.png'
    device = 'cuda:0'

    categories = [
        ('bottle', 'broken_large', 0),
        ('hazelnut', 'crack', 0),
        ('tile', 'crack', 0),
    ]

    generate_figure(data_root, out_path, categories, device)
