"""Generate multi-model visualization for README.

Rows = models (PatchCore, PaDiM, SPADE)
Columns = categories (Bottle, Hazelnut, Tile)
Each cell: Image | GT | Anomaly Map | Prediction
All models are memory-bank based — no gradient training needed.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import cm

os.environ.pop('MMENGINE_DIST_INFO', None)

import baoiad  # noqa: E402 — trigger registry
from baoiad.models.detectors.patchcore import PatchCore
from baoiad.models.detectors.padim import PaDiMDetector
from baoiad.models.detectors.spade import SPADEDetector
from baoiad.datasets import MVTecADDataset
from baoiad.datasets.transforms import LoadImage, ResizeAD, ScaleNormalizeAD, PackADInputs
from baoiad.structures import ADDataSample
from mmengine.dataset import pseudo_collate


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


def normalize_map(amap):
    a = amap.astype(np.float64)
    vmin, vmax = a.min(), a.max()
    if vmax - vmin > 1e-8:
        return (a - vmin) / (vmax - vmin)
    return np.zeros_like(a)


def hstack_with_pad(panels, pad=2):
    bg = 255
    parts = [panels[0]]
    for p in panels[1:]:
        parts.append(np.full((panels[0].shape[0], pad, 3), bg, dtype=np.uint8))
        parts.append(p)
    return np.hstack(parts)


def make_label(text, h, w=80, font_scale=0.4):
    bar = np.full((h, w, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = text.split('\n') if '\n' in text else [text]
    th_max = max(cv2.getTextSize(l, font, font_scale, 1)[0][1] for l in lines)
    start_y = (h - len(lines) * (th_max + 4)) // 2 + th_max
    for i, line in enumerate(lines):
        (lw, _), _ = cv2.getTextSize(line, font, font_scale, 1)
        cv2.putText(bar, line, ((w - lw) // 2, start_y + i * (th_max + 4)),
                    font, font_scale, (30, 30, 30), 1, cv2.LINE_AA)
    return bar


def make_col_header(text, w, h=36, font_scale=0.55):
    panel = np.full((h, w, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = text.split('\n')
    th_max = max(cv2.getTextSize(l, font, font_scale, 1)[0][1] for l in lines)
    start_y = (h - len(lines) * (th_max + 4)) // 2 + th_max
    for i, line in enumerate(lines):
        (lw, _), _ = cv2.getTextSize(line, font, font_scale, 1)
        cv2.putText(panel, line, ((w - lw) // 2, start_y + i * (th_max + 4)),
                    font, font_scale, (30, 30, 30), 1, cv2.LINE_AA)
    return panel


# ---------- data helpers ----------

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


def prepare_tensor(image_np, device='cuda:0'):
    img = image_np.astype(np.float32) / 255.0
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    return torch.from_numpy(img).unsqueeze(0).float().to(device)


def build_dataloader(data_root, category):
    pipeline = [LoadImage(), ResizeAD(size=256), ScaleNormalizeAD(), PackADInputs()]
    ds = MVTecADDataset(
        data_root=data_root, split='train',
        cls_names=[category], pipeline=pipeline,
    )
    loader = DataLoader(ds, batch_size=8, num_workers=0,
                        shuffle=False, collate_fn=pseudo_collate)
    return loader, len(ds)


def get_inputs(batch, device):
    if isinstance(batch, list):
        return torch.stack([b['inputs'] for b in batch]).to(device)
    inputs = batch['inputs']
    if isinstance(inputs, list):
        inputs = torch.stack(inputs)
    return inputs.to(device)


# ---------- model builders ----------

def build_patchcore(device):
    return PatchCore(
        backbone=dict(
            type='TIMMBackbone', model_name='wide_resnet50_2', pretrained=True,
            features_only=True, out_indices=(2, 3), frozen=True,
        ),
        neck=dict(type='MultiScalePooling', output_size=28),
        head=dict(
            type='MemoryBankHead', coreset_ratio=0.1, num_neighbors=9,
            distance='euclidean', input_size=(256, 256), blur_sigma=4.0,
            reweight_scores=False, image_score_source='postprocessed',
            patch_score_neighbors=1, patch_score_reduction='first',
            coreset_sampling_method='approx_greedy', coreset_projection_dim=128,
            coreset_starting_points=10, coreset_device='auto',
        ),
        freeze_backbone=True,
    ).to(device).eval()


def build_padim(device):
    return PaDiMDetector(
        backbone=dict(
            type='TIMMBackbone', model_name='wide_resnet50_2', pretrained=True,
            features_only=True, out_indices=(1, 2, 3), frozen=True,
        ),
        sigma=4.0,
    ).to(device).eval()


def build_spade(device):
    return SPADEDetector(
        backbone=dict(
            type='TIMMBackbone', model_name='wide_resnet50_2', pretrained=True,
            features_only=True, out_indices=(1, 2, 3), frozen=True,
        ),
        k=5,
    ).to(device).eval()


# ---------- train, predict, reset ----------

def train_model(model, data_root, category, device):
    loader, n = build_dataloader(data_root, category)
    if isinstance(model, PatchCore):
        model.head._train_features = []
        with torch.no_grad():
            for batch in loader:
                inputs = get_inputs(batch, device)
                feats = model.extract_feat(inputs)
                model.head.collect_features(feats)
        model.build_memory_bank()
        print(f'  Memory bank: {model.head.memory_bank.shape}')
    else:
        with torch.no_grad():
            for batch in loader:
                inputs = get_inputs(batch, device)
                model(inputs, [], mode='loss')
        model.build_memory_bank()
        print(f'  Memory bank built from {n} train images')


def predict_model(model, image_np, device='cuda:0'):
    tensor = prepare_tensor(image_np, device)
    ds = ADDataSample()
    ds.set_metainfo(dict(img_path=''))
    with torch.no_grad():
        if isinstance(model, PatchCore):
            feats = model.extract_feat(tensor)
            results = model.head.predict(feats, [ds])
        else:
            results = model(tensor, [ds], mode='predict')
    pred = results[0]
    amap = pred.pred_anomaly_map.squeeze().cpu().numpy()
    score = float(pred.pred_score)
    return amap, score


def reset_model(model):
    if isinstance(model, PatchCore):
        model.head._train_features = []
        model.head.memory_bank = None
        model.head._nn_index = None
    elif isinstance(model, PaDiMDetector):
        model._features = []
        model.mean = None
        model.cov_inv = None
    elif isinstance(model, SPADEDetector):
        model._layer_features = [[], [], []]
        model._gap_features = []
        model.memory_bank_0 = None
        model.memory_bank_1 = None
        model.memory_bank_2 = None
        model.memory_bank_gap = None


# ---------- main ----------

def generate_figure(data_root, out_path, device='cuda:0'):
    sub_sz = 100  # sub-panel size
    sub_pad = 2
    cat_pad = 10  # gap between category columns
    bg = 255
    label_w = 80

    models_info = [
        ('PatchCore', build_patchcore),
        ('PaDiM', build_padim),
        ('SPADE', build_spade),
    ]
    categories = [
        ('Bottle', 'bottle', 'broken_large', 0),
        ('Hazelnut', 'hazelnut', 'crack', 0),
        ('Tile', 'tile', 'crack', 0),
    ]
    n_cats = len(categories)
    n_subs = 4  # Image, GT, Anomaly Map, Prediction per cell

    cell_w = n_subs * sub_sz + (n_subs - 1) * sub_pad
    total_w = label_w + n_cats * cell_w + (n_cats - 1) * cat_pad

    rows = []

    # --- Category headers ---
    empty_label = np.full((36, label_w, 3), bg, dtype=np.uint8)
    cat_parts = [make_col_header(categories[0][0], cell_w)]
    for cat_label, _, _, _ in categories[1:]:
        cat_parts.append(np.full((36, cat_pad, 3), bg, dtype=np.uint8))
        cat_parts.append(make_col_header(cat_label, cell_w))
    rows.append(np.hstack([empty_label] + cat_parts))
    rows.append(np.full((2, total_w, 3), 200, dtype=np.uint8))

    # --- Run all models on all categories ---
    all_panels = {}  # (model_idx, cat_idx) -> [img, gt, heatmap, pred] at sub_sz

    for m_idx, (model_name, builder) in enumerate(models_info):
        model = builder(device)
        for c_idx, (_, cat, defect, img_idx) in enumerate(categories):
            print(f'\n{model_name} on {cat}/{defect}...')
            train_model(model, data_root, cat, device)

            img_path = os.path.join(data_root, cat, 'test', defect, f'{img_idx:03d}.png')
            gt_path = os.path.join(data_root, cat, 'ground_truth', defect,
                                   f'{img_idx:03d}_mask.png')

            image = load_image(img_path, 256)
            gt_mask = load_gt_mask(gt_path, 256)
            if image is None:
                print(f'  SKIP: {img_path}')
                all_panels[(m_idx, c_idx)] = [
                    np.full((sub_sz, sub_sz, 3), bg, dtype=np.uint8)] * 4
                continue

            amap, score = predict_model(model, image, device)
            print(f'  Score: {score:.4f}')

            # Resize everything to display size
            image_sub = cv2.resize(image, (sub_sz, sub_sz))
            gt_sub = cv2.resize(gt_mask, (sub_sz, sub_sz)) if gt_mask is not None else None
            amap_sub = cv2.resize(amap, (sub_sz, sub_sz))
            amap_norm = normalize_map(amap_sub)

            panel_img = image_sub.copy()
            panel_gt = np.full((sub_sz, sub_sz, 3), bg, dtype=np.uint8)
            if gt_sub is not None:
                panel_gt[gt_sub > 0] = (0, 0, 200)
            panel_heatmap = heatmap_overlay(image_sub, amap_sub)
            pred_mask = (amap_norm >= 0.5).astype(np.uint8)
            panel_pred = np.full((sub_sz, sub_sz, 3), bg, dtype=np.uint8)
            panel_pred[pred_mask > 0] = (60, 60, 220)

            all_panels[(m_idx, c_idx)] = [
                panel_img, panel_gt, panel_heatmap, panel_pred]

            reset_model(model)

    # --- Build data rows ---
    for m_idx, (model_name, _) in enumerate(models_info):
        model_label = make_label(model_name, sub_sz, label_w)
        cat_cells = []
        for c_idx in range(n_cats):
            cell = hstack_with_pad(all_panels[(m_idx, c_idx)], sub_pad)
            cat_cells.append(cell)
        # Join category cells with wider padding
        row_parts = [cat_cells[0]]
        for c in cat_cells[1:]:
            row_parts.append(np.full((sub_sz, cat_pad, 3), bg, dtype=np.uint8))
            row_parts.append(c)
        rows.append(np.hstack([model_label] + row_parts))
        rows.append(np.full((2, total_w, 3), 220, dtype=np.uint8))

    figure = np.vstack(rows)
    figure = cv2.copyMakeBorder(figure, 3, 3, 3, 3, cv2.BORDER_CONSTANT,
                                value=(180, 180, 180))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, figure)
    print(f'\nSaved: {out_path}  ({figure.shape[1]}x{figure.shape[0]})')


if __name__ == '__main__':
    generate_figure('data/mvtec_ad',
                    'resources/vis_examples/anomaly_detection_results.png')
