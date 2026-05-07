# Dataset Zoo

BaoIAD supports 10 industrial anomaly detection datasets. Each dataset has a base config file under [`configs/_base_/datasets/`](../../configs/_base_/datasets/) that defines the dataset class, data root, default image size, data pipeline, and evaluation metric.

## Supported Datasets

| Dataset | Config | Dataset Class | `data_root` | Default Batch Size | Categories | Modality |
|---------|--------|---------------|-------------|--------------------|------------|----------|
| MVTec AD | [`mvtec_ad.py`](../../configs/_base_/datasets/mvtec_ad.py) | `MVTecADDataset` | `data/mvtec_ad` | 32 | 15 | RGB |
| VisA | [`visa.py`](../../configs/_base_/datasets/visa.py) | `VisADataset` | `data/visa` | 8 | 12 | RGB |
| BTech | [`btech.py`](../../configs/_base_/datasets/btech.py) | `BTechDataset` | `data/btech` | 8 | 3 | RGB |
| MVTec 3D AD | [`mvtec_3d_ad.py`](../../configs/_base_/datasets/mvtec_3d_ad.py) | `MVTec3DDataset` | `data/mvtec_3d_ad` | 8 | 10 | RGB + 3D |
| MVTec LOCO AD | [`mvtec_loco_ad.py`](../../configs/_base_/datasets/mvtec_loco_ad.py) | `MVTecLOCODataset` | `data/mvtec_loco_ad` | 8 | 5 | RGB |
| MPDD | [`mpdd.py`](../../configs/_base_/datasets/mpdd.py) | `MPDDDataset` | `data/mpdd` | 32 | 6 | RGB |
| MVTec AD 2 | [`mvtec_ad2.py`](../../configs/_base_/datasets/mvtec_ad2.py) | `MVTecAD2Dataset` | `data/mvtec_ad_2` | 32 | 25+ | RGB |
| Kolektor | [`kolektor.py`](../../configs/_base_/datasets/kolektor.py) | `KolektorDataset` | `data/kolektor` | 32 | 3 | RGB |
| VAD | [`vad.py`](../../configs/_base_/datasets/vad.py) | `VADDataset` | `data/vad` | 32 | 5 | RGB |
| RealIAD | [`realiad.py`](../../configs/_base_/datasets/realiad.py) | `RealIADDataset` | `data/Real-IAD` | 32 | 30 | RGB |

All datasets default to `img_size = 256` and use the same data pipeline (`LoadImage → LoadMask → ResizeAD → NormalizeAD → PackADInputs`).

## Common Config Fields

Every dataset base config defines these fields:

| Field | Description |
|-------|-------------|
| `data_root` | Relative path to the dataset directory (resolved against `BAOIAD_DATA_ROOT`) |
| `train_dataloader` | Batch size, num workers, sampler, dataset class, and pipeline for training |
| `test_dataloader` | Same structure for testing |
| `val_dataloader` | Set to `test_dataloader` for most datasets (no separate validation split) |
| `test_evaluator` | `AnomalyDetectionMetric` (computes image-level and pixel-level metrics) |

### Category Selection

By default, all configs use `multi_class=True`, which iterates over all categories in the dataset. To train or test on a single category, override from the command line:

```bash
python tools/train.py <config> --cfg-options \
    train_dataloader.dataset.cls_names="['bottle']" \
    train_dataloader.dataset.multi_class=False \
    test_dataloader.dataset.cls_names="['bottle']" \
    test_dataloader.dataset.multi_class=False
```

### Train / Test Split

All datasets use a `split` argument (`'train'` or `'test'`) to select the data subset. MVTec AD 2 additionally has a `'val'` split and a `test_type='public'` field.

## Directory Structure Examples

### MVTec AD

```
data/mvtec_ad/
├── bottle/
│   ├── train/
│   │   └── good/
│   │       ├── 000.png
│   │       ├── 001.png
│   │       └── ...
│   └── test/
│       ├── good/
│       │   └── ...
│       ├── broken_large/
│       │   ├── 000.png
│       │   └── ...
│       └── broken_small/
│           └── ...
├── cable/
│   ├── train/
│   │   └── good/
│   └── test/
│       ├── good/
│       ├── bent_wire/
│       └── ...
└── ...                   # 15 categories total
```

- **Train split**: Only `good/` (normal) images.
- **Test split**: Both `good/` (normal) and defect-type subdirectories (anomalous).
- **Ground truth masks**: Stored alongside test images as `<name>_mask.png` or in a dedicated `ground_truth/` directory.

### VisA

```
data/visa/
├── candle/
│   ├── Data/
│   │   ├── Images/
│   │   │   └── <split>/
│   │   │       ├── Normal/
│   │   │       └── Anomaly/
│   │   └── Masks/
│   │       └── <split>/
│   │           └── Anomaly/
│   └── ...
├── capsules/
│   └── ...
└── ...                   # 12 categories total
```

- **Train split**: Normal images only.
- **Test split**: Both normal and anomaly images with corresponding pixel-level masks.

## Evaluation

All dataset configs use `AnomalyDetectionMetric` as the evaluator. It computes:

- **Image-level**: AUROC, F1-max, AP, ECE, FPR@95TPR
- **Pixel-level**: AUROC, F1-max, AP, AUPRO, AUPIMO, ECE

Results are printed to the log and saved in the work directory.

## See Also

- [Prepare Datasets](user_guides/prepare_dataset.md) — Download links and setup instructions for each dataset
- [Config System](user_guides/config.md) — How dataset configs fit into the config hierarchy
