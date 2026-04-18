# Dataset Zoo

BaoIAD supports 10 industrial anomaly detection datasets out of the box.

## Supported Datasets

| Dataset | Categories | Defect Types | Pixel Masks | Config |
|---------|-----------|-------------|-------------|--------|
| MVTec AD | 15 | 73 | Yes | `configs/_base_/datasets/mvtec_ad.py` |
| VisA | 12 | 96 | Yes | `configs/_base_/datasets/visa.py` |
| BTech | 3 | 7 | Yes | `configs/_base_/datasets/btech.py` |
| MVTec 3D AD | 10 | 48 | Yes | `configs/_base_/datasets/mvtec_3d_ad.py` |
| MVTec LOCO | 5 | 18 | Yes | `configs/_base_/datasets/mvtec_loco_ad.py` |
| MPDD | 6 | 12 | Yes | `configs/_base_/datasets/mpdd.py` |
| MVTec AD 2 | 16 | -- | Yes | `configs/_base_/datasets/mvtec_ad2.py` |
| Kolektor | 3 | 7 | Yes | `configs/_base_/datasets/kolektor.py` |
| VAD | 6 | -- | Yes | `configs/_base_/datasets/vad.py` |
| RealIAD | -- | -- | Yes | `configs/_base_/datasets/realiad.py` |

## Download Instructions

### MVTec AD

Download from [the MVTec AD website](https://www.mvtec.com/company/research/datasets/mvtec-ad) (requires license agreement).

```bash
# After downloading, extract to data directory
mkdir -p data/mvtec_ad
unzip mvtec_anomaly_detection.tar.xz -d data/mvtec_ad
```

### VisA

Download from [the VisA repository](https://github.com/amazon-science/spot-diff).

```bash
mkdir -p data/visa
# Extract downloaded archive to data/visa
```

### BTech

Download from [the BTech website](https://avires.dimi.uniud.it/papers/btad/btad.zip).

### MVTec 3D AD

Download from [the MVTec 3D AD website](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad).

### Other Datasets

See each dataset's official website for download instructions. Place all datasets under the `data/` directory or set `BAOIAD_DATA_ROOT`.

## Data Directory Structure

BaoIAD expects datasets to follow a standard directory layout:

```
data/
├── mvtec_ad/
│   ├── bottle/
│   │   ├── train/
│   │   │   └── good/
│   │   │       ├── 000.png
│   │   │       └── ...
│   │   └── test/
│   │       ├── good/
│   │       │   └── ...
│   │       └── broken_large/
│   │           └── ...
│   ├── cable/
│   └── ...
├── visa/
│   ├── candle/
│   │   ├── train/
│   │   │   └── good/
│   │   └── test/
│   └── ...
└── ...
```

### MVTec AD Category List

bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper

### VisA Category List

candle, capsules, cashew, chewinggum, fryum, macaroni1, macaroni2, pcb1, pcb2, pcb3, pcb4, pipe_fryum

## Configuring Data Paths

### Default Path

By default, BaoIAD looks for data in the `data/` directory relative to the project root. You can override this:

```bash
# Via environment variable
export BAOIAD_DATA_ROOT=/path/to/data

# Via config override
python tools/train.py <config> --cfg-options \
    train_dataloader.dataset.data_root=/path/to/mvtec_ad
```

### Per-Dataset Config

Each base dataset config specifies its own `data_root` relative to `BAOIAD_DATA_ROOT`:

```python
# configs/_base_/datasets/mvtec_ad.py
data_root = 'data/mvtec_ad'
```

## Multi-Class vs Single-Class Training

BaoIAD supports two training modes:

- **Multi-class** (`multi_class=True`): Train one model on all categories simultaneously (used by UniAD, ViTAD, InvAD)
- **Single-class** (`multi_class=False`): Train a separate model per category (default for most methods)

```bash
# Single category (default mode)
python tools/train.py <config> --work-dir runs/bottle \
    --cfg-options train_dataloader.dataset.cls_names="['bottle']" train_dataloader.dataset.multi_class=False

# All categories (multi-class)
python tools/train.py <config> --work-dir runs/all \
    --cfg-options train_dataloader.dataset.multi_class=True
```
