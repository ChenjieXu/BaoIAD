# Prepare Datasets

This guide explains how to download, organize, and configure each dataset supported by BaoIAD.

## Set the Data Root

BaoIAD reads datasets from a top-level directory configured via the
`BAOIAD_DATA_ROOT` environment variable. Each base config appends its dataset
name to that directory. For example, MVTec AD resolves to
`$BAOIAD_DATA_ROOT/mvtec_ad`. Without the environment variable, it resolves to
`<repo>/data/mvtec_ad`.

```bash
# Option 1: Source the provided script (defaults to <repo>/data/)
source tools/env.sh

# Option 2: Export manually
export BAOIAD_DATA_ROOT=/path/to/your/datasets
```

Relative `BAOIAD_DATA_ROOT` values are resolved against the repository root,
not the process's current working directory. A command-line `data_root`
override is the final dataset path and takes precedence over the environment.

## Dataset Config Files

Each dataset has a base config in [`configs/_base_/datasets/`](../../../configs/_base_/datasets/):

| Dataset | Config |
|---------|--------|
| MVTec AD | `configs/_base_/datasets/mvtec_ad.py` |
| VisA | `configs/_base_/datasets/visa.py` |
| BTech | `configs/_base_/datasets/btech.py` |
| MVTec 3D AD | `configs/_base_/datasets/mvtec_3d_ad.py` |
| MVTec LOCO AD | `configs/_base_/datasets/mvtec_loco_ad.py` |
| MPDD | `configs/_base_/datasets/mpdd.py` |
| MVTec AD 2 | `configs/_base_/datasets/mvtec_ad2.py` |
| Kolektor | `configs/_base_/datasets/kolektor.py` |
| VAD | `configs/_base_/datasets/vad.py` |
| RealIAD | `configs/_base_/datasets/realiad.py` |

## Taxonomy Terms

- **BaoIAD object entries** are the values exposed through the primary dataset
  class's `ALL_CATEGORIES` selector. Kolektor and VAD use one synthetic adapter
  entry each; those values are not claims about an upstream official taxonomy.
- **Defect categories** are abnormal subtypes. The current primary loaders do
  not contain exhaustive static defect lists: they discover labels from files
  or metadata, or map them to a binary adapter label. BaoIAD therefore does not
  publish a static defect-category count for these datasets.
- **Base config entries** count unique public base-config and primary
  dataset-class pairs. Each row above contributes one entry, for 10 total.

You can override the `data_root` for any dataset from the command line:

```bash
python tools/train.py <config> --cfg-options \
    train_dataloader.dataset.data_root=/custom/path/mvtec_ad \
    test_dataloader.dataset.data_root=/custom/path/mvtec_ad
```

## Download and Setup

### MVTec AD

- **Source**: [https://www.mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- **License**: Academic / non-commercial (requires agreement)
- **BaoIAD object entries**: 15 (bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper)
- **Expected layout**:

```
data/mvtec_ad/
├── bottle/
│   ├── train/
│   │   └── good/           # Normal training images (PNG)
│   └── test/
│       ├── good/           # Normal test images
│       ├── broken_large/   # Defect type 1
│       ├── broken_small/   # Defect type 2
│       └── contamination/  # Defect type 3
├── cable/
│   ├── train/good/
│   └── test/
│       ├── good/
│       ├── bent_wire/
│       └── ...
└── ...
```

Ground truth masks for test anomalies are stored as `<stem>_mask.png` alongside the images or in a `ground_truth/` subdirectory.

### VisA

- **Source**: [https://github.com/amazon-science/spot-diff](https://github.com/amazon-science/spot-diff)
- **BaoIAD object entries**: 12 (candle, capsules, cashew, chewinggum, fryum, macaroni1, macaroni2, pcb1, pcb2, pcb3, pcb4, pipe_fryum)
- **Expected layout**:

```
data/visa/
├── candle/
│   └── Data/
│       ├── Images/
│       │   └── <split>/
│       │       ├── Normal/
│       │       └── Anomaly/
│       └── Masks/
│           └── <split>/
│               └── Anomaly/
├── capsules/
│   └── ...
└── ...
```

### BTech

- **Source**: [https://avires.dimi.uniud.it/papers/btad/btad.zip](https://avires.dimi.uniud.it/papers/btad/btad.zip)
- **BaoIAD object entries**: 3 (01, 02, 03)
- **Expected layout**: Same structure as MVTec AD (train/good + test/defect_types).
- **`data_root`**: `data/btech`

### MVTec 3D AD

- **Source**: [https://www.mvtec.com/company/research/datasets/mvtec-3d-ad](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad)
- **BaoIAD object entries**: 10
- **Modality**: RGB + 3D (point clouds / organized depth maps)
- **`data_root`**: `data/mvtec_3d_ad`
- **Note**: BaoIAD's base config uses the RGB modality. The 3D data is available in the dataset but may require method-specific pipeline changes.

### MVTec LOCO AD

- **Source**: [https://www.mvtec.com/company/research/datasets/mvtec-loco](https://www.mvtec.com/company/research/datasets/mvtec-loco)
- **BaoIAD object entries**: 5 (breakfast_box, juice_bottle, pushpins, screw_bag, splicing_connectors)
- **`data_root`**: `data/mvtec_loco_ad`
- **Note**: Contains both logical and structural anomalies.

### MPDD

- **Source**: [https://github.com/stepanje/MPDD](https://github.com/stepanje/MPDD)
- **BaoIAD object entries**: 6
- **`data_root`**: `data/mpdd`

### MVTec AD 2

- **Source**: [https://www.mvtec.com/company/research/datasets/mvtec-ad-2](https://www.mvtec.com/company/research/datasets/mvtec-ad-2)
- **BaoIAD object entries**: 8 (can, fabric, fruit_jelly, rice, sheet_metal, vial, wallplugs, walnuts)
- **`data_root`**: `data/mvtec_ad_2`
- **Note**: This dataset has separate `train`, `val`, and `test` splits. The base config uses `test_type='public'`.

### Kolektor

- **Source**: [https://www.vicos.si/resources/kolektorsdd/](https://www.vicos.si/resources/kolektorsdd/)
- **BaoIAD object entries**: 1 synthetic adapter entry (`kolektor`)
- **`data_root`**: `data/kolektor`

### VAD

- **Source**: [https://github.com/hq-deng/RD4AD](https://github.com/hq-deng/RD4AD) (via the VAD subset)
- **BaoIAD object entries**: 1 synthetic adapter entry (`vad`)
- **`data_root`**: `data/vad`

### RealIAD

- **Source**: [https://github.com/TencentYoutuResearch/AnomalyDetection-RealIAD](https://github.com/TencentYoutuResearch/AnomalyDetection-RealIAD)
- **BaoIAD object entries**: 30
- **`data_root`**: `data/Real-IAD`
- **Note**: Requires additional JSON annotation files. Set `json_path='realiad_jsons/realiad_jsons'` (already configured in the base config). The `resolution` field selects image resolution (`'256'` by default).

## Custom Datasets

To use your own dataset, follow the MVTec AD directory convention:

1. Create a directory under `$BAOIAD_DATA_ROOT/` with your dataset name.
2. Organize categories as `<category>/train/good/` and `<category>/test/<defect_type>/`.
3. Provide ground truth masks as `<stem>_mask.png` in the defect-type directories.
4. Either create a new dataset config in `configs/_base_/datasets/` or override `data_root` and `cls_names` via `--cfg-options`:

```bash
python tools/train.py configs/patchcore/patchcore_wrn50_256_mvtec_strict.py \
    --work-dir runs/custom \
    --cfg-options \
    train_dataloader.dataset.data_root=data/my_custom_dataset \
    train_dataloader.dataset.cls_names="['category_a', 'category_b']" \
    train_dataloader.dataset.multi_class=True \
    test_dataloader.dataset.data_root=data/my_custom_dataset \
    test_dataloader.dataset.cls_names="['category_a', 'category_b']" \
    test_dataloader.dataset.multi_class=True
```

For more complex layouts, create a new dataset class in `baoiad/datasets/` that inherits from `BaseADDataset` and register it in the `DATASETS` registry. See the existing dataset implementations in [`baoiad/datasets/`](../../../baoiad/datasets/) for reference.
