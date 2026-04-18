# 数据集库

## 支持的数据集

| 数据集 | 类别数 | 像素掩码 | 配置 |
|--------|--------|----------|------|
| MVTec AD | 15 | 是 | `configs/_base_/datasets/mvtec_ad.py` |
| VisA | 12 | 是 | `configs/_base_/datasets/visa.py` |
| BTech | 3 | 是 | `configs/_base_/datasets/btech.py` |
| MVTec 3D AD | 10 | 是 | `configs/_base_/datasets/mvtec_3d_ad.py` |
| MVTec LOCO | 5 | 是 | `configs/_base_/datasets/mvtec_loco_ad.py` |
| MPDD | 6 | 是 | `configs/_base_/datasets/mpdd.py` |
| MVTec AD 2 | 16 | 是 | `configs/_base_/datasets/mvtec_ad2.py` |
| Kolektor | 3 | 是 | `configs/_base_/datasets/kolektor.py` |
| VAD | 6 | 是 | `configs/_base_/datasets/vad.py` |
| RealIAD | -- | 是 | `configs/_base_/datasets/realiad.py` |

## 数据目录结构

```
data/
├── mvtec_ad/
│   ├── bottle/
│   │   ├── train/good/
│   │   └── test/
│   │       ├── good/
│   │       └── broken_large/
│   └── ...
├── visa/
└── ...
```

## 配置数据路径

```bash
export BAOIAD_DATA_ROOT=/path/to/data
```
