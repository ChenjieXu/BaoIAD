# 数据集库

## 支持的数据集

表中明确区分三种口径：

- **BaoIAD 对象入口**与主数据集类的 `ALL_CATEGORIES` 选择器一致。
  Kolektor 和 VAD 各使用一个适配器合成入口，不代表对上游官方对象分类的声明。
- **缺陷类别口径**只在主 loader 存在穷举的静态异常子类表时计数。
  当前 loader 都从目录或元数据动态读取，或映射为二值标签，因此不声称静态缺陷数量。
- **基础配置入口**统计唯一的公开 base config 与主数据集类组合，
  本版本共 10 个。

| 数据集 | BaoIAD 对象入口 | 基础配置入口 | 缺陷类别口径 | 像素掩码 | 配置 |
|--------|------------------:|---------------:|--------------|----------|------|
| MVTec AD | 15 | 1 | 未静态穷举 | 是 | `configs/_base_/datasets/mvtec_ad.py` |
| VisA | 12 | 1 | 未静态穷举 | 是 | `configs/_base_/datasets/visa.py` |
| BTech | 3 | 1 | 未静态穷举 | 是 | `configs/_base_/datasets/btech.py` |
| MVTec 3D AD | 10 | 1 | 未静态穷举 | 是 | `configs/_base_/datasets/mvtec_3d_ad.py` |
| MVTec LOCO AD | 5 | 1 | 未静态穷举 | 是 | `configs/_base_/datasets/mvtec_loco_ad.py` |
| MPDD | 6 | 1 | 未静态穷举 | 是 | `configs/_base_/datasets/mpdd.py` |
| MVTec AD 2 | 8 | 1 | 未静态穷举 | 是 | `configs/_base_/datasets/mvtec_ad2.py` |
| Kolektor | 1（适配器） | 1 | 未静态穷举 | 是 | `configs/_base_/datasets/kolektor.py` |
| VAD | 1（适配器） | 1 | 未静态穷举 | 是 | `configs/_base_/datasets/vad.py` |
| RealIAD | 30 | 1 | 未静态穷举 | 是 | `configs/_base_/datasets/realiad.py` |

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
