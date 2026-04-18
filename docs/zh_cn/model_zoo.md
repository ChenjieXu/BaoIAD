# 模型库

BaoIAD 集成了 50 多种异常检测方法，按其底层范式分类。

## 记忆库方法

| 方法 | 配置目录 | 骨干网络 |
|------|----------|----------|
| PatchCore | `configs/patchcore/` | WRN-50-2 |
| SPADE | `configs/spade/` | WRN-50-2 |
| PaDiM | `configs/padim/` | WRN-50-2 / ResNet-18 |
| DFM | `configs/dfm/` | WRN-50-2 |
| DFKDE | `configs/dfkde/` | WRN-50-2 |
| RegAD | `configs/regad/` | WRN-50-2 |
| GraphCore | `configs/graphcore/` | ViG |

## 知识蒸馏方法

| 方法 | 配置目录 | 骨干网络 |
|------|----------|----------|
| RD | `configs/rd/` | WRN-50-2 |
| RD++ | `configs/rdpp/` | WRN-50-2 |
| STFPM | `configs/stfpm/` | WRN-50-2 / ResNet-18 |
| EfficientAD | `configs/efficientad/` | PDN |
| Dinomaly | `configs/dinomaly/` | DINOv2 |

## 归一化流方法

| 方法 | 配置目录 | 骨干网络 |
|------|----------|----------|
| CSFlow | `configs/csflow/` | WRN-50-2 |
| FastFlow | `configs/fastflow/` | WRN-50-2 / ResNet-18 |
| CFlow | `configs/cflow/` | WRN-50-2 |
| UFlow | `configs/uflow/` | WRN-50-2 |
| DifferNet | `configs/differnet/` | WRN-50-2 |
| PyramidFlow | `configs/pyramidflow/` | WRN-50-2 |

## 重建方法

| 方法 | 配置目录 | 骨干网络 |
|------|----------|----------|
| DRAEM | `configs/draem/` | -- |
| MemSeg | `configs/memseg/` | WRN-50-2 |
| DeSTSeg | `configs/destseg/` | -- |
| MemAE | `configs/memae/` | -- |
| FRE | `configs/fre/` | WRN-50-2 |
| GANomaly | `configs/ganomaly/` | -- |
| DSR | `configs/dsr/` | -- |

## 视觉语言方法

| 方法 | 配置目录 | 骨干网络 |
|------|----------|----------|
| WinCLIP | `configs/winclip/` | OpenCLIP ViT-L/14 |
| AnomalyCLIP | `configs/anomalyclip/` | OpenCLIP ViT-L/14 |
| AnoVL | `configs/anovl/` | OpenCLIP ViT-L/14 |
| MuSc | `configs/musc/` | OpenCLIP ViT-L/14 |
| AdaCLIP | `configs/adaclip/` | OpenCLIP ViT-L/14 |
| AACLIP | `configs/aaclip/` | OpenCLIP ViT-L/14 |
| AnomalyDINO | `configs/anomalydino/` | DINOv2 ViT-L/14 |

## 判别器方法

| 方法 | 配置目录 | 骨干网络 |
|------|----------|----------|
| SimpleNet | `configs/simplenet/` | WRN-50-2 |
| SuperSimpleNet | `configs/supersimplenet/` | WRN-50-2 |
| CFA | `configs/cfa/` | WRN-50-2 |

## 其他方法

InvAD、ViTAD、UniAD、MambaAD、NSA、ResAD、CutPaste、GLASS、AST、PNI、RealNet、ComposeAD、UniNet、UniVAD、SAA+
