# BaoIAD

<p align="center">
  <img src="resources/baoiad-hero.png" width="900" alt="BaoIAD Industrial Anomaly Detection Benchmark">
</p>

<p align="center">
  <a href="https://baoiad.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/docs-latest-blue" alt="docs"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="license"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/pytorch-2.0%2B-orange" alt="pytorch"></a>
  <a href="docs/alignment/README.md"><img src="https://img.shields.io/badge/methods-37-purple" alt="methods"></a>
  <a href="https://github.com/ChenjieXu/BaoIAD"><img src="https://img.shields.io/badge/NeurIPS%202026-Under%20E%26D%20Review-red" alt="NeurIPS 2026 under review"></a>
</p>

<p align="center">
  <a href="README.md">English</a> | 简体中文
</p>

> **📢 BaoIAD 目前正在 NeurIPS 2026 Evaluations and Datasets 审稿中。**

<p align="center">
  <a href="https://baoiad.readthedocs.io/en/latest/">📘 文档</a> |
  <a href="docs/zh_cn/get_started.md">🛠️ 安装</a> |
  <a href="docs/zh_cn/user_guides/train_test.md">🚀 训练与测试</a> |
  <a href="docs/zh_cn/model_zoo.md">📚 模型库</a> |
  <a href="docs/alignment/README.md">🧭 对齐证据</a>
</p>

## 简介

BaoIAD 是一个基于 [MMEngine](https://github.com/open-mmlab/mmengine) 的工业异常检测统一基准与工具箱。仓库提供配置化训练、测试、批量基准流程，并为常用 IAD 方法固定保存仓库内严格对齐证据。

主要特性：

- **配置化方法入口**：每个方法目录都有 `configs/<method>/README.md`，列出可运行配置。
- **按家族组织的模型库**：方法按建模思路分组，方便快速定位相关基线。
- **严格对齐证据**：[`docs/alignment/`](docs/alignment/) 保存每个方法的 reference freeze、实现路径检查、行为 probe 和归档 benchmark stop-line。
- **统一基准工具**：`tools/train.py`、`tools/test.py` 和 `tools/benchmark.py` 使用一致的配置约定。

## 架构图

<p align="center">
  <img src="resources/architecture_zh.png" width="900" alt="BaoIAD 架构概览">
</p>

## 方法家族

BaoIAD 当前保留 37 个方法，按 9 个 family 组织。方法名链接到 config README，`evidence` 链接到仓库内固定的对齐记录。同样的分组总览也保留在 [模型库](docs/zh_cn/model_zoo.md)。

| **自监督合成** | **重建 / ViT** | **判别式方法** |
| --- | --- | --- |
| [GLASS](configs/glass/README.md) (ECCV'2024; [evidence](docs/alignment/glass.md))<br>[DRAEM](configs/draem/README.md) (ICCV'2021; [evidence](docs/alignment/draem.md))<br>[DSR](configs/dsr/README.md) (ECCV'2022; [evidence](docs/alignment/dsr.md))<br>[CutPaste](configs/cutpaste/README.md) (CVPR'2021; [evidence](docs/alignment/cutpaste.md))<br>[NSA](configs/nsa/README.md) (ECCV'2022; [evidence](docs/alignment/nsa.md)) | [Dinomaly](configs/dinomaly/README.md) (CVPR'2025; [evidence](docs/alignment/dinomaly.md))<br>[ViTAD](configs/vitad/README.md) (AAAI'2024; [evidence](docs/alignment/vitad.md))<br>[MemSeg](configs/memseg/README.md) (EAAI'2023; [evidence](docs/alignment/memseg.md))<br>[UniAD](configs/uniad/README.md) (NeurIPS'2022; [evidence](docs/alignment/uniad.md))<br>[GANomaly](configs/ganomaly/README.md) (ACCV'2018; [evidence](docs/alignment/ganomaly.md)) | [SimpleNet](configs/simplenet/README.md) (CVPR'2023; [evidence](docs/alignment/simplenet.md))<br>[SuperSimpleNet](configs/supersimplenet/README.md) (ICPR'2024; [evidence](docs/alignment/supersimplenet.md))<br>[CFA](configs/cfa/README.md) (IEEE Access'2022; [evidence](docs/alignment/cfa.md)) |

| **知识蒸馏** | **混合 / 统一框架** | **归一化流** |
| --- | --- | --- |
| [RD++](configs/rdpp/README.md) (CVPR'2023; [evidence](docs/alignment/rdpp.md))<br>[AST](configs/ast/README.md) (WACV'2023; [evidence](docs/alignment/ast.md))<br>[RD](configs/rd/README.md) (CVPR'2022; [evidence](docs/alignment/rd.md))<br>[EfficientAD](configs/efficientad/README.md) (WACV'2024; [evidence](docs/alignment/efficientad.md))<br>[DeSTSeg](configs/destseg/README.md) (CVPR'2023; [evidence](docs/alignment/destseg.md)) | [UniNet](configs/uninet/README.md) (CVPR'2025; [evidence](docs/alignment/uninet.md)) | [U-Flow](configs/uflow/README.md) (JMIV'2024; [evidence](docs/alignment/uflow.md))<br>[CFlow](configs/cflow/README.md) (WACV'2022; [evidence](docs/alignment/cflow.md))<br>[DifferNet](configs/differnet/README.md) (WACV'2021; [evidence](docs/alignment/differnet.md))<br>[FastFlow](configs/fastflow/README.md) (arXiv'2021; [evidence](docs/alignment/fastflow.md))<br>[PyramidFlow](configs/pyramidflow/README.md) (CVPR'2023; [evidence](docs/alignment/pyramidflow.md)) |

| **特征记忆 / 密度估计** | **视觉语言 / 基础模型** | **少样本 / 配准** |
| --- | --- | --- |
| [PatchCore](configs/patchcore/README.md) (CVPR'2022; [evidence](docs/alignment/patchcore.md))<br>[PaDiM](configs/padim/README.md) (ICPR'2021; [evidence](docs/alignment/padim.md))<br>[DFM](configs/dfm/README.md) (ICPR'2021; [evidence](docs/alignment/dfm.md))<br>[DFKDE](configs/dfkde/README.md) (Anomalib / ICIP'2022; [evidence](docs/alignment/dfkde.md)) | [MuSc](configs/musc/README.md) (ICLR'2024; [evidence](docs/alignment/musc.md))<br>[AACLIP](configs/aaclip/README.md) (CVPR'2025; [evidence](docs/alignment/aaclip.md))<br>[AnoVL](configs/anovl/README.md) (arXiv'2023; [evidence](docs/alignment/anovl.md))<br>[AnomalyCLIP](configs/anomalyclip/README.md) (ICLR'2024; [evidence](docs/alignment/anomalyclip.md))<br>[WinCLIP](configs/winclip/README.md) (CVPR'2023; [evidence](docs/alignment/winclip.md))<br>[AdaCLIP](configs/adaclip/README.md) (ECCV'2024; [evidence](docs/alignment/adaclip.md))<br>[SAA+](configs/saaplus/README.md) (arXiv'2023; [evidence](docs/alignment/saaplus.md)) | [AnomalyDINO](configs/anomalydino/README.md) (WACV'2025; [evidence](docs/alignment/anomalydino.md))<br>[RegAD](configs/regad/README.md) (ECCV'2022; [evidence](docs/alignment/regad.md)) |

## 支持的数据集

| 数据集 | 类别数 | 配置 |
|---------|------:|--------|
| MVTec AD | 15 | `configs/_base_/datasets/mvtec_ad.py` |
| VisA | 12 | `configs/_base_/datasets/visa.py` |
| BTech | 3 | `configs/_base_/datasets/btech.py` |
| MVTec 3D AD | 10 | `configs/_base_/datasets/mvtec_3d_ad.py` |
| MVTec LOCO | 5 | `configs/_base_/datasets/mvtec_loco_ad.py` |
| MPDD | 6 | `configs/_base_/datasets/mpdd.py` |
| MVTec AD 2 | 16 | `configs/_base_/datasets/mvtec_ad2.py` |
| Kolektor | 3 | `configs/_base_/datasets/kolektor.py` |
| VAD | 6 | `configs/_base_/datasets/vad.py` |
| RealIAD | — | `configs/_base_/datasets/realiad.py` |

## 评价指标

- **图像级**：AUROC、F1-max、AP、ECE、FPR@95TPR
- **像素级**：AUROC、F1-max、AP、AUPRO、AUPIMO、ECE

## 安装

详细步骤见 [安装文档](docs/zh_cn/get_started.md)。

```bash
# 创建环境
conda create -n baoiad python=3.10 -y && conda activate baoiad

# 安装 BaoIAD
git clone https://github.com/ChenjieXu/BaoIAD.git
cd BaoIAD
pip install -e .

# 可选依赖组
pip install -e ".[flow]"    # 归一化流方法
pip install -e ".[vl]"      # 视觉语言方法
pip install -e ".[all]"     # 常用可选依赖
```

设置数据路径：

```bash
source tools/env.sh
# 或手动设置：export BAOIAD_DATA_ROOT=/path/to/data
```

## 快速开始

```bash
# 在 MVTec AD 上训练 PatchCore
python tools/train.py configs/patchcore/patchcore_wrn50_256_mvtec_strict.py --work-dir runs/patchcore

# 训练单个类别
python tools/train.py configs/patchcore/patchcore_wrn50_256_mvtec_strict.py \
    --work-dir runs/patchcore_bottle \
    --cfg-options train_dataloader.dataset.cls_names="['bottle']" train_dataloader.dataset.multi_class=False

# 使用 checkpoint 测试
python tools/test.py configs/patchcore/patchcore_wrn50_256_mvtec_strict.py runs/patchcore/best.pth

# 基准评测多个方法
python tools/benchmark.py --data_root data/mvtec_ad --methods patchcore rd --categories all \
    --output runs/benchmark_results.json
```

## 可视化

BaoIAD 内置异常检测结果可视化功能，测试时启用即可：

```bash
python tools/test.py <config> <checkpoint> \
    --cfg-options default_hooks.visualization.enable=True
```

<div align="center">
<img src="resources/vis_examples/anomaly_detection_results.png" width="100%" alt="异常检测结果可视化">

*示例：PatchCore、PaDiM 和 SPADE (WideResNet-50) 在 MVTec AD 数据集上的检测结果。*
</div>

## 文档导航

- [英文文档](docs/en/index.rst)
- [中文文档](docs/zh_cn/index.rst)
- [模型库](docs/zh_cn/model_zoo.md)
- [对齐证据](docs/alignment/README.md)

## 校验

```bash
python tools/check_method_inventory.py
python tools/benchmark.py --methods all --help
```

`python tools/benchmark.py --methods all` 会解析基准工具使用的 37 方法仓库清单。

## 贡献

欢迎提交 issue、pull request 和可复现性修复。请在提交时说明改动内容以及已经运行的验证。

## 引用

如果您在研究中使用了本工具箱或基准评测，请引用本 GitHub 仓库。

```bibtex
@misc{xu2026baoiad,
  title        = {BaoIAD: Towards Trustworthy and Reproducible Benchmarking for Industrial Anomaly Detection},
  author       = {Chenjie Xu},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/ChenjieXu/BaoIAD}
}
```

## 许可证

本项目基于 [Apache 2.0 许可证](LICENSE) 发布。

## 致谢

BaoIAD 基于 [MMEngine](https://github.com/open-mmlab/mmengine) 和 [MMCV](https://github.com/open-mmlab/mmcv) 构建。感谢 OpenMMLab 社区提供基础训练设施，也感谢所有已实现异常检测方法的原始作者公开代码和协议。
