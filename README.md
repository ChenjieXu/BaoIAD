<div align="center">
  <img src="resources/baoiad-logo.svg" width="65%" alt="BaoIAD Logo"/>

  [![docs](https://img.shields.io/badge/docs-latest-blue)](https://baoiad.readthedocs.io/en/latest/)
  [![license](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
  [![python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
  [![pytorch](https://img.shields.io/badge/pytorch-2.0%2B-orange)](https://pytorch.org/)
  [![methods](https://img.shields.io/badge/methods-50%2B-purple)](docs/en/model_zoo.md)
  [![paper](https://img.shields.io/badge/NeurIPS%202026-Under%20E%26D%20Review-red)](https://github.com/ChenjieXu/BaoIAD)

  English | [简体中文](README_zh-CN.md)

> **📢 BaoIAD is currently under NeurIPS 2026 Evaluations and Datasets Track review.**

  [📘 Documentation](https://baoiad.readthedocs.io/en/latest/) |
  [🛠️ Installation](docs/en/get_started.md) |
  [🚀 Train & Test](docs/en/user_guides/train_test.md) |
  [📊 Benchmark](docs/en/user_guides/benchmark.md) |
  [🔧 Customize Models](docs/en/advanced_guides/customize_models.md)
</div>

<br>

## Introduction

BaoIAD is a unified benchmarking framework for **industrial anomaly detection (IAD)**, built on [MMEngine](https://github.com/open-mmlab/mmengine) (OpenMMLab style). It integrates **50+ anomaly detection methods** under a single config-driven interface for fair comparison across 10+ datasets.

Major features:

- **Modular Design**: We decompose the AD framework into different components (backbone, neck, head) following OpenMMLab conventions. One can easily construct a customized detection framework by combining different modules via config files.
- **50+ Methods Out of the Box**: Memory bank, knowledge distillation, normalizing flow, reconstruction, vision-language, and discriminator-based methods are all supported with reference-aligned implementations.
- **Fair Comparison**: Standardized backbone configurations, consistent evaluation metrics (image-level + pixel-level), and unified data pipelines ensure fair cross-method benchmarking.
- **Config-Driven**: All experiments are controlled through Python config files with inheritance — no code changes needed to swap methods, datasets, or hyperparameters.

## Overview of Benchmark and Model Zoo

| | | | |
| --- | --- | --- | --- |
| **Memory Bank** | **Knowledge Distillation** | **Normalizing Flow** | **Reconstruction** |
| [PatchCore](configs/patchcore/) (CVPR'2022)<br>[SPADE](configs/spade/) (ICPR'2021)<br>[PaDiM](configs/padim/) (ICPR'2021)<br>[DFM](configs/dfm/) (IWCF'2022)<br>[DFKDE](configs/dfkde/) (IWCF'2022)<br>[RegAD](configs/regad/) (ECCV'2022)<br>[GraphCore](configs/graphcore/) (ECCV'2024) | [RD](configs/rd/) (CVPR'2022)<br>[RD++](configs/rdpp/) (ICCV'2023)<br>[STFPM](configs/stfpm/) (WACV'2021)<br>[EfficientAD](configs/efficientad/) (CVPR'2024)<br>[Dinomaly](configs/dinomaly/) (arXiv'2024)<br>[AST](configs/ast/) (ECCV'2022) | [CSFlow](configs/csflow/) (WACV'2022)<br>[FastFlow](configs/fastflow/) (CVPR'2022)<br>[CFlow](configs/cflow/) (WACV'2022)<br>[UFlow](configs/uflow/) (PR'2023)<br>[DifferNet](configs/differnet/) (WACV'2021)<br>[PyramidFlow](configs/pyramidflow/) (CVPR'2023) | [DRAEM](configs/draem/) (ICCV'2021)<br>[MemSeg](configs/memseg/) (NeurIPS'2022)<br>[DeSTSeg](configs/destseg/) (CVPR'2023)<br>[MemAE](configs/memae/) (ICCV'2019)<br>[FRE](configs/fre/) (ICPR'2021)<br>[GANomaly](configs/ganomaly/) (ACCV'2018)<br>[DSR](configs/dsr/) (ECCV'2022) |

| | | | |
| --- | --- | --- | --- |
| **Vision-Language** | **Discriminator** | **Unified Multi-Class** | **Self-Supervised** |
| [WinCLIP](configs/winclip/) (CVPR'2023)<br>[AnomalyCLIP](configs/anomalyclip/) (ICLR'2024)<br>[AnoVL](configs/anovl/) (AAAI'2024)<br>[MuSc](configs/musc/) (CVPR'2024)<br>[AdaCLIP](configs/adaclip/) (arXiv'2024)<br>[AACLIP](configs/aaclip/) (arXiv'2024)<br>[AnomalyDINO](configs/anomalydino/) (arXiv'2024) | [SimpleNet](configs/simplenet/) (CVPR'2023)<br>[SuperSimpleNet](configs/supersimplenet/) (arXiv'2024)<br>[CFA](configs/cfa/) (Access'2022) | [InvAD](configs/invad/) (AAAI'2024)<br>[ViTAD](configs/vitad/) (ECCV'2024)<br>[UniAD](configs/uniad/) (NeurIPS'2022)<br>[MambaAD](configs/mambaad/) (arXiv'2024)<br>[UniNet](configs/uninet/) (arXiv'2024)<br>[UniVAD](configs/univad/) (arXiv'2024) | [NSA](configs/nsa/) (CVPR'2022)<br>[CutPaste](configs/cutpaste/) (CVPR'2021)<br>[GLASS](configs/glass/) (arXiv'2024)<br>[PNI](configs/pni/) (ICLR'2024)<br>[RealNet](configs/realnet/) (CVPR'2024)<br>[ResAD](configs/resad/) (arXiv'2024)<br>[ComposeAD](configs/compose_ad/) |

### Supported Datasets

| Dataset | Categories | Config |
|---------|-----------|--------|
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

### Evaluation Metrics

- **Image-level**: AUROC, F1-max, AP, ECE, FPR@95TPR
- **Pixel-level**: AUROC, F1-max, AP, AUPRO, AUPIMO, ECE

## Installation

Please refer to [Installation](docs/en/get_started.md) for detailed instructions.

```bash
# Create environment
conda create -n baoiad python=3.10 -y && conda activate baoiad

# Install BaoIAD
git clone https://github.com/ChenjieXu/BaoIAD.git
cd BaoIAD
pip install -e .

# With optional dependencies
pip install -e ".[flow]"    # Normalizing flow methods (CSFlow, FastFlow, etc.)
pip install -e ".[vl]"      # Vision-language methods (WinCLIP, AnomalyCLIP, etc.)
pip install -e ".[all]"     # Everything
```

Set up data path:

```bash
source tools/env.sh
# Or: export BAOIAD_DATA_ROOT=/path/to/data
```

## Getting Started

```bash
# Train PatchCore on MVTec AD
python tools/train.py configs/patchcore/patchcore_wrn50_256_mvtec_strict.py --work-dir runs/patchcore

# Train single category
python tools/train.py configs/patchcore/patchcore_wrn50_256_mvtec_strict.py \
    --work-dir runs/patchcore_bottle \
    --cfg-options train_dataloader.dataset.cls_names="['bottle']" train_dataloader.dataset.multi_class=False

# Test with checkpoint
python tools/test.py configs/patchcore/patchcore_wrn50_256_mvtec_strict.py runs/patchcore/best.pth

# Benchmark multiple methods
python tools/benchmark.py --data_root data/mvtec_ad --methods patchcore rd --categories all \
    --output runs/benchmark_results.json
```

## Visualization

BaoIAD provides built-in visualization for anomaly detection results. Enable it during testing:

```bash
python tools/test.py <config> <checkpoint> \
    --cfg-options default_hooks.visualization.enable=True
```

<div align="center">
<img src="resources/vis_examples/anomaly_detection_results.png" width="100%" alt="Anomaly detection visualization">

*Examples: PatchCore, PaDiM, and SPADE (WideResNet-50) on MVTec AD*
</div>

## Contributing

We appreciate all contributions to improve BaoIAD. Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for the contributing guideline.

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```bibtex
@article{xu2026baoiad,
  title={BaoIAD: Towards Trustworthy and Reproducible Benchmarking for Industrial Anomaly Detection},
  author={Chenjie Xu},
  journal={GitHub repository},
  year={2026},
  howpublished={\url{https://github.com/ChenjieXu/BaoIAD}}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement

BaoIAD is built on top of [MMEngine](https://github.com/open-mmlab/mmengine) and [MMCV](https://github.com/open-mmlab/mmcv). We thank the OpenMMLab community for providing the foundational training infrastructure. We also thank all the original authors of the implemented methods for making their code publicly available.
