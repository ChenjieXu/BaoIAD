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
  English | <a href="README_zh-CN.md">简体中文</a>
</p>

> **📢 BaoIAD is currently under NeurIPS 2026 Evaluations and Datasets review.**

<p align="center">
  <a href="https://baoiad.readthedocs.io/en/latest/">📘 Documentation</a> |
  <a href="docs/en/get_started.md">🛠️ Installation</a> |
  <a href="docs/en/user_guides/train_test.md">🚀 Train & Test</a> |
  <a href="docs/en/model_zoo.md">📚 Model Zoo</a> |
  <a href="docs/alignment/README.md">🧭 Alignment Evidence</a>
</p>

## Introduction

BaoIAD is a unified industrial anomaly detection (IAD) benchmark and toolbox built on [MMEngine](https://github.com/open-mmlab/mmengine). It provides config-driven training, testing, benchmarking, and repository-local strict-alignment evidence for commonly used IAD methods.

Major features:

- **Config-driven methods**: each method has a dedicated `configs/<method>/README.md` with runnable config entry points.
- **Grouped model zoo**: methods are organized by modeling family so users can find related baselines quickly.
- **Strict-alignment evidence**: [`docs/alignment/`](docs/alignment/) preserves per-method reference freezes, implementation checks, behavior probes, and archived benchmark stop-lines.
- **Unified benchmark tooling**: `tools/train.py`, `tools/test.py`, and `tools/benchmark.py` share the same config conventions.

## Architecture

<p align="center">
  <img src="resources/architecture.png" width="900" alt="BaoIAD architecture overview">
</p>

## Method Families

BaoIAD keeps 37 methods in 9 families. Method names link to config READMEs, and `evidence` links point to repository-local alignment records. The same grouped overview is also available in the [Model Zoo](docs/en/model_zoo.md).

| **Self-supervised synthesis** | **Reconstruction / ViT** | **Discriminative** |
| --- | --- | --- |
| [GLASS](configs/glass/README.md) (ECCV'2024; [evidence](docs/alignment/glass.md))<br>[DRAEM](configs/draem/README.md) (ICCV'2021; [evidence](docs/alignment/draem.md))<br>[DSR](configs/dsr/README.md) (ECCV'2022; [evidence](docs/alignment/dsr.md))<br>[CutPaste](configs/cutpaste/README.md) (CVPR'2021; [evidence](docs/alignment/cutpaste.md))<br>[NSA](configs/nsa/README.md) (ECCV'2022; [evidence](docs/alignment/nsa.md)) | [Dinomaly](configs/dinomaly/README.md) (CVPR'2025; [evidence](docs/alignment/dinomaly.md))<br>[ViTAD](configs/vitad/README.md) (AAAI'2024; [evidence](docs/alignment/vitad.md))<br>[MemSeg](configs/memseg/README.md) (EAAI'2023; [evidence](docs/alignment/memseg.md))<br>[UniAD](configs/uniad/README.md) (NeurIPS'2022; [evidence](docs/alignment/uniad.md))<br>[GANomaly](configs/ganomaly/README.md) (ACCV'2018; [evidence](docs/alignment/ganomaly.md)) | [SimpleNet](configs/simplenet/README.md) (CVPR'2023; [evidence](docs/alignment/simplenet.md))<br>[SuperSimpleNet](configs/supersimplenet/README.md) (ICPR'2024; [evidence](docs/alignment/supersimplenet.md))<br>[CFA](configs/cfa/README.md) (IEEE Access'2022; [evidence](docs/alignment/cfa.md)) |

| **Knowledge distillation** | **Hybrid / unified** | **Normalizing flow** |
| --- | --- | --- |
| [RD++](configs/rdpp/README.md) (CVPR'2023; [evidence](docs/alignment/rdpp.md))<br>[AST](configs/ast/README.md) (WACV'2023; [evidence](docs/alignment/ast.md))<br>[RD](configs/rd/README.md) (CVPR'2022; [evidence](docs/alignment/rd.md))<br>[EfficientAD](configs/efficientad/README.md) (WACV'2024; [evidence](docs/alignment/efficientad.md))<br>[DeSTSeg](configs/destseg/README.md) (CVPR'2023; [evidence](docs/alignment/destseg.md)) | [UniNet](configs/uninet/README.md) (CVPR'2025; [evidence](docs/alignment/uninet.md)) | [U-Flow](configs/uflow/README.md) (JMIV'2024; [evidence](docs/alignment/uflow.md))<br>[CFlow](configs/cflow/README.md) (WACV'2022; [evidence](docs/alignment/cflow.md))<br>[DifferNet](configs/differnet/README.md) (WACV'2021; [evidence](docs/alignment/differnet.md))<br>[FastFlow](configs/fastflow/README.md) (arXiv'2021; [evidence](docs/alignment/fastflow.md))<br>[PyramidFlow](configs/pyramidflow/README.md) (CVPR'2023; [evidence](docs/alignment/pyramidflow.md)) |

| **Feature-memory / density** | **Vision-language / foundation** | **Few-shot / registration** |
| --- | --- | --- |
| [PatchCore](configs/patchcore/README.md) (CVPR'2022; [evidence](docs/alignment/patchcore.md))<br>[PaDiM](configs/padim/README.md) (ICPR'2021; [evidence](docs/alignment/padim.md))<br>[DFM](configs/dfm/README.md) (ICPR'2021; [evidence](docs/alignment/dfm.md))<br>[DFKDE](configs/dfkde/README.md) (Anomalib / ICIP'2022; [evidence](docs/alignment/dfkde.md)) | [MuSc](configs/musc/README.md) (ICLR'2024; [evidence](docs/alignment/musc.md))<br>[AACLIP](configs/aaclip/README.md) (CVPR'2025; [evidence](docs/alignment/aaclip.md))<br>[AnoVL](configs/anovl/README.md) (arXiv'2023; [evidence](docs/alignment/anovl.md))<br>[AnomalyCLIP](configs/anomalyclip/README.md) (ICLR'2024; [evidence](docs/alignment/anomalyclip.md))<br>[WinCLIP](configs/winclip/README.md) (CVPR'2023; [evidence](docs/alignment/winclip.md))<br>[AdaCLIP](configs/adaclip/README.md) (ECCV'2024; [evidence](docs/alignment/adaclip.md))<br>[SAA+](configs/saaplus/README.md) (arXiv'2023; [evidence](docs/alignment/saaplus.md)) | [AnomalyDINO](configs/anomalydino/README.md) (WACV'2025; [evidence](docs/alignment/anomalydino.md))<br>[RegAD](configs/regad/README.md) (ECCV'2022; [evidence](docs/alignment/regad.md)) |

## Supported Datasets

| Dataset | Categories | Config |
|---------|-----------:|--------|
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

## Evaluation Metrics

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

# Optional dependency groups
pip install -e ".[flow]"    # Normalizing-flow methods
pip install -e ".[vl]"      # Vision-language methods
pip install -e ".[all]"     # Common optional method dependencies
```

Set up the data path:

```bash
source tools/env.sh
# Or: export BAOIAD_DATA_ROOT=/path/to/data
```

## Getting Started

```bash
# Train PatchCore on MVTec AD
python tools/train.py configs/patchcore/patchcore_wrn50_256_mvtec_strict.py --work-dir runs/patchcore

# Train a single category
python tools/train.py configs/patchcore/patchcore_wrn50_256_mvtec_strict.py \
    --work-dir runs/patchcore_bottle \
    --cfg-options train_dataloader.dataset.cls_names="['bottle']" train_dataloader.dataset.multi_class=False

# Test with a checkpoint
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

*Examples: PatchCore, PaDiM, and SPADE (WideResNet-50) on MVTec AD.*
</div>

## Documentation Map

- [English docs](docs/en/index.rst)
- [Chinese docs](docs/zh_cn/index.rst)
- [Model zoo](docs/en/model_zoo.md)
- [Alignment evidence](docs/alignment/README.md)

## Validation

```bash
python tools/check_method_inventory.py
python tools/benchmark.py --methods all --help
```

`python tools/benchmark.py --methods all` resolves the 37-method repository inventory used by the benchmark helper.

## Contributing

Contributions, issue reports, and reproducibility fixes are welcome. Please open an issue or pull request with a clear description of the change and the validation you ran.

## Citation

If you use this toolbox or benchmark in your research, please cite this GitHub repository.

```bibtex
@misc{xu2026baoiad,
  title        = {BaoIAD: Towards Trustworthy and Reproducible Benchmarking for Industrial Anomaly Detection},
  author       = {Chenjie Xu},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/ChenjieXu/BaoIAD}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement

BaoIAD is built on top of [MMEngine](https://github.com/open-mmlab/mmengine) and [MMCV](https://github.com/open-mmlab/mmcv). We thank the OpenMMLab community for providing the foundational training infrastructure. We also thank the original authors of the implemented anomaly-detection methods for making their code and protocols publicly available.
