# Model Zoo

BaoIAD integrates 50+ anomaly detection methods organized by their underlying paradigm.

## Memory Bank Methods

Methods that build a memory bank of normal features and detect anomalies by distance to the bank.

| Method | Config Directory | Backbone | Paper |
|--------|-----------------|----------|-------|
| PatchCore | `configs/patchcore/` | WRN-50-2 | [CVPR 2022](https://arxiv.org/abs/2110.07347) |
| SPADE | `configs/spade/` | WRN-50-2 | [CVPR 2021](https://arxiv.org/abs/2005.02357) |
| PaDiM | `configs/padim/` | WRN-50-2 / ResNet-18 | [ICPR 2021](https://arxiv.org/abs/2011.08785) |
| DFM | `configs/dfm/` | WRN-50-2 | [ICML Workshop 2020](https://arxiv.org/abs/1909.11786) |
| DFKDE | `configs/dfkde/` | WRN-50-2 | [arXiv 2019](https://arxiv.org/abs/1909.11786) |
| RegAD | `configs/regad/` | WRN-50-2 | [ECCV 2022](https://arxiv.org/abs/2203.10450) |
| GraphCore | `configs/graphcore/` | ViG | [CVPR 2024](https://arxiv.org/abs/2401.09441) |

## Knowledge Distillation Methods

Methods that use a frozen teacher network to guide a student network; anomalies are detected from the teacher-student discrepancy.

| Method | Config Directory | Backbone | Paper |
|--------|-----------------|----------|-------|
| RD | `configs/rd/` | WRN-50-2 | [CVPR 2022](https://arxiv.org/abs/2201.01402) |
| RD++ | `configs/rdpp/` | WRN-50-2 | [WACV 2024](https://arxiv.org/abs/2309.14935) |
| STFPM | `configs/stfpm/` | WRN-50-2 / ResNet-18 | [arXiv 2020](https://arxiv.org/abs/2003.02052) |
| EfficientAD | `configs/efficientad/` | PDN | [CVPR 2024](https://arxiv.org/abs/2303.14535) |
| Dinomaly | `configs/dinomaly/` | DINOv2 | [arXiv 2024](https://arxiv.org/abs/2405.14525) |

## Normalizing Flow Methods

Methods that model normal feature distributions with normalizing flows; anomalies are detected by low likelihood.

| Method | Config Directory | Backbone | Paper |
|--------|-----------------|----------|-------|
| CSFlow | `configs/csflow/` | WRN-50-2 | [WACV 2022](https://arxiv.org/abs/2110.10046) |
| FastFlow | `configs/fastflow/` | WRN-50-2 / ResNet-18 | [arXiv 2021](https://arxiv.org/abs/2111.07677) |
| CFlow | `configs/cflow/` | WRN-50-2 | [WACV 2022](https://arxiv.org/abs/2108.03335) |
| UFlow | `configs/uflow/` | WRN-50-2 | [WACV 2023](https://arxiv.org/abs/2211.03415) |
| DifferNet | `configs/differnet/` | WRN-50-2 | [WACV 2021](https://arxiv.org/abs/2008.12577) |
| PyramidFlow | `configs/pyramidflow/` | WRN-50-2 | [ICCV 2023](https://arxiv.org/abs/2308.06348) |

:::{note}
Normalizing flow methods require `FrEIA>=0.2`. Install with `pip install -e ".[flow]"`.
:::

## Reconstruction Methods

Methods that learn to reconstruct normal images; anomalies are detected from high reconstruction error.

| Method | Config Directory | Backbone | Paper |
|--------|-----------------|----------|-------|
| DRAEM | `configs/draem/` | -- | [ICCV 2021](https://arxiv.org/abs/2108.07610) |
| MemSeg | `configs/memseg/` | WRN-50-2 | [NeurIPS 2022](https://arxiv.org/abs/2204.08545) |
| DeSTSeg | `configs/destseg/` | -- | [CVPR 2023](https://arxiv.org/abs/2304.06108) |
| MemAE | `configs/memae/` | -- | [ICCV 2019](https://arxiv.org/abs/1909.07493) |
| FRE | `configs/fre/` | WRN-50-2 | [arXiv 2023](https://arxiv.org/abs/2309.06348) |
| GANomaly | `configs/ganomaly/` | -- | [ACCV 2018](https://arxiv.org/abs/1805.06825) |
| DSR | `configs/dsr/` | -- | [ECCV 2022](https://arxiv.org/abs/2205.14841) |

## Vision-Language Methods

Methods that leverage pretrained vision-language models (e.g., CLIP) for zero-shot or few-shot anomaly detection.

| Method | Config Directory | Backbone | Paper |
|--------|-----------------|----------|-------|
| WinCLIP | `configs/winclip/` | OpenCLIP ViT-L/14 | [CVPR 2023](https://arxiv.org/abs/2303.14814) |
| AnomalyCLIP | `configs/anomalyclip/` | OpenCLIP ViT-L/14 | [ICLR 2024](https://arxiv.org/abs/2308.15958) |
| AnoVL | `configs/anovl/` | OpenCLIP ViT-L/14 | [arXiv 2023](https://arxiv.org/abs/2305.10758) |
| MuSc | `configs/musc/` | OpenCLIP ViT-L/14 | [CVPR 2024](https://arxiv.org/abs/2405.06638) |
| AdaCLIP | `configs/adaclip/` | OpenCLIP ViT-L/14 | [arXiv 2024](https://arxiv.org/abs/2401.01468) |
| AACLIP | `configs/aaclip/` | OpenCLIP ViT-L/14 | [arXiv 2024](https://arxiv.org/abs/2403.06881) |
| AnomalyDINO | `configs/anomalydino/` | DINOv2 ViT-L/14 | [arXiv 2024](https://arxiv.org/abs/2405.14525) |

:::{note}
Vision-language methods require `open_clip_torch`. Install with `pip install -e ".[vl]"`.
:::

## Discriminator Methods

Methods that train feature discriminators to distinguish normal features from perturbed/noisy features.

| Method | Config Directory | Backbone | Paper |
|--------|-----------------|----------|-------|
| SimpleNet | `configs/simplenet/` | WRN-50-2 | [CVPR 2023](https://arxiv.org/abs/2304.09652) |
| SuperSimpleNet | `configs/supersimplenet/` | WRN-50-2 | [CVPR 2024](https://arxiv.org/abs/2404.02400) |
| CFA | `configs/cfa/` | WRN-50-2 | [IEEE Access 2022](https://arxiv.org/abs/2206.04325) |

## Other Methods

| Method | Config Directory | Backbone | Paper |
|--------|-----------------|----------|-------|
| InvAD | `configs/invad/` | WRN-50-2 | [arXiv 2023](https://arxiv.org/abs/2310.18261) |
| ViTAD | `configs/vitad/` | ViT-B/16 | [arXiv 2023](https://arxiv.org/abs/2304.03163) |
| UniAD | `configs/uniad/` | EfficientNet-B4 | [NeurIPS 2022](https://arxiv.org/abs/2206.09610) |
| MambaAD | `configs/mambaad/` | EfficientNet-B4 | [arXiv 2024](https://arxiv.org/abs/2404.06364) |
| NSA | `configs/nsa/` | -- | [ICLR 2022](https://arxiv.org/abs/2109.15222) |
| ResAD | `configs/resad/` | WRN-50-2 | [arXiv 2024](https://arxiv.org/abs/2405.07706) |
| CutPaste | `configs/cutpaste/` | WRN-50-2 | [CVPR 2021](https://arxiv.org/abs/2104.01215) |
| GLASS | `configs/glass/` | WRN-50-2 | [arXiv 2024](https://arxiv.org/abs/2402.12673) |
| AST | `configs/ast/` | EfficientNet-B4/B5 | [WACV 2024](https://arxiv.org/abs/2309.04738) |
| PNI | `configs/pni/` | WRN-50-2 | [arXiv 2023](https://arxiv.org/abs/2308.10255) |
| RealNet | `configs/realnet/` | WRN-50-2 | [arXiv 2024](https://arxiv.org/abs/2401.09863) |
| ComposeAD | `configs/compose_ad/` | WRN-50-2 | [arXiv 2024](https://arxiv.org/abs/2406.03734) |
| UniNet | `configs/uninet/` | WRN-50-2 | [arXiv 2024](https://arxiv.org/abs/2403.02529) |
| UniVAD | `configs/univad/` | ViT-L/14 | [arXiv 2024](https://arxiv.org/abs/2401.02526) |
| SAA+ | `configs/saaplus/` | GroundingDINO + SAM | [arXiv 2023](https://arxiv.org/abs/2305.09888) |

## Benchmark Results

Benchmark results are available in the `runs/alignment/` directory after running benchmarks with `tools/benchmark.py`. See the [Benchmark Guide](user_guides/benchmark.md) for details on reproducing results.
