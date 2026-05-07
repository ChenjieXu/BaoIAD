# Model Zoo

BaoIAD exposes 37 repo-local method entries grouped into 9 families. Each method name links to its config README, and each `evidence` link points to the repository-local alignment record. Venue labels follow the current BaoIAD paper bibliography.

## Family Overview

| **Self-supervised synthesis** | **Reconstruction / ViT** | **Discriminative** |
| --- | --- | --- |
| [GLASS](../../configs/glass/README.md) (ECCV'2024; [evidence](../alignment/glass.md))<br>[DRAEM](../../configs/draem/README.md) (ICCV'2021; [evidence](../alignment/draem.md))<br>[DSR](../../configs/dsr/README.md) (ECCV'2022; [evidence](../alignment/dsr.md))<br>[CutPaste](../../configs/cutpaste/README.md) (CVPR'2021; [evidence](../alignment/cutpaste.md))<br>[NSA](../../configs/nsa/README.md) (ECCV'2022; [evidence](../alignment/nsa.md)) | [Dinomaly](../../configs/dinomaly/README.md) (CVPR'2025; [evidence](../alignment/dinomaly.md))<br>[ViTAD](../../configs/vitad/README.md) (AAAI'2024; [evidence](../alignment/vitad.md))<br>[MemSeg](../../configs/memseg/README.md) (EAAI'2023; [evidence](../alignment/memseg.md))<br>[UniAD](../../configs/uniad/README.md) (NeurIPS'2022; [evidence](../alignment/uniad.md))<br>[GANomaly](../../configs/ganomaly/README.md) (ACCV'2018; [evidence](../alignment/ganomaly.md)) | [SimpleNet](../../configs/simplenet/README.md) (CVPR'2023; [evidence](../alignment/simplenet.md))<br>[SuperSimpleNet](../../configs/supersimplenet/README.md) (ICPR'2024; [evidence](../alignment/supersimplenet.md))<br>[CFA](../../configs/cfa/README.md) (IEEE Access'2022; [evidence](../alignment/cfa.md)) |

| **Knowledge distillation** | **Hybrid / unified** | **Normalizing flow** |
| --- | --- | --- |
| [RD++](../../configs/rdpp/README.md) (CVPR'2023; [evidence](../alignment/rdpp.md))<br>[AST](../../configs/ast/README.md) (WACV'2023; [evidence](../alignment/ast.md))<br>[RD](../../configs/rd/README.md) (CVPR'2022; [evidence](../alignment/rd.md))<br>[EfficientAD](../../configs/efficientad/README.md) (WACV'2024; [evidence](../alignment/efficientad.md))<br>[DeSTSeg](../../configs/destseg/README.md) (CVPR'2023; [evidence](../alignment/destseg.md)) | [UniNet](../../configs/uninet/README.md) (CVPR'2025; [evidence](../alignment/uninet.md)) | [U-Flow](../../configs/uflow/README.md) (JMIV'2024; [evidence](../alignment/uflow.md))<br>[CFlow](../../configs/cflow/README.md) (WACV'2022; [evidence](../alignment/cflow.md))<br>[DifferNet](../../configs/differnet/README.md) (WACV'2021; [evidence](../alignment/differnet.md))<br>[FastFlow](../../configs/fastflow/README.md) (arXiv'2021; [evidence](../alignment/fastflow.md))<br>[PyramidFlow](../../configs/pyramidflow/README.md) (CVPR'2023; [evidence](../alignment/pyramidflow.md)) |

| **Feature-memory / density** | **Vision-language / foundation** | **Few-shot / registration** |
| --- | --- | --- |
| [PatchCore](../../configs/patchcore/README.md) (CVPR'2022; [evidence](../alignment/patchcore.md))<br>[PaDiM](../../configs/padim/README.md) (ICPR'2021; [evidence](../alignment/padim.md))<br>[DFM](../../configs/dfm/README.md) (ICPR'2021; [evidence](../alignment/dfm.md))<br>[DFKDE](../../configs/dfkde/README.md) (Anomalib / ICIP'2022; [evidence](../alignment/dfkde.md)) | [MuSc](../../configs/musc/README.md) (ICLR'2024; [evidence](../alignment/musc.md))<br>[AACLIP](../../configs/aaclip/README.md) (CVPR'2025; [evidence](../alignment/aaclip.md))<br>[AnoVL](../../configs/anovl/README.md) (arXiv'2023; [evidence](../alignment/anovl.md))<br>[AnomalyCLIP](../../configs/anomalyclip/README.md) (ICLR'2024; [evidence](../alignment/anomalyclip.md))<br>[WinCLIP](../../configs/winclip/README.md) (CVPR'2023; [evidence](../alignment/winclip.md))<br>[AdaCLIP](../../configs/adaclip/README.md) (ECCV'2024; [evidence](../alignment/adaclip.md))<br>[SAA+](../../configs/saaplus/README.md) (arXiv'2023; [evidence](../alignment/saaplus.md)) | [AnomalyDINO](../../configs/anomalydino/README.md) (WACV'2025; [evidence](../alignment/anomalydino.md))<br>[RegAD](../../configs/regad/README.md) (ECCV'2022; [evidence](../alignment/regad.md)) |

## Recommended Configs

The canonical configs for each method, sourced from [`baoiad/method_inventory.py`](../../baoiad/method_inventory.py). The "MVTec config" column lists the config used for MVTec AD benchmarking; "VisA config" lists the VisA equivalent.

### Self-supervised Synthesis

| Slug | Display | MVTec Config | VisA Config | README | Evidence |
|------|---------|-------------|-------------|--------|----------|
| `glass` | GLASS | [`configs/glass/glass_wrn50_288_mvtec_strict.py`](../../configs/glass/glass_wrn50_288_mvtec_strict.py) | [`configs/glass/glass_wrn50_288_visa.py`](../../configs/glass/glass_wrn50_288_visa.py) | [README](../../configs/glass/README.md) | [alignment](../alignment/glass.md) |
| `draem` | DRAEM | [`configs/draem/draem_256_mvtec_strict.py`](../../configs/draem/draem_256_mvtec_strict.py) | [`configs/draem/draem_256_visa.py`](../../configs/draem/draem_256_visa.py) | [README](../../configs/draem/README.md) | [alignment](../alignment/draem.md) |
| `dsr` | DSR | [`configs/dsr/dsr_256_mvtec_strict.py`](../../configs/dsr/dsr_256_mvtec_strict.py) | [`configs/dsr/dsr_256_visa.py`](../../configs/dsr/dsr_256_visa.py) | [README](../../configs/dsr/README.md) | [alignment](../alignment/dsr.md) |
| `cutpaste` | CutPaste | [`configs/cutpaste/cutpaste_rn18_256_mvtec_strict.py`](../../configs/cutpaste/cutpaste_rn18_256_mvtec_strict.py) | [`configs/cutpaste/cutpaste_rn18_256_visa.py`](../../configs/cutpaste/cutpaste_rn18_256_visa.py) | [README](../../configs/cutpaste/README.md) | [alignment](../alignment/cutpaste.md) |
| `nsa` | NSA | [`configs/nsa/nsa_rn18_256_mvtec_strict.py`](../../configs/nsa/nsa_rn18_256_mvtec_strict.py) | [`configs/nsa/nsa_rn18_256_visa.py`](../../configs/nsa/nsa_rn18_256_visa.py) | [README](../../configs/nsa/README.md) | [alignment](../alignment/nsa.md) |

### Reconstruction / ViT

| Slug | Display | MVTec Config | VisA Config | README | Evidence |
|------|---------|-------------|-------------|--------|----------|
| `dinomaly` | Dinomaly | [`configs/dinomaly/dinomaly_392_mvtec_strict.py`](../../configs/dinomaly/dinomaly_392_mvtec_strict.py) | [`configs/dinomaly/dinomaly_392_visa.py`](../../configs/dinomaly/dinomaly_392_visa.py) | [README](../../configs/dinomaly/README.md) | [alignment](../alignment/dinomaly.md) |
| `vitad` | ViTAD | [`configs/vitad/vitad_256_mvtec_strict.py`](../../configs/vitad/vitad_256_mvtec_strict.py) | [`configs/vitad/vitad_256_visa.py`](../../configs/vitad/vitad_256_visa.py) | [README](../../configs/vitad/README.md) | [alignment](../alignment/vitad.md) |
| `memseg` | MemSeg | [`configs/memseg/memseg_rn18_256_mvtec_strict.py`](../../configs/memseg/memseg_rn18_256_mvtec_strict.py) | [`configs/memseg/memseg_rn18_256_visa.py`](../../configs/memseg/memseg_rn18_256_visa.py) | [README](../../configs/memseg/README.md) | [alignment](../alignment/memseg.md) |
| `uniad` | UniAD | [`configs/uniad/uniad_wrn50_256_mvtec_strict.py`](../../configs/uniad/uniad_wrn50_256_mvtec_strict.py) | [`configs/uniad/uniad_wrn50_256_visa.py`](../../configs/uniad/uniad_wrn50_256_visa.py) | [README](../../configs/uniad/README.md) | [alignment](../alignment/uniad.md) |
| `ganomaly` | GANomaly | [`configs/ganomaly/ganomaly_256_mvtec_strict.py`](../../configs/ganomaly/ganomaly_256_mvtec_strict.py) | [`configs/ganomaly/ganomaly_256_visa.py`](../../configs/ganomaly/ganomaly_256_visa.py) | [README](../../configs/ganomaly/README.md) | [alignment](../alignment/ganomaly.md) |

### Discriminative

| Slug | Display | MVTec Config | VisA Config | README | Evidence |
|------|---------|-------------|-------------|--------|----------|
| `simplenet` | SimpleNet | [`configs/simplenet/simplenet_wrn50_288_mvtec_strict.py`](../../configs/simplenet/simplenet_wrn50_288_mvtec_strict.py) | [`configs/simplenet/simplenet_wrn50_288_visa.py`](../../configs/simplenet/simplenet_wrn50_288_visa.py) | [README](../../configs/simplenet/README.md) | [alignment](../alignment/simplenet.md) |
| `supersimplenet` | SuperSimpleNet | [`configs/supersimplenet/supersimplenet_256_mvtec_strict.py`](../../configs/supersimplenet/supersimplenet_256_mvtec_strict.py) | [`configs/supersimplenet/supersimplenet_256_visa.py`](../../configs/supersimplenet/supersimplenet_256_visa.py) | [README](../../configs/supersimplenet/README.md) | [alignment](../alignment/supersimplenet.md) |
| `cfa` | CFA | [`configs/cfa/cfa_256_mvtec_strict.py`](../../configs/cfa/cfa_256_mvtec_strict.py) | [`configs/cfa/cfa_256_visa.py`](../../configs/cfa/cfa_256_visa.py) | [README](../../configs/cfa/README.md) | [alignment](../alignment/cfa.md) |

### Knowledge Distillation

| Slug | Display | MVTec Config | VisA Config | README | Evidence |
|------|---------|-------------|-------------|--------|----------|
| `rdpp` | RD++ | [`configs/rdpp/rdpp_wrn50_256_mvtec_strict.py`](../../configs/rdpp/rdpp_wrn50_256_mvtec_strict.py) | [`configs/rdpp/rdpp_wrn50_256_visa.py`](../../configs/rdpp/rdpp_wrn50_256_visa.py) | [README](../../configs/rdpp/README.md) | [alignment](../alignment/rdpp.md) |
| `ast` | AST | [`configs/ast/ast_effnet_b5_768_mvtec_strict.py`](../../configs/ast/ast_effnet_b5_768_mvtec_strict.py) | [`configs/ast/ast_effnet_b5_768_visa.py`](../../configs/ast/ast_effnet_b5_768_visa.py) | [README](../../configs/ast/README.md) | [alignment](../alignment/ast.md) |
| `rd` | RD | [`configs/rd/rd_wrn50_256_mvtec_strict.py`](../../configs/rd/rd_wrn50_256_mvtec_strict.py) | [`configs/rd/rd_wrn50_256_visa.py`](../../configs/rd/rd_wrn50_256_visa.py) | [README](../../configs/rd/README.md) | [alignment](../alignment/rd.md) |
| `efficientad` | EfficientAD | [`configs/efficientad/efficientad_256_mvtec_strict.py`](../../configs/efficientad/efficientad_256_mvtec_strict.py) | [`configs/efficientad/efficientad_256_visa.py`](../../configs/efficientad/efficientad_256_visa.py) | [README](../../configs/efficientad/README.md) | [alignment](../alignment/efficientad.md) |
| `destseg` | DeSTSeg | [`configs/destseg/destseg_rn18_256_mvtec_strict.py`](../../configs/destseg/destseg_rn18_256_mvtec_strict.py) | [`configs/destseg/destseg_rn18_256_visa.py`](../../configs/destseg/destseg_rn18_256_visa.py) | [README](../../configs/destseg/README.md) | [alignment](../alignment/destseg.md) |

### Hybrid / Unified

| Slug | Display | MVTec Config | VisA Config | README | Evidence |
|------|---------|-------------|-------------|--------|----------|
| `uninet` | UniNet | [`configs/uninet/uninet_256_mvtec_strict.py`](../../configs/uninet/uninet_256_mvtec_strict.py) | [`configs/uninet/uninet_256_visa.py`](../../configs/uninet/uninet_256_visa.py) | [README](../../configs/uninet/README.md) | [alignment](../alignment/uninet.md) |

### Normalizing Flow

| Slug | Display | MVTec Config | VisA Config | README | Evidence |
|------|---------|-------------|-------------|--------|----------|
| `uflow` | U-Flow | [`configs/uflow/uflow_mcait_448_mvtec_strict.py`](../../configs/uflow/uflow_mcait_448_mvtec_strict.py) | [`configs/uflow/uflow_mcait_448_visa.py`](../../configs/uflow/uflow_mcait_448_visa.py) | [README](../../configs/uflow/README.md) | [alignment](../alignment/uflow.md) |
| `cflow` | CFlow | [`configs/cflow/cflow_mvtec_strict.py`](../../configs/cflow/cflow_mvtec_strict.py) | [`configs/cflow/cflow_visa.py`](../../configs/cflow/cflow_visa.py) | [README](../../configs/cflow/README.md) | [alignment](../alignment/cflow.md) |
| `differnet` | DifferNet | [`configs/differnet/differnet_alexnet_256_mvtec_strict.py`](../../configs/differnet/differnet_alexnet_256_mvtec_strict.py) | [`configs/differnet/differnet_alexnet_256_visa.py`](../../configs/differnet/differnet_alexnet_256_visa.py) | [README](../../configs/differnet/README.md) | [alignment](../alignment/differnet.md) |
| `fastflow` | FastFlow | [`configs/fastflow/fastflow_wrn50_256_mvtec_strict.py`](../../configs/fastflow/fastflow_wrn50_256_mvtec_strict.py) | [`configs/fastflow/fastflow_wrn50_256_visa.py`](../../configs/fastflow/fastflow_wrn50_256_visa.py) | [README](../../configs/fastflow/README.md) | [alignment](../alignment/fastflow.md) |
| `pyramidflow` | PyramidFlow | [`configs/pyramidflow/pyramidflow_fnf_256_mvtec_strict.py`](../../configs/pyramidflow/pyramidflow_fnf_256_mvtec_strict.py) | [`configs/pyramidflow/pyramidflow_resnet18_1024_visa.py`](../../configs/pyramidflow/pyramidflow_resnet18_1024_visa.py) | [README](../../configs/pyramidflow/README.md) | [alignment](../alignment/pyramidflow.md) |

### Feature-Memory / Density

| Slug | Display | MVTec Config | VisA Config | README | Evidence |
|------|---------|-------------|-------------|--------|----------|
| `patchcore` | PatchCore | [`configs/patchcore/patchcore_wrn50_256_mvtec_strict.py`](../../configs/patchcore/patchcore_wrn50_256_mvtec_strict.py) | [`configs/patchcore/patchcore_wrn50_256_visa.py`](../../configs/patchcore/patchcore_wrn50_256_visa.py) | [README](../../configs/patchcore/README.md) | [alignment](../alignment/patchcore.md) |
| `padim` | PaDiM | [`configs/padim/padim_wrn50_256_mvtec_strict.py`](../../configs/padim/padim_wrn50_256_mvtec_strict.py) | [`configs/padim/padim_wrn50_256_visa.py`](../../configs/padim/padim_wrn50_256_visa.py) | [README](../../configs/padim/README.md) | [alignment](../alignment/padim.md) |
| `dfm` | DFM | [`configs/dfm/dfm_256_mvtec_strict.py`](../../configs/dfm/dfm_256_mvtec_strict.py) | [`configs/dfm/dfm_256_visa.py`](../../configs/dfm/dfm_256_visa.py) | [README](../../configs/dfm/README.md) | [alignment](../alignment/dfm.md) |
| `dfkde` | DFKDE | [`configs/dfkde/dfkde_256_mvtec_strict.py`](../../configs/dfkde/dfkde_256_mvtec_strict.py) | [`configs/dfkde/dfkde_256_visa.py`](../../configs/dfkde/dfkde_256_visa.py) | [README](../../configs/dfkde/README.md) | [alignment](../alignment/dfkde.md) |

### Vision-Language / Foundation

| Slug | Display | MVTec Config | VisA Config | README | Evidence |
|------|---------|-------------|-------------|--------|----------|
| `musc` | MuSc | [`configs/musc/musc_vitl14_336_518_mvtec_strict.py`](../../configs/musc/musc_vitl14_336_518_mvtec_strict.py) | [`configs/musc/musc_vitl14_336_518_visa.py`](../../configs/musc/musc_vitl14_336_518_visa.py) | [README](../../configs/musc/README.md) | [alignment](../alignment/musc.md) |
| `aaclip` | AACLIP | [`configs/aaclip/aaclip_vitl14_336_518_mvtec_strict.py`](../../configs/aaclip/aaclip_vitl14_336_518_mvtec_strict.py) | [`configs/aaclip/aaclip_vitl14_336_518_visa_fullshot_stage1.py`](../../configs/aaclip/aaclip_vitl14_336_518_visa_fullshot_stage1.py) + [`stage2`](../../configs/aaclip/aaclip_vitl14_336_518_visa_fullshot_stage2.py) | [README](../../configs/aaclip/README.md) | [alignment](../alignment/aaclip.md) |
| `anovl` | AnoVL | [`configs/anovl/anovl_vitb16plus_240_mvtec_strict.py`](../../configs/anovl/anovl_vitb16plus_240_mvtec_strict.py) | [`configs/anovl/anovl_vitb16plus_240_visa.py`](../../configs/anovl/anovl_vitb16plus_240_visa.py) | [README](../../configs/anovl/README.md) | [alignment](../alignment/anovl.md) |
| `anomalyclip` | AnomalyCLIP | [`configs/anomalyclip/anomalyclip_vitl14_336_518_mvtec_strict.py`](../../configs/anomalyclip/anomalyclip_vitl14_336_518_mvtec_strict.py) | [`configs/anomalyclip/anomalyclip_vitl14_336_518_visa.py`](../../configs/anomalyclip/anomalyclip_vitl14_336_518_visa.py) | [README](../../configs/anomalyclip/README.md) | [alignment](../alignment/anomalyclip.md) |
| `winclip` | WinCLIP | [`configs/winclip/winclip_256_mvtec.py`](../../configs/winclip/winclip_256_mvtec.py) | [`configs/winclip/winclip_256_visa.py`](../../configs/winclip/winclip_256_visa.py) | [README](../../configs/winclip/README.md) | [alignment](../alignment/winclip.md) |
| `adaclip` | AdaCLIP | [`configs/adaclip/adaclip_vitl14_336_518_mvtec_strict.py`](../../configs/adaclip/adaclip_vitl14_336_518_mvtec_strict.py) | [`configs/adaclip/adaclip_vitl14_336_518_visa.py`](../../configs/adaclip/adaclip_vitl14_336_518_visa.py) | [README](../../configs/adaclip/README.md) | [alignment](../alignment/adaclip.md) |
| `saaplus` | SAA+ | [`configs/saaplus/saaplus_400_mvtec_strict.py`](../../configs/saaplus/saaplus_400_mvtec_strict.py) | [`configs/saaplus/saaplus_400_visa.py`](../../configs/saaplus/saaplus_400_visa.py) | [README](../../configs/saaplus/README.md) | [alignment](../alignment/saaplus.md) |

### Few-Shot / Registration

| Slug | Display | MVTec Config | VisA Config | README | Evidence |
|------|---------|-------------|-------------|--------|----------|
| `anomalydino` | AnomalyDINO | [`configs/anomalydino/anomalydino_vitb14_448_mvtec_strict.py`](../../configs/anomalydino/anomalydino_vitb14_448_mvtec_strict.py) | [`configs/anomalydino/anomalydino_vitb14_448_visa.py`](../../configs/anomalydino/anomalydino_vitb14_448_visa.py) | [README](../../configs/anomalydino/README.md) | [alignment](../alignment/anomalydino.md) |
| `regad` | RegAD | [`configs/regad/regad_wrn50_256_mvtec_strict.py`](../../configs/regad/regad_wrn50_256_mvtec_strict.py) | [`configs/regad/regad_wrn50_256_visa.py`](../../configs/regad/regad_wrn50_256_visa.py) | [README](../../configs/regad/README.md) | [alignment](../alignment/regad.md) |

## Method Caveats

Some methods have special requirements or quirks. See the [Method Caveats](user_guides/method_caveats.md) page for details on:
- Optional dependencies (FrEIA, open_clip, faiss, geomloss, imgaug)
- Special training scripts (AST, RegAD, ViTAD)
- Memory bank lifecycle
- Known quirks (NSA category-specific epochs, CutPaste RepeatDataset wrapper, etc.)
