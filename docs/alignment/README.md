# Alignment records

This directory records repository-local strict-alignment evidence for the 37 BaoIAD methods. Each method page preserves the migrated report and checklist evidence, translated to English where needed, including reference freezes, code-path checks, probes, and archived benchmark stop-lines.

Use each method README for runnable configs plus MVTec AD, VisA, and speed summaries. Use these alignment records when reviewing why a method is considered strictly aligned and which fixed evidence supports that state.

## Methods by family

### Self-supervised synthesis

| Method | Config README | Alignment record |
|---|---|---|
| GLASS | [`configs/glass/README.md`](../../configs/glass/README.md) | [`docs/alignment/glass.md`](glass.md) |
| DRAEM | [`configs/draem/README.md`](../../configs/draem/README.md) | [`docs/alignment/draem.md`](draem.md) |
| DSR | [`configs/dsr/README.md`](../../configs/dsr/README.md) | [`docs/alignment/dsr.md`](dsr.md) |
| CutPaste | [`configs/cutpaste/README.md`](../../configs/cutpaste/README.md) | [`docs/alignment/cutpaste.md`](cutpaste.md) |
| NSA | [`configs/nsa/README.md`](../../configs/nsa/README.md) | [`docs/alignment/nsa.md`](nsa.md) |

### Reconstruction / ViT

| Method | Config README | Alignment record |
|---|---|---|
| Dinomaly | [`configs/dinomaly/README.md`](../../configs/dinomaly/README.md) | [`docs/alignment/dinomaly.md`](dinomaly.md) |
| ViTAD | [`configs/vitad/README.md`](../../configs/vitad/README.md) | [`docs/alignment/vitad.md`](vitad.md) |
| MemSeg | [`configs/memseg/README.md`](../../configs/memseg/README.md) | [`docs/alignment/memseg.md`](memseg.md) |
| UniAD | [`configs/uniad/README.md`](../../configs/uniad/README.md) | [`docs/alignment/uniad.md`](uniad.md) |
| GANomaly | [`configs/ganomaly/README.md`](../../configs/ganomaly/README.md) | [`docs/alignment/ganomaly.md`](ganomaly.md) |

### Discriminative

| Method | Config README | Alignment record |
|---|---|---|
| SimpleNet | [`configs/simplenet/README.md`](../../configs/simplenet/README.md) | [`docs/alignment/simplenet.md`](simplenet.md) |
| SuperSimpleNet | [`configs/supersimplenet/README.md`](../../configs/supersimplenet/README.md) | [`docs/alignment/supersimplenet.md`](supersimplenet.md) |
| CFA | [`configs/cfa/README.md`](../../configs/cfa/README.md) | [`docs/alignment/cfa.md`](cfa.md) |

### Knowledge distillation

| Method | Config README | Alignment record |
|---|---|---|
| RD++ | [`configs/rdpp/README.md`](../../configs/rdpp/README.md) | [`docs/alignment/rdpp.md`](rdpp.md) |
| AST | [`configs/ast/README.md`](../../configs/ast/README.md) | [`docs/alignment/ast.md`](ast.md) |
| RD | [`configs/rd/README.md`](../../configs/rd/README.md) | [`docs/alignment/rd.md`](rd.md) |
| EfficientAD | [`configs/efficientad/README.md`](../../configs/efficientad/README.md) | [`docs/alignment/efficientad.md`](efficientad.md) |
| DeSTSeg | [`configs/destseg/README.md`](../../configs/destseg/README.md) | [`docs/alignment/destseg.md`](destseg.md) |

### Hybrid / unified

| Method | Config README | Alignment record |
|---|---|---|
| UniNet | [`configs/uninet/README.md`](../../configs/uninet/README.md) | [`docs/alignment/uninet.md`](uninet.md) |

### Normalizing flow

| Method | Config README | Alignment record |
|---|---|---|
| U-Flow | [`configs/uflow/README.md`](../../configs/uflow/README.md) | [`docs/alignment/uflow.md`](uflow.md) |
| CFlow | [`configs/cflow/README.md`](../../configs/cflow/README.md) | [`docs/alignment/cflow.md`](cflow.md) |
| DifferNet | [`configs/differnet/README.md`](../../configs/differnet/README.md) | [`docs/alignment/differnet.md`](differnet.md) |
| FastFlow | [`configs/fastflow/README.md`](../../configs/fastflow/README.md) | [`docs/alignment/fastflow.md`](fastflow.md) |
| PyramidFlow | [`configs/pyramidflow/README.md`](../../configs/pyramidflow/README.md) | [`docs/alignment/pyramidflow.md`](pyramidflow.md) |

### Feature-memory / density

| Method | Config README | Alignment record |
|---|---|---|
| PatchCore | [`configs/patchcore/README.md`](../../configs/patchcore/README.md) | [`docs/alignment/patchcore.md`](patchcore.md) |
| PaDiM | [`configs/padim/README.md`](../../configs/padim/README.md) | [`docs/alignment/padim.md`](padim.md) |
| DFM | [`configs/dfm/README.md`](../../configs/dfm/README.md) | [`docs/alignment/dfm.md`](dfm.md) |
| DFKDE | [`configs/dfkde/README.md`](../../configs/dfkde/README.md) | [`docs/alignment/dfkde.md`](dfkde.md) |

### Vision-language / foundation

| Method | Config README | Alignment record |
|---|---|---|
| MuSc | [`configs/musc/README.md`](../../configs/musc/README.md) | [`docs/alignment/musc.md`](musc.md) |
| AACLIP | [`configs/aaclip/README.md`](../../configs/aaclip/README.md) | [`docs/alignment/aaclip.md`](aaclip.md) |
| AnoVL | [`configs/anovl/README.md`](../../configs/anovl/README.md) | [`docs/alignment/anovl.md`](anovl.md) |
| AnomalyCLIP | [`configs/anomalyclip/README.md`](../../configs/anomalyclip/README.md) | [`docs/alignment/anomalyclip.md`](anomalyclip.md) |
| WinCLIP | [`configs/winclip/README.md`](../../configs/winclip/README.md) | [`docs/alignment/winclip.md`](winclip.md) |
| AdaCLIP | [`configs/adaclip/README.md`](../../configs/adaclip/README.md) | [`docs/alignment/adaclip.md`](adaclip.md) |
| SAA+ | [`configs/saaplus/README.md`](../../configs/saaplus/README.md) | [`docs/alignment/saaplus.md`](saaplus.md) |

### Few-shot / registration

| Method | Config README | Alignment record |
|---|---|---|
| AnomalyDINO | [`configs/anomalydino/README.md`](../../configs/anomalydino/README.md) | [`docs/alignment/anomalydino.md`](anomalydino.md) |
| RegAD | [`configs/regad/README.md`](../../configs/regad/README.md) | [`docs/alignment/regad.md`](regad.md) |
