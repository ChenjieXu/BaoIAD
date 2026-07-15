# Implementation provenance and reproducibility notes

This directory contains public, repository-local summaries for the 37 methods in the BaoIAD inventory. Each page is derived from the machine-readable method status inventory and records source provenance, implementation differences, runtime conditions, and known limitations.

Evidence completeness differs by method. These pages do not assert uniform reference parity or independently reproducible benchmark proof. The current inventory marks every method as either **Partially verified** or **Historical evidence**; the exact limitations remain attached to each method.

Public records:

- [Method status inventory](method_status.json)
- [Known exceptions](exceptions.json)

Status meanings:

- **Partially verified**: the underlying record identifies incomplete coverage, a missing rerun or source, proxy-only evidence, or another restricted validation path.
- **Historical evidence**: an implementation narrative exists, but the raw validation artifacts needed for independent verification are not distributed in the public repository.

Inventory assessment: `2026-07-15` against `upstream/master@e93614a01204c441fc85511765879ae031a360bb`.

## Methods by family

### Self-supervised synthesis

| Method | Validation status | Config README | Public summary |
|---|---|---|---|
| GLASS | Partially verified | [`configs/glass/README.md`](../../configs/glass/README.md) | [`docs/alignment/glass.md`](glass.md) |
| DRAEM | Historical evidence | [`configs/draem/README.md`](../../configs/draem/README.md) | [`docs/alignment/draem.md`](draem.md) |
| DSR | Historical evidence | [`configs/dsr/README.md`](../../configs/dsr/README.md) | [`docs/alignment/dsr.md`](dsr.md) |
| CutPaste | Partially verified | [`configs/cutpaste/README.md`](../../configs/cutpaste/README.md) | [`docs/alignment/cutpaste.md`](cutpaste.md) |
| NSA | Historical evidence | [`configs/nsa/README.md`](../../configs/nsa/README.md) | [`docs/alignment/nsa.md`](nsa.md) |

### Reconstruction / ViT

| Method | Validation status | Config README | Public summary |
|---|---|---|---|
| Dinomaly | Historical evidence | [`configs/dinomaly/README.md`](../../configs/dinomaly/README.md) | [`docs/alignment/dinomaly.md`](dinomaly.md) |
| ViTAD | Partially verified | [`configs/vitad/README.md`](../../configs/vitad/README.md) | [`docs/alignment/vitad.md`](vitad.md) |
| MemSeg | Partially verified | [`configs/memseg/README.md`](../../configs/memseg/README.md) | [`docs/alignment/memseg.md`](memseg.md) |
| UniAD | Historical evidence | [`configs/uniad/README.md`](../../configs/uniad/README.md) | [`docs/alignment/uniad.md`](uniad.md) |
| GANomaly | Partially verified | [`configs/ganomaly/README.md`](../../configs/ganomaly/README.md) | [`docs/alignment/ganomaly.md`](ganomaly.md) |

### Discriminative

| Method | Validation status | Config README | Public summary |
|---|---|---|---|
| SimpleNet | Historical evidence | [`configs/simplenet/README.md`](../../configs/simplenet/README.md) | [`docs/alignment/simplenet.md`](simplenet.md) |
| SuperSimpleNet | Partially verified | [`configs/supersimplenet/README.md`](../../configs/supersimplenet/README.md) | [`docs/alignment/supersimplenet.md`](supersimplenet.md) |
| CFA | Partially verified | [`configs/cfa/README.md`](../../configs/cfa/README.md) | [`docs/alignment/cfa.md`](cfa.md) |

### Knowledge distillation

| Method | Validation status | Config README | Public summary |
|---|---|---|---|
| RD++ | Historical evidence | [`configs/rdpp/README.md`](../../configs/rdpp/README.md) | [`docs/alignment/rdpp.md`](rdpp.md) |
| AST | Partially verified | [`configs/ast/README.md`](../../configs/ast/README.md) | [`docs/alignment/ast.md`](ast.md) |
| RD | Historical evidence | [`configs/rd/README.md`](../../configs/rd/README.md) | [`docs/alignment/rd.md`](rd.md) |
| EfficientAD | Partially verified | [`configs/efficientad/README.md`](../../configs/efficientad/README.md) | [`docs/alignment/efficientad.md`](efficientad.md) |
| DeSTSeg | Partially verified | [`configs/destseg/README.md`](../../configs/destseg/README.md) | [`docs/alignment/destseg.md`](destseg.md) |

### Hybrid / unified

| Method | Validation status | Config README | Public summary |
|---|---|---|---|
| UniNet | Historical evidence | [`configs/uninet/README.md`](../../configs/uninet/README.md) | [`docs/alignment/uninet.md`](uninet.md) |

### Normalizing flow

| Method | Validation status | Config README | Public summary |
|---|---|---|---|
| U-Flow | Historical evidence | [`configs/uflow/README.md`](../../configs/uflow/README.md) | [`docs/alignment/uflow.md`](uflow.md) |
| CFlow | Partially verified | [`configs/cflow/README.md`](../../configs/cflow/README.md) | [`docs/alignment/cflow.md`](cflow.md) |
| DifferNet | Historical evidence | [`configs/differnet/README.md`](../../configs/differnet/README.md) | [`docs/alignment/differnet.md`](differnet.md) |
| FastFlow | Partially verified | [`configs/fastflow/README.md`](../../configs/fastflow/README.md) | [`docs/alignment/fastflow.md`](fastflow.md) |
| PyramidFlow | Partially verified | [`configs/pyramidflow/README.md`](../../configs/pyramidflow/README.md) | [`docs/alignment/pyramidflow.md`](pyramidflow.md) |

### Feature-memory / density

| Method | Validation status | Config README | Public summary |
|---|---|---|---|
| PatchCore | Historical evidence | [`configs/patchcore/README.md`](../../configs/patchcore/README.md) | [`docs/alignment/patchcore.md`](patchcore.md) |
| PaDiM | Historical evidence | [`configs/padim/README.md`](../../configs/padim/README.md) | [`docs/alignment/padim.md`](padim.md) |
| DFM | Partially verified | [`configs/dfm/README.md`](../../configs/dfm/README.md) | [`docs/alignment/dfm.md`](dfm.md) |
| DFKDE | Historical evidence | [`configs/dfkde/README.md`](../../configs/dfkde/README.md) | [`docs/alignment/dfkde.md`](dfkde.md) |

### Vision-language / foundation

| Method | Validation status | Config README | Public summary |
|---|---|---|---|
| MuSc | Partially verified | [`configs/musc/README.md`](../../configs/musc/README.md) | [`docs/alignment/musc.md`](musc.md) |
| AACLIP | Partially verified | [`configs/aaclip/README.md`](../../configs/aaclip/README.md) | [`docs/alignment/aaclip.md`](aaclip.md) |
| AnoVL | Historical evidence | [`configs/anovl/README.md`](../../configs/anovl/README.md) | [`docs/alignment/anovl.md`](anovl.md) |
| AnomalyCLIP | Historical evidence | [`configs/anomalyclip/README.md`](../../configs/anomalyclip/README.md) | [`docs/alignment/anomalyclip.md`](anomalyclip.md) |
| WinCLIP | Historical evidence | [`configs/winclip/README.md`](../../configs/winclip/README.md) | [`docs/alignment/winclip.md`](winclip.md) |
| AdaCLIP | Partially verified | [`configs/adaclip/README.md`](../../configs/adaclip/README.md) | [`docs/alignment/adaclip.md`](adaclip.md) |
| SAA+ | Historical evidence | [`configs/saaplus/README.md`](../../configs/saaplus/README.md) | [`docs/alignment/saaplus.md`](saaplus.md) |

### Few-shot / registration

| Method | Validation status | Config README | Public summary |
|---|---|---|---|
| AnomalyDINO | Partially verified | [`configs/anomalydino/README.md`](../../configs/anomalydino/README.md) | [`docs/alignment/anomalydino.md`](anomalydino.md) |
| RegAD | Partially verified | [`configs/regad/README.md`](../../configs/regad/README.md) | [`docs/alignment/regad.md`](regad.md) |
