# Third-party notices and release audit

BaoIAD-owned code and documentation may be distributed under Apache-2.0. That
repository-level license does not relicense copied, ported, vendored, closely
derived, or bundled third-party material. Each such item retains its upstream
terms, if any.

The authoritative audit inventory is
[`third_party/provenance.json`](third_party/provenance.json). The resource-level
record is [`resources/asset_approvals.json`](resources/asset_approvals.json).
These records describe the evidence available in the repository; they are not
legal advice or a grant of rights.

## Current release status

The following categories remain **pending external review** and therefore
block the public WAIC release until the recorded disposition is completed:

- code copied or closely ported from upstream projects;
- implementations closely derived from fixed upstream source trees that do not
  contain an identifiable redistribution license;
- a U-Flow NFA implementation derived from AGPL-3.0 source, which is marked for
  removal or clean-room replacement rather than Apache-only distribution;
- specifically referenced third-party pretrained weights whose terms and
  distribution boundary require confirmation; and
- all bundled artwork, diagrams, and dataset-derived example images.

Attribution alone does not resolve incompatible or absent redistribution
permission. Files marked `remove_before_release`, `replace_before_release`, or
`rewrite_before_release` must not ship until that disposition is complete.

## Code and adapted tests

| BaoIAD path or group | Upstream evidence | License evidence | Current disposition |
| --- | --- | --- | --- |
| `baoiad/utils/uflow_nfa.py` and the U-Flow strict hook | [`mtailanian/uflow@d621784`](https://github.com/mtailanian/uflow/tree/d6217844836790773f2c4b91ff3046c59b23f027) | AGPL-3.0-only | Remove the optional NFA path and prove an independent origin for the hook, or clean-room rewrite the affected code before Apache-only release. |
| `baoiad/models/detectors/ast.py` | [`marco-rudolph/AST@8c243ad`](https://github.com/marco-rudolph/AST/tree/8c243ad9adac68e874f87edc6618aa5ea2827228) | No LICENSE/COPYING/NOTICE in the frozen tree | Written permission, clean-room replacement, or omission is required. |
| `baoiad/models/detectors/differnet.py` | [`marco-rudolph/differnet@9bdf026`](https://github.com/marco-rudolph/differnet/tree/9bdf02686297a093fb206ffeba64b1c0e78182b6) | No LICENSE/COPYING/NOTICE in the frozen tree | Written permission, clean-room replacement, or omission is required. |
| `baoiad/models/detectors/cutpaste.py` | [`Runinho/pytorch-cutpaste@10d8bf7`](https://github.com/Runinho/pytorch-cutpaste/tree/10d8bf71df76d3a97f0106efee1d76f81d983149) | No LICENSE/COPYING/NOTICE in the frozen tree | Written permission, clean-room replacement, or omission is required. |
| DRAEM detector and dataset-side augmentation | [`VitjanZ/DRAEM@2dbf673`](https://github.com/VitjanZ/DRAEM/tree/2dbf67397ab5c10a1494e5ae70ab59a25d7c35ef) | MIT; FocalLoss, anomalib Solarize, and ADer blend sources are inventoried separately | Retain the DRAEM MIT notice and detector/dataset modification record, resolve the secondary-source entries, and complete the wrapper comparison. |
| Shared DRAEM/AA-CLIP/GLASS focal-loss lineage | [`hsuxu/Loss_ToolBox-PyTorch`](https://github.com/hsuxu/Loss_ToolBox-PyTorch) | Exact historical source revision and SPDX remain under review because current Apache-2.0 and historical MIT snapshots differ | Freeze the exact historical FocalLoss revision and applicable notice before approval. |
| DRAEM Solarize and beta-blending adaptations | [`open-edge-platform/anomalib@4f6af1a`](https://github.com/open-edge-platform/anomalib/tree/4f6af1acb0ee7b81f54cda036dd9f1c27f63b69a) and [`zhangzjn/ADer@902937a`](https://github.com/zhangzjn/ADer/tree/902937a7ed7fa7689674a4ac9b8fe9a72a40c402) | Apache-2.0 for anomalib; no identifiable redistribution license in the frozen ADer tree | Retain the anomalib notice; obtain permission or independently replace the ADer-derived blend. |
| GANomaly detector | [`samet-akcay/ganomaly@78da4ea`](https://github.com/samet-akcay/ganomaly/tree/78da4ea9a99f5b02ab60dd651a18def929176d77) | MIT | Retain the GANomaly MIT notice, identify the adapted network and initialization code, and complete the MMEngine-wrapper range review. |
| CFA detector | [`open-edge-platform/anomalib@4f6af1a`](https://github.com/open-edge-platform/anomalib/tree/4f6af1acb0ee7b81f54cda036dd9f1c27f63b69a) plus Walsvid CoordConv and the separately inventoried anomalib GaussianBlur2d helper | Apache-2.0 plus the embedded Walsvid MIT CoordConv notice; historical CoordConv revision not frozen | Retain both notice sets, mark modifications, freeze the CoordConv revision, and complete external sign-off. |
| Dinomaly backbone, detector, and StableAdamW optimizer | [`guojiajeremy/Dinomaly@c5c76d0`](https://github.com/guojiajeremy/Dinomaly/tree/c5c76d01a2bd7212f1c4b7dfdad14902d0f48cfe), with DINOv2 and PyTorch recorded separately | Apache-2.0; original DINOv2 revision and PyTorch AdamW revision remain unresolved | Retain the Dinomaly/Jia Guo, Meta, and applicable PyTorch notices, mark modifications, freeze the original source revisions, and complete the wrapper comparisons. |
| SimpleNet and MuSc PatchCore inheritance | [`amazon-science/patchcore-inspection@8a7748c`](https://github.com/amazon-science/patchcore-inspection/tree/8a7748c84b7fee463cb2b27a466ff6c0b7d60882) | Apache-2.0 plus Amazon NOTICE | Retain the PatchCore license and NOTICE and identify both inherited helper variants as modified. |
| SimpleNet detector and split optimizer constructor | [`DonaldRR/SimpleNet@351a2b8`](https://github.com/DonaldRR/SimpleNet/tree/351a2b8d4e8cfc944dbccbf9bc6ceda930c6f26b) | MIT; PatchCore inheritance is inventoried separately | Retain the SimpleNet MIT notice, identify detector/optimizer modifications, preserve the PatchCore notice, and complete external review. |
| SuperSimpleNet detector | [`open-edge-platform/anomalib@4f6af1a`](https://github.com/open-edge-platform/anomalib/tree/4f6af1acb0ee7b81f54cda036dd9f1c27f63b69a) SuperSimpleNet sources | Apache-2.0 and MIT (Intel and original Blaž Rolih notices) | Retain both notice sets, mark the anomalib-derived implementation as modified, and identify BaoIAD framework/backbone adaptations. |
| DFM detector | [`open-edge-platform/anomalib@4f6af1a`](https://github.com/open-edge-platform/anomalib/tree/4f6af1acb0ee7b81f54cda036dd9f1c27f63b69a) DFM and PCA sources | Apache-2.0 | Retain the Intel/anomalib notice, mark the PCA/Gaussian/MMEngine adaptations as modified, and complete external sign-off. |
| RegAD detector and support-set augmentation | [`MediaBrain-SJTU/RegAD@5e2c1f8`](https://github.com/MediaBrain-SJTU/RegAD/tree/5e2c1f8c18d302b0354471567846fee3ed2ff063) | MIT | Retain the RegAD MIT notice, identify Torch/device/MMEngine adaptations, and complete the exact detector/support-bank wrapper review. |
| UniNet detector | [`open-edge-platform/anomalib@4f6af1a`](https://github.com/open-edge-platform/anomalib/tree/4f6af1acb0ee7b81f54cda036dd9f1c27f63b69a) UniNet and shared ResNet decoder | Apache-2.0 and MIT (Intel, Shun Wei, and applicable RD4AD notices) | Retain all applicable Apache-2.0/MIT notices, mark modifications, and complete external sign-off. |
| AUPRO metric | [`open-edge-platform/anomalib@4f6af1a`](https://github.com/open-edge-platform/anomalib/tree/4f6af1acb0ee7b81f54cda036dd9f1c27f63b69a) | Apache-2.0 | Retain the Intel/anomalib notice and mark the Torch-to-NumPy/SciPy port and variable-shape adaptation as modified. |
| `baoiad/utils/sampling.py` k-center-greedy sampler | [`openvinotoolkit/anomalib@0ef8ab1`](https://github.com/open-edge-platform/anomalib/tree/0ef8ab1e43340bddf4d92d1f046c3d34a83af6b0) | Apache-2.0 | Retain the anomalib notice and mark the sampling-size API and optional scikit-learn fallback modifications. |
| FastFlow detector | [`open-edge-platform/anomalib@4f6af1a`](https://github.com/open-edge-platform/anomalib/tree/4f6af1acb0ee7b81f54cda036dd9f1c27f63b69a) FastFlow sources | Apache-2.0; source preserves original @gathierry and Intel notices | Retain both notice sets and mark the BaoIAD registry/runtime adaptation as modified. |
| PaDiM detector | [`openvinotoolkit/anomalib@0ef8ab1`](https://github.com/open-edge-platform/anomalib/tree/0ef8ab1e43340bddf4d92d1f046c3d34a83af6b0) | Apache-2.0 | Retain the Intel/anomalib notice, mark modifications, and complete the exact Gaussian-statistics/memory-bank comparison. |
| DSR detector code and Perlin helper | [`open-edge-platform/anomalib@4f6af1a`](https://github.com/open-edge-platform/anomalib/tree/4f6af1acb0ee7b81f54cda036dd9f1c27f63b69a) DSR, Perlin, and anomaly-generator sources | Apache-2.0; source preserves original VitjanZ and Intel notices | Retain both notice sets, mark the NumPy Perlin port and wrapper modifications, and complete wrapper review; VQ-VAE weights remain separately inventoried. |
| EfficientAD detector code | [`open-edge-platform/anomalib@4f6af1a`](https://github.com/open-edge-platform/anomalib/tree/4f6af1acb0ee7b81f54cda036dd9f1c27f63b69a) EfficientAD sources | Apache-2.0 | Retain the Intel/anomalib notice and mark the architecture, lifecycle, and download-fallback adaptations; teacher weights remain separately inventoried. |
| AA-CLIP detector and dataset adapter | [`Mwxinnn/AA-CLIP@53db195`](https://github.com/Mwxinnn/AA-CLIP/tree/53db195f230442aa118c246876c94ba1c76139cc) | Apache-2.0; upstream FocalLoss names an unfrozen secondary source | Retain the AA-CLIP notice, mark local modifications, freeze/review the Hsuxu/Loss_ToolBox-PyTorch FocalLoss source, and keep dynamically loaded reference assets outside the release unless separately approved. |
| PyramidFlow, ViTAD, and held UniAD/ViTAD support paths listed in the manifest | [`zhangzjn/ADer@902937a`](https://github.com/zhangzjn/ADer/tree/902937a7ed7fa7689674a4ac9b8fe9a72a40c402) | No LICENSE/COPYING/NOTICE in the frozen tree | Trace each retained section to a permissive original source, obtain permission, or replace/omit it. |
| DeSTSeg model and transforms | [`apple/ml-destseg@f6ea31f`](https://github.com/apple/ml-destseg/tree/f6ea31fb5b097698b195f85b1d5e3efaedce9eb6) | Apple sample-code license; non-SPDX custom terms | Legal review is required. If approved, retain the complete upstream terms and modification notices; otherwise omit or rewrite. |
| Reverse Distillation detector | [`hq-deng/RD4AD@6554076`](https://github.com/hq-deng/RD4AD/tree/6554076872c65f8784f6ece8cfb39ce77e1aee12) | MIT | Retain the RD4AD MIT notice, identify the adapted decoder, OCBE, loss, and anomaly-map code, and complete the remaining wrapper range review. |
| MemSeg detector/train loop/runtime hook and Coordinate Attention block | [`TooTouch/MemSeg@836bd46`](https://github.com/TooTouch/MemSeg/tree/836bd465a9b14422f92666dc29dc36edce2692d0) and [`houqb/CoordAttention@7619bea`](https://github.com/houqb/CoordAttention/tree/7619bea9acbe260b3793833cc78cef3f124c8112) | MIT | Retain both MIT notices, identify detector/loop/runtime-hook modifications, and freeze the exact derived ranges and historical Coordinate Attention source revision. |
| AdaCLIP detector, prompts, and dataset adapter | [`caoyunkang/AdaCLIP@b762ac4`](https://github.com/caoyunkang/AdaCLIP/tree/b762ac40c3f33c77e7e513e48cb436f059d456da) | MIT | Retain the MIT notice, identify prompt/code modifications, and freeze exact derived ranges. |
| AnoVL prompts, TextAdapter, TTA loss, and multi-view augmentation in `anovl.py` plus prompts in `anomalyclip.py` | [`hq-deng/AnoVL@3a70bfd`](https://github.com/hq-deng/AnoVL/tree/3a70bfdaea6baf1eeb140c5de8155b535bd94833) | MIT | Retain the AnoVL MIT notice and identify the copied prompt tables, TextAdapter, TTA loss, augmentation helpers, assembly logic, and local modifications in both files. |
| AnomalyCLIP detector adapter | [`zqhang/AnomalyCLIP@3911738`](https://github.com/zqhang/AnomalyCLIP/tree/3911738c0867544f545a076ad78f3f11d9ecbfdf) | MIT | Retain the AnomalyCLIP MIT notice, identify prompt-checkpoint, text-encoding, similarity, and pipeline modifications, disclose the deep-prompt behavior difference, and complete the remaining range review. |
| Canonical `anomalyclip_official.py` wrapper | [`zqhang/AnomalyCLIP@3911738`](https://github.com/zqhang/AnomalyCLIP/tree/3911738c0867544f545a076ad78f3f11d9ecbfdf) | MIT | Retain the AnomalyCLIP MIT notice, identify the BaoIAD wrapper modifications, keep undistributed reference code/checkpoints outside the release, and complete the exact comparison. |
| AnomalyDINO detector | [`dammsi/AnomalyDINO@b9d1c26`](https://github.com/dammsi/AnomalyDINO/tree/b9d1c2648e3a5247437d4d953d907a8f3d994457) | Apache-2.0 | Retain the AnomalyDINO notice, identify BaoIAD/MMEngine adaptations, and complete external sign-off. |
| DFKDE detector | [`open-edge-platform/anomalib@4f6af1a`](https://github.com/open-edge-platform/anomalib/tree/4f6af1acb0ee7b81f54cda036dd9f1c27f63b69a) DFKDE and shared classification sources | Apache-2.0 | Retain the Intel/anomalib notice, identify the modified PCA/KDE/MMEngine adapter, and complete external sign-off. |
| NSA detector and dataset adapter | [`hmsch/natural-synthetic-anomalies@9195916`](https://github.com/hmsch/natural-synthetic-anomalies/tree/919591685307ce030fe27cb77687509dc277189c) | MIT | Retain the MIT notice, identify local modifications, and freeze exact derived ranges. |
| GLASS detector/dataset/helpers/training loop and CFlow detector/training loop | [`cqylunlun/GLASS@6af03b9`](https://github.com/cqylunlun/GLASS/tree/6af03b9d7f7b33a1aebd69cd4c30a41bf020a2d1) and [`gudovskiy/cflow-ad@b2ebf9e`](https://github.com/gudovskiy/cflow-ad/tree/b2ebf9e673a0aa46992a3b18367ec066a57bba89) | MIT and BSD-3-Clause; GLASS FocalLoss names an unfrozen secondary source | Retain both upstream notices, freeze/review the Hsuxu/Loss_ToolBox-PyTorch FocalLoss source, observe the BSD non-endorsement clause, identify all detector/dataset/helper/loop modifications, and complete remaining wrapper review. |
| MuSc LNAMD/MSM/RsCIN regions | [`xrli-U/MuSc@72d58ad`](https://github.com/xrli-U/MuSc/tree/72d58ad56c0cafa2b056bd0aa7676f9c21fccbc4) | MIT; PatchCore inheritance is inventoried separately | Retain the MIT and PatchCore notices and mark modifications. |
| U-Flow non-NFA port and WinCLIP port | [`open-edge-platform/anomalib@4f6af1a`](https://github.com/open-edge-platform/anomalib/tree/4f6af1acb0ee7b81f54cda036dd9f1c27f63b69a) or a revision still to be frozen for WinCLIP | Apache-2.0 | Retain applicable notices, identify modifications, and freeze exact source revisions. |
| RD++ detector, category-epoch training loop, and `baoiad/utils/rdpp_noise.py` | [`tientrandinh/Revisiting-Reverse-Distillation@7f2ceb7`](https://github.com/tientrandinh/Revisiting-Reverse-Distillation/tree/7f2ceb7c87e602617b8600e1a498f7ef7f5247d6) plus [`lmas/opensimplex@ba5cf7f`](https://github.com/lmas/opensimplex/tree/ba5cf7faca5f1bde8e322c90d9c1244458551ce1) | MIT for both audited sources | Retain both MIT notices, identify detector/loop/noise-helper modifications, and complete the remaining detector-wrapper range review. |
| SAA/SAA+ detector, prompts, and saliency wrapper | [`caoyunkang/Segment-Any-Anomaly@ff564ed`](https://github.com/caoyunkang/Segment-Any-Anomaly/tree/ff564ed09bef91d86452f62aa1564e778580513e) | MIT | Retain the MIT notice and mark modifications; weights remain separate artifacts. |
| AUPIMO implementation/reference-vector test | [`jpcbertoldo/aupimo@188ec86`](https://github.com/jpcbertoldo/aupimo/tree/188ec86b5b7d6badad9aa1ae2bc25f94e98a92ca) | MIT | Retain the MIT notice and identify the adapted test vector. |

Line ranges, secondary sources, and reviewer state are recorded in the
machine-readable manifest. This table is deliberately a summary and must not
be treated as evidence that pending items have been approved.

## Dependencies

This document does not attempt to duplicate the license list for every
transitive Python dependency, because that list changes with the resolved
environment. Package metadata and lock/constraint files define dependencies;
a release-specific SBOM or dependency-license report should be generated when
company policy requires it.

## Referenced datasets and weights

Datasets and model weights are not bundled merely because a configuration or
download URL references them. Users must obtain those artifacts from their
owners and comply with the applicable terms. A reference in BaoIAD does not
grant redistribution, commercial-use, trademark, privacy, or publicity rights.

The machine-readable manifest now records the explicit references currently
identified in the public code, including:

- EfficientAD teacher weights and the [Imagenette v2 archive](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz);
- DSR VQ-VAE, GroundingDINO, and SAM checkpoints, plus DTD;
- the canonical Dinomaly [DINOv2 ViT-B/14 register checkpoint](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth);
- the canonical ViTAD [DINO ViT-S/16 checkpoint](https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth); and
- the two optional ViTAD DeiT-distilled checkpoints: [small](https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth) and [tiny](https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth).

These artifacts are not bundled. Their artifact-level licenses and complete
SHA-256 values are not yet frozen, so every entry remains release-blocking.
Local `.refs` checkpoints, the SAA RACM weight, and other dynamically resolved
backbones must still be inventoried or removed during the G004 runtime and
network-boundary pass before the release gate can close.

## Bundled resources

See [`resources/README.md`](resources/README.md) for the exact file hashes,
known origins, intended surfaces, reproduction limits, and fail-closed
disposition of the six bundled resource files.
