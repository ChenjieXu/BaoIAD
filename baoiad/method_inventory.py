"""Repo-local BaoIAD method inventory.

This module is intentionally pure data: it must stay importable without optional
training dependencies and must not contain result metrics, speed values, review
state labels, or external manuscript-repository paths.
"""
from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class MethodEntry:
    slug: str
    display: str
    family: str
    config_paths: tuple[str, ...]
    readme_path: str
    alignment_path: str

METHODS: tuple[MethodEntry, ...] = (
    MethodEntry(
        slug='glass',
        display='GLASS',
        family='Self-supervised synthesis',
        config_paths=('configs/glass/glass_wrn50_288_mvtec_strict.py', 'configs/glass/glass_wrn50_288_visa.py'),
        readme_path='configs/glass/README.md',
        alignment_path='docs/alignment/glass.md',
    ),
    MethodEntry(
        slug='dinomaly',
        display='Dinomaly',
        family='Reconstruction / ViT',
        config_paths=('configs/dinomaly/dinomaly_392_mvtec_strict.py', 'configs/dinomaly/dinomaly_392_visa.py'),
        readme_path='configs/dinomaly/README.md',
        alignment_path='docs/alignment/dinomaly.md',
    ),
    MethodEntry(
        slug='simplenet',
        display='SimpleNet',
        family='Discriminative',
        config_paths=('configs/simplenet/simplenet_wrn50_288_mvtec_strict.py', 'configs/simplenet/simplenet_wrn50_288_visa.py'),
        readme_path='configs/simplenet/README.md',
        alignment_path='docs/alignment/simplenet.md',
    ),
    MethodEntry(
        slug='rdpp',
        display='RD++',
        family='Knowledge distillation',
        config_paths=('configs/rdpp/rdpp_wrn50_256_mvtec_strict.py', 'configs/rdpp/rdpp_wrn50_256_visa.py'),
        readme_path='configs/rdpp/README.md',
        alignment_path='docs/alignment/rdpp.md',
    ),
    MethodEntry(
        slug='ast',
        display='AST',
        family='Knowledge distillation',
        config_paths=('configs/ast/ast_effnet_b5_768_mvtec_strict.py', 'configs/ast/ast_effnet_b5_768_visa.py'),
        readme_path='configs/ast/README.md',
        alignment_path='docs/alignment/ast.md',
    ),
    MethodEntry(
        slug='rd',
        display='RD',
        family='Knowledge distillation',
        config_paths=('configs/rd/rd_wrn50_256_mvtec_strict.py', 'configs/rd/rd_wrn50_256_visa.py'),
        readme_path='configs/rd/README.md',
        alignment_path='docs/alignment/rd.md',
    ),
    MethodEntry(
        slug='uninet',
        display='UniNet',
        family='Hybrid / unified',
        config_paths=('configs/uninet/uninet_256_mvtec_strict.py', 'configs/uninet/uninet_256_visa.py'),
        readme_path='configs/uninet/README.md',
        alignment_path='docs/alignment/uninet.md',
    ),
    MethodEntry(
        slug='supersimplenet',
        display='SuperSimpleNet',
        family='Discriminative',
        config_paths=('configs/supersimplenet/supersimplenet_256_mvtec_strict.py', 'configs/supersimplenet/supersimplenet_256_visa.py'),
        readme_path='configs/supersimplenet/README.md',
        alignment_path='docs/alignment/supersimplenet.md',
    ),
    MethodEntry(
        slug='vitad',
        display='ViTAD',
        family='Reconstruction / ViT',
        config_paths=('configs/vitad/vitad_256_mvtec_strict.py', 'configs/vitad/vitad_256_visa.py'),
        readme_path='configs/vitad/README.md',
        alignment_path='docs/alignment/vitad.md',
    ),
    MethodEntry(
        slug='uflow',
        display='U-Flow',
        family='Normalizing flow',
        config_paths=('configs/uflow/uflow_mcait_448_mvtec_strict.py', 'configs/uflow/uflow_mcait_448_visa.py'),
        readme_path='configs/uflow/README.md',
        alignment_path='docs/alignment/uflow.md',
    ),
    MethodEntry(
        slug='efficientad',
        display='EfficientAD',
        family='Knowledge distillation',
        config_paths=('configs/efficientad/efficientad_256_mvtec_strict.py', 'configs/efficientad/efficientad_256_visa.py'),
        readme_path='configs/efficientad/README.md',
        alignment_path='docs/alignment/efficientad.md',
    ),
    MethodEntry(
        slug='patchcore',
        display='PatchCore',
        family='Feature-memory / density',
        config_paths=('configs/patchcore/patchcore_wrn50_256_mvtec_strict.py', 'configs/patchcore/patchcore_wrn50_256_visa.py'),
        readme_path='configs/patchcore/README.md',
        alignment_path='docs/alignment/patchcore.md',
    ),
    MethodEntry(
        slug='destseg',
        display='DeSTSeg',
        family='Knowledge distillation',
        config_paths=('configs/destseg/destseg_rn18_256_mvtec_strict.py', 'configs/destseg/destseg_rn18_256_visa.py'),
        readme_path='configs/destseg/README.md',
        alignment_path='docs/alignment/destseg.md',
    ),
    MethodEntry(
        slug='musc',
        display='MuSc',
        family='Vision-language / foundation',
        config_paths=('configs/musc/musc_vitl14_336_518_mvtec_strict.py', 'configs/musc/musc_vitl14_336_518_visa.py'),
        readme_path='configs/musc/README.md',
        alignment_path='docs/alignment/musc.md',
    ),
    MethodEntry(
        slug='memseg',
        display='MemSeg',
        family='Reconstruction / ViT',
        config_paths=('configs/memseg/memseg_rn18_256_mvtec_strict.py', 'configs/memseg/memseg_rn18_256_visa.py'),
        readme_path='configs/memseg/README.md',
        alignment_path='docs/alignment/memseg.md',
    ),
    MethodEntry(
        slug='anomalydino',
        display='AnomalyDINO',
        family='Few-shot / registration',
        config_paths=('configs/anomalydino/anomalydino_vitb14_448_mvtec_strict.py', 'configs/anomalydino/anomalydino_vitb14_448_visa.py'),
        readme_path='configs/anomalydino/README.md',
        alignment_path='docs/alignment/anomalydino.md',
    ),
    MethodEntry(
        slug='cflow',
        display='CFlow',
        family='Normalizing flow',
        config_paths=('configs/cflow/cflow_mvtec_strict.py', 'configs/cflow/cflow_visa.py'),
        readme_path='configs/cflow/README.md',
        alignment_path='docs/alignment/cflow.md',
    ),
    MethodEntry(
        slug='draem',
        display='DRAEM',
        family='Self-supervised synthesis',
        config_paths=('configs/draem/draem_256_mvtec_strict.py', 'configs/draem/draem_256_visa.py'),
        readme_path='configs/draem/README.md',
        alignment_path='docs/alignment/draem.md',
    ),
    MethodEntry(
        slug='padim',
        display='PaDiM',
        family='Feature-memory / density',
        config_paths=('configs/padim/padim_wrn50_256_mvtec_strict.py', 'configs/padim/padim_wrn50_256_visa.py'),
        readme_path='configs/padim/README.md',
        alignment_path='docs/alignment/padim.md',
    ),
    MethodEntry(
        slug='cfa',
        display='CFA',
        family='Discriminative',
        config_paths=('configs/cfa/cfa_256_mvtec_strict.py', 'configs/cfa/cfa_256_visa.py'),
        readme_path='configs/cfa/README.md',
        alignment_path='docs/alignment/cfa.md',
    ),
    MethodEntry(
        slug='aaclip',
        display='AACLIP',
        family='Vision-language / foundation',
        config_paths=('configs/aaclip/aaclip_vitl14_336_256_mvtec.py', 'configs/aaclip/aaclip_vitl14_336_518_mvtec_strict.py', 'configs/aaclip/aaclip_vitl14_336_518_visa_32shot_stage1.py', 'configs/aaclip/aaclip_vitl14_336_518_visa_32shot_stage2.py', 'configs/aaclip/aaclip_vitl14_336_518_visa_fullshot_stage1.py', 'configs/aaclip/aaclip_vitl14_336_518_visa_fullshot_stage2.py'),
        readme_path='configs/aaclip/README.md',
        alignment_path='docs/alignment/aaclip.md',
    ),
    MethodEntry(
        slug='differnet',
        display='DifferNet',
        family='Normalizing flow',
        config_paths=('configs/differnet/differnet_alexnet_256_mvtec_strict.py', 'configs/differnet/differnet_alexnet_256_visa.py'),
        readme_path='configs/differnet/README.md',
        alignment_path='docs/alignment/differnet.md',
    ),
    MethodEntry(
        slug='dfm',
        display='DFM',
        family='Feature-memory / density',
        config_paths=('configs/dfm/dfm_256_mvtec_strict.py', 'configs/dfm/dfm_256_visa.py'),
        readme_path='configs/dfm/README.md',
        alignment_path='docs/alignment/dfm.md',
    ),
    MethodEntry(
        slug='fastflow',
        display='FastFlow',
        family='Normalizing flow',
        config_paths=('configs/fastflow/fastflow_wrn50_256_mvtec_strict.py', 'configs/fastflow/fastflow_wrn50_256_visa.py'),
        readme_path='configs/fastflow/README.md',
        alignment_path='docs/alignment/fastflow.md',
    ),
    MethodEntry(
        slug='uniad',
        display='UniAD',
        family='Reconstruction / ViT',
        config_paths=('configs/uniad/uniad_wrn50_256_mvtec_strict.py', 'configs/uniad/uniad_wrn50_256_visa.py'),
        readme_path='configs/uniad/README.md',
        alignment_path='docs/alignment/uniad.md',
    ),
    MethodEntry(
        slug='anovl',
        display='AnoVL',
        family='Vision-language / foundation',
        config_paths=('configs/anovl/anovl_vitb16plus_240_mvtec_strict.py', 'configs/anovl/anovl_vitb16plus_240_visa.py'),
        readme_path='configs/anovl/README.md',
        alignment_path='docs/alignment/anovl.md',
    ),
    MethodEntry(
        slug='anomalyclip',
        display='AnomalyCLIP',
        family='Vision-language / foundation',
        config_paths=('configs/anomalyclip/anomalyclip_vitl14_336_518_mvtec_strict.py', 'configs/anomalyclip/anomalyclip_vitl14_336_518_visa.py'),
        readme_path='configs/anomalyclip/README.md',
        alignment_path='docs/alignment/anomalyclip.md',
    ),
    MethodEntry(
        slug='dsr',
        display='DSR',
        family='Self-supervised synthesis',
        config_paths=('configs/dsr/dsr_256_mvtec_strict.py', 'configs/dsr/dsr_256_visa.py'),
        readme_path='configs/dsr/README.md',
        alignment_path='docs/alignment/dsr.md',
    ),
    MethodEntry(
        slug='winclip',
        display='WinCLIP',
        family='Vision-language / foundation',
        config_paths=('configs/winclip/winclip_256_mvtec.py', 'configs/winclip/winclip_256_visa.py'),
        readme_path='configs/winclip/README.md',
        alignment_path='docs/alignment/winclip.md',
    ),
    MethodEntry(
        slug='regad',
        display='RegAD',
        family='Few-shot / registration',
        config_paths=('configs/regad/regad_wrn50_256_mvtec_strict.py', 'configs/regad/regad_wrn50_256_visa.py'),
        readme_path='configs/regad/README.md',
        alignment_path='docs/alignment/regad.md',
    ),
    MethodEntry(
        slug='cutpaste',
        display='CutPaste',
        family='Self-supervised synthesis',
        config_paths=('configs/cutpaste/cutpaste_rn18_256_mvtec_strict.py', 'configs/cutpaste/cutpaste_rn18_256_visa.py'),
        readme_path='configs/cutpaste/README.md',
        alignment_path='docs/alignment/cutpaste.md',
    ),
    MethodEntry(
        slug='pyramidflow',
        display='PyramidFlow',
        family='Normalizing flow',
        config_paths=('configs/pyramidflow/pyramidflow_fnf_256_mvtec_strict.py', 'configs/pyramidflow/pyramidflow_resnet18_1024_mvtec_strict.py', 'configs/pyramidflow/pyramidflow_resnet18_1024_visa.py'),
        readme_path='configs/pyramidflow/README.md',
        alignment_path='docs/alignment/pyramidflow.md',
    ),
    MethodEntry(
        slug='nsa',
        display='NSA',
        family='Self-supervised synthesis',
        config_paths=('configs/nsa/nsa_rn18_256_mvtec_strict.py', 'configs/nsa/nsa_rn18_256_visa.py'),
        readme_path='configs/nsa/README.md',
        alignment_path='docs/alignment/nsa.md',
    ),
    MethodEntry(
        slug='adaclip',
        display='AdaCLIP',
        family='Vision-language / foundation',
        config_paths=('configs/adaclip/adaclip_vitl14_336_518_mvtec_strict.py', 'configs/adaclip/adaclip_vitl14_336_518_visa.py'),
        readme_path='configs/adaclip/README.md',
        alignment_path='docs/alignment/adaclip.md',
    ),
    MethodEntry(
        slug='dfkde',
        display='DFKDE',
        family='Feature-memory / density',
        config_paths=('configs/dfkde/dfkde_256_mvtec_strict.py', 'configs/dfkde/dfkde_256_visa.py'),
        readme_path='configs/dfkde/README.md',
        alignment_path='docs/alignment/dfkde.md',
    ),
    MethodEntry(
        slug='saaplus',
        display='SAA+',
        family='Vision-language / foundation',
        config_paths=('configs/saaplus/saaplus_400_mvtec_strict.py', 'configs/saaplus/saaplus_400_visa.py'),
        readme_path='configs/saaplus/README.md',
        alignment_path='docs/alignment/saaplus.md',
    ),
    MethodEntry(
        slug='ganomaly',
        display='GANomaly',
        family='Reconstruction / ViT',
        config_paths=('configs/ganomaly/ganomaly_256_mvtec_strict.py', 'configs/ganomaly/ganomaly_256_visa.py'),
        readme_path='configs/ganomaly/README.md',
        alignment_path='docs/alignment/ganomaly.md',
    ),
)

METHODS_BY_SLUG: dict[str, MethodEntry] = {entry.slug: entry for entry in METHODS}

def method_slugs() -> tuple[str, ...]:
    return tuple(entry.slug for entry in METHODS)

def families() -> tuple[str, ...]:
    return tuple(dict.fromkeys(entry.family for entry in METHODS))

def methods_by_family() -> dict[str, tuple[MethodEntry, ...]]:
    grouped: dict[str, list[MethodEntry]] = {}
    for entry in METHODS:
        grouped.setdefault(entry.family, []).append(entry)
    return {family: tuple(items) for family, items in grouped.items()}
