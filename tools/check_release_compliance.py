#!/usr/bin/env python3
"""Validate BaoIAD's public-release compliance inventory.

The default mode validates that known release blockers are represented
truthfully and fail closed. ``--release-gate`` additionally requires every
blocking approval, exception, asset, and provenance item to be resolved.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
APACHE_2_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
METHOD_STATUS_PATH = Path("docs/alignment/method_status.json")
ALIGNMENT_EXCEPTIONS_PATH = Path("docs/alignment/exceptions.json")
PROVENANCE_PATH = Path(".github/release/provenance.json")
ASSET_APPROVALS_PATH = Path("resources/asset_approvals.json")
EXTERNAL_APPROVALS_PATH = Path("docs/release/external_approvals.json")
EXPECTED_MANIFEST_DIGESTS = {
    METHOD_STATUS_PATH: "998bfb8cac35ffa9415e3bd17a0948e8d1119198f8c0a7c0902569f287003ca4",
    ALIGNMENT_EXCEPTIONS_PATH: "a74d6e88c79e2c3b9d5bb03f67bbed03d2acd5db512fbf8b31705d7826032199",
    PROVENANCE_PATH: "39183ed56227f9013766f79d6ebd7f00b8a0494c3dc6f865313149e3b4c11229",
    ASSET_APPROVALS_PATH: "d1a27f9981e7643bd261235f789606418e45cf01a1a5327e9fd66ad53f0e76d1",
    EXTERNAL_APPROVALS_PATH: "440a6d10e713c9ab480f2a25580de62054032192ae850fc42fc87b250b8409a2",
}

EXPECTED_ASSETS = {
    "resources/architecture.png": "70fa4e6d8a7fc5acc450867dfa919c43dc6c3b6d555e42b89ab61ab4e03e8fb3",
    "resources/architecture_zh.png": "c79cd1c9569e52c89b0625935e6381fe9324a93ff3a093ba207a3a7d4f89d5ad",
    "resources/baoiad-hero.png": "abbae3f88a0a3e120f3df1c31de5c444eeb87a7681d575eae426f05b53367edc",
    "resources/baoiad-logo.svg": "83fdac1470f5a6b58a7b3f1a7847ccb6a34337ef7b165e0e7b36b64f90d91652",
    "resources/vis_examples/anomaly_detection_results.png": (
        "ea0134cf17b023eb4499918ee9c225c8ce34435fb243a020cb2acf144293ac12"
    ),
    "resources/vis_examples/normal_sample.png": (
        "c3d179dd0dfdc2e5bbd859088b4ed74b50939b0364c04e69b0f40cb2f33a56a3"
    ),
}
EXPECTED_ASSET_METADATA = {
    "resources/architecture.png": (
        "image/png",
        "1600x941",
        "unverified_project_diagram",
        None,
        None,
    ),
    "resources/architecture_zh.png": (
        "image/png",
        "1600x941",
        "unverified_project_diagram",
        None,
        None,
    ),
    "resources/baoiad-hero.png": (
        "image/png",
        "1600x500",
        "unverified_project_artwork",
        None,
        None,
    ),
    "resources/baoiad-logo.svg": (
        "image/svg+xml",
        "480x120",
        "unverified_project_artwork",
        None,
        None,
    ),
    "resources/vis_examples/anomaly_detection_results.png": (
        "image/png",
        "1324x350",
        "generated_from_third_party_dataset",
        "https://www.mvtec.com/company/research/datasets/mvtec-ad",
        "scripts/gen_vis_multi_model.py",
    ),
    "resources/vis_examples/normal_sample.png": (
        "image/png",
        "519x300",
        "generated_from_third_party_dataset",
        "https://www.mvtec.com/company/research/datasets/mvtec-ad",
        "scripts/gen_vis_examples.py",
    ),
}
EXPECTED_ASSET_ORIGIN_DIGESTS = {
    "resources/architecture.png": "b789f6ef9a84fbb5c249e125e7dd316b6ba5af0fdfdc3f720e63d82461910ec2",
    "resources/architecture_zh.png": "b789f6ef9a84fbb5c249e125e7dd316b6ba5af0fdfdc3f720e63d82461910ec2",
    "resources/baoiad-hero.png": "a90642d43f6b7e29895e159a30226249d86de4f536046f07d86b740c1c83007a",
    "resources/baoiad-logo.svg": "a55ce2229b925771db869c36df226b692b081808fcd925c54bfde49025ace1a3",
    "resources/vis_examples/anomaly_detection_results.png": "1c7920ac76f503813a7237db356acbf99b626744e5ef736ad60a3e181816616f",
    "resources/vis_examples/normal_sample.png": "09e3e75e17c8e201c9f235f8cf77ae62aac4212321078a453ca9589ba05a8617",
}
REQUIRED_APPROVAL_IDS = {
    "APP-BRAND-ASSETS",
    "APP-COMMUNITY-CONDUCT",
    "APP-COMPANY-IDENTITY",
    "APP-SECURITY-CHANNEL",
    "APP-THIRD-PARTY",
}
EXPECTED_PAPER_URLS = {
    "aaclip": "https://arxiv.org/abs/2503.06661",
    "adaclip": "https://arxiv.org/abs/2407.15795",
    "anomalyclip": "https://arxiv.org/abs/2310.18961",
    "anomalydino": "https://arxiv.org/abs/2405.14529",
    "anovl": "https://arxiv.org/abs/2308.15939",
    "ast": "https://arxiv.org/abs/2210.07829",
    "cfa": "https://arxiv.org/abs/2206.04325",
    "cflow": "https://arxiv.org/abs/2107.12571",
    "cutpaste": "https://arxiv.org/abs/2104.04015",
    "destseg": "https://arxiv.org/abs/2211.11317",
    "dfkde": None,
    "dfm": "https://arxiv.org/abs/1909.11786",
    "differnet": "https://arxiv.org/abs/2008.12577",
    "dinomaly": "https://arxiv.org/abs/2405.14325",
    "draem": "https://arxiv.org/abs/2108.07610",
    "dsr": "https://arxiv.org/abs/2208.01521",
    "efficientad": "https://arxiv.org/abs/2303.14535",
    "fastflow": "https://arxiv.org/abs/2111.07677",
    "ganomaly": "https://arxiv.org/abs/1805.06725",
    "glass": "https://arxiv.org/abs/2407.09359",
    "memseg": "https://arxiv.org/abs/2205.00908",
    "musc": "https://arxiv.org/abs/2401.16753",
    "nsa": "https://arxiv.org/abs/2109.15222",
    "padim": "https://arxiv.org/abs/2011.08785",
    "patchcore": "https://arxiv.org/abs/2106.08265",
    "pyramidflow": "https://arxiv.org/abs/2303.02595",
    "rd": "https://arxiv.org/abs/2201.10703",
    "rdpp": (
        "https://openaccess.thecvf.com/content/CVPR2023/html/"
        "Tien_Revisiting_Reverse_Distillation_for_Anomaly_Detection_"
        "CVPR_2023_paper.html"
    ),
    "regad": "https://arxiv.org/abs/2207.07361",
    "saaplus": "https://arxiv.org/abs/2305.10724",
    "simplenet": "https://arxiv.org/abs/2303.15140",
    "supersimplenet": "https://arxiv.org/abs/2408.03143",
    "uflow": "https://arxiv.org/abs/2211.12353",
    "uniad": "https://arxiv.org/abs/2206.03687",
    "uninet": (
        "https://openaccess.thecvf.com/content/CVPR2025/html/"
        "Wei_UniNet_A_Contrastive_Learning-guided_Unified_Framework_with_"
        "Feature_Selection_for_CVPR_2025_paper.html"
    ),
    "vitad": "https://arxiv.org/abs/2312.07495",
    "winclip": "https://arxiv.org/abs/2303.14814",
}
EXPECTED_METHOD_SOURCES = {
    "aaclip": (
        "https://github.com/Mwxinnn/AA-CLIP",
        "53db195f230442aa118c246876c94ba1c76139cc",
        "public_revision",
    ),
    "adaclip": (
        "https://github.com/caoyunkang/AdaCLIP",
        "b762ac40c3f33c77e7e513e48cb436f059d456da",
        "public_revision",
    ),
    "anomalyclip": (
        "https://github.com/zqhang/AnomalyCLIP",
        "3911738c0867544f545a076ad78f3f11d9ecbfdf",
        "public_revision",
    ),
    "anomalydino": (
        "https://github.com/dammsi/AnomalyDINO",
        "b9d1c2648e3a5247437d4d953d907a8f3d994457",
        "public_revision",
    ),
    "anovl": (
        "https://github.com/hq-deng/AnoVL",
        "3a70bfdaea6baf1eeb140c5de8155b535bd94833",
        "public_revision",
    ),
    "ast": (
        "https://github.com/marco-rudolph/AST",
        "8c243ad9adac68e874f87edc6618aa5ea2827228",
        "public_revision",
    ),
    "cfa": (
        "https://github.com/open-edge-platform/anomalib",
        "4f6af1acb0ee7b81f54cda036dd9f1c27f63b69a",
        "public_revision",
    ),
    "cflow": (
        "https://github.com/gudovskiy/cflow-ad",
        "b2ebf9e673a0aa46992a3b18367ec066a57bba89",
        "public_revision",
    ),
    "cutpaste": (
        "https://github.com/Runinho/pytorch-cutpaste",
        "10d8bf71df76d3a97f0106efee1d76f81d983149",
        "public_revision",
    ),
    "destseg": (
        "https://github.com/apple/ml-destseg",
        "f6ea31fb5b097698b195f85b1d5e3efaedce9eb6",
        "public_revision",
    ),
    "dfkde": (
        "https://github.com/open-edge-platform/anomalib",
        "4f6af1acb0ee7b81f54cda036dd9f1c27f63b69a",
        "public_revision",
    ),
    "dfm": (
        "https://github.com/open-edge-platform/anomalib",
        "4f6af1acb0ee7b81f54cda036dd9f1c27f63b69a",
        "public_revision",
    ),
    "differnet": (
        "https://github.com/marco-rudolph/differnet",
        "9bdf02686297a093fb206ffeba64b1c0e78182b6",
        "public_revision",
    ),
    "dinomaly": (
        "https://github.com/guojiajeremy/Dinomaly",
        "c5c76d01a2bd7212f1c4b7dfdad14902d0f48cfe",
        "public_revision",
    ),
    "draem": (
        "https://github.com/VitjanZ/DRAEM",
        "2dbf67397ab5c10a1494e5ae70ab59a25d7c35ef",
        "public_revision",
    ),
    "dsr": (
        "https://github.com/open-edge-platform/anomalib",
        "4f6af1acb0ee7b81f54cda036dd9f1c27f63b69a",
        "public_revision",
    ),
    "efficientad": (
        "https://github.com/open-edge-platform/anomalib",
        "4f6af1acb0ee7b81f54cda036dd9f1c27f63b69a",
        "public_revision",
    ),
    "fastflow": (
        "https://github.com/open-edge-platform/anomalib",
        "4f6af1acb0ee7b81f54cda036dd9f1c27f63b69a",
        "public_revision",
    ),
    "ganomaly": (
        "https://github.com/samet-akcay/ganomaly",
        "78da4ea9a99f5b02ab60dd651a18def929176d77",
        "public_revision",
    ),
    "glass": (
        "https://github.com/cqylunlun/GLASS",
        "6af03b9d7f7b33a1aebd69cd4c30a41bf020a2d1",
        "public_revision",
    ),
    "memseg": (
        "https://github.com/TooTouch/MemSeg",
        "836bd465a9b14422f92666dc29dc36edce2692d0",
        "public_revision",
    ),
    "musc": (
        "https://github.com/xrli-U/MuSc",
        "72d58ad56c0cafa2b056bd0aa7676f9c21fccbc4",
        "public_revision",
    ),
    "nsa": (
        "https://github.com/hmsch/natural-synthetic-anomalies",
        "919591685307ce030fe27cb77687509dc277189c",
        "audit_revision_only",
    ),
    "padim": (
        "https://github.com/open-edge-platform/anomalib",
        "0ef8ab1e43340bddf4d92d1f046c3d34a83af6b0",
        "public_revision",
    ),
    "patchcore": (None, None, "local_snapshot_only"),
    "pyramidflow": (
        "https://github.com/gasharper/PyramidFlow",
        None,
        "repository_unavailable",
    ),
    "rd": (
        "https://github.com/hq-deng/RD4AD",
        "6554076872c65f8784f6ece8cfb39ce77e1aee12",
        "public_revision",
    ),
    "rdpp": (
        "https://github.com/tientrandinh/Revisiting-Reverse-Distillation",
        "7f2ceb7c87e602617b8600e1a498f7ef7f5247d6",
        "public_revision",
    ),
    "regad": (
        "https://github.com/MediaBrain-SJTU/RegAD",
        "5e2c1f8c18d302b0354471567846fee3ed2ff063",
        "public_revision",
    ),
    "saaplus": (
        "https://github.com/caoyunkang/Segment-Any-Anomaly",
        "ff564ed09bef91d86452f62aa1564e778580513e",
        "public_revision",
    ),
    "simplenet": (
        "https://github.com/DonaldRR/SimpleNet",
        "351a2b8d4e8cfc944dbccbf9bc6ceda930c6f26b",
        "public_revision",
    ),
    "supersimplenet": (
        "https://github.com/open-edge-platform/anomalib",
        "4f6af1acb0ee7b81f54cda036dd9f1c27f63b69a",
        "public_revision",
    ),
    "uflow": (
        "https://github.com/mtailanian/uflow",
        "d6217844836790773f2c4b91ff3046c59b23f027",
        "public_revision",
    ),
    "uniad": (
        "https://github.com/zhangzjn/ADer",
        "902937a7ed7fa7689674a4ac9b8fe9a72a40c402",
        "public_revision",
    ),
    "uninet": (
        "https://github.com/open-edge-platform/anomalib",
        "4f6af1acb0ee7b81f54cda036dd9f1c27f63b69a",
        "public_revision",
    ),
    "vitad": (
        "https://github.com/zhangzjn/ADer",
        "902937a7ed7fa7689674a4ac9b8fe9a72a40c402",
        "public_revision",
    ),
    "winclip": (
        "https://github.com/open-edge-platform/anomalib",
        None,
        "public_repository_unpinned",
    ),
}
EXPECTED_RUNTIME_STATES = {
    "aaclip": "blocked_by_undistributed_assets",
    "adaclip": "not_assessed",
    "anomalyclip": "blocked_by_undistributed_assets",
    "anomalydino": "not_assessed",
    "anovl": "blocked_by_undistributed_assets",
    "ast": "not_assessed",
    "cfa": "not_assessed",
    "cflow": "blocked_by_undistributed_assets",
    "cutpaste": "not_assessed",
    "destseg": "blocked_by_undistributed_assets",
    "dfkde": "not_assessed",
    "dfm": "not_assessed",
    "differnet": "not_assessed",
    "dinomaly": "network_dependent",
    "draem": "not_assessed",
    "dsr": "network_dependent",
    "efficientad": "network_dependent",
    "fastflow": "blocked_by_optional_dependency",
    "ganomaly": "not_assessed",
    "glass": "blocked_by_undistributed_assets",
    "memseg": "not_assessed",
    "musc": "blocked_by_undistributed_assets",
    "nsa": "not_assessed",
    "padim": "not_assessed",
    "patchcore": "not_assessed",
    "pyramidflow": "not_assessed",
    "rd": "not_assessed",
    "rdpp": "blocked_by_optional_dependency",
    "regad": "not_assessed",
    "saaplus": "network_dependent",
    "simplenet": "not_assessed",
    "supersimplenet": "not_assessed",
    "uflow": "blocked_by_optional_dependency",
    "uniad": "not_assessed",
    "uninet": "not_assessed",
    "vitad": "network_dependent",
    "winclip": "network_dependent",
}
REQUIRED_LIMITATION_TERMS = {
    "cflow": ("FrEIA", "fallback"),
    "destseg": ("DTD", "not distributed"),
    "fastflow": ("FrEIA", "optional"),
    "glass": ("pandas", "not declared"),
    "nsa": ("historical derivation revision", "audit"),
    "rdpp": ("geomloss", "fails closed"),
    "regad": ("strict_require_official_support_set=False", "fallback"),
    "uflow": ("FrEIA", "optional"),
    "vitad": ("pretrained DINO weights", "randomly initialized"),
}
REQUIRED_PROVENANCE_PATHS = {
    "baoiad/models/detectors/anovl.py",
    "baoiad/models/detectors/ast.py",
    "baoiad/models/detectors/cfa.py",
    "baoiad/models/detectors/cflow.py",
    "baoiad/models/detectors/cutpaste.py",
    "baoiad/models/detectors/destseg.py",
    "baoiad/models/detectors/differnet.py",
    "baoiad/models/detectors/dfm.py",
    "baoiad/models/backbones/dinomaly_backbone.py",
    "baoiad/models/detectors/dinomaly.py",
    "baoiad/models/detectors/draem.py",
    "baoiad/datasets/draem_dataset.py",
    "baoiad/models/detectors/ganomaly.py",
    "baoiad/models/detectors/anomalyclip_official.py",
    "baoiad/models/detectors/anomalydino.py",
    "baoiad/models/detectors/dfkde.py",
    "baoiad/models/detectors/glass.py",
    "baoiad/datasets/glass_dataset.py",
    "baoiad/utils/glass_utils.py",
    "baoiad/models/detectors/musc.py",
    "baoiad/models/detectors/pyramidflow.py",
    "baoiad/models/detectors/rdpp.py",
    "baoiad/models/detectors/reverse_distillation.py",
    "baoiad/models/detectors/regad.py",
    "baoiad/models/detectors/saa.py",
    "baoiad/models/detectors/saa_prompts.py",
    "baoiad/models/backbones/saa_saliency_backbone.py",
    "baoiad/models/detectors/uflow.py",
    "baoiad/models/detectors/uninet.py",
    "baoiad/models/detectors/simplenet.py",
    "baoiad/models/detectors/supersimplenet.py",
    "baoiad/engine/optimizers/simplenet_optim_wrapper_constructor.py",
    "baoiad/models/detectors/uniad_detector.py",
    "baoiad/models/detectors/vitad.py",
    "baoiad/models/detectors/winclip.py",
    "baoiad/models/detectors/memseg.py",
    "baoiad/engine/hooks/memseg_strict_hook.py",
    "baoiad/engine/loops/memseg_train_loop.py",
    "baoiad/engine/loops/rdpp_train_loop.py",
    "baoiad/engine/optimizers/stable_adamw.py",
    "baoiad/models/detectors/adaclip.py",
    "baoiad/models/detectors/anomalyclip.py",
    "baoiad/datasets/adaclip_aux.py",
    "baoiad/models/detectors/nsa.py",
    "baoiad/datasets/nsa_dataset.py",
    "baoiad/models/backbones/vitad_backbone.py",
    "baoiad/datasets/transforms/destseg.py",
    "baoiad/datasets/samplers.py",
    "baoiad/engine/hooks/vitad_strict_hook.py",
    "baoiad/engine/loops/vitad_train_loop.py",
    "baoiad/engine/optimizers/vitad_optim_wrapper_constructor.py",
    "baoiad/evaluation/aupimo.py",
    "baoiad/utils/rdpp_noise.py",
    "baoiad/utils/sampling.py",
    "baoiad/utils/uflow_nfa.py",
    "tests/test_evaluation/test_aupimo.py",
    *EXPECTED_ASSETS,
}
REQUIRED_SECONDARY_SOURCE_IDS = {
    "TP-CODE-CFA-GAUSSIAN-BLUR",
    "TP-CODE-DINOMALY-DINOV2",
    "TP-CODE-DINOMALY-PYTORCH-ADAMW",
    "TP-CODE-DRAEM-ADER-BLEND",
    "TP-CODE-DRAEM-ANOMALIB-SOLARIZE",
    "TP-CODE-DSR-PERLIN",
    "TP-CODE-HSUXU-FOCAL",
    "TP-CODE-OPENSIMPLEX",
    "TP-CODE-PATCHCORE-INHERITANCE",
}
EXPECTED_EXTERNAL_ARTIFACTS = {
    "TP-DATA-IMAGENETTE": (
        "referenced_external_dataset",
        "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz",
        "baoiad/models/detectors/efficientad.py",
    ),
    "TP-WEIGHT-DINOMALY-DINOV2-REG-BASE14": (
        "referenced_external_weight",
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/"
        "dinov2_vitb14_reg4_pretrain.pth",
        "baoiad/models/backbones/dinomaly_backbone.py",
    ),
    "TP-WEIGHT-VITAD-DINO-SMALL": (
        "referenced_external_weight",
        "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/"
        "dino_deitsmall16_pretrain.pth",
        "baoiad/models/backbones/vitad_backbone.py",
    ),
    "TP-WEIGHT-VITAD-DEIT-SMALL-DISTILLED": (
        "referenced_external_weight",
        "https://dl.fbaipublicfiles.com/deit/"
        "deit_small_distilled_patch16_224-649709d9.pth",
        "baoiad/models/backbones/vitad_backbone.py",
    ),
    "TP-WEIGHT-VITAD-DEIT-TINY-DISTILLED": (
        "referenced_external_weight",
        "https://dl.fbaipublicfiles.com/deit/"
        "deit_tiny_distilled_patch16_224-b40b3cf7.pth",
        "baoiad/models/backbones/vitad_backbone.py",
    ),
}
ALLOWED_VERIFICATION_STATES = {"historical_evidence", "partial"}
EXPECTED_VERIFICATION_STATE_DEFINITIONS = {
    "historical_evidence": "The repository contains an implementation narrative, but the referenced raw validation artifacts are not distributed and the claim is not independently verifiable from a public clone.",
    "partial": "The alignment narrative itself records incomplete coverage, a missing fresh rerun, proxy-only evidence, a missing source/checkpoint, or a restricted evaluation path.",
}
EXPECTED_PARTIAL_METHODS = {
    "aaclip",
    "adaclip",
    "anomalydino",
    "ast",
    "cfa",
    "cflow",
    "cutpaste",
    "destseg",
    "dfm",
    "efficientad",
    "fastflow",
    "ganomaly",
    "glass",
    "memseg",
    "musc",
    "pyramidflow",
    "regad",
    "supersimplenet",
    "vitad",
}
ALLOWED_RUNTIME_STATES = {
    "blocked_by_optional_dependency",
    "not_assessed",
    "blocked_by_undistributed_assets",
    "network_dependent",
}
EXPECTED_RUNTIME_STATE_DEFINITIONS = {
    "not_assessed": "Runtime availability has not been fully assessed; no clean-clone blocker was established beyond the separately recorded evidence and license limitations.",
    "blocked_by_undistributed_assets": "The canonical path requires local assets, checkpoints, support sets, or datasets that are not distributed with the repository.",
    "blocked_by_optional_dependency": "The canonical implementation imports or requires a project extra that a core installation does not provide.",
    "network_dependent": "The canonical path may download a model or other external artifact when it is absent locally.",
}
ALLOWED_REVIEW_STATES = {"pending_external_review", "approved", "rejected"}
ALLOWED_LICENSE_STATES = {"confirmed", "incompatible", "needs_external_review"}
ALLOWED_DISPOSITIONS = {
    "external_reference_only",
    "keep_with_attribution",
    "omit_unless_approved",
    "remove_before_release",
    "replace_before_release",
    "rewrite_before_release",
}
DERIVED_PROVENANCE_KINDS = {
    "adapted_test_vector",
    "closely_derived_code",
    "closely_derived_code_and_prompts",
    "closely_derived_code_and_training_control",
    "closely_derived_training_control",
    "direct_copy_and_modification",
    "ported_code",
    "ported_code_and_prompts",
    "vendored_code",
}
ALLOWED_PROVENANCE_KINDS = DERIVED_PROVENANCE_KINDS | {
    "bundled_assets",
    "dataset_derived_assets",
    "referenced_external_dataset",
    "referenced_external_weight",
}
ALLOWED_INCORPORATION_MODES = {
    "behavioral_port_and_adapter",
    "behavioral_port_pending_exact_comparison",
    "bundled_binary_or_svg_assets",
    "closely_derived_reimplementation",
    "closely_derived_reimplementation_and_adapter",
    "copied_then_modified",
    "external_download_reference_not_bundled",
    "generated_derivative_images",
    "language_level_port_and_modification",
    "modified_port",
    "multiple_closely_derived_implementations",
    "paper_based_implementation_and_adapted_test",
    "ported_and_modified",
    "vendored_then_modified",
}
ACCEPTED_LIMITATION_IDS = {"ALIGN-PARTIAL-VALIDATION"}
MACHINE_GATED_ALIGNMENT_IDS = {"ALIGN-CLEAN-CLONE"}
CLEAN_CLONE_MANUAL_METHODS = {
    # RegAD remains importable, but the strict-named config silently substitutes a
    # non-official support-set sampler when the undistributed official set is absent.
    "regad"
}
APPROVAL_PLACEHOLDER = re.compile(
    r"\b(?:pending|tbd|to be frozen|still to be frozen|unknown|none|n/?a|"
    r"fake|placeholder|unverified|not-a-range)\b",
    re.IGNORECASE,
)
DIRECT_SOURCE_MARKER = re.compile(
    r"copied from|ported from|vendored from|reference-aligned copy|"
    r"port of the official|follows the official reference closely|"
    r"following (?:the )?official implementation|from anomalib|"
    r"(?:adapted|derived) from (?:the )?(?:upstream|official)|"
    r"matches (?:the )?reference (?:[A-Za-z0-9_-]+ )?(?:code|implementation)|"
    r"matches [^\n]{0,120}(?:official|anomalib)[^\n]{0,120}"
    r"(?:repository|implementation|code|optimizer)|"
    r"matching (?:the )?(?:upstream|official|anomalib(?:'s)?)|"
    r"official [^\n]{0,80}train loop helpers|"
    r"aligned (?:with|to) (?:the )?(?:official|anomalib|ADer)|"
    r"faithful reimplementation of https?://|mirrors? (?:the )?(?:official|upstream)|"
    r"from (?:the )?official implementation|"
    r"aligned to (?:the )?(?:original )?official implementation|from ADer",
    re.IGNORECASE,
)


def _read_json(root: Path, relative: Path, errors: list[str]) -> dict[str, Any]:
    path = root / relative
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        errors.append(f"missing manifest: {relative}")
        return {}
    except json.JSONDecodeError as exc:
        errors.append(f"invalid JSON {relative}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{relative}: top-level value must be an object")
        return {}
    return value


def _valid_terminal_evidence(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and APPROVAL_PLACEHOLDER.search(value) is None
    )


def _is_bare_placeholder(value: Any) -> bool:
    return isinstance(value, str) and bool(
        APPROVAL_PLACEHOLDER.fullmatch(value.strip())
    )


def _canonical_json_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _manifest_digest_error(document: dict[str, Any], path: Path) -> str | None:
    actual = _canonical_json_digest(document)
    expected = EXPECTED_MANIFEST_DIGESTS[path]
    if actual == expected:
        return None
    return (
        f"{path}: audited manifest projection changed; update the release checker "
        f"only after review (expected {expected}, found {actual})"
    )


def _load_inventory(root: Path):
    path = root / "baoiad" / "method_inventory.py"
    spec = importlib.util.spec_from_file_location(
        "baoiad_release_method_inventory", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return tuple(module.METHODS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _broken_alignment_links(root: Path) -> list[str]:
    broken: list[str] = []
    link_pattern = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
    for document in sorted((root / "docs" / "alignment").glob("*.md")):
        text = document.read_text(encoding="utf-8")
        for raw_target in link_pattern.findall(text):
            target = raw_target.strip().strip("<>")
            if not target or target.startswith(("#", "mailto:")):
                continue
            if re.match(r"^[a-z][a-z0-9+.-]*://", target, re.IGNORECASE):
                continue
            path_text = unquote(target.split("#", 1)[0].strip())
            if not path_text:
                continue
            resolved = (document.parent / path_text).resolve()
            if not resolved.exists():
                broken.append(f"{document.relative_to(root).as_posix()} -> {path_text}")
    return broken


def _undistributed_alignment_artifact_mentions(root: Path) -> list[str]:
    """Return internal evidence markers that cannot ship as public proof."""
    markers = (
        ".refs/",
        "runs/alignment",
        "runs/benchmark",
        "manuscript evidence workspace",
        "playbook",
        "agent handoff",
    )
    mentions: list[str] = []
    for document in sorted((root / "docs" / "alignment").glob("*.md")):
        text = document.read_text(encoding="utf-8").lower()
        for marker in markers:
            if marker in text:
                mentions.append(f"{document.relative_to(root).as_posix()} -> {marker}")
    return mentions


def _config_model_types(path: Path) -> tuple[list[str], list[str]]:
    """Return explicit ``model.type`` strings and parse errors without importing configs."""
    errors: list[str] = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        return [], [f"cannot parse config {path}: {exc}"]

    model_types: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "model" for target in targets
        ):
            continue
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "dict"
        ):
            candidates = [
                keyword.value for keyword in value.keywords if keyword.arg == "type"
            ]
        elif isinstance(value, ast.Dict):
            candidates = [
                item
                for key, item in zip(value.keys, value.values)
                if isinstance(key, ast.Constant) and key.value == "type"
            ]
        else:
            candidates = []
        for candidate in candidates:
            if isinstance(candidate, ast.Constant) and isinstance(candidate.value, str):
                model_types.append(candidate.value)
            else:
                errors.append(f"config {path}: model.type must be a string literal")
    return model_types, errors


def _method_readme_paper_link_mismatches(
    root: Path, inventory: tuple[Any, ...]
) -> dict[str, tuple[str | None, str | None]]:
    mismatches: dict[str, tuple[str | None, str | None]] = {}
    url_pattern = re.compile(r"https://[^)\s]+")
    for entry in inventory:
        path = root / entry.readme_path
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            mismatches[entry.slug] = (None, EXPECTED_PAPER_URLS[entry.slug])
            continue
        paper_line = next((line for line in lines if "**Paper**" in line), "")
        urls = url_pattern.findall(paper_line)
        actual = urls[0] if urls else None
        expected = EXPECTED_PAPER_URLS[entry.slug]
        if actual != expected:
            mismatches[entry.slug] = (actual, expected)
    return mismatches


def validate_license(root: Path = ROOT) -> list[str]:
    path = root / "LICENSE"
    if not path.is_file():
        return ["missing LICENSE"]
    actual = _sha256(path)
    if actual != APACHE_2_SHA256:
        return [
            "LICENSE must be the unmodified Apache-2.0 text "
            f"(expected sha256 {APACHE_2_SHA256}, found {actual})"
        ]
    return []


def validate_method_status_document(
    document: dict[str, Any],
    inventory: tuple[Any, ...],
    root: Path = ROOT,
) -> list[str]:
    errors: list[str] = []
    if digest_error := _manifest_digest_error(document, METHOD_STATUS_PATH):
        errors.append(digest_error)
    if document.get("schema_version") != 1:
        errors.append("method status: schema_version must be 1")
    if document.get("inventory_ref") != "baoiad/method_inventory.py":
        errors.append(
            "method status: inventory_ref must point to baoiad/method_inventory.py"
        )
    if document.get("state_definitions") != EXPECTED_VERIFICATION_STATE_DEFINITIONS:
        errors.append(
            "method status: state_definitions must match the audited release schema"
        )
    if document.get("runtime_state_definitions") != EXPECTED_RUNTIME_STATE_DEFINITIONS:
        errors.append(
            "method status: runtime_state_definitions must match the release schema"
        )

    records = document.get("methods")
    if not isinstance(records, list):
        return errors + ["method status: methods must be a list"]
    by_slug: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            errors.append("method status: every method record must be an object")
            continue
        slug = record.get("slug")
        if not isinstance(slug, str) or not slug:
            errors.append("method status: every record needs a non-empty slug")
            continue
        if slug in by_slug:
            errors.append(f"method status: duplicate slug {slug}")
        by_slug[slug] = record

    expected = {entry.slug: entry for entry in inventory}
    if set(by_slug) != set(expected):
        missing = sorted(set(expected) - set(by_slug))
        extra = sorted(set(by_slug) - set(expected))
        errors.append(
            f"method status: slug set mismatch; missing={missing}, extra={extra}"
        )

    for slug, entry in expected.items():
        record = by_slug.get(slug)
        if record is None:
            continue
        for key, expected_value in (
            ("display", entry.display),
            ("family", entry.family),
            ("alignment_path", entry.alignment_path),
        ):
            if record.get(key) != expected_value:
                errors.append(
                    f"method status {slug}: {key} must match inventory "
                    f"({expected_value!r})"
                )
        for key in ("registry_name", "detector_module", "implementation_differences"):
            if not isinstance(record.get(key), str) or not record[key].strip():
                errors.append(f"method status {slug}: {key} must be non-empty")

        registry_name = record.get("registry_name")
        explicit_config_types: list[str] = []
        for config_path in entry.config_paths:
            config_types, parse_errors = _config_model_types(root / config_path)
            errors.extend(
                f"method status {slug}: {message}" for message in parse_errors
            )
            explicit_config_types.extend(config_types)
        if not explicit_config_types:
            errors.append(
                f"method status {slug}: no inventory config declares a literal model.type"
            )
        elif any(value != registry_name for value in explicit_config_types):
            errors.append(
                f"method status {slug}: registry_name must match config model.type values "
                f"({sorted(set(explicit_config_types))!r})"
            )

        detector_module = record.get("detector_module")
        if isinstance(detector_module, str) and detector_module.startswith("baoiad."):
            module_path = root / Path(*detector_module.split(".")).with_suffix(".py")
            if not module_path.is_file():
                errors.append(
                    f"method status {slug}: detector_module does not resolve to a file"
                )
            else:
                try:
                    module_tree = ast.parse(
                        module_path.read_text(encoding="utf-8"),
                        filename=str(module_path),
                    )
                except (OSError, SyntaxError) as exc:
                    errors.append(
                        f"method status {slug}: cannot parse detector_module: {exc}"
                    )
                else:
                    class_names = {
                        node.name
                        for node in module_tree.body
                        if isinstance(node, ast.ClassDef)
                    }
                    if registry_name not in class_names:
                        errors.append(
                            f"method status {slug}: detector_module does not define "
                            f"registry class {registry_name!r}"
                        )
        elif isinstance(detector_module, str):
            errors.append(
                f"method status {slug}: detector_module must be inside baoiad"
            )

        paper_url = record.get("paper_url")
        if paper_url != EXPECTED_PAPER_URLS.get(slug):
            errors.append(
                f"method status {slug}: paper_url must match verified primary source "
                f"({EXPECTED_PAPER_URLS.get(slug)!r})"
            )

        source = record.get("source")
        if not isinstance(source, dict):
            errors.append(f"method status {slug}: source must be an object")
        else:
            if set(source) != {"url", "revision", "traceability"}:
                errors.append(
                    f"method status {slug}: source fields must be url/revision/traceability"
                )
            url = source.get("url")
            if url is not None and (
                not isinstance(url, str) or not url.startswith("https://")
            ):
                errors.append(f"method status {slug}: source.url must be HTTPS or null")
            revision = source.get("revision")
            if revision is not None and (
                not isinstance(revision, str) or not revision.strip()
            ):
                errors.append(
                    f"method status {slug}: source.revision must be non-empty or null"
                )
            traceability = source.get("traceability")
            if traceability not in {
                "audit_revision_only",
                "public_revision",
                "public_repository_unpinned",
                "local_snapshot_only",
                "paper_only",
                "repository_unavailable",
            }:
                errors.append(f"method status {slug}: invalid source.traceability")
            if traceability == "public_revision" and (url is None or revision is None):
                errors.append(
                    f"method status {slug}: public_revision requires URL and revision"
                )
            if traceability == "public_repository_unpinned" and (
                url is None or revision is not None
            ):
                errors.append(
                    f"method status {slug}: public_repository_unpinned requires URL and null revision"
                )
            if traceability == "repository_unavailable" and (
                url is None or revision is not None
            ):
                errors.append(
                    f"method status {slug}: repository_unavailable requires URL and null revision"
                )
            if traceability == "audit_revision_only" and (
                url is None or revision is None
            ):
                errors.append(
                    f"method status {slug}: audit_revision_only requires URL and revision"
                )
            actual_source = (url, revision, traceability)
            if actual_source != EXPECTED_METHOD_SOURCES.get(slug):
                errors.append(
                    f"method status {slug}: source metadata must match the audited freeze "
                    f"({EXPECTED_METHOD_SOURCES.get(slug)!r})"
                )

        validation = record.get("validation")
        if not isinstance(validation, dict):
            errors.append(f"method status {slug}: validation must be an object")
        else:
            verification_state = validation.get("state")
            if verification_state not in ALLOWED_VERIFICATION_STATES:
                errors.append(f"method status {slug}: invalid validation.state")
            expected_verification_state = (
                "partial" if slug in EXPECTED_PARTIAL_METHODS else "historical_evidence"
            )
            if verification_state != expected_verification_state:
                errors.append(
                    f"method status {slug}: validation.state must match the audited freeze "
                    f"({expected_verification_state!r})"
                )
            if not isinstance(validation.get("public_evidence"), bool):
                errors.append(f"method status {slug}: public_evidence must be boolean")
            if validation.get("public_evidence"):
                errors.append(
                    f"method status {slug}: public_evidence cannot be true while referenced "
                    "raw .refs/runs artifacts are absent"
                )
            runtime_state = validation.get("runtime_state")
            if runtime_state not in ALLOWED_RUNTIME_STATES:
                errors.append(f"method status {slug}: invalid validation.runtime_state")
            if runtime_state != EXPECTED_RUNTIME_STATES.get(slug):
                errors.append(
                    f"method status {slug}: runtime_state must match the audited freeze "
                    f"({EXPECTED_RUNTIME_STATES.get(slug)!r})"
                )
            limitations = validation.get("limitations")
            if (
                not isinstance(limitations, list)
                or not limitations
                or not all(
                    isinstance(item, str) and item.strip() for item in limitations
                )
            ):
                errors.append(
                    f"method status {slug}: limitations must be non-empty strings"
                )
            elif slug in REQUIRED_LIMITATION_TERMS:
                joined = " ".join(limitations).casefold()
                missing_terms = [
                    term
                    for term in REQUIRED_LIMITATION_TERMS[slug]
                    if term.casefold() not in joined
                ]
                if missing_terms:
                    errors.append(
                        f"method status {slug}: limitations omit audited terms "
                        f"{missing_terms!r}"
                    )

        review = record.get("license_review")
        if not isinstance(review, dict):
            errors.append(f"method status {slug}: license_review must be an object")
        else:
            status = review.get("status")
            if status not in ALLOWED_REVIEW_STATES:
                errors.append(f"method status {slug}: invalid license_review.status")
            if review.get("approval_id") != "APP-THIRD-PARTY":
                errors.append(
                    f"method status {slug}: license review must link APP-THIRD-PARTY"
                )
            if status != "approved" and review.get("release_blocking") is not True:
                errors.append(
                    f"method status {slug}: non-approved license review must block release"
                )
            if status == "approved":
                if review.get("release_blocking") is not False:
                    errors.append(
                        f"method status {slug}: approved license review cannot remain blocking"
                    )
                if not _valid_terminal_evidence(review.get("evidence_reference")):
                    errors.append(
                        f"method status {slug}: approved license review needs evidence; placeholders are not accepted"
                    )
            elif (
                status == "pending_external_review"
                and review.get("evidence_reference") is not None
            ):
                errors.append(
                    f"method status {slug}: pending license review cannot claim evidence"
                )
            elif status == "rejected" and not _valid_terminal_evidence(
                review.get("evidence_reference")
            ):
                errors.append(
                    f"method status {slug}: rejected license review needs non-placeholder evidence"
                )
    return errors


def validate_method_status(root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    document = _read_json(root, METHOD_STATUS_PATH, errors)
    if document:
        errors.extend(
            validate_method_status_document(document, _load_inventory(root), root)
        )
    return errors


def validate_alignment_exceptions_document(
    document: dict[str, Any],
    root: Path = ROOT,
) -> list[str]:
    errors: list[str] = []
    if digest_error := _manifest_digest_error(document, ALIGNMENT_EXCEPTIONS_PATH):
        errors.append(digest_error)
    if document.get("schema_version") != 1:
        errors.append("alignment exceptions: schema_version must be 1")
    records = document.get("exceptions")
    if not isinstance(records, list) or not records:
        return errors + ["alignment exceptions: exceptions must be a non-empty list"]
    required_ids = {
        "ALIGN-ABSENT-EVIDENCE",
        "ALIGN-BROKEN-LINKS",
        "ALIGN-CLEAN-CLONE",
        "ALIGN-PAPER-LINKS",
        "ALIGN-PARTIAL-VALIDATION",
    }
    ids = {record.get("id") for record in records if isinstance(record, dict)}
    if not required_ids <= ids:
        errors.append(
            f"alignment exceptions: missing required ids {sorted(required_ids - ids)}"
        )
    valid_slugs = {entry.slug for entry in _load_inventory(root)}
    method_status_records = json.loads(
        (root / METHOD_STATUS_PATH).read_text(encoding="utf-8")
    )["methods"]
    for record in records:
        if not isinstance(record, dict):
            errors.append("alignment exceptions: every entry must be an object")
            continue
        identifier = record.get("id", "<missing>")
        methods = record.get("methods")
        if methods != "all_37":
            if not isinstance(methods, list) or not methods:
                errors.append(
                    f"alignment exception {identifier}: methods must be all_37 or list"
                )
            elif not set(methods) <= valid_slugs:
                errors.append(
                    f"alignment exception {identifier}: unknown methods "
                    f"{sorted(set(methods) - valid_slugs)}"
                )
        for key in ("kind", "observed", "public_disposition", "resolution_goal"):
            if not isinstance(record.get(key), str) or not record[key].strip():
                errors.append(
                    f"alignment exception {identifier}: {key} must be non-empty"
                )
        blocking = record.get("release_blocking")
        status = record.get("status")
        if identifier == "ALIGN-PAPER-LINKS":
            mismatches = _method_readme_paper_link_mismatches(
                root, _load_inventory(root)
            )
            if status == "open" and isinstance(methods, list):
                if set(methods) != set(mismatches):
                    errors.append(
                        "alignment exception ALIGN-PAPER-LINKS: methods must match "
                        f"the README scan ({sorted(mismatches)!r})"
                    )
            if status == "resolved" and mismatches:
                errors.append(
                    "alignment exception ALIGN-PAPER-LINKS: "
                    f"{len(mismatches)} README paper-link mismatches remain"
                )
        if identifier == "ALIGN-ABSENT-EVIDENCE" and status == "resolved":
            internal_mentions = _undistributed_alignment_artifact_mentions(root)
            if internal_mentions:
                errors.append(
                    "alignment exception ALIGN-ABSENT-EVIDENCE: "
                    f"{len(internal_mentions)} undistributed artifact markers remain"
                )
        if identifier == "ALIGN-BROKEN-LINKS" and isinstance(methods, list):
            broken_link_methods = {
                Path(item.split(" -> ", 1)[0]).stem
                for item in _broken_alignment_links(root)
            }
            if status == "open" and set(methods) != broken_link_methods:
                errors.append(
                    "alignment exception ALIGN-BROKEN-LINKS: methods must match "
                    f"the broken-link scan ({sorted(broken_link_methods)!r})"
                )
        if identifier == "ALIGN-CLEAN-CLONE" and isinstance(methods, list):
            expected_clean_clone_exceptions = {
                item["slug"]
                for item in method_status_records
                if item["validation"]["runtime_state"] != "not_assessed"
            } | CLEAN_CLONE_MANUAL_METHODS
            if set(methods) != expected_clean_clone_exceptions:
                errors.append(
                    "alignment exception ALIGN-CLEAN-CLONE: methods must match "
                    "non-clean-clone runtime states plus documented manual exceptions "
                    f"({sorted(expected_clean_clone_exceptions)!r})"
                )
        if identifier == "ALIGN-PARTIAL-VALIDATION" and isinstance(methods, list):
            if set(methods) != EXPECTED_PARTIAL_METHODS:
                errors.append(
                    "alignment exception ALIGN-PARTIAL-VALIDATION: methods must match "
                    f"the partial validation state set ({sorted(EXPECTED_PARTIAL_METHODS)!r})"
                )
        if not isinstance(blocking, bool):
            errors.append(
                f"alignment exception {identifier}: release_blocking must be boolean"
            )
        if blocking:
            if status != "open":
                errors.append(
                    f"alignment exception {identifier}: blocker must have status open"
                )
            if record.get("resolution_evidence") is not None:
                errors.append(
                    f"alignment exception {identifier}: open blocker cannot claim resolution evidence"
                )
            continue

        evidence = record.get("resolution_evidence")
        if not _valid_terminal_evidence(evidence):
            errors.append(
                f"alignment exception {identifier}: non-blocker needs non-placeholder resolution evidence"
            )
        if status == "accepted_public_limitation":
            if identifier not in ACCEPTED_LIMITATION_IDS:
                errors.append(
                    f"alignment exception {identifier}: cannot be an accepted public limitation"
                )
        elif status != "resolved":
            errors.append(
                f"alignment exception {identifier}: non-blocker must be resolved"
            )
        if identifier in MACHINE_GATED_ALIGNMENT_IDS:
            errors.append(
                f"alignment exception {identifier}: requires a goal-specific machine gate before closure"
            )
        if identifier == "ALIGN-BROKEN-LINKS" and status == "resolved":
            broken = _broken_alignment_links(root)
            if broken:
                errors.append(
                    f"alignment exception {identifier}: {len(broken)} broken relative links remain"
                )
    return errors


def validate_alignment_exceptions(root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    document = _read_json(root, ALIGNMENT_EXCEPTIONS_PATH, errors)
    if document:
        errors.extend(validate_alignment_exceptions_document(document, root))
    return errors


def validate_provenance_document(
    document: dict[str, Any],
    root: Path = ROOT,
) -> list[str]:
    errors: list[str] = []
    if digest_error := _manifest_digest_error(document, PROVENANCE_PATH):
        errors.append(digest_error)
    if document.get("schema_version") != 1:
        errors.append("provenance: schema_version must be 1")
    entries = document.get("entries")
    if not isinstance(entries, list) or not entries:
        return errors + ["provenance: entries must be a non-empty list"]

    ids: set[str] = set()
    entries_by_id: dict[str, dict[str, Any]] = {}
    covered_paths: set[str] = set()
    derived_source_paths: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            errors.append("provenance: every entry must be an object")
            continue
        identifier = entry.get("id")
        if not isinstance(identifier, str) or not identifier:
            errors.append("provenance: every entry needs a non-empty id")
            identifier = "<missing>"
        elif identifier in ids:
            errors.append(f"provenance: duplicate id {identifier}")
        ids.add(identifier)
        entries_by_id[identifier] = entry
        kind = entry.get("kind")
        if not isinstance(kind, str) or not kind.strip():
            errors.append(f"provenance {identifier}: kind must be non-empty")
            kind = ""
        elif kind not in ALLOWED_PROVENANCE_KINDS:
            errors.append(f"provenance {identifier}: invalid provenance kind")
        paths = entry.get("paths")
        if (
            not isinstance(paths, list)
            or not paths
            or not all(isinstance(item, str) and item for item in paths)
        ):
            errors.append(f"provenance {identifier}: paths must be non-empty strings")
            paths = []
        for raw in paths:
            covered_paths.add(raw)
            if kind in DERIVED_PROVENANCE_KINDS:
                derived_source_paths.add(raw)
            if not (root / raw).exists():
                errors.append(
                    f"provenance {identifier}: tracked path does not exist: {raw}"
                )

        ranges = entry.get("ranges")
        if (
            not isinstance(ranges, list)
            or not ranges
            or not all(isinstance(item, str) and item for item in ranges)
        ):
            errors.append(f"provenance {identifier}: ranges must be non-empty strings")
            ranges = []
        for raw in paths:
            matching_ranges = [item for item in ranges if item.startswith(f"{raw}:")]
            if not matching_ranges:
                errors.append(
                    f"provenance {identifier}: no range or whole-file scope for {raw}"
                )
            else:
                for item in matching_ranges:
                    scope = item.removeprefix(f"{raw}:")
                    numeric_scope = re.match(
                        r"^((?:[0-9]+(?:-(?:[0-9]+|end))?)"
                        r"(?:,(?:[0-9]+(?:-(?:[0-9]+|end))?))*)",
                        scope,
                    )
                    valid_scope = bool(
                        numeric_scope
                        and (
                            len(scope) == len(numeric_scope.group(1))
                            or scope[len(numeric_scope.group(1))] in {" ", "<", ";"}
                        )
                    ) or bool(
                        re.match(
                            r"^(?:entire (?:file|adapted detector path|adapted wrapper|"
                            r"official-semantics adapter|ViTAD-specific sampler path)\b|"
                            r"whole file\b|paper-based implementation\b)",
                            scope,
                        )
                    )
                    if not valid_scope:
                        errors.append(
                            f"provenance {identifier}: invalid range or whole-file scope for {raw}"
                        )
                        continue
                    if numeric_scope and (root / raw).suffix == ".py":
                        line_count = len(
                            (root / raw).read_text(encoding="utf-8").splitlines()
                        )
                        line_numbers = [
                            int(value)
                            for value in re.findall(r"[0-9]+", numeric_scope.group(1))
                        ]
                        if any(
                            value < 1 or value > line_count for value in line_numbers
                        ):
                            errors.append(
                                f"provenance {identifier}: range exceeds file length for {raw}"
                            )

        source_revision: Any = None
        source_url: Any = None
        source = entry.get("source")
        if not isinstance(source, dict) or set(source) != {"name", "url", "revision"}:
            errors.append(
                f"provenance {identifier}: source fields must be name/url/revision"
            )
        else:
            name = source.get("name")
            if not isinstance(name, str) or not name.strip():
                errors.append(f"provenance {identifier}: source.name must be non-empty")
            source_url = source.get("url")
            if source_url is not None and (
                not isinstance(source_url, str) or not source_url.startswith("https://")
            ):
                errors.append(
                    f"provenance {identifier}: source.url must be HTTPS or null"
                )
            source_revision = source.get("revision")
            if source_revision is not None and (
                not isinstance(source_revision, str) or not source_revision.strip()
            ):
                errors.append(
                    f"provenance {identifier}: source.revision must be non-empty or null"
                )

        license_record = entry.get("license")
        if not isinstance(license_record, dict):
            errors.append(f"provenance {identifier}: license must be an object")
            license_state = None
        else:
            license_state = license_record.get("status")
            if license_state not in ALLOWED_LICENSE_STATES:
                errors.append(f"provenance {identifier}: invalid license.status")
            for key in ("spdx", "evidence"):
                if key not in license_record:
                    errors.append(f"provenance {identifier}: license.{key} is required")
            if (
                not isinstance(license_record.get("evidence"), str)
                or not license_record["evidence"].strip()
            ):
                errors.append(
                    f"provenance {identifier}: license.evidence must be non-empty"
                )
            spdx = license_record.get("spdx")
            if license_state == "confirmed" and (
                not isinstance(spdx, str)
                or not spdx.strip()
                or _is_bare_placeholder(spdx)
            ):
                errors.append(
                    f"provenance {identifier}: confirmed license needs a non-placeholder SPDX expression"
                )
            if (
                isinstance(spdx, str)
                and spdx.startswith("AGPL-")
                and license_state != "incompatible"
            ):
                errors.append(
                    f"provenance {identifier}: AGPL SPDX must remain incompatible"
                )

        incorporation = entry.get("incorporation")
        incorporation_mode = None
        if (
            not isinstance(incorporation, dict)
            or not isinstance(incorporation.get("mode"), str)
            or not isinstance(incorporation.get("modified"), bool)
        ):
            errors.append(
                f"provenance {identifier}: incorporation needs mode and modified boolean"
            )
        else:
            incorporation_mode = incorporation.get("mode")
            if incorporation_mode not in ALLOWED_INCORPORATION_MODES:
                errors.append(f"provenance {identifier}: invalid incorporation mode")
        disposition = entry.get("disposition")
        if disposition not in ALLOWED_DISPOSITIONS:
            errors.append(f"provenance {identifier}: invalid disposition")
        obligations = entry.get("obligations")
        if (
            not isinstance(obligations, list)
            or not obligations
            or not all(isinstance(item, str) and item.strip() for item in obligations)
        ):
            errors.append(
                f"provenance {identifier}: obligations must be non-empty strings"
            )

        signoff = entry.get("reviewer_signoff")
        if not isinstance(signoff, dict):
            errors.append(
                f"provenance {identifier}: reviewer_signoff must be an object"
            )
            signoff_status = None
        else:
            signoff_status = signoff.get("status")
            if signoff_status not in ALLOWED_REVIEW_STATES:
                errors.append(
                    f"provenance {identifier}: invalid reviewer signoff status"
                )
            if not isinstance(signoff.get("role"), str) or not signoff["role"].strip():
                errors.append(
                    f"provenance {identifier}: reviewer role must be non-empty"
                )
            if signoff.get("approval_id") != "APP-THIRD-PARTY":
                errors.append(
                    f"provenance {identifier}: signoff must link APP-THIRD-PARTY"
                )

        if signoff_status == "approved":
            if license_state != "confirmed":
                errors.append(
                    f"provenance {identifier}: approved signoff requires confirmed license status"
                )
            if not _valid_terminal_evidence(signoff.get("evidence_reference")):
                errors.append(
                    f"provenance {identifier}: approved signoff needs evidence; placeholders are not accepted"
                )
            if _is_bare_placeholder(signoff.get("role")):
                errors.append(
                    f"provenance {identifier}: approved signoff needs a real reviewer role"
                )
            is_code_or_test = kind in DERIVED_PROVENANCE_KINDS
            if is_code_or_test and (
                not isinstance(source_revision, str)
                or re.fullmatch(r"[0-9a-f]{40}", source_revision) is None
                or len(set(source_revision)) == 1
            ):
                errors.append(
                    f"provenance {identifier}: approved code/test provenance needs a pinned source revision"
                )
            if is_code_or_test and (
                not isinstance(source_url, str)
                or not source_url.startswith("https://github.com/")
            ):
                errors.append(
                    f"provenance {identifier}: approved code/test provenance needs an audited GitHub source URL"
                )
            if is_code_or_test and disposition != "keep_with_attribution":
                errors.append(
                    f"provenance {identifier}: approved retained code/test provenance must use keep_with_attribution"
                )
            if is_code_or_test and (
                not isinstance(incorporation_mode, str)
                or "external_reference" in incorporation_mode
                or "not_bundled" in incorporation_mode
            ):
                errors.append(
                    f"provenance {identifier}: approved bundled code/test needs a derived-code incorporation mode"
                )
            if any(APPROVAL_PLACEHOLDER.search(item) for item in ranges):
                errors.append(
                    f"provenance {identifier}: approved provenance cannot retain pending range placeholders"
                )
            if any(_is_bare_placeholder(item) for item in obligations):
                errors.append(
                    f"provenance {identifier}: approved provenance cannot retain placeholder obligations"
                )
            obligations_evidence = entry.get("obligations_evidence")
            if (
                not isinstance(obligations_evidence, list)
                or not obligations_evidence
                or not all(
                    _valid_terminal_evidence(item) for item in obligations_evidence
                )
            ):
                errors.append(
                    f"provenance {identifier}: approved provenance needs obligations evidence"
                )
            if disposition == "keep_with_attribution":
                notice_files = entry.get("notice_files")
                if (
                    not isinstance(notice_files, list)
                    or not notice_files
                    or not all(isinstance(item, str) and item for item in notice_files)
                ):
                    errors.append(
                        f"provenance {identifier}: approved attribution needs repository license notice files"
                    )
                else:
                    notice_hashes = entry.get("notice_sha256")
                    if not isinstance(notice_hashes, dict) or set(notice_hashes) != set(
                        notice_files
                    ):
                        errors.append(
                            f"provenance {identifier}: approved attribution needs exact notice SHA-256 values"
                        )
                        notice_hashes = {}
                    for raw in notice_files:
                        notice_path = Path(raw)
                        if (
                            not raw.startswith(".github/release/licenses/")
                            or ".." in notice_path.parts
                            or not (root / notice_path).is_file()
                        ):
                            errors.append(
                                f"provenance {identifier}: invalid repository license notice file {raw!r}"
                            )
                            continue
                        if (root / notice_path).stat().st_size < 200:
                            errors.append(
                                f"provenance {identifier}: repository license notice file is implausibly short: {raw!r}"
                            )
                        expected_notice_hash = notice_hashes.get(raw)
                        if (
                            not isinstance(expected_notice_hash, str)
                            or re.fullmatch(r"[0-9a-f]{64}", expected_notice_hash)
                            is None
                            or _sha256(root / notice_path) != expected_notice_hash
                        ):
                            errors.append(
                                f"provenance {identifier}: repository license notice SHA-256 mismatch for {raw!r}"
                            )
        elif signoff_status == "pending_external_review":
            if signoff.get("evidence_reference") is not None:
                errors.append(
                    f"provenance {identifier}: pending signoff cannot claim evidence"
                )
        elif signoff_status == "rejected" and not _valid_terminal_evidence(
            signoff.get("evidence_reference")
        ):
            errors.append(
                f"provenance {identifier}: rejected signoff needs non-placeholder evidence"
            )

        unresolved = (
            license_state != "confirmed"
            or signoff_status != "approved"
            or entry.get("disposition")
            in {
                "remove_before_release",
                "replace_before_release",
                "rewrite_before_release",
            }
        )
        if unresolved and entry.get("release_blocking") is not True:
            if signoff_status != "approved":
                errors.append(
                    f"provenance {identifier}: unresolved item must block release; "
                    "non-approved signoff must block release"
                )
            else:
                errors.append(
                    f"provenance {identifier}: unresolved item must block release"
                )
        if not unresolved and entry.get("release_blocking") is not False:
            errors.append(
                f"provenance {identifier}: resolved item cannot remain release-blocking"
            )
        if license_state == "incompatible" and entry.get("disposition") not in {
            "remove_before_release",
            "replace_before_release",
            "rewrite_before_release",
        }:
            errors.append(
                f"provenance {identifier}: incompatible code must be removed, replaced, or rewritten"
            )

    missing_secondary_sources = sorted(REQUIRED_SECONDARY_SOURCE_IDS - ids)
    if missing_secondary_sources:
        errors.append(
            "provenance: required secondary-source entries are missing: "
            f"{missing_secondary_sources}"
        )

    for identifier, (
        expected_kind,
        expected_url,
        expected_path,
    ) in EXPECTED_EXTERNAL_ARTIFACTS.items():
        entry = entries_by_id.get(identifier)
        if entry is None:
            errors.append(
                f"provenance: required external artifact entry is missing: {identifier}"
            )
            continue
        source = entry.get("source")
        incorporation = entry.get("incorporation")
        actual_binding = (
            entry.get("kind"),
            source.get("url") if isinstance(source, dict) else None,
            expected_path in entry.get("paths", []),
            entry.get("disposition"),
            incorporation.get("mode") if isinstance(incorporation, dict) else None,
        )
        expected_binding = (
            expected_kind,
            expected_url,
            True,
            "external_reference_only",
            "external_download_reference_not_bundled",
        )
        if actual_binding != expected_binding:
            errors.append(
                f"provenance {identifier}: external artifact binding changed from "
                "the audited source/path/reference-only state"
            )

    required_derived_paths = {
        path for path in REQUIRED_PROVENANCE_PATHS if path.endswith(".py")
    }
    missing_derived = sorted(required_derived_paths - derived_source_paths)
    if missing_derived:
        errors.append(
            "provenance: required derived paths are not covered by code/test "
            f"provenance: {missing_derived}"
        )
    required_other_paths = REQUIRED_PROVENANCE_PATHS - required_derived_paths
    missing_other = sorted(required_other_paths - covered_paths)
    if missing_other:
        errors.append(f"provenance: required paths are not covered: {missing_other}")

    for path in (root / "baoiad").rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if DIRECT_SOURCE_MARKER.search(text):
            relative = path.relative_to(root).as_posix()
            if relative not in derived_source_paths:
                errors.append(
                    f"provenance: copied/ported/vendored marker lacks coverage: {relative}"
                )
    return errors


def validate_provenance(root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    document = _read_json(root, PROVENANCE_PATH, errors)
    if document:
        errors.extend(validate_provenance_document(document, root))
    return errors


def validate_asset_approvals_document(
    document: dict[str, Any],
    root: Path = ROOT,
) -> list[str]:
    errors: list[str] = []
    if digest_error := _manifest_digest_error(document, ASSET_APPROVALS_PATH):
        errors.append(digest_error)
    if document.get("schema_version") != 1:
        errors.append("asset approvals: schema_version must be 1")
    assets = document.get("assets")
    if not isinstance(assets, list):
        return errors + ["asset approvals: assets must be a list"]
    paths = [
        asset.get("path")
        for asset in assets
        if isinstance(asset, dict) and isinstance(asset.get("path"), str)
    ]
    duplicate_paths = sorted({path for path in paths if paths.count(path) > 1})
    if duplicate_paths:
        errors.append(f"asset approvals: duplicate path entries {duplicate_paths}")
    for asset in assets:
        if not isinstance(asset, dict):
            errors.append("asset approvals: every asset must be an object")
            continue
        raw_path = asset.get("path", "<missing>")
        raw_scopes = asset.get("scopes")
        if isinstance(raw_scopes, dict) and not all(
            value is None or isinstance(value, bool) for value in raw_scopes.values()
        ):
            errors.append(f"asset {raw_path}: scope values must be boolean or null")
    by_path = {
        asset.get("path"): asset
        for asset in assets
        if isinstance(asset, dict) and isinstance(asset.get("path"), str)
    }
    if set(by_path) != set(EXPECTED_ASSETS):
        errors.append(
            "asset approvals: path set mismatch; "
            f"missing={sorted(set(EXPECTED_ASSETS) - set(by_path))}, "
            f"extra={sorted(set(by_path) - set(EXPECTED_ASSETS))}"
        )
    for path, expected_hash in EXPECTED_ASSETS.items():
        asset = by_path.get(path)
        if asset is None:
            continue
        actual_hash = _sha256(root / path)
        if actual_hash != expected_hash:
            errors.append(
                f"asset {path}: content changed; expected {expected_hash}, found {actual_hash}"
            )
        if asset.get("sha256") != actual_hash:
            errors.append(f"asset {path}: manifest sha256 does not match file")
        for key in ("media_type", "dimensions", "purpose"):
            if not isinstance(asset.get(key), str) or not asset[key].strip():
                errors.append(f"asset {path}: {key} must be non-empty")
        origin = asset.get("origin")
        if not isinstance(origin, dict):
            errors.append(f"asset {path}: origin must be an object")
            origin = {}
        elif set(origin) != {
            "kind",
            "creator_or_owner",
            "source_url",
            "source_revision",
            "source_file",
            "generator",
            "embedded_third_party_content",
        }:
            errors.append(f"asset {path}: origin fields must match the release schema")
        expected_metadata = EXPECTED_ASSET_METADATA[path]
        actual_metadata = (
            asset.get("media_type"),
            asset.get("dimensions"),
            origin.get("kind"),
            origin.get("source_url"),
            origin.get("source_file"),
        )
        if actual_metadata != expected_metadata:
            errors.append(
                f"asset {path}: media, dimensions, or origin metadata changed from the audited freeze"
            )
        if _canonical_json_digest(origin) != EXPECTED_ASSET_ORIGIN_DIGESTS[path]:
            errors.append(
                f"asset {path}: origin evidence changed from the audited freeze"
            )
        source_file = origin.get("source_file")
        if isinstance(source_file, str) and not (root / source_file).is_file():
            errors.append(f"asset {path}: origin source_file does not exist")
        if not isinstance(origin.get("embedded_third_party_content"), str):
            errors.append(
                f"asset {path}: embedded_third_party_content must be a string"
            )

        rights = asset.get("rights")
        rights_status = None
        if not isinstance(rights, dict):
            errors.append(f"asset {path}: rights must be an object")
        else:
            rights_status = rights.get("status")
            if rights_status not in ALLOWED_REVIEW_STATES:
                errors.append(f"asset {path}: invalid rights status")
            if "evidence_reference" not in rights:
                errors.append(f"asset {path}: rights.evidence_reference is required")
            if rights_status == "approved" and (
                not _valid_terminal_evidence(rights.get("license_or_basis"))
                or not _valid_terminal_evidence(rights.get("evidence_reference"))
            ):
                errors.append(
                    f"asset {path}: approved rights need license basis and evidence"
                )
            if (
                origin.get("creator_or_owner") is not None
                and rights_status != "approved"
            ):
                errors.append(
                    f"asset {path}: creator or owner claim requires approved rights evidence"
                )
            if (
                rights_status == "pending_external_review"
                and rights.get("evidence_reference") is not None
            ):
                errors.append(f"asset {path}: pending rights cannot claim evidence")
            if rights_status == "rejected" and not _valid_terminal_evidence(
                rights.get("evidence_reference")
            ):
                errors.append(f"asset {path}: rejected rights need evidence")

        approvals = asset.get("approvals")
        approval_statuses: list[str | None] = []
        if not isinstance(approvals, dict):
            errors.append(f"asset {path}: approvals must be an object")
        else:
            for role in ("technical", "brand", "legal_or_oss"):
                approval = approvals.get(role)
                if not isinstance(approval, dict):
                    errors.append(f"asset {path}: missing {role} approval")
                    continue
                status = approval.get("status")
                approval_statuses.append(status)
                if status not in ALLOWED_REVIEW_STATES:
                    errors.append(f"asset {path}: invalid {role} approval status")
                if status == "approved" and not _valid_terminal_evidence(
                    approval.get("evidence_reference")
                ):
                    errors.append(
                        f"asset {path}: approved {role} approval needs evidence"
                    )
                if (
                    status == "pending_external_review"
                    and approval.get("evidence_reference") is not None
                ):
                    errors.append(
                        f"asset {path}: pending {role} approval cannot claim evidence"
                    )
                if status == "rejected" and not _valid_terminal_evidence(
                    approval.get("evidence_reference")
                ):
                    errors.append(
                        f"asset {path}: rejected {role} approval needs evidence"
                    )
        scopes = asset.get("scopes")
        if not isinstance(scopes, dict) or set(scopes) != {
            "github_repository",
            "github_social_preview",
            "press_marketing",
            "readthedocs",
            "waic_event",
        }:
            errors.append(f"asset {path}: scopes must enumerate every public surface")
        elif not all(
            value is None or isinstance(value, bool) for value in scopes.values()
        ):
            errors.append(f"asset {path}: scope values must be boolean or null")
        if path.startswith("resources/vis_examples/") and isinstance(scopes, dict):
            if scopes.get("github_social_preview") is not False:
                errors.append(
                    f"asset {path}: MVTec-derived image cannot be a social preview"
                )
        if asset.get("approval_id") != "APP-BRAND-ASSETS":
            errors.append(f"asset {path}: approval_id must be APP-BRAND-ASSETS")
        disposition = asset.get("disposition")
        if disposition not in {
            "approved_for_scopes",
            "omit_unless_approved",
            "replace_before_release",
        }:
            errors.append(f"asset {path}: invalid disposition")

        scopes_resolved = isinstance(scopes, dict) and all(
            isinstance(value, bool) for value in scopes.values()
        )
        unresolved = (
            rights_status != "approved"
            or any(status != "approved" for status in approval_statuses)
            or not scopes_resolved
            or disposition in {"omit_unless_approved", "replace_before_release"}
        )
        if unresolved and asset.get("release_blocking") is not True:
            errors.append(f"asset {path}: unresolved authorization must block release")
        if not unresolved:
            if asset.get("release_blocking") is not False:
                errors.append(
                    f"asset {path}: resolved authorization cannot remain blocking"
                )
            if disposition != "approved_for_scopes":
                errors.append(f"asset {path}: resolved asset needs approved_for_scopes")
            if not any(scopes.values()):
                errors.append(
                    f"asset {path}: approved asset needs at least one true scope"
                )
            if scopes.get("github_repository") is not True:
                errors.append(
                    f"asset {path}: retained file requires github_repository scope"
                )
        elif statuses_resolved := all(
            status == "approved" for status in approval_statuses
        ):
            if (
                rights_status == "approved"
                and statuses_resolved
                and not scopes_resolved
            ):
                errors.append(
                    f"asset {path}: resolved asset scopes must be explicit booleans"
                )
    return errors


def validate_asset_approvals(root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    document = _read_json(root, ASSET_APPROVALS_PATH, errors)
    if document:
        errors.extend(validate_asset_approvals_document(document, root))
    readme = root / "resources" / "README.md"
    if not readme.is_file():
        errors.append("missing resources/README.md")
    else:
        text = readme.read_text(encoding="utf-8")
        for path in EXPECTED_ASSETS:
            if path.removeprefix("resources/") not in text:
                errors.append(f"resources/README.md does not list {path}")
    return errors


def validate_external_approvals_document(document: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if digest_error := _manifest_digest_error(document, EXTERNAL_APPROVALS_PATH):
        errors.append(digest_error)
    if document.get("schema_version") != 1:
        errors.append("external approvals: schema_version must be 1")
    approvals = document.get("approvals")
    if not isinstance(approvals, list):
        return errors + ["external approvals: approvals must be a list"]
    identifiers = [
        item.get("id")
        for item in approvals
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]
    duplicate_ids = sorted(
        {identifier for identifier in identifiers if identifiers.count(identifier) > 1}
    )
    if duplicate_ids:
        errors.append(f"external approvals: duplicate id entries {duplicate_ids}")
    by_id = {
        item.get("id"): item
        for item in approvals
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    if set(by_id) != REQUIRED_APPROVAL_IDS:
        errors.append(
            "external approvals: id set mismatch; "
            f"missing={sorted(REQUIRED_APPROVAL_IDS - set(by_id))}, "
            f"extra={sorted(set(by_id) - REQUIRED_APPROVAL_IDS)}"
        )
    for identifier, approval in by_id.items():
        status = approval.get("status")
        if status not in {"pending", "approved", "rejected"}:
            errors.append(f"external approval {identifier}: invalid status")
        for key in ("subject", "owner_role", "evidence_required", "resolution_goal"):
            if not isinstance(approval.get(key), str) or not approval[key].strip():
                errors.append(
                    f"external approval {identifier}: {key} must be non-empty"
                )
        if status == "pending":
            if approval.get("evidence_reference") is not None:
                errors.append(
                    f"external approval {identifier}: pending item cannot claim evidence"
                )
        if status != "approved" and approval.get("release_blocking") is not True:
            if status == "pending":
                errors.append(
                    f"external approval {identifier}: pending item must block release"
                )
            errors.append(
                f"external approval {identifier}: non-approved item must block release"
            )
        evidence_reference = approval.get("evidence_reference")
        if status in {"approved", "rejected"} and not _valid_terminal_evidence(
            evidence_reference
        ):
            errors.append(
                f"external approval {identifier}: {status} item needs a string evidence reference"
            )
        if status == "approved" and approval.get("release_blocking") is not False:
            errors.append(
                f"external approval {identifier}: approved item cannot remain release-blocking"
            )
    return errors


def validate_external_approvals(root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    document = _read_json(root, EXTERNAL_APPROVALS_PATH, errors)
    if document:
        errors.extend(validate_external_approvals_document(document))
    return errors


def validate_human_notices(root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    path = root / "THIRD_PARTY_NOTICES.md"
    if not path.is_file():
        return ["missing THIRD_PARTY_NOTICES.md"]
    text = path.read_text(encoding="utf-8")
    for term in (PROVENANCE_PATH.as_posix(), "Apache-2.0"):
        if term not in text:
            errors.append(f"THIRD_PARTY_NOTICES.md must mention {term!r}")
    provenance_path = root / PROVENANCE_PATH
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        entries = provenance.get("entries", [])
        if (
            any(
                isinstance(entry, dict)
                and isinstance(entry.get("reviewer_signoff"), dict)
                and entry["reviewer_signoff"].get("status") == "pending_external_review"
                for entry in entries
            )
            and "pending external review" not in text
        ):
            errors.append(
                "THIRD_PARTY_NOTICES.md must mention 'pending external review' while reviews are pending"
            )
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            signoff = entry.get("reviewer_signoff")
            source = entry.get("source")
            if (
                isinstance(signoff, dict)
                and signoff.get("status") == "approved"
                and entry.get("disposition") == "keep_with_attribution"
                and isinstance(source, dict)
            ):
                source_name = source.get("name")
                if (
                    isinstance(source_name, str)
                    and source_name.strip()
                    and source_name not in text
                ):
                    errors.append(
                        f"THIRD_PARTY_NOTICES.md must name approved source {source_name!r}"
                    )
    return errors


def validate_all(root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    errors.extend(validate_license(root))
    errors.extend(validate_method_status(root))
    errors.extend(validate_alignment_exceptions(root))
    errors.extend(validate_provenance(root))
    errors.extend(validate_asset_approvals(root))
    errors.extend(validate_external_approvals(root))
    errors.extend(validate_human_notices(root))
    return errors


def open_release_blockers(root: Path = ROOT) -> list[str]:
    blockers: list[str] = []
    approvals = json.loads((root / EXTERNAL_APPROVALS_PATH).read_text(encoding="utf-8"))
    for item in approvals["approvals"]:
        if item["release_blocking"] or item["status"] != "approved":
            blockers.append(f"external approval {item['id']}: {item['status']}")

    methods = json.loads((root / METHOD_STATUS_PATH).read_text(encoding="utf-8"))
    for item in methods["methods"]:
        review = item["license_review"]
        if review["status"] != "approved" or review["release_blocking"]:
            blockers.append(f"method {item['slug']} license review: {review['status']}")

    provenance = json.loads((root / PROVENANCE_PATH).read_text(encoding="utf-8"))
    for item in provenance["entries"]:
        unresolved_disposition = item["disposition"] in {
            "remove_before_release",
            "replace_before_release",
            "rewrite_before_release",
        }
        unresolved_license = item["license"]["status"] != "confirmed"
        unresolved_signoff = item["reviewer_signoff"]["status"] != "approved"
        if item["release_blocking"] or (
            unresolved_disposition or unresolved_license or unresolved_signoff
        ):
            blockers.append(f"provenance {item['id']}: unresolved")

    assets = json.loads((root / ASSET_APPROVALS_PATH).read_text(encoding="utf-8"))
    for item in assets["assets"]:
        rights_resolved = (
            item["rights"]["status"] == "approved"
            and bool(item["rights"].get("license_or_basis"))
            and bool(item["rights"].get("evidence_reference"))
        )
        approvals_resolved = all(
            record["status"] == "approved" and bool(record.get("evidence_reference"))
            for record in item["approvals"].values()
        )
        scopes_resolved = all(
            isinstance(value, bool) for value in item["scopes"].values()
        )
        disposition_resolved = item["disposition"] == "approved_for_scopes"
        if (
            item["release_blocking"]
            or not rights_resolved
            or not approvals_resolved
            or not scopes_resolved
            or not disposition_resolved
        ):
            blockers.append(f"asset {item['path']}: unresolved")

    exceptions = json.loads(
        (root / ALIGNMENT_EXCEPTIONS_PATH).read_text(encoding="utf-8")
    )
    for item in exceptions["exceptions"]:
        if item["release_blocking"] or item["status"] == "open":
            blockers.append(f"alignment exception {item['id']}: open")
    return blockers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-gate",
        action="store_true",
        help="also fail when any recorded release blocker remains unresolved",
    )
    args = parser.parse_args(argv)

    errors = validate_all(ROOT)
    if errors:
        print("FAIL release compliance inventory validation")
        for error in errors:
            print(f"- {error}")
        return 1

    blockers = open_release_blockers(ROOT)
    if args.release_gate and blockers:
        print("FAIL public release gate")
        for blocker in blockers:
            print(f"- {blocker}")
        return 1

    print("PASS release compliance inventory validation")
    print("method records: 37")
    print(f"open release blockers: {len(blockers)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
