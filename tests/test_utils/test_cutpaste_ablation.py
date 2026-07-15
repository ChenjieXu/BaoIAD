"""Tests for CutPaste ablation tooling."""

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "cutpaste_ablation.py"
pytestmark = pytest.mark.optional
if not TOOL.is_file():
    pytest.skip(
        "legacy research-only ablation tool is excluded from the public release",
        allow_module_level=True,
    )


def _load_module():
    spec = importlib.util.spec_from_file_location("baoiad_cutpaste_ablation", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_cutpaste_baseline_overrides():
    module = _load_module()
    overrides = module.build_cutpaste_variant_overrides(
        "bottle",
        data_root="data/mvtec_ad",
        variant="baseline",
    )

    assert overrides["train_cfg.val_interval"] == 256
    assert overrides["train_dataloader.dataset.dataset.cls_names"] == ["bottle"]
    assert overrides["test_dataloader.dataset.cls_names"] == ["bottle"]
    assert overrides["default_hooks.logger.interval"] == 256


def test_build_cutpaste_val10_overrides():
    module = _load_module()
    overrides = module.build_cutpaste_variant_overrides(
        "hazelnut",
        data_root="data/mvtec_ad",
        variant="val10",
    )

    assert overrides["train_cfg.val_interval"] == 10
    assert "model.backbone.checkpoint_path" not in overrides


def test_build_cutpaste_localpth_overrides():
    module = _load_module()
    overrides = module.build_cutpaste_variant_overrides(
        "carpet",
        data_root="data/mvtec_ad",
        variant="localpth",
        local_checkpoint_path="/tmp/tf_efficientnet_b4_aa-818f208c.pth",
    )

    assert overrides["train_cfg.val_interval"] == 256
    assert overrides["model.backbone.pretrained"] is False
    assert (
        overrides["model.backbone.checkpoint_path"]
        == "/tmp/tf_efficientnet_b4_aa-818f208c.pth"
    )


def test_build_cutpaste_localpth_val10_overrides():
    module = _load_module()
    overrides = module.build_cutpaste_variant_overrides(
        "screw",
        data_root="data/mvtec_ad",
        variant="localpth_val10",
        local_checkpoint_path="/tmp/tf_efficientnet_b4_aa-818f208c.pth",
    )

    assert overrides["train_cfg.val_interval"] == 10
    assert overrides["model.backbone.pretrained"] is False
    assert (
        overrides["model.backbone.checkpoint_path"]
        == "/tmp/tf_efficientnet_b4_aa-818f208c.pth"
    )


def test_build_cutpaste_localpth_gde_prelogits_overrides():
    module = _load_module()
    overrides = module.build_cutpaste_variant_overrides(
        "hazelnut",
        data_root="data/mvtec_ad",
        variant="localpth_gde_prelogits",
        local_checkpoint_path="/tmp/tf_efficientnet_b4_aa-818f208c.pth",
    )

    assert overrides["train_cfg.val_interval"] == 256
    assert overrides["model.backbone.pretrained"] is False
    assert (
        overrides["model.backbone.checkpoint_path"]
        == "/tmp/tf_efficientnet_b4_aa-818f208c.pth"
    )
    assert overrides["model.backbone.features_only"] is False
    assert overrides["model.train_embedding_source"] == "features_only"
    assert overrides["model.density_embedding_source"] == "pre_logits"


def test_build_cutpaste_localpth_val10_gde_prelogits_overrides():
    module = _load_module()
    overrides = module.build_cutpaste_variant_overrides(
        "screw",
        data_root="data/mvtec_ad",
        variant="localpth_val10_gde_prelogits",
        local_checkpoint_path="/tmp/tf_efficientnet_b4_aa-818f208c.pth",
    )

    assert overrides["train_cfg.val_interval"] == 10
    assert overrides["model.backbone.pretrained"] is False
    assert (
        overrides["model.backbone.checkpoint_path"]
        == "/tmp/tf_efficientnet_b4_aa-818f208c.pth"
    )
    assert overrides["model.backbone.features_only"] is False
    assert overrides["model.train_embedding_source"] == "features_only"
    assert overrides["model.density_embedding_source"] == "pre_logits"


def test_build_cutpaste_localpth_prelogits_overrides():
    module = _load_module()
    overrides = module.build_cutpaste_variant_overrides(
        "hazelnut",
        data_root="data/mvtec_ad",
        variant="localpth_prelogits",
        local_checkpoint_path="/tmp/tf_efficientnet_b4_aa-818f208c.pth",
    )

    assert overrides["train_cfg.val_interval"] == 256
    assert overrides["model.backbone.pretrained"] is False
    assert (
        overrides["model.backbone.checkpoint_path"]
        == "/tmp/tf_efficientnet_b4_aa-818f208c.pth"
    )
    assert overrides["model.backbone.features_only"] is False
    assert overrides["model.train_embedding_source"] == "pre_logits"
    assert overrides["model.density_embedding_source"] == "pre_logits"


def test_build_cutpaste_localpth_val10_prelogits_overrides():
    module = _load_module()
    overrides = module.build_cutpaste_variant_overrides(
        "screw",
        data_root="data/mvtec_ad",
        variant="localpth_val10_prelogits",
        local_checkpoint_path="/tmp/tf_efficientnet_b4_aa-818f208c.pth",
    )

    assert overrides["train_cfg.val_interval"] == 10
    assert overrides["model.backbone.pretrained"] is False
    assert (
        overrides["model.backbone.checkpoint_path"]
        == "/tmp/tf_efficientnet_b4_aa-818f208c.pth"
    )
    assert overrides["model.backbone.features_only"] is False
    assert overrides["model.train_embedding_source"] == "pre_logits"
    assert overrides["model.density_embedding_source"] == "pre_logits"


def test_build_cutpaste_localpth_layer3_overrides():
    module = _load_module()
    overrides = module.build_cutpaste_variant_overrides(
        "carpet",
        data_root="data/mvtec_ad",
        variant="localpth_layer3",
        local_checkpoint_path="/tmp/tf_efficientnet_b4_aa-818f208c.pth",
    )

    assert overrides["train_cfg.val_interval"] == 256
    assert overrides["model.backbone.pretrained"] is False
    assert (
        overrides["model.backbone.checkpoint_path"]
        == "/tmp/tf_efficientnet_b4_aa-818f208c.pth"
    )
    assert overrides["model.backbone.out_indices"] == [3]


def test_build_cutpaste_localpth_val10_layer3_overrides():
    module = _load_module()
    overrides = module.build_cutpaste_variant_overrides(
        "screw",
        data_root="data/mvtec_ad",
        variant="localpth_val10_layer3",
        local_checkpoint_path="/tmp/tf_efficientnet_b4_aa-818f208c.pth",
    )

    assert overrides["train_cfg.val_interval"] == 10
    assert overrides["model.backbone.pretrained"] is False
    assert (
        overrides["model.backbone.checkpoint_path"]
        == "/tmp/tf_efficientnet_b4_aa-818f208c.pth"
    )
    assert overrides["model.backbone.out_indices"] == [3]


def test_build_cutpaste_scar_boost_overrides():
    module = _load_module()
    overrides = module.build_cutpaste_variant_overrides(
        "carpet",
        data_root="data/mvtec_ad",
        variant="scar_boost",
    )

    assert overrides["train_cfg.val_interval"] == 256
    assert overrides["model.scar_width"] == [6, 24]
    assert overrides["model.scar_length"] == [16, 48]
    assert overrides["model.scar_min_changed_ratio"] == 0.005
    assert overrides["model.scar_max_attempts"] == 8


def test_build_cutpaste_localpth_scar_boost_overrides():
    module = _load_module()
    overrides = module.build_cutpaste_variant_overrides(
        "screw",
        data_root="data/mvtec_ad",
        variant="localpth_scar_boost",
        local_checkpoint_path="/tmp/tf_efficientnet_b4_aa-818f208c.pth",
    )

    assert overrides["train_cfg.val_interval"] == 256
    assert overrides["model.backbone.pretrained"] is False
    assert (
        overrides["model.backbone.checkpoint_path"]
        == "/tmp/tf_efficientnet_b4_aa-818f208c.pth"
    )
    assert overrides["model.scar_width"] == [6, 24]
    assert overrides["model.scar_length"] == [16, 48]
    assert overrides["model.scar_min_changed_ratio"] == 0.005
    assert overrides["model.scar_max_attempts"] == 8


def test_safe_probability_metric_returns_none_on_invalid_scores():
    module = _load_module()

    def _raise(_):
        raise ValueError("bad")

    result = module._safe_probability_metric(_raise, [1.2])

    assert result is None
