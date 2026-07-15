"""Tests for CutPaste checkpoint diagnose helpers."""

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "cutpaste_checkpoint_diagnose.py"
pytestmark = pytest.mark.optional
if not TOOL.is_file():
    pytest.skip(
        "legacy research-only diagnostic tool is excluded from the public release",
        allow_module_level=True,
    )


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "baoiad_cutpaste_checkpoint_diagnose", TOOL
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_checkpoint_label_prefers_iter_token():
    module = _load_module()

    assert module._checkpoint_label("/tmp/iter_20.pth") == "iter_20"
    assert module._checkpoint_label("/tmp/epoch_1.pth") == "epoch_1"
    assert module._checkpoint_label("/tmp/model_final.pth") == "model_final"


def test_find_first_stop_line_label_detects_low_auroc():
    module = _load_module()
    ordered = [
        {
            "label": "iter_10",
            "metrics": {"image_auroc": 0.82, "pixel_auroc": 0.61},
            "score_gap": {
                "score_gap_mean": 0.9,
                "normal": {"mean": 0.1},
                "anomaly": {"mean": 1.0},
            },
        },
        {
            "label": "iter_20",
            "metrics": {"image_auroc": 0.17, "pixel_auroc": 0.41},
            "score_gap": {
                "score_gap_mean": -0.2,
                "normal": {"mean": 0.8},
                "anomaly": {"mean": 0.6},
            },
        },
    ]

    assert module._find_first_stop_line_label(ordered) == "iter_20"


def test_build_transition_summary_reports_metric_deltas():
    module = _load_module()
    ordered = [
        {
            "label": "iter_10",
            "metrics": {"image_auroc": 0.82, "pixel_auroc": 0.61},
            "score_gap": {
                "score_gap_mean": 0.9,
                "normal": {"mean": 0.1},
                "anomaly": {"mean": 1.0},
            },
            "gde": {
                "fit_num_samples": 5,
                "mean": {"mean": 0.2, "std": 0.3},
                "cov_inv": {"mean": 0.4, "std": 0.5},
            },
            "embeddings": {
                "train_raw": {"l2_norm": {"mean": 2.0}},
                "train_normalized": {"per_dim_var": {"mean": 0.02}},
                "test_good_raw": {"l2_norm": {"mean": 1.5}},
                "test_good_normalized": {"per_dim_var": {"mean": 0.01}},
                "test_anomaly_raw": {"l2_norm": {"mean": 3.0}},
                "test_anomaly_normalized": {"per_dim_var": {"mean": 0.04}},
            },
        },
        {
            "label": "iter_20",
            "metrics": {"image_auroc": 0.17, "pixel_auroc": 0.41},
            "score_gap": {
                "score_gap_mean": -0.2,
                "normal": {"mean": 0.8},
                "anomaly": {"mean": 0.6},
            },
            "gde": {
                "fit_num_samples": 7,
                "mean": {"mean": 0.5, "std": 0.1},
                "cov_inv": {"mean": 0.8, "std": 0.9},
            },
            "embeddings": {
                "train_raw": {"l2_norm": {"mean": 2.5}},
                "train_normalized": {"per_dim_var": {"mean": 0.03}},
                "test_good_raw": {"l2_norm": {"mean": 1.8}},
                "test_good_normalized": {"per_dim_var": {"mean": 0.02}},
                "test_anomaly_raw": {"l2_norm": {"mean": 2.2}},
                "test_anomaly_normalized": {"per_dim_var": {"mean": 0.05}},
            },
        },
    ]

    transitions = module._build_transition_summary(ordered)

    assert transitions["iter_10 -> iter_20"]["image_auroc_delta"] == -0.6499999999999999
    assert transitions["iter_10 -> iter_20"]["score_gap_delta"] == -1.1
    assert (
        transitions["iter_10 -> iter_20"]["drift_stats_delta"][
            "gde_fit_num_samples_delta"
        ]
        == 2.0
    )
    assert (
        transitions["iter_10 -> iter_20"]["drift_stats_delta"][
            "test_anomaly_raw_l2_mean_delta"
        ]
        == -0.7999999999999998
    )


def test_build_compare_summary_keeps_source_label():
    module = _load_module()
    ordered = [
        {
            "label": "iter_10",
            "checkpoint_path": "/tmp/iter_10.pth",
            "metrics": {"image_auroc": 0.82},
            "score_gap": {
                "score_gap_mean": 0.9,
                "normal": {"mean": 0.1},
                "anomaly": {"mean": 1.0},
            },
            "gde": {
                "fit_num_samples": 5,
                "mean": {"mean": 0.2, "std": 0.3},
                "cov_inv": {"mean": 0.4, "std": 0.5},
            },
            "embeddings": {
                "train_raw": {"l2_norm": {"mean": 2.0}},
                "train_normalized": {"per_dim_var": {"mean": 0.02}},
                "test_good_raw": {"l2_norm": {"mean": 1.5}},
                "test_good_normalized": {"per_dim_var": {"mean": 0.01}},
                "test_anomaly_raw": {"l2_norm": {"mean": 3.0}},
                "test_anomaly_normalized": {"per_dim_var": {"mean": 0.04}},
            },
        },
    ]

    compare = module._build_compare_summary(
        class_name="screw",
        config_path="/tmp/cutpaste.py",
        ordered_results=ordered,
        source_label="baoiad",
    )

    assert compare["class_name"] == "screw"
    assert compare["source_label"] == "baoiad"
    assert compare["ordered_checkpoint_labels"] == ["iter_10"]
    assert compare["checkpoints"]["iter_10"]["score_gap_mean"] == 0.9
    assert (
        compare["checkpoints"]["iter_10"]["drift_stats"]["gde_fit_num_samples"] == 5.0
    )


def test_build_cross_source_compare_reports_shared_label_deltas():
    module = _load_module()
    current = {
        "source_label": "baoiad",
        "ordered_checkpoint_labels": ["iter_10", "iter_20"],
        "first_stop_line_label": "iter_20",
        "checkpoints": {
            "iter_10": {
                "checkpoint_path": "/tmp/current_iter_10.pth",
                "metrics": {"image_auroc": 0.82, "image_ap": 0.80},
                "score_gap_mean": 0.90,
                "good_score_mean": 0.10,
                "anomaly_score_mean": 1.00,
                "drift_stats": {"gde_fit_num_samples": 5.0, "gde_mean_mean": 0.2},
            },
            "iter_20": {
                "checkpoint_path": "/tmp/current_iter_20.pth",
                "metrics": {"image_auroc": 0.17, "image_ap": 0.20},
                "score_gap_mean": -0.20,
                "good_score_mean": 0.80,
                "anomaly_score_mean": 0.60,
                "drift_stats": {"gde_fit_num_samples": 7.0, "gde_mean_mean": 0.5},
            },
        },
    }
    reference = {
        "source_label": "official_reference",
        "ordered_checkpoint_labels": ["iter_10", "iter_20"],
        "first_stop_line_label": None,
        "checkpoints": {
            "iter_10": {
                "checkpoint_path": "/tmp/ref_iter_10.pth",
                "metrics": {"image_auroc": 0.80, "image_ap": 0.78},
                "score_gap_mean": 0.70,
                "good_score_mean": 0.20,
                "anomaly_score_mean": 0.90,
                "drift_stats": {"gde_fit_num_samples": 4.0, "gde_mean_mean": 0.1},
            },
            "iter_20": {
                "checkpoint_path": "/tmp/ref_iter_20.pth",
                "metrics": {"image_auroc": 0.79, "image_ap": 0.76},
                "score_gap_mean": 0.60,
                "good_score_mean": 0.25,
                "anomaly_score_mean": 0.85,
                "drift_stats": {"gde_fit_num_samples": 5.0, "gde_mean_mean": 0.2},
            },
        },
    }

    summary = module._build_cross_source_compare(current, reference)

    assert summary["current_source_label"] == "baoiad"
    assert summary["reference_source_label"] == "official_reference"
    assert summary["shared_checkpoint_labels"] == ["iter_10", "iter_20"]
    assert summary["current_first_stop_line_label"] == "iter_20"
    assert (
        summary["checkpoints"]["iter_10"]["metrics_delta"]["image_auroc_delta"]
        == 0.019999999999999907
    )
    assert summary["checkpoints"]["iter_20"]["score_gap_mean_delta"] == -0.8
    assert (
        summary["checkpoints"]["iter_10"]["drift_stats_delta"][
            "gde_fit_num_samples_delta"
        ]
        == 1.0
    )
    assert (
        summary["checkpoints"]["iter_20"]["drift_stats_delta"]["gde_mean_mean_delta"]
        == 0.3
    )


def test_build_cross_source_compare_honors_reference_label_override():
    module = _load_module()
    current = {
        "source_label": "baoiad",
        "ordered_checkpoint_labels": ["iter_10"],
        "checkpoints": {
            "iter_10": {
                "checkpoint_path": "/tmp/current_iter_10.pth",
                "metrics": {"image_auroc": 0.82},
                "score_gap_mean": 0.90,
                "good_score_mean": 0.10,
                "anomaly_score_mean": 1.00,
                "drift_stats": {"gde_fit_num_samples": 5.0},
            },
        },
    }
    reference = {
        "source_label": "official_reference",
        "ordered_checkpoint_labels": ["iter_10"],
        "checkpoints": {
            "iter_10": {
                "checkpoint_path": "/tmp/ref_iter_10.pth",
                "metrics": {"image_auroc": 0.80},
                "score_gap_mean": 0.70,
                "good_score_mean": 0.20,
                "anomaly_score_mean": 0.90,
                "drift_stats": {"gde_fit_num_samples": 4.0},
            },
        },
    }

    summary = module._build_cross_source_compare(
        current, reference, reference_source_label="manual_ref"
    )

    assert summary["reference_source_label"] == "manual_ref"
