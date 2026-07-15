"""Tests for GLASS strict-alignment guards."""

import importlib
import importlib.util
from functools import lru_cache
from pathlib import Path

import pytest

pytestmark = pytest.mark.optional
pd = pytest.importorskip("pandas", reason='requires the "glass" optional extra')
pytest.importorskip("openpyxl", reason='requires the "glass" optional extra')

collect_glass_asset_report = importlib.import_module(
    "baoiad.utils.glass_assets"
).collect_glass_asset_report


ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _load_benchmark_module():
    benchmark_path = ROOT / "tools" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("baoiad_benchmark", benchmark_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_glass_benchmark_prefers_strict_config():
    benchmark = _load_benchmark_module()
    config_path = benchmark.find_config("glass")
    assert config_path.endswith("glass_wrn50_288_mvtec_strict.py")


def test_collect_glass_asset_report_requires_distribution_masks_and_dtd(tmp_path):
    assets_root = tmp_path / "glass_assets" / "mvtec"
    fg_mask_root = assets_root / "fg_mask"
    for cls_name in ["bottle", "cable", "pill", "screw", "transistor"]:
        cls_dir = fg_mask_root / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        (cls_dir / "000.png").write_bytes(b"0")

    assets_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"Class": "mvtec_bottle", "Distribution": 1, "Foreground": 1},
            {"Class": "mvtec_cable", "Distribution": 1, "Foreground": 1},
            {"Class": "mvtec_pill", "Distribution": 1, "Foreground": 1},
            {"Class": "mvtec_screw", "Distribution": 1, "Foreground": 1},
            {"Class": "mvtec_transistor", "Distribution": 1, "Foreground": 1},
            {"Class": "mvtec_carpet", "Distribution": 0, "Foreground": 0},
        ]
    ).to_excel(assets_root / "mvtec_distribution.xlsx", index=False)

    dtd_root = tmp_path / "data" / "dtd" / "dtd" / "images" / "banded"
    dtd_root.mkdir(parents=True)
    (dtd_root / "tex.jpg").write_bytes(b"jpg")

    report = collect_glass_asset_report(
        glass_assets_root=assets_root,
        dtd_root=tmp_path / "data" / "dtd",
    )

    assert report["ok"] is True
    assert report["dtd_texture_count"] == 1
    assert report["missing_fg_mask_classes"] == []
