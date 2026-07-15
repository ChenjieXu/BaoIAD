"""Tests for import-safe dataset-root resolution."""

from __future__ import annotations

import importlib.util
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from baoiad.config import apply_data_root_overrides
from baoiad.paths import get_data_root, resolve_data_root

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATASET_CONFIGS = {
    "btech.py": "btech",
    "kolektor.py": "kolektor",
    "mpdd.py": "mpdd",
    "mvtec_3d_ad.py": "mvtec_3d_ad",
    "mvtec_ad.py": "mvtec_ad",
    "mvtec_ad2.py": "mvtec_ad_2",
    "mvtec_loco_ad.py": "mvtec_loco_ad",
    "realiad.py": "Real-IAD",
    "vad.py": "vad",
    "visa.py": "visa",
}


def _load_tool_module(filename: str):
    path = REPOSITORY_ROOT / "tools" / filename
    spec = importlib.util.spec_from_file_location(
        f"_data_root_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_root_is_repository_local_and_cwd_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    root = get_data_root(environ={}, repository_root=REPOSITORY_ROOT)

    assert root == REPOSITORY_ROOT / "data"


@pytest.mark.parametrize("suffix", ["", os.sep, os.sep * 2])
def test_absolute_environment_root_normalizes_trailing_separators(
    tmp_path: Path, suffix: str
) -> None:
    environment = {"BAOIAD_DATA_ROOT": f"{tmp_path}{suffix}"}

    resolved = resolve_data_root(
        "mvtec_ad", environ=environment, repository_root=REPOSITORY_ROOT
    )

    assert resolved == tmp_path / "mvtec_ad"


def test_relative_environment_root_resolves_from_repository() -> None:
    resolved = resolve_data_root(
        "visa",
        environ={"BAOIAD_DATA_ROOT": "shared-data"},
        repository_root=REPOSITORY_ROOT,
    )

    assert resolved == REPOSITORY_ROOT / "shared-data" / "visa"


@pytest.mark.parametrize("value", ["", "   ", "bad\x00path"])
def test_malformed_environment_root_fails(value: str) -> None:
    with pytest.raises(ValueError, match="data root"):
        get_data_root(
            environ={"BAOIAD_DATA_ROOT": value}, repository_root=REPOSITORY_ROOT
        )


def test_cli_override_is_the_final_dataset_path_and_wins(tmp_path: Path) -> None:
    override = tmp_path / "explicit" / "mvtec_ad"

    resolved = resolve_data_root(
        "ignored-dataset-name",
        override=override,
        environ={"BAOIAD_DATA_ROOT": "/ignored/environment/root"},
        repository_root=REPOSITORY_ROOT,
    )

    assert resolved == override


def test_loaded_config_paths_are_resolved_before_cli_override(tmp_path: Path) -> None:
    config = {
        "train_dataloader": {
            "dataset": {
                "data_root": "data/mvtec_ad",
                "support": {"data_root": "data/visa"},
            }
        },
        "absolute": {"data_root": "/already/explicit"},
        "unrelated": {"data_root": "custom/location"},
    }

    apply_data_root_overrides(
        config,
        environ={"BAOIAD_DATA_ROOT": str(tmp_path)},
        repository_root=REPOSITORY_ROOT,
    )
    config["train_dataloader"]["dataset"]["data_root"] = "/cli/mvtec_ad"

    assert config["train_dataloader"]["dataset"]["data_root"] == "/cli/mvtec_ad"
    assert config["train_dataloader"]["dataset"]["support"]["data_root"] == str(
        tmp_path / "visa"
    )
    assert config["absolute"]["data_root"] == "/already/explicit"
    assert config["unrelated"]["data_root"] == "custom/location"


@pytest.mark.parametrize("dataset_name", ["", "..", "../outside", "/absolute"])
def test_invalid_dataset_name_fails(dataset_name: str) -> None:
    with pytest.raises(ValueError, match="invalid dataset name"):
        resolve_data_root(dataset_name, environ={}, repository_root=REPOSITORY_ROOT)


def test_all_base_dataset_configs_use_environment_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BAOIAD_DATA_ROOT", str(tmp_path))
    config_dir = REPOSITORY_ROOT / "configs" / "_base_" / "datasets"

    for filename, dataset_name in DATASET_CONFIGS.items():
        namespace = runpy.run_path(str(config_dir / filename))
        assert Path(namespace["data_root"]) == tmp_path / dataset_name


@pytest.mark.parametrize(
    ("script_name", "config_name"),
    [
        ("train_regad_strict.py", "regad/regad_wrn50_256_mvtec_strict.py"),
        ("train_vitad_exact_order.py", "vitad/vitad_256_mvtec_strict.py"),
    ],
)
def test_specialized_trainers_apply_environment_root_outside_repo_cwd(
    script_name: str,
    config_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved_root = tmp_path / "shared-data"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BAOIAD_DATA_ROOT", str(resolved_root))
    module = _load_tool_module(script_name)
    args = SimpleNamespace(
        config=str(REPOSITORY_ROOT / "configs" / config_name),
        cfg_options=None,
        work_dir=None,
    )

    cfg = module._load_cfg(args)

    expected_dataset = str(resolved_root / "mvtec_ad")
    assert cfg.train_dataloader.dataset.data_root == expected_dataset
    assert cfg.test_dataloader.dataset.data_root == expected_dataset
    if script_name == "train_regad_strict.py":
        assert cfg.model.data_root == expected_dataset
        assert cfg.support_set_root == str(
            resolved_root / "regad_official" / "support_set"
        )


def test_top_level_import_does_not_require_runtime_dependencies() -> None:
    code = """
import importlib.abc
import sys

blocked = {'numpy', 'torch', 'mmcv', 'mmengine', 'FrEIA', 'open_clip'}

class BlockRuntimeDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise ModuleNotFoundError(fullname, name=fullname)
        return None

sys.meta_path.insert(0, BlockRuntimeDependencies())
import baoiad
print(baoiad.__version__)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()


def test_env_script_uses_its_own_location_when_sourced(tmp_path: Path) -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is unavailable")
    env_script = REPOSITORY_ROOT / "tools" / "env.sh"
    result = subprocess.run(
        [
            bash,
            "-c",
            'unset BAOIAD_DATA_ROOT; cd "$1"; source "$2"; printf %s "$BAOIAD_DATA_ROOT"',
            "bash",
            str(tmp_path),
            str(env_script),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == str(REPOSITORY_ROOT / "data")
