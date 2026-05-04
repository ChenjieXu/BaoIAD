"""Test all configs can be loaded and parsed."""

import glob
import os

import pytest
from mmengine import Config

import baoiad  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_configs = sorted(glob.glob(os.path.join(ROOT, 'configs', '*', '*.py')))
_configs = [c for c in _configs if '_base_' not in c]


@pytest.mark.parametrize('cfg_path', _configs,
                         ids=[os.path.basename(c) for c in _configs])
def test_config_loads(cfg_path):
    """Each config should load without error and contain a model definition."""
    cfg = Config.fromfile(cfg_path)
    assert hasattr(cfg, 'model') or 'model' in cfg


@pytest.mark.parametrize('cfg_path', _configs,
                         ids=[os.path.basename(c) for c in _configs])
def test_config_has_dataloader(cfg_path):
    """Each config should define train and test dataloaders."""
    cfg = Config.fromfile(cfg_path)
    assert hasattr(cfg, 'train_dataloader') or 'train_dataloader' in cfg
    assert hasattr(cfg, 'test_dataloader') or 'test_dataloader' in cfg


@pytest.mark.parametrize('cfg_path', _configs,
                         ids=[os.path.basename(c) for c in _configs])
def test_config_has_evaluator(cfg_path):
    """Each config should define a test evaluator."""
    cfg = Config.fromfile(cfg_path)
    assert hasattr(cfg, 'test_evaluator') or 'test_evaluator' in cfg
