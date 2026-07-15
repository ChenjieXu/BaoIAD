"""Tests for BaoIAD registries."""

from baoiad import register_all_modules
from baoiad.registry import (
    DATASETS,
    HOOKS,
    LOOPS,
    METRICS,
    MODELS,
    TRANSFORMS,
    VISUALIZERS,
)

register_all_modules()


class TestRegistries:
    def test_registries_exist(self):
        for reg in [MODELS, DATASETS, METRICS, TRANSFORMS, LOOPS, HOOKS, VISUALIZERS]:
            assert reg is not None
            assert reg.scope == "baoiad"

    def test_model_registration(self):
        cls = MODELS.get("PatchCore")
        assert cls is not None
        from baoiad.models.detectors.patchcore import PatchCore

        assert cls is PatchCore

    def test_dataset_registration(self):
        cls = DATASETS.get("MVTecADDataset")
        assert cls is not None

    def test_metric_registration(self):
        cls = METRICS.get("AnomalyDetectionMetric")
        assert cls is not None

    def test_transform_registration(self):
        for name in [
            "LoadImage",
            "LoadMask",
            "ResizeAD",
            "NormalizeAD",
            "PackADInputs",
        ]:
            assert TRANSFORMS.get(name) is not None

    def test_hook_registration(self):
        assert HOOKS.get("MemoryBankHook") is not None

    def test_loop_registration(self):
        assert LOOPS.get("ADTestLoop") is not None
