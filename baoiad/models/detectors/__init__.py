"""Detector registrations for the public BaoIAD method inventory."""

from __future__ import annotations

from typing import Any

from baoiad.optional import OptionalDependencyError, optional_dependency_message
from baoiad.registry import MODELS


def _missing_optional_detector(
    class_name: str, *, extra: str, feature: str, import_name: str
) -> type:
    message = optional_dependency_message(
        extra=extra, feature=feature, import_name=import_name
    )

    class MissingOptionalDetector:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise OptionalDependencyError(message, name=import_name)

    MissingOptionalDetector.__name__ = class_name
    MissingOptionalDetector.__qualname__ = class_name
    MODELS.register_module(name=class_name, module=MissingOptionalDetector, force=True)
    return MissingOptionalDetector


from baoiad.models.detectors.aaclip import AACLIPDetector  # noqa: E402,F401
from baoiad.models.detectors.adaclip import AdaCLIPDetector  # noqa: E402,F401
from baoiad.models.detectors.anomalyclip import AnomalyCLIPDetector  # noqa: E402,F401
from baoiad.models.detectors.anomalyclip_official import (  # noqa: E402,F401
    AnomalyCLIPOfficialDetector,
)
from baoiad.models.detectors.anomalydino import AnomalyDINODetector  # noqa: E402,F401
from baoiad.models.detectors.anovl import AnoVLDetector  # noqa: E402,F401
from baoiad.models.detectors.ast import ASTDetector  # noqa: E402,F401
from baoiad.models.detectors.cfa import CFADetector  # noqa: E402,F401
from baoiad.models.detectors.cflow import CFlowDetector  # noqa: E402,F401
from baoiad.models.detectors.cutpaste import CutPasteDetector  # noqa: E402,F401
from baoiad.models.detectors.destseg import DeSTSegDetector  # noqa: E402,F401
from baoiad.models.detectors.dfkde import DFKDEDetector  # noqa: E402,F401
from baoiad.models.detectors.dfm import DFMDetector  # noqa: E402,F401
from baoiad.models.detectors.differnet import DifferNetDetector  # noqa: E402,F401
from baoiad.models.detectors.dinomaly import DinomalyDetector  # noqa: E402,F401
from baoiad.models.detectors.draem import DRAEMDetector  # noqa: E402,F401
from baoiad.models.detectors.dsr import DSRDetector  # noqa: E402,F401
from baoiad.models.detectors.efficientad import EfficientADDetector  # noqa: E402,F401

try:
    from baoiad.models.detectors.fastflow import FastFlowDetector  # noqa: F401
except ModuleNotFoundError as exc:
    if (exc.name or "").split(".")[0] != "FrEIA":
        raise
    FastFlowDetector = _missing_optional_detector(
        "FastFlowDetector",
        extra="flow",
        feature="FastFlow",
        import_name="FrEIA",
    )

from baoiad.models.detectors.ganomaly import GanomalyDetector  # noqa: E402,F401
from baoiad.models.detectors.glass import GLASSDetector  # noqa: E402,F401
from baoiad.models.detectors.memseg import MemSegDetector  # noqa: E402,F401
from baoiad.models.detectors.musc import MuScDetector  # noqa: E402,F401
from baoiad.models.detectors.nsa import NSADetector  # noqa: E402,F401
from baoiad.models.detectors.padim import PaDiMDetector  # noqa: E402,F401
from baoiad.models.detectors.patchcore import PatchCore  # noqa: E402,F401
from baoiad.models.detectors.pyramidflow import PyramidFlowDetector  # noqa: E402,F401
from baoiad.models.detectors.rdpp import RDPPDetector  # noqa: E402,F401
from baoiad.models.detectors.regad import RegADDetector  # noqa: E402,F401
from baoiad.models.detectors.reverse_distillation import (  # noqa: E402,F401
    ReverseDistillation,
)
from baoiad.models.detectors.saa import SAADetector  # noqa: E402,F401
from baoiad.models.detectors.simplenet import SimpleNetDetector  # noqa: E402,F401
from baoiad.models.detectors.supersimplenet import (  # noqa: E402,F401
    SuperSimpleNetDetector,
)

try:
    from baoiad.models.detectors.uflow import UFlowDetector  # noqa: F401
except ModuleNotFoundError as exc:
    if (exc.name or "").split(".")[0] not in {
        "FrEIA",
        "mpmath",
        "networkx",
        "skimage",
    }:
        raise
    UFlowDetector = _missing_optional_detector(
        "UFlowDetector",
        extra="flow",
        feature="U-Flow",
        import_name=exc.name or "FrEIA",
    )

from baoiad.models.detectors.uniad_detector import UniADDetector  # noqa: E402,F401
from baoiad.models.detectors.uninet import UniNetDetector  # noqa: E402,F401
from baoiad.models.detectors.vitad import ViTADDetector  # noqa: E402,F401
from baoiad.models.detectors.winclip import WinClipDetector  # noqa: E402,F401

__all__ = [
    "AACLIPDetector",
    "AdaCLIPDetector",
    "AnomalyCLIPDetector",
    "AnomalyCLIPOfficialDetector",
    "AnomalyDINODetector",
    "AnoVLDetector",
    "ASTDetector",
    "CFADetector",
    "CFlowDetector",
    "CutPasteDetector",
    "DeSTSegDetector",
    "DFKDEDetector",
    "DFMDetector",
    "DifferNetDetector",
    "DinomalyDetector",
    "DRAEMDetector",
    "DSRDetector",
    "EfficientADDetector",
    "FastFlowDetector",
    "GanomalyDetector",
    "GLASSDetector",
    "MemSegDetector",
    "MuScDetector",
    "NSADetector",
    "PaDiMDetector",
    "PatchCore",
    "PyramidFlowDetector",
    "RDPPDetector",
    "RegADDetector",
    "ReverseDistillation",
    "SAADetector",
    "SimpleNetDetector",
    "SuperSimpleNetDetector",
    "UFlowDetector",
    "UniADDetector",
    "UniNetDetector",
    "ViTADDetector",
    "WinClipDetector",
]
