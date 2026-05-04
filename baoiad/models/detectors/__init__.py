# ruff: noqa: E402
"""Detector registrations for the paper-facing BaoIAD method inventory."""

from baoiad.models.detectors.aaclip import AACLIPDetector  # noqa: F401
from baoiad.models.detectors.adaclip import AdaCLIPDetector  # noqa: F401
from baoiad.models.detectors.anomalyclip import AnomalyCLIPDetector  # noqa: F401
from baoiad.models.detectors.anomalyclip_official import AnomalyCLIPOfficialDetector  # noqa: F401
try:
    from baoiad.models.detectors.anomalydino import AnomalyDINODetector  # noqa: F401
    _has_anomalydino = True
except ImportError:
    _has_anomalydino = False
from baoiad.models.detectors.anovl import AnoVLDetector  # noqa: F401
try:
    from baoiad.models.detectors.ast import ASTDetector  # noqa: F401
    _has_ast = True
except ImportError:
    _has_ast = False
from baoiad.models.detectors.cfa import CFADetector  # noqa: F401
from baoiad.models.detectors.cflow import CFlowDetector  # noqa: F401
from baoiad.models.detectors.cutpaste import CutPasteDetector  # noqa: F401
from baoiad.models.detectors.destseg import DeSTSegDetector  # noqa: F401
from baoiad.models.detectors.dfkde import DFKDEDetector  # noqa: F401
from baoiad.models.detectors.dfm import DFMDetector  # noqa: F401
from baoiad.models.detectors.differnet import DifferNetDetector  # noqa: F401
from baoiad.models.detectors.dinomaly import DinomalyDetector  # noqa: F401
from baoiad.models.detectors.draem import DRAEMDetector  # noqa: F401
from baoiad.models.detectors.dsr import DSRDetector  # noqa: F401
from baoiad.models.detectors.efficientad import EfficientADDetector  # noqa: F401
from baoiad.models.detectors.fastflow import FastFlowDetector  # noqa: F401
from baoiad.models.detectors.ganomaly import GanomalyDetector  # noqa: F401
from baoiad.models.detectors.glass import GLASSDetector  # noqa: F401
from baoiad.models.detectors.memseg import MemSegDetector  # noqa: F401
from baoiad.models.detectors.musc import MuScDetector  # noqa: F401
from baoiad.models.detectors.nsa import NSADetector  # noqa: F401
from baoiad.models.detectors.padim import PaDiMDetector  # noqa: F401
from baoiad.models.detectors.patchcore import PatchCore  # noqa: F401
from baoiad.models.detectors.pyramidflow import PyramidFlowDetector  # noqa: F401
from baoiad.models.detectors.rdpp import RDPPDetector  # noqa: F401
from baoiad.models.detectors.regad import RegADDetector  # noqa: F401
from baoiad.models.detectors.reverse_distillation import ReverseDistillation  # noqa: F401
try:
    from baoiad.models.detectors.saa import SAADetector  # noqa: F401
    _has_saa = True
except Exception:
    _has_saa = False
from baoiad.models.detectors.simplenet import SimpleNetDetector  # noqa: F401
from baoiad.models.detectors.supersimplenet import SuperSimpleNetDetector  # noqa: F401
from baoiad.models.detectors.uflow import UFlowDetector  # noqa: F401
from baoiad.models.detectors.uniad_detector import UniADDetector  # noqa: F401
try:
    from baoiad.models.detectors.uninet import UniNetDetector  # noqa: F401
    _has_uninet = True
except ImportError:
    _has_uninet = False
from baoiad.models.detectors.vitad import ViTADDetector  # noqa: F401
from baoiad.models.detectors.winclip import WinClipDetector  # noqa: F401

__all__ = [
    'AACLIPDetector', 'AdaCLIPDetector', 'AnomalyCLIPDetector',
    'AnomalyCLIPOfficialDetector', 'AnoVLDetector', 'CFADetector',
    'CFlowDetector', 'CutPasteDetector', 'DeSTSegDetector', 'DFKDEDetector',
    'DFMDetector', 'DifferNetDetector', 'DinomalyDetector', 'DRAEMDetector',
    'DSRDetector', 'EfficientADDetector', 'FastFlowDetector', 'GanomalyDetector',
    'GLASSDetector', 'MemSegDetector', 'MuScDetector', 'NSADetector',
    'PaDiMDetector', 'PatchCore', 'PyramidFlowDetector', 'RDPPDetector',
    'RegADDetector', 'ReverseDistillation', 'SimpleNetDetector',
    'SuperSimpleNetDetector', 'UFlowDetector', 'UniADDetector', 'ViTADDetector',
    'WinClipDetector',
]
if _has_anomalydino:
    __all__.append('AnomalyDINODetector')
if _has_ast:
    __all__.append('ASTDetector')
if _has_saa:
    __all__.append('SAADetector')
if _has_uninet:
    __all__.append('UniNetDetector')
