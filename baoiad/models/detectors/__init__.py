# ruff: noqa: E402
# ADer methods (13)
from baoiad.models.detectors.patchcore import PatchCore  # noqa: F401
from baoiad.models.detectors.compose_ad import ComposeAD  # noqa: F401
from baoiad.models.detectors.reverse_distillation import ReverseDistillation  # noqa: F401
from baoiad.models.detectors.simplenet import SimpleNetDetector  # noqa: F401
from baoiad.models.detectors.draem import DRAEMDetector  # noqa: F401
from baoiad.models.detectors.padim import PaDiMDetector  # noqa: F401
from baoiad.models.detectors.stfpm import STFPMDetector  # noqa: F401
from baoiad.models.detectors.efficientad import EfficientADDetector  # noqa: F401
from baoiad.models.detectors.rdpp import RDPPDetector  # noqa: F401
from baoiad.models.detectors.invad import InvADDetector  # noqa: F401
from baoiad.models.detectors.vitad import ViTADDetector  # noqa: F401
from baoiad.models.detectors.uniad_detector import UniADDetector  # noqa: F401
try:
    from baoiad.models.detectors.pni import PNI  # noqa: F401
    _has_pni = True
except ImportError:
    _has_pni = False

# FrEIA-dependent detectors (optional: requires FrEIA>=0.2)
_has_freia = True
try:
    from baoiad.models.detectors.cflow import CFlowDetector  # noqa: F401
    from baoiad.models.detectors.fastflow import FastFlowDetector  # noqa: F401
    from baoiad.models.detectors.csflow import CSFlowDetector  # noqa: F401
    from baoiad.models.detectors.uflow import UFlowDetector  # noqa: F401
    from baoiad.models.detectors.differnet import DifferNetDetector  # noqa: F401
    from baoiad.models.detectors.pyramidflow import PyramidFlowDetector  # noqa: F401
    from baoiad.models.detectors.resad import ResADDetector  # noqa: F401
except ImportError:
    _has_freia = False
try:
    from baoiad.models.detectors.ast import ASTDetector  # noqa: F401
    _has_ast = True
except ImportError:
    _has_ast = False

# Anomalib methods
from baoiad.models.detectors.cfa import CFADetector  # noqa: F401
from baoiad.models.detectors.dfkde import DFKDEDetector  # noqa: F401
from baoiad.models.detectors.dfm import DFMDetector  # noqa: F401
from baoiad.models.detectors.dinomaly import DinomalyDetector  # noqa: F401
from baoiad.models.detectors.dsr import DSRDetector  # noqa: F401
from baoiad.models.detectors.fre import FREDetector  # noqa: F401
from baoiad.models.detectors.ganomaly import GanomalyDetector  # noqa: F401
from baoiad.models.detectors.winclip import WinClipDetector  # noqa: F401
from baoiad.models.detectors.supersimplenet import SuperSimpleNetDetector  # noqa: F401
try:
    from baoiad.models.detectors.uninet import UniNetDetector  # noqa: F401
    _has_uninet = True
except ImportError:
    _has_uninet = False

# P0 methods (5)
from baoiad.models.detectors.spade import SPADEDetector  # noqa: F401
from baoiad.models.detectors.cutpaste import CutPasteDetector  # noqa: F401
try:
    from baoiad.models.detectors.memae import MemAEDetector  # noqa: F401
    _has_memae = True
except ImportError:
    _has_memae = False
from baoiad.models.detectors.destseg import DeSTSegDetector  # noqa: F401
from baoiad.models.detectors.realnet import RealNetDetector  # noqa: F401
from baoiad.models.detectors.memseg import MemSegDetector  # noqa: F401

# P1 methods
from baoiad.models.detectors.musc import MuScDetector  # noqa: F401
from baoiad.models.detectors.anomalyclip import AnomalyCLIPDetector  # noqa: F401
from baoiad.models.detectors.anomalyclip_official import AnomalyCLIPOfficialDetector  # noqa: F401
from baoiad.models.detectors.aaclip import AACLIPDetector  # noqa: F401
from baoiad.models.detectors.adaclip import AdaCLIPDetector  # noqa: F401
from baoiad.models.detectors.regad import RegADDetector  # noqa: F401
try:
    from baoiad.models.detectors.graphcore import GraphCoreDetector  # noqa: F401
    _has_graphcore = True
except ImportError:
    _has_graphcore = False
from baoiad.models.detectors.nsa import NSADetector  # noqa: F401
from baoiad.models.detectors.anovl import AnoVLDetector  # noqa: F401
try:
    from baoiad.models.detectors.anomalydino import AnomalyDINODetector  # noqa: F401
    _has_anomalydino = True
except ImportError:
    _has_anomalydino = False
from baoiad.models.detectors.glass import GLASSDetector  # noqa: F401
from baoiad.models.detectors.mambaad import MambaADDetector  # noqa: F401
try:
    from baoiad.models.detectors.univad import UniVADDetector  # noqa: F401
    _has_univad = True
except ImportError:
    _has_univad = False

# SAA/SAA+ (requires groundingdino + segment_anything)
try:
    from baoiad.models.detectors.saa import SAADetector  # noqa: F401
    _has_saa = True
except Exception:
    _has_saa = False

__all__ = [
    # ADer
    'PatchCore', 'ReverseDistillation', 'SimpleNetDetector', 'DRAEMDetector',
    'PaDiMDetector', 'STFPMDetector',
    'EfficientADDetector', 'RDPPDetector', 'InvADDetector', 'ViTADDetector',
    'UniADDetector',
    # Anomalib
    'CFADetector', 'DFKDEDetector', 'DFMDetector',
    'DinomalyDetector', 'DSRDetector', 'FREDetector', 'GanomalyDetector',
    'WinClipDetector',
    'SuperSimpleNetDetector',
    # P0
    'SPADEDetector', 'CutPasteDetector',
    'DeSTSegDetector', 'RealNetDetector', 'MemSegDetector',
    # P1
    'MuScDetector', 'AnomalyCLIPDetector', 'AnomalyCLIPOfficialDetector',
    'AACLIPDetector', 'AdaCLIPDetector',
    'RegADDetector', 'NSADetector', 'AnoVLDetector',
    'GLASSDetector',
    'ComposeAD',
    'MambaADDetector',
]
if _has_freia:
    __all__.extend([
        'CFlowDetector', 'FastFlowDetector', 'CSFlowDetector',
        'UFlowDetector', 'DifferNetDetector', 'PyramidFlowDetector',
        'ResADDetector',
    ])
if _has_ast:
    __all__.append('ASTDetector')
if _has_uninet:
    __all__.append('UniNetDetector')
if _has_pni:
    __all__.append('PNI')
if _has_memae:
    __all__.append('MemAEDetector')
if _has_graphcore:
    __all__.append('GraphCoreDetector')
if _has_univad:
    __all__.append('UniVADDetector')
if _has_saa:
    __all__.append('SAADetector')
if _has_anomalydino:
    __all__.append('AnomalyDINODetector')
