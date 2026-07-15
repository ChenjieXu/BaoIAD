"""Global registries for BaoIAD."""

from mmengine.registry import DATA_SAMPLERS as MMENGINE_DATA_SAMPLERS
from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmengine.registry import HOOKS as MMENGINE_HOOKS
from mmengine.registry import LOOPS as MMENGINE_LOOPS
from mmengine.registry import METRICS as MMENGINE_METRICS
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import (
    OPTIM_WRAPPER_CONSTRUCTORS as MMENGINE_OPTIM_WRAPPER_CONSTRUCTORS,
)
from mmengine.registry import OPTIM_WRAPPERS as MMENGINE_OPTIM_WRAPPERS
from mmengine.registry import OPTIMIZERS as MMENGINE_OPTIMIZERS
from mmengine.registry import PARAM_SCHEDULERS as MMENGINE_PARAM_SCHEDULERS
from mmengine.registry import RUNNERS as MMENGINE_RUNNERS
from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS
from mmengine.registry import VISUALIZERS as MMENGINE_VISUALIZERS
from mmengine.registry import Registry

MODELS = Registry(
    "models", parent=MMENGINE_MODELS, scope="baoiad", locations=["baoiad.models"]
)
DATASETS = Registry(
    "datasets", parent=MMENGINE_DATASETS, scope="baoiad", locations=["baoiad.datasets"]
)
TRANSFORMS = Registry(
    "transforms",
    parent=MMENGINE_TRANSFORMS,
    scope="baoiad",
    locations=["baoiad.datasets.transforms"],
)
METRICS = Registry(
    "metrics", parent=MMENGINE_METRICS, scope="baoiad", locations=["baoiad.evaluation"]
)
LOOPS = Registry(
    "loops", parent=MMENGINE_LOOPS, scope="baoiad", locations=["baoiad.engine.loops"]
)
HOOKS = Registry(
    "hooks", parent=MMENGINE_HOOKS, scope="baoiad", locations=["baoiad.engine.hooks"]
)
RUNNERS = Registry(
    "runners", parent=MMENGINE_RUNNERS, scope="baoiad", locations=["baoiad.engine"]
)
VISUALIZERS = Registry(
    "visualizers",
    parent=MMENGINE_VISUALIZERS,
    scope="baoiad",
    locations=["baoiad.visualization"],
)
DATA_SAMPLERS = Registry(
    "data sampler",
    parent=MMENGINE_DATA_SAMPLERS,
    scope="baoiad",
    locations=["baoiad.datasets.samplers"],
)
OPTIM_WRAPPER_CONSTRUCTORS = Registry(
    "optimizer wrapper constructor",
    parent=MMENGINE_OPTIM_WRAPPER_CONSTRUCTORS,
    scope="baoiad",
    locations=["baoiad.engine.optimizers"],
)
OPTIM_WRAPPERS = Registry(
    "optim_wrapper",
    parent=MMENGINE_OPTIM_WRAPPERS,
    scope="baoiad",
    locations=["baoiad.engine.optimizers"],
)
OPTIMIZERS = Registry(
    "optimizer",
    parent=MMENGINE_OPTIMIZERS,
    scope="baoiad",
    locations=["baoiad.engine.optimizers"],
)
PARAM_SCHEDULERS = Registry(
    "parameter scheduler",
    parent=MMENGINE_PARAM_SCHEDULERS,
    scope="baoiad",
    locations=["baoiad.engine.schedulers"],
)
