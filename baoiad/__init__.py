"""BaoIAD: Unified Industrial Anomaly Detection Benchmark.

The top-level package deliberately stays lightweight. Runtime registries are
loaded explicitly through :func:`register_all_modules` or by importing
``baoiad.registration`` from an MMEngine config.
"""

from baoiad.paths import get_data_root

__version__ = "1.1.0"
BAOIAD_DATA_ROOT = str(get_data_root())

_MODULES_REGISTERED = False


def register_all_modules() -> None:
    """Import BaoIAD runtime modules and populate the MMEngine registries."""
    global _MODULES_REGISTERED
    if _MODULES_REGISTERED:
        return

    from baoiad.utils.compat import ensure_legacy_imgaug_compat

    ensure_legacy_imgaug_compat()
    from . import datasets, engine, evaluation, models, structures, visualization  # noqa: F401

    _MODULES_REGISTERED = True


__all__ = ["BAOIAD_DATA_ROOT", "__version__", "get_data_root", "register_all_modules"]
