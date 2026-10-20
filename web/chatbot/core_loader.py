"""Makes the existing dialog-act code importable on Vercel without changing it.

The recommender imports ``encoders`` and ``datasets`` at module level, which pull in torch and transformers.
Those packages are far too large for a Vercel Function and the web chat never uses them: it runs the
logistic-regression classifier (a plain ``CountVectorizer``) with the Levenshtein fallback. Lightweight stand-ins
are therefore registered in ``sys.modules`` before the real code is imported.
"""
import os
import pathlib
import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent.parent
BUNDLED_ROOT = WEB_DIR / "_core"  # filled by scripts/bundle_core.py during the Vercel build
CORE_ROOT = BUNDLED_ROOT if (BUNDLED_ROOT / "src").is_dir() else WEB_DIR.parent

RESTAURANTS_PATH = CORE_ROOT / "data" / "restaurant_info_extended.csv"
TEMPLATES_PATH = CORE_ROOT / "data" / "templates.json"
MODEL_PATH = CORE_ROOT / "model-zoo" / "Logistic regression" / "log_reg_simple_grouped.pkl"


class _Unavailable:
    """Stand-in for a torch-based class; it can be referenced, but not used."""

    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError(f"{type(self).__name__} needs torch, which is not installed in the web deployment.")


def _stub(name: str, *class_names: str, **attributes: object) -> None:
    module = types.ModuleType(name)
    for class_name in class_names:
        setattr(module, class_name, type(class_name, (_Unavailable,), {}))
    module.__dict__.update(attributes)
    sys.modules[name] = module


@contextmanager
def posix_paths_on_windows() -> Iterator[None]:
    """The pickled model holds a PosixPath, which cannot be unpickled on Windows (local development only)."""
    if os.name != "nt":
        yield
        return
    modules = [module for module in (pathlib, sys.modules.get("pathlib._local")) if module is not None]
    originals = [module.PosixPath for module in modules]
    for module in modules:
        module.PosixPath = pathlib.WindowsPath
    try:
        yield
    finally:
        for module, original in zip(modules, originals):
            module.PosixPath = original


def install() -> None:
    if str(CORE_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(CORE_ROOT / "src"))
    try:
        import torch  # noqa: F401  (full install, e.g. the poetry environment: use the real modules)
        return
    except ImportError:
        pass
    _stub("uu.msc.ai.mair.dialog.core.encoders", "Encoder", "SimpleEncoder", "FrozenDistilBertEncoder",
          "DistilBertTokenEncoder", EMBED_LEN=768)
    _stub("uu.msc.ai.mair.dialog.core.datasets", "DatasetFactory", "DialogActsDataset")
