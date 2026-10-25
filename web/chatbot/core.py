"""Where the web interface finds the dialog-act data, and a Windows fix for loading the pickled model.

The dialog-act code itself is the installed ``dialog`` package (``poetry install --with web`` in the repository
root), so it is imported normally.
"""
import os
import pathlib
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

RESTAURANTS_PATH = REPO_ROOT / "data" / "restaurant_info_extended.csv"
TEMPLATES_PATH = REPO_ROOT / "data" / "templates.json"
MODEL_PATH = REPO_ROOT / "model-zoo" / "Logistic regression" / "log_reg_simple_grouped.pkl"


@contextmanager
def posix_paths_on_windows() -> Iterator[None]:
    """The pickled model holds a PosixPath, which cannot be unpickled on Windows."""
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
