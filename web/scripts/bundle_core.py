"""Copies the parts of the existing project the web chat needs into web/_core.

Vercel builds this directory as its own project (Root Directory = web), so the function bundle only contains
files inside web/. Run during the Vercel build via ``[tool.vercel.scripts]`` in pyproject.toml; locally it is
optional, because chatbot.core_loader falls back to the repository root.
"""
import shutil
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent.parent
REPO = WEB_DIR.parent
TARGET = WEB_DIR / "_core"
FILES = ("data/restaurant_info_extended.csv", "data/templates.json",
         "model-zoo/Logistic regression/log_reg_simple_grouped.pkl")


def main() -> None:
    if not (REPO / "src" / "uu").is_dir():
        raise SystemExit(f"Cannot find the dialog-act source in {REPO}. On Vercel, enable "
                         "'Include files outside the root directory in the Build Step'.")
    shutil.rmtree(TARGET, ignore_errors=True)
    shutil.copytree(REPO / "src" / "uu", TARGET / "src" / "uu", ignore=shutil.ignore_patterns("__pycache__"))
    for name in FILES:
        (TARGET / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name, TARGET / name)
    print(f"Bundled dialog-act core into {TARGET}")


if __name__ == "__main__":
    main()
