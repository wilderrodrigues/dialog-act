import os
from pathlib import Path

def get_root() -> Path:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return Path(current_dir).parent.parent