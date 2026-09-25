"""Make the repo root importable (common/, stage1_voice/, ...) when running pytest."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
