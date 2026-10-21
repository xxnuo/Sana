"""Path compatibility layer for the shared GB10 model assembler."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_GB10 = Path(__file__).resolve().parent.parent / "GB10" / "paths.py"
_SPEC = importlib.util.spec_from_file_location("_h3_gb10_paths", _GB10)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load shared H3 paths module: {_GB10}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

for _name in dir(_MODULE):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_MODULE, _name)
