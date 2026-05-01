"""Local helper for importing the sibling `dynn` repository."""

from __future__ import annotations

import sys
from pathlib import Path


_DYNN_REPO = Path(__file__).resolve().parents[3] / "dynn"
if str(_DYNN_REPO) not in sys.path:
    sys.path.insert(0, str(_DYNN_REPO))

import dynn

__all__ = ["dynn"]
