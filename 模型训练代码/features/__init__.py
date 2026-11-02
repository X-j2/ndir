# -*- coding: utf-8 -*-
"""Stage 1 SI feature and quality packages.

The repository keeps the frozen Stage 0 implementation under ``tools/``.
``features.quality_states`` intentionally reuses the frozen
``tools.quality_state.HistorySummary`` contract, so make ``tools/`` importable
when this package is imported from the repository root.
"""
from __future__ import annotations

import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))
