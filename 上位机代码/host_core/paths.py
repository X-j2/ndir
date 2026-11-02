"""Locate the reviewed training repository without coupling to cwd."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def training_root() -> Path | None:
    env = os.environ.get("CFREM_TRAINING_ROOT")
    candidates = []
    if env:
        candidates.append(Path(env))
    # host_core/paths.py -> host repo -> 任务书 -> 模型训练代码
    candidates.append(Path(__file__).resolve().parents[2] / "模型训练代码")
    candidates.append(Path(__file__).resolve().parents[1] / "模型训练代码")
    for p in candidates:
        if (p / "features" / "si_features.py").exists():
            return p
    return None


def ensure_training_path() -> Path | None:
    root = training_root()
    if root is None:
        return None
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    tools = root / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    return root
