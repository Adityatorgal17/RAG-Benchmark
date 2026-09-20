"""Runtime configuration loader for the project.

This file reads the shared YAML settings used across the RAG pipeline (model IDs,
retrieval parameters, data paths, and experiment config). It keeps the rest of the
codebase decoupled from hard-coded values and makes changes to runtime behavior easy
and centralized.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.yaml"


def _deep_update(ns, data):
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                child = getattr(ns, key, None)
                if child is None or not isinstance(child, SimpleNamespace):
                    child = SimpleNamespace()
                    setattr(ns, key, child)
                _deep_update(child, value)
            else:
                setattr(ns, key, value)


def load_config(path: str | Path | None = None) -> SimpleNamespace:
    config_file = Path(path) if path is not None else CONFIG_PATH
    with config_file.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    ns = SimpleNamespace()
    _deep_update(ns, raw)
    return ns
