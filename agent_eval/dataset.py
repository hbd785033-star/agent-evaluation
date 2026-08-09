"""Versioned YAML dataset loading and validation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models import TaskCase


def load_dataset(path: str | Path) -> tuple[str, list[TaskCase]]:
    dataset_path = Path(path)
    raw: Any = yaml.safe_load(dataset_path.read_text(encoding="utf-8")) or {}
    if isinstance(raw, list):
        rows = raw
        version = "unversioned"
    elif isinstance(raw, dict):
        rows = raw.get("tasks", raw.get("cases", []))
        version = str(raw.get("version", "unknown"))
    else:
        raise TypeError("dataset root must be a list or mapping")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"dataset has no tasks/cases: {dataset_path}")
    tasks = [TaskCase.from_mapping(row) for row in rows]
    ids = [task.id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("dataset task IDs must be unique")
    return version, tasks
