"""Strict ExecutionRecord 0.1 ingestion boundary."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import RunRecord, TaskCase

_FIELDS = {
    "schema_version",
    "task_id",
    "run_id",
    "model",
    "provider",
    "harness",
    "status",
    "started_at",
    "finished_at",
    "latency_seconds",
    "input_tokens",
    "output_tokens",
    "cached_tokens",
    "cost_usd",
    "tool_calls",
    "files_changed",
    "output",
    "workspace_root",
    "isolation_level",
    "metadata",
}
_REQUIRED = _FIELDS - {"workspace_root"}


def _string(raw: dict[str, Any], name: str) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"ExecutionRecord.{name} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    raw: dict[str, Any]

    @classmethod
    def from_mapping(cls, raw: Any) -> ExecutionRecord:
        if not isinstance(raw, dict):
            raise TypeError("ExecutionRecord must be an object")
        version = raw.get("schema_version")
        if version != "0.1":
            raise ValueError(f"unsupported ExecutionRecord schema_version: {version!r}")
        unknown = sorted(set(raw) - _FIELDS)
        if unknown:
            raise ValueError(f"unknown ExecutionRecord fields: {unknown}")
        missing = sorted(_REQUIRED - set(raw))
        if missing:
            raise ValueError(f"missing ExecutionRecord fields: {missing}")
        for name in (
            "task_id",
            "run_id",
            "model",
            "provider",
            "harness",
            "started_at",
            "finished_at",
        ):
            _string(raw, name)
        if raw["status"] not in {"completed", "failed", "cancelled", "timeout"}:
            raise ValueError("ExecutionRecord.status is invalid")
        if raw["isolation_level"] not in {"none", "workspace", "os"}:
            raise ValueError("ExecutionRecord.isolation_level is invalid")
        for name in ("input_tokens", "output_tokens", "cached_tokens"):
            value = raw[name]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise TypeError(f"ExecutionRecord.{name} must be a non-negative integer")
        for name in ("latency_seconds", "cost_usd"):
            value = raw[name]
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            ):
                raise TypeError(f"ExecutionRecord.{name} must be a non-negative number")
        if raw["cached_tokens"] > raw["input_tokens"]:
            raise ValueError("ExecutionRecord.cached_tokens cannot exceed input_tokens")
        if not isinstance(raw["output"], str):
            raise TypeError("ExecutionRecord.output must be a string")
        if not isinstance(raw["tool_calls"], list) or not all(
            isinstance(item, dict) for item in raw["tool_calls"]
        ):
            raise TypeError("ExecutionRecord.tool_calls must be a list of objects")
        if not isinstance(raw["files_changed"], list) or not all(
            isinstance(item, str) for item in raw["files_changed"]
        ):
            raise TypeError("ExecutionRecord.files_changed must be a list of strings")
        if raw.get("workspace_root") is not None and not isinstance(raw["workspace_root"], str):
            raise TypeError("ExecutionRecord.workspace_root must be a string or null")
        if not isinstance(raw["metadata"], dict):
            raise TypeError("ExecutionRecord.metadata must be an object")
        return cls(copy.deepcopy(raw))

    @property
    def trial(self) -> int:
        value = self.raw["metadata"].get("trial", 1)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise TypeError("ExecutionRecord.metadata.trial must be a positive integer")
        return value

    def to_run_record(self) -> RunRecord:
        raw = self.raw
        metadata = dict(raw["metadata"])
        metadata.update(
            {
                "execution_schema_version": "0.1",
                "started_at": raw["started_at"],
                "finished_at": raw["finished_at"],
            }
        )
        return RunRecord(
            task_id=raw["task_id"],
            model=raw["model"],
            provider=raw["provider"],
            harness=raw["harness"],
            trial=self.trial,
            output=raw["output"],
            tool_calls=copy.deepcopy(raw["tool_calls"]),
            files_changed=list(raw["files_changed"]),
            input_tokens=raw["input_tokens"],
            output_tokens=raw["output_tokens"],
            cached_tokens=raw["cached_tokens"],
            cost_usd=float(raw["cost_usd"]),
            latency_seconds=float(raw["latency_seconds"]),
            exit_status=raw["status"],
            error=None
            if raw["status"] == "completed"
            else str(metadata.get("failure_reason", raw["status"])),
            run_id=raw["run_id"],
            workspace_root=raw.get("workspace_root"),
            isolation_level=raw["isolation_level"],
            metadata=metadata,
        )


def load_execution_records(path: str | Path) -> list[ExecutionRecord]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = (
        raw
        if isinstance(raw, list)
        else raw.get("records", [raw])
        if isinstance(raw, dict)
        else None
    )
    if not isinstance(rows, list) or not rows:
        raise ValueError("ExecutionRecord input must contain at least one record")
    records = [ExecutionRecord.from_mapping(row) for row in rows]
    keys = [(record.raw["task_id"], record.trial) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate ExecutionRecord task_id/trial")
    run_ids = [record.raw["run_id"] for record in records]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("duplicate ExecutionRecord run_id")
    return records


class ExecutionRecordAdapter:
    """Convert strict external records into AE's independent RunRecord model."""

    def __init__(self, records: list[ExecutionRecord]) -> None:
        converted = [record.to_run_record() for record in records]
        self._records = {(record.task_id, record.trial): record for record in converted}
        first = converted[0]
        self.model = first.model
        self.provider = first.provider
        self.harness = first.harness
        roots = {record.workspace_root for record in converted if record.workspace_root}
        self.workspace_root = Path(next(iter(roots))).resolve() if len(roots) == 1 else None
        levels = {record.isolation_level for record in converted}
        self.isolation_level = next(iter(levels)) if len(levels) == 1 else "none"

    def run(self, task: TaskCase, trial: int) -> RunRecord:
        try:
            return copy.deepcopy(self._records[(task.id, trial)])
        except KeyError as exc:
            raise KeyError(f"missing ExecutionRecord for {task.id} trial {trial}") from exc

    def cleanup(self, record: RunRecord) -> None:
        return None
