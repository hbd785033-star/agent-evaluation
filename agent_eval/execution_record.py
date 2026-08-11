"""Strict ExecutionRecord 0.1 ingestion boundary."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from datetime import datetime
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


def _timestamp(raw: dict[str, Any], name: str) -> datetime:
    value = _string(raw, name)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"ExecutionRecord.{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"ExecutionRecord.{name} must include a timezone")
    return parsed


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
        ):
            _string(raw, name)
        started_at = _timestamp(raw, "started_at")
        finished_at = _timestamp(raw, "finished_at")
        if finished_at < started_at:
            raise ValueError("ExecutionRecord.finished_at must not precede started_at")
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

    def to_run_record(
        self,
        *,
        trusted_workspace_root: Path | None = None,
        trusted_isolation_level: str = "none",
    ) -> RunRecord:
        raw = self.raw
        metadata = dict(raw["metadata"])
        metadata.update(
            {
                "execution_schema_version": "0.1",
                "started_at": raw["started_at"],
                "finished_at": raw["finished_at"],
                "claimed_workspace_root": raw.get("workspace_root"),
                "claimed_isolation_level": raw["isolation_level"],
            }
        )
        if trusted_isolation_level not in {"none", "workspace", "os"}:
            raise ValueError("trusted_isolation_level is invalid")
        workspace_root: str | None = None
        claimed_workspace = raw.get("workspace_root")
        if trusted_workspace_root is not None and claimed_workspace:
            trust_root = trusted_workspace_root.resolve()
            candidate = Path(claimed_workspace).resolve(strict=False)
            try:
                candidate.relative_to(trust_root)
            except ValueError as exc:
                raise ValueError("ExecutionRecord workspace_root escapes trusted root") from exc
            workspace_root = str(candidate)
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
            workspace_root=workspace_root,
            isolation_level=trusted_isolation_level,
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

    def __init__(
        self,
        records: list[ExecutionRecord],
        *,
        trusted_workspace_root: str | Path | None = None,
        trusted_isolation_level: str = "none",
    ) -> None:
        trust_root = (
            Path(trusted_workspace_root).resolve() if trusted_workspace_root is not None else None
        )
        converted = [
            record.to_run_record(
                trusted_workspace_root=trust_root,
                trusted_isolation_level=trusted_isolation_level,
            )
            for record in records
        ]
        self._records = {(record.task_id, record.trial): record for record in converted}
        first = converted[0]
        self.model = first.model
        self.provider = first.provider
        self.harness = first.harness
        self.workspace_root = trust_root
        self.isolation_level = trusted_isolation_level

    def run(self, task: TaskCase, trial: int) -> RunRecord:
        try:
            return copy.deepcopy(self._records[(task.id, trial)])
        except KeyError as exc:
            raise KeyError(f"missing ExecutionRecord for {task.id} trial {trial}") from exc

    def cleanup(self, record: RunRecord) -> None:
        return None
