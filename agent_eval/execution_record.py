"""Strict, truth-preserving ExecutionRecord 0.1 ingestion boundary."""

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
_REQUIRED = {"task_id", "status", "started_at", "finished_at", "latency_seconds"}
_DEFAULTS: dict[str, Any] = {
    "schema_version": "0.1",
    "run_id": None,
    "model": None,
    "provider": None,
    "harness": "adaptive-agent-orchestrator",
    "input_tokens": None,
    "output_tokens": None,
    "cached_tokens": None,
    "cost_usd": None,
    "tool_calls": None,
    "files_changed": None,
    "output": None,
    "workspace_root": None,
    "isolation_level": None,
    "metadata": {},
}
_EXPERIMENT_LABEL = "execution-record"


def _string(raw: dict[str, Any], name: str) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"ExecutionRecord.{name} must be a non-empty string")
    return value


def _optional_string(raw: dict[str, Any], name: str) -> str | None:
    value = raw.get(name)
    if value is not None and not isinstance(value, str):
        raise TypeError(f"ExecutionRecord.{name} must be a string or null")
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


def _optional_nonnegative_int(raw: dict[str, Any], name: str) -> int | None:
    value = raw.get(name)
    if value is not None and (
        not isinstance(value, int) or isinstance(value, bool) or value < 0
    ):
        raise TypeError(f"ExecutionRecord.{name} must be a non-negative integer or null")
    return value


def _nonnegative_number(raw: dict[str, Any], name: str) -> int | float:
    value = raw.get(name)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise TypeError(f"ExecutionRecord.{name} must be a non-negative number")
    return value


def _optional_nonnegative_number(raw: dict[str, Any], name: str) -> int | float | None:
    value = raw.get(name)
    if value is not None:
        _nonnegative_number(raw, name)
    return value


def _nested_string(raw: Any, *path: str) -> str | None:
    current = raw
    for name in path:
        if not isinstance(current, dict):
            return None
        current = current.get(name)
    return current if isinstance(current, str) and current.strip() else None


@dataclass(frozen=True, slots=True)
class WorkspaceAuthority:
    """Control-plane workspace bindings for imported execution records."""

    trusted_workspace_root: Path
    isolation_level: str
    workspaces: dict[tuple[str, int], Path]
    schema_version: str = "0.1"

    @classmethod
    def from_mapping(cls, raw: Any) -> WorkspaceAuthority:
        if not isinstance(raw, dict):
            raise TypeError("workspace authority must be an object")
        expected = {
            "schema_version",
            "trusted_workspace_root",
            "isolation_level",
            "workspaces",
        }
        unknown = sorted(set(raw) - expected)
        missing = sorted(expected - set(raw))
        if unknown or missing:
            raise ValueError(
                f"invalid workspace authority fields: missing={missing}, unknown={unknown}"
            )
        if raw["schema_version"] != "0.1":
            raise ValueError("workspace authority schema_version must be 0.1")
        root_raw = raw["trusted_workspace_root"]
        if not isinstance(root_raw, str) or not root_raw.strip():
            raise TypeError("workspace authority trusted_workspace_root must be a string")
        root = Path(root_raw).resolve()
        if not root.is_dir():
            raise ValueError("workspace authority trusted_workspace_root must exist")
        isolation_level = raw["isolation_level"]
        if isolation_level not in {"none", "workspace", "os"}:
            raise ValueError("workspace authority isolation_level is invalid")
        rows = raw["workspaces"]
        if not isinstance(rows, list) or not rows:
            raise TypeError("workspace authority workspaces must be a non-empty list")
        workspaces: dict[tuple[str, int], Path] = {}
        for row in rows:
            if not isinstance(row, dict) or set(row) != {
                "task_id",
                "trial",
                "workspace_root",
            }:
                raise ValueError("workspace authority entry fields are invalid")
            task_id = row["task_id"]
            trial = row["trial"]
            workspace_raw = row["workspace_root"]
            if not isinstance(task_id, str) or not task_id.strip():
                raise TypeError("workspace authority task_id must be a non-empty string")
            if not isinstance(trial, int) or isinstance(trial, bool) or trial < 1:
                raise TypeError("workspace authority trial must be a positive integer")
            if not isinstance(workspace_raw, str) or not workspace_raw.strip():
                raise TypeError("workspace authority workspace_root must be a string")
            workspace = Path(workspace_raw).resolve()
            try:
                workspace.relative_to(root)
            except ValueError as exc:
                raise ValueError("workspace authority entry escapes trusted root") from exc
            if not workspace.is_dir():
                raise ValueError("workspace authority workspace_root must exist")
            key = (task_id, trial)
            if key in workspaces:
                raise ValueError(f"duplicate workspace authority binding: {key}")
            workspaces[key] = workspace
        return cls(root, isolation_level, workspaces)

    @classmethod
    def load(cls, path: str | Path) -> WorkspaceAuthority:
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    raw: dict[str, Any]

    @classmethod
    def from_mapping(cls, raw: Any) -> ExecutionRecord:
        if not isinstance(raw, dict):
            raise TypeError("ExecutionRecord must be an object")
        unknown = sorted(set(raw) - _FIELDS)
        if unknown:
            raise ValueError(f"unknown ExecutionRecord fields: {unknown}")
        missing = sorted(_REQUIRED - set(raw))
        if missing:
            raise ValueError(f"missing ExecutionRecord fields: {missing}")
        normalized = copy.deepcopy(_DEFAULTS)
        normalized.update(copy.deepcopy(raw))
        version = normalized.get("schema_version")
        if version != "0.1":
            raise ValueError(f"unsupported ExecutionRecord schema_version: {version!r}")
        _string(normalized, "task_id")
        _string(normalized, "harness")
        for name in ("run_id", "model", "provider"):
            _optional_string(normalized, name)
        started_at = _timestamp(normalized, "started_at")
        finished_at = _timestamp(normalized, "finished_at")
        if finished_at < started_at:
            raise ValueError("ExecutionRecord.finished_at must not precede started_at")
        if normalized["status"] not in {"completed", "failed", "cancelled", "timeout"}:
            raise ValueError("ExecutionRecord.status is invalid")
        isolation = normalized["isolation_level"]
        if isolation is not None and isolation not in {"none", "workspace", "os"}:
            raise ValueError("ExecutionRecord.isolation_level is invalid")
        for name in ("input_tokens", "output_tokens", "cached_tokens"):
            _optional_nonnegative_int(normalized, name)
        _nonnegative_number(normalized, "latency_seconds")
        _optional_nonnegative_number(normalized, "cost_usd")
        cached = normalized["cached_tokens"]
        inputs = normalized["input_tokens"]
        if cached is not None and inputs is not None and cached > inputs:
            raise ValueError("ExecutionRecord.cached_tokens cannot exceed input_tokens")
        if normalized["output"] is not None and not isinstance(normalized["output"], str):
            raise TypeError("ExecutionRecord.output must be a string or null")
        tool_calls = normalized["tool_calls"]
        if tool_calls is not None and (
            not isinstance(tool_calls, list)
            or not all(isinstance(item, dict) for item in tool_calls)
        ):
            raise TypeError("ExecutionRecord.tool_calls must be a list of objects or null")
        files_changed = normalized["files_changed"]
        if files_changed is not None and (
            not isinstance(files_changed, list)
            or not all(isinstance(item, str) for item in files_changed)
        ):
            raise TypeError("ExecutionRecord.files_changed must be a list of strings or null")
        if normalized["workspace_root"] is not None and not isinstance(
            normalized["workspace_root"], str
        ):
            raise TypeError("ExecutionRecord.workspace_root must be a string or null")
        if not isinstance(normalized["metadata"], dict):
            raise TypeError("ExecutionRecord.metadata must be an object")
        return cls(normalized)

    @property
    def trial(self) -> int:
        value = self.raw["metadata"].get("trial", 1)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise TypeError("ExecutionRecord.metadata.trial must be a positive integer")
        return value

    def to_run_record(
        self,
        *,
        authoritative_workspace: Path | None = None,
        authoritative_isolation_level: str = "none",
        experiment_model: str = _EXPERIMENT_LABEL,
        experiment_provider: str = _EXPERIMENT_LABEL,
        experiment_harness: str = _EXPERIMENT_LABEL,
    ) -> RunRecord:
        raw = self.raw
        source_metadata = copy.deepcopy(raw["metadata"])
        planned_runtime = _nested_string(source_metadata, "planned", "runtime_plan", "executor")
        selected_runtime = _nested_string(
            source_metadata, "planned", "runtime_selection", "selected_runtime"
        )
        observed = source_metadata.get("observed")
        observed_runtime = (
            _nested_string(observed, "runtime_adapter")
            if isinstance(observed, dict) and observed.get("runtime_adapter_invoked") is True
            else None
        )
        metadata = dict(source_metadata)
        metadata.update(
            {
                "execution_schema_version": "0.1",
                "started_at": raw["started_at"],
                "finished_at": raw["finished_at"],
                "claimed_workspace_root": raw.get("workspace_root"),
                "claimed_isolation_level": raw.get("isolation_level"),
                "experiment_identity": {
                    "model": experiment_model,
                    "provider": experiment_provider,
                    "harness": experiment_harness,
                },
                "evidence_completeness": {
                    "execution_identity": "partial" if raw["run_id"] is None else "complete",
                    "usage": (
                        "complete"
                        if all(raw[name] is not None for name in (
                            "input_tokens",
                            "output_tokens",
                            "cached_tokens",
                            "cost_usd",
                        ))
                        else "partial"
                    ),
                    "tool_calls": "unavailable" if raw["tool_calls"] is None else "complete",
                    "files_changed": (
                        "unavailable" if raw["files_changed"] is None else "complete"
                    ),
                    "output": "unavailable" if raw["output"] is None else "complete",
                    "trajectory": "unavailable",
                },
            }
        )
        if authoritative_isolation_level not in {"none", "workspace", "os"}:
            raise ValueError("authoritative_isolation_level is invalid")
        workspace_root = (
            str(authoritative_workspace.resolve()) if authoritative_workspace is not None else None
        )
        return RunRecord(
            task_id=raw["task_id"],
            model=experiment_model,
            provider=experiment_provider,
            harness=experiment_harness,
            trial=self.trial,
            output=raw["output"],
            tool_calls=copy.deepcopy(raw["tool_calls"]),
            files_changed=copy.deepcopy(raw["files_changed"]),
            trajectory=None,
            input_tokens=raw["input_tokens"],
            output_tokens=raw["output_tokens"],
            cached_tokens=raw["cached_tokens"],
            cost_usd=(float(raw["cost_usd"]) if raw["cost_usd"] is not None else None),
            cost_semantics=("producer_estimated" if raw["cost_usd"] is not None else None),
            latency_seconds=float(raw["latency_seconds"]),
            exit_status=raw["status"],
            error=(
                None
                if raw["status"] == "completed"
                else str(metadata.get("failure_reason") or raw["status"])
            ),
            run_id=raw["run_id"],
            observed_model=raw["model"],
            observed_provider=raw["provider"],
            observed_harness=raw["harness"],
            planned_runtime=planned_runtime,
            selected_runtime=selected_runtime,
            observed_runtime=observed_runtime,
            workspace_root=workspace_root,
            isolation_level=authoritative_isolation_level,
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
    run_ids = [
        record.raw["run_id"]
        for record in records
        if isinstance(record.raw["run_id"], str) and record.raw["run_id"].strip()
    ]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("duplicate ExecutionRecord run_id")
    return records


class ExecutionRecordAdapter:
    """Convert external records while keeping experiment and execution identity separate."""

    def __init__(
        self,
        records: list[ExecutionRecord],
        *,
        authority: WorkspaceAuthority | None = None,
    ) -> None:
        converted: list[RunRecord] = []
        seen_keys: set[tuple[str, int]] = set()
        for record in records:
            key = (record.raw["task_id"], record.trial)
            if key in seen_keys:
                raise ValueError(f"duplicate ExecutionRecord task_id/trial: {key}")
            seen_keys.add(key)
            if authority is not None and key not in authority.workspaces:
                raise ValueError(f"workspace authority missing binding: {key}")
            converted.append(
                record.to_run_record(
                    authoritative_workspace=(
                        authority.workspaces[key] if authority is not None else None
                    ),
                    authoritative_isolation_level=(
                        authority.isolation_level if authority is not None else "none"
                    ),
                )
            )
        if not converted:
            raise ValueError("ExecutionRecordAdapter requires at least one record")
        self._records = {(record.task_id, record.trial): record for record in converted}
        self.model = _EXPERIMENT_LABEL
        self.provider = _EXPERIMENT_LABEL
        self.harness = _EXPERIMENT_LABEL
        self.workspace_root = authority.trusted_workspace_root if authority is not None else None
        self.isolation_level = authority.isolation_level if authority is not None else "none"

    def run(self, task: TaskCase, trial: int) -> RunRecord:
        try:
            return copy.deepcopy(self._records[(task.id, trial)])
        except KeyError as exc:
            raise KeyError(f"missing ExecutionRecord for {task.id} trial {trial}") from exc

    def cleanup(self, record: RunRecord) -> None:
        return None
