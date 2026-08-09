"""Stable contracts for real agent evaluation runs."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


def _is_finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


@dataclass(slots=True)
class TaskCase:
    id: str
    prompt: str
    category: str = "general"
    description: str = ""
    allowed_files: list[str] = field(default_factory=list)
    forbidden_files: list[str] = field(default_factory=list)
    forbidden_actions: list[str] = field(default_factory=list)
    success_criteria: dict[str, list[str]] = field(default_factory=dict)
    limits: dict[str, float | int] = field(default_factory=dict)
    trials: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> TaskCase:
        if not isinstance(raw, dict):
            raise TypeError("task must be an object")
        if not isinstance(raw.get("id"), str) or not raw["id"].strip():
            raise TypeError("task.id must be a non-empty string")

        def string_field(name: str, default: str) -> str:
            value = raw.get(name, default)
            if not isinstance(value, str):
                raise TypeError(f"task.{name} must be a string")
            return value

        def string_list(name: str) -> list[str]:
            value = raw.get(name, [])
            if not isinstance(value, list) or not all(
                isinstance(item, str) for item in value
            ):
                raise TypeError(f"task.{name} must be a list of strings")
            return list(value)

        success_criteria = raw.get("success_criteria", {})
        if not isinstance(success_criteria, dict) or not all(
            isinstance(key, str)
            and isinstance(value, list)
            and all(isinstance(item, str) for item in value)
            for key, value in success_criteria.items()
        ):
            raise TypeError(
                "task.success_criteria must map strings to lists of strings"
            )
        limits = raw.get("limits", {})
        if not isinstance(limits, dict) or not all(
            isinstance(key, str)
            and _is_finite_number(value)
            and value >= 0
            for key, value in limits.items()
        ):
            raise TypeError("task.limits must map strings to non-negative numbers")
        trials = raw.get("trials", 1)
        if not isinstance(trials, int) or isinstance(trials, bool) or trials < 1:
            raise TypeError("task.trials must be a positive integer")
        known = {
            "id", "prompt", "category", "description", "allowed_files",
            "forbidden_files", "forbidden_actions", "success_criteria",
            "limits", "trials",
        }
        return cls(
            id=raw["id"],
            prompt=string_field("prompt", ""),
            category=string_field("category", "general"),
            description=string_field("description", ""),
            allowed_files=string_list("allowed_files"),
            forbidden_files=string_list("forbidden_files"),
            forbidden_actions=string_list("forbidden_actions"),
            success_criteria={key: list(value) for key, value in success_criteria.items()},
            limits=dict(limits),
            trials=trials,
            metadata={key: value for key, value in raw.items() if key not in known},
        )


@dataclass(slots=True)
class RunRecord:
    """Provider-neutral record of one real model/harness trial."""

    task_id: str
    model: str
    provider: str
    harness: str
    trial: int
    output: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    exit_status: str = "completed"
    error: str | None = None
    run_id: str | None = None
    dataset_version: str = ""
    sandbox_id: str = ""
    workspace_root: str | None = None
    isolation_level: str = "none"
    metadata: dict[str, Any] = field(default_factory=dict)

    def integrity_errors(self) -> list[str]:
        """Return schema and accounting invariant violations without raising."""
        errors: list[str] = []
        if not isinstance(self.output, str):
            errors.append("output must be a string")
        if self.error is not None and not isinstance(self.error, str):
            errors.append("error must be a string or null")
        if self.run_id is not None and not isinstance(self.run_id, str):
            errors.append("run_id must be a string or null")
        for name in ("task_id", "model", "provider", "harness"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{name} must be a non-empty string")
        if not isinstance(self.trial, int) or isinstance(self.trial, bool) or self.trial < 1:
            errors.append("trial must be a positive integer")
        if not isinstance(self.dataset_version, str):
            errors.append("dataset_version must be a string")
        if not isinstance(self.sandbox_id, str):
            errors.append("sandbox_id must be a string")
        if self.workspace_root is not None and not isinstance(self.workspace_root, str):
            errors.append("workspace_root must be a string or null")
        if self.isolation_level not in {"none", "workspace", "os"}:
            errors.append("isolation_level is invalid")
        if not isinstance(self.tool_calls, list) or not all(
            isinstance(item, dict) for item in self.tool_calls
        ):
            errors.append("tool_calls must be a list of objects")
        elif any(
            key in item and not isinstance(item[key], dict)
            for item in self.tool_calls
            for key in ("arguments", "args")
        ):
            errors.append("tool call arguments must be objects")
        if not isinstance(self.files_changed, list) or not all(
            isinstance(item, str) for item in self.files_changed
        ):
            errors.append("files_changed must be a list of strings")
        elif any("\x00" in item for item in self.files_changed):
            errors.append("files_changed paths must not contain NUL")
        if not isinstance(self.trajectory, list) or not all(
            isinstance(item, dict) for item in self.trajectory
        ):
            errors.append("trajectory must be a list of objects")
        if not isinstance(self.metadata, dict):
            errors.append("metadata must be an object")
        for name in ("input_tokens", "output_tokens", "cached_tokens"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                errors.append(f"{name} must be a non-negative integer")
        for name in ("cost_usd", "latency_seconds"):
            value = getattr(self, name)
            if not _is_finite_number(value) or value < 0:
                errors.append(f"{name} must be a finite non-negative number")
        if (
            isinstance(self.cached_tokens, int)
            and isinstance(self.input_tokens, int)
            and self.cached_tokens > self.input_tokens
        ):
            errors.append("cached_tokens cannot exceed input_tokens")
        if self.exit_status not in {"completed", "failed", "cancelled", "timeout"}:
            errors.append("exit_status is invalid")
        if self.exit_status == "completed" and not self.run_id:
            errors.append("completed record requires run_id")
        if isinstance(self.metadata, dict):
            for name in ("retries", "sub_agents"):
                if name not in self.metadata:
                    continue
                value = self.metadata[name]
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    errors.append(f"metadata.{name} must be a non-negative integer")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RunRecord:
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in raw.items() if key in fields})


@dataclass(slots=True)
class EvaluatedRun:
    record: RunRecord
    passed: bool
    layers: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "record": self.record.to_dict(),
            "passed": self.passed,
            "layers": self.layers,
        }
